from __future__ import annotations

import time

import pytest

from config import Settings
from state import ApprovalDecision, RunRequest, RunStatus
from tests.conftest import (
    ALERT_TASK,
    CRM_TASK,
    EMAIL_TASK,
    GITHUB_TASK,
    RESEARCH_TASK,
)
from tools import CodeFixTool, ToolRegistry
from workflow import Orchestrator


def test_research_run_completes(orchestrator: Orchestrator) -> None:
    response = orchestrator.start(RunRequest(task=RESEARCH_TASK, auto_approve=True))
    assert response.status == RunStatus.COMPLETED
    assert response.task_type == "research"
    assert response.state["attempts"] == 1
    output = response.state["output"]
    assert output["kind"] == "report"
    assert "Comparison" in output["body"]
    assert "Recommendation" in output["body"]
    assert response.state["validation"]["valid"] is True
    assert {log["agent"] for log in response.state["logs"]} >= {
        "reader",
        "planner",
        "executor",
        "validator",
        "output",
    }


def test_github_run_pauses_then_approves(orchestrator: Orchestrator) -> None:
    response = orchestrator.start(RunRequest(task=GITHUB_TASK, auto_approve=False))
    assert response.status == RunStatus.AWAITING_APPROVAL
    assert response.approval is not None
    assert response.approval.risk_level == "medium"
    assert response.state["output"] == {}
    assert response.state["approval_status"] == "pending"
    waiting = [log for log in response.state["logs"] if log["event"] == "awaiting_approval"]
    assert waiting

    resumed = orchestrator.approve(
        ApprovalDecision(run_id=response.run_id, approved=True, reviewer="qa")
    )
    assert resumed.status == RunStatus.COMPLETED
    assert resumed.state["approval_status"] == "approved"
    output = resumed.state["output"]
    assert output["kind"] == "pull_request"
    assert output["links"][0]["url"].startswith("https://github.com/acme/webapp/pull/")
    artifact = resumed.state["result"]["artifact"]
    assert artifact["diff"].startswith("--- a/app/users.py")
    assert "+++ b/app/users.py" in artifact["diff"]

    with pytest.raises(Exception):
        orchestrator.approve(
            ApprovalDecision(run_id=response.run_id, approved=True)
        )


def test_rejection_produces_no_deliverable(orchestrator: Orchestrator) -> None:
    response = orchestrator.start(RunRequest(task=EMAIL_TASK, auto_approve=False))
    assert response.status == RunStatus.AWAITING_APPROVAL
    rejected = orchestrator.approve(
        ApprovalDecision(
            run_id=response.run_id,
            approved=False,
            reviewer="legal",
            comment="Needs a policy review.",
        )
    )
    assert rejected.status == RunStatus.REJECTED
    assert rejected.state["output"] == {}
    assert rejected.state["approval"]["reviewer"] == "legal"
    assert rejected.state["approval_status"] == "rejected"


def test_auto_approve_skips_interrupt(orchestrator: Orchestrator) -> None:
    response = orchestrator.start(RunRequest(task=EMAIL_TASK, auto_approve=True))
    assert response.status == RunStatus.COMPLETED
    assert response.state["approval_status"] == "auto_approved"
    assert response.state["output"]["kind"] == "email"


def test_alert_and_crm_run_without_approval(orchestrator: Orchestrator) -> None:
    alert = orchestrator.start(RunRequest(task=ALERT_TASK, auto_approve=False))
    assert alert.status == RunStatus.COMPLETED
    assert alert.state["output"]["kind"] == "triage"

    crm = orchestrator.start(RunRequest(task=CRM_TASK, auto_approve=False))
    assert crm.status == RunStatus.COMPLETED
    assert crm.state["output"]["kind"] == "crm"
    contacts = crm.state["result"]["artifact"]["contacts"]
    assert contacts and contacts[0]["company"] == "Acme Corp"


class FlakyCodeFixTool(CodeFixTool):
    """Returns an incomplete artifact on the first call to exercise corrections."""

    def __init__(self) -> None:
        self.calls = 0

    def run(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        result = super().run(**kwargs)
        if self.calls == 1:
            artifact = dict(result.artifact)
            artifact["diff"] = ""
            return result.model_copy(update={"artifact": artifact})
        return result


def test_validator_drives_executor_retry(settings: Settings) -> None:
    registry = ToolRegistry(settings)
    flaky = FlakyCodeFixTool()
    registry.register(flaky)
    orchestrator = Orchestrator(settings=settings, tools=registry, llm=None)
    response = orchestrator.start(
        RunRequest(task=GITHUB_TASK, auto_approve=True, max_attempts=3)
    )
    assert response.status == RunStatus.COMPLETED
    assert response.state["attempts"] == 2
    assert flaky.calls == 2
    executor_logs = [
        log
        for log in response.state["logs"]
        if log["agent"] == "executor" and log["event"] == "completed"
    ]
    assert len(executor_logs) == 2


class SlowTool(CodeFixTool):
    def run(self, **kwargs):  # type: ignore[no-untyped-def]
        time.sleep(0.4)
        return super().run(**kwargs)


def test_agent_timeout_marks_run_failed(settings: Settings) -> None:
    fast_timeout = Settings(
        crm_db_path=settings.crm_db_path,
        agent_max_retries=0,
        agent_timeout_seconds=0.05,
        max_attempts=1,
    )
    registry = ToolRegistry(fast_timeout)
    registry.register(SlowTool())
    orchestrator = Orchestrator(settings=fast_timeout, tools=registry, llm=None)
    response = orchestrator.start(RunRequest(task=GITHUB_TASK, auto_approve=True))
    assert response.status == RunStatus.FAILED
    assert "timeout" in (response.error or "").lower()
    assert any(
        log["event"] == "failed" and log["agent"] == "executor"
        for log in response.state["logs"]
    )


def test_unknown_run_raises(orchestrator: Orchestrator) -> None:
    with pytest.raises(Exception):
        orchestrator.get("does-not-exist")
