from __future__ import annotations

import pytest

from agents import PlannerAgent, ReaderAgent, ValidatorAgent
from config import Settings
from state import (
    ActionResult,
    Plan,
    PlanStep,
    ReaderOutput,
    Strategy,
    TaskType,
    initial_state,
)
from tests.conftest import ALERT_TASK, CRM_TASK, EMAIL_TASK, GITHUB_TASK, RESEARCH_TASK


@pytest.fixture()
def reader(settings: Settings) -> ReaderAgent:
    return ReaderAgent(settings)


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        (GITHUB_TASK, TaskType.GITHUB_ISSUE),
        (EMAIL_TASK, TaskType.EMAIL),
        (RESEARCH_TASK, TaskType.RESEARCH),
        (ALERT_TASK, TaskType.ALERT),
        (CRM_TASK, TaskType.CRM),
        ("Write a short welcome message for new employees.", TaskType.GENERIC),
    ],
)
def test_reader_classifies_task_types(reader: ReaderAgent, task: str, expected: TaskType) -> None:
    output = reader.run(initial_state(task))
    assert isinstance(output, ReaderOutput)
    assert output.task_type == expected


def test_reader_extracts_entities_and_risk(reader: ReaderAgent) -> None:
    github = reader.run(initial_state(GITHUB_TASK))
    assert github.entities["repos"] == ["acme/webapp"]
    assert github.entities["files"] == ["app/users.py"]
    assert github.entities["error_type"] == "TypeError"
    assert github.needs_approval is True

    email = reader.run(initial_state(EMAIL_TASK))
    assert "priya@atlas.example" in email.entities["emails"]
    assert email.entities["subject"] == "Refund request for duplicate charge"
    assert email.entities["sender_name"] == "Priya Nair"
    assert email.risk_level.value == "high"
    assert email.needs_approval is True


def test_planner_uses_complex_strategy_for_github(reader: ReaderAgent, settings: Settings) -> None:
    planner = PlannerAgent(settings)
    state = initial_state(GITHUB_TASK)
    state.context = reader.run(state).model_dump(mode="json")
    plan = planner.run(state)
    assert isinstance(plan, Plan)
    assert plan.strategy == Strategy.COMPLEX
    assert [step.tool for step in plan.steps][:2] == ["code_fix", "test_plan"]
    assert "review" in [step.tool for step in plan.steps]
    assert plan.requires_approval is True


def test_planner_uses_simple_strategy_for_research(reader: ReaderAgent, settings: Settings) -> None:
    planner = PlannerAgent(settings)
    state = initial_state("Research the trade-offs between REST and gRPC.")
    state.context = reader.run(state).model_dump(mode="json")
    plan = planner.run(state)
    assert plan.strategy in {Strategy.SIMPLE, Strategy.COMPLEX}
    assert plan.steps[0].tool == "research"
    assert plan.requires_approval is False


def test_validator_rejects_incomplete_email(settings: Settings) -> None:
    validator = ValidatorAgent(settings)
    state = initial_state(EMAIL_TASK)
    state.plan = Plan(
        strategy=Strategy.SIMPLE,
        steps=[PlanStep(id=1, description="Draft a reply", tool="email_reply")],
        validation_criteria=["Recipient and subject are present"],
        requires_approval=True,
    ).model_dump(mode="json")
    state.result = ActionResult(
        action="email_reply",
        tool="email_reply",
        success=True,
        summary="draft",
        artifact={"to": "", "subject": "", "body": "short"},
    ).model_dump(mode="json")
    result = validator.offline(state)
    assert result.valid is False
    assert result.score <= 0.5
    assert any("recipient" in issue.lower() for issue in result.issues)
    assert "TODO" not in result.feedback


def test_validator_accepts_complete_research(settings: Settings) -> None:
    validator = ValidatorAgent(settings)
    state = initial_state(RESEARCH_TASK)
    state.plan = Plan(
        strategy=Strategy.SIMPLE,
        steps=[PlanStep(id=1, description="Research", tool="research")],
        validation_criteria=["Structured brief"],
    ).model_dump(mode="json")
    body = "## One\n" + ("A" * 120) + "\n## Two\n" + ("B" * 120) + "\n## Three\n" + ("C" * 120)
    state.result = ActionResult(
        action="research",
        tool="research",
        success=True,
        summary="brief",
        artifact={
            "body": body,
            "findings": ["one", "two", "three"],
            "recommendation": "Run a bounded pilot with clear metrics.",
            "limitations": "Offline synthesis.",
        },
    ).model_dump(mode="json")
    result = validator.offline(state)
    assert result.valid is True
    assert result.score == 1.0
