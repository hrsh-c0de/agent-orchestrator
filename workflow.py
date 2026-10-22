"""LangGraph orchestration: routing, retries, timeouts, logging, approval.

Flow:
    START -> reader -> planner -> executor -> validator
    validator -> executor          (invalid, retry budget left)
    validator -> approval|output   (valid)
    validator -> give_up           (retry budget exhausted)
    approval  -> output | END      (approved / rejected)

The graph state is :class:`state.AgentState`. A checkpointer keyed by run id
keeps paused runs resumable: the approval node raises ``interrupt`` and the API
resumes it with ``Command(resume=...)``.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import datetime, timezone
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents import Agents, build_agents
from config import Settings
from config import settings as default_settings
from llm import LLMClient, build_llm
from persistence import (
    RunRecord,
    RunStore,
    build_checkpointer,
    build_run_store,
)
from state import (
    ActionResult,
    AgentState,
    ApprovalDecision,
    ApprovalPrompt,
    ApprovalStatus,
    RunRequest,
    RunResponse,
    RunStatus,
    initial_state,
    make_log,
    state_json_safe,
    utc_now_iso,
)
from tools import ToolRegistry

logger = logging.getLogger(__name__)

try:  # LangGraph control-flow exception must never be retried or swallowed.
    from langgraph.errors import GraphInterrupt
except ImportError:  # pragma: no cover - older/newer naming
    class GraphInterrupt(BaseException):  # type: ignore[no-redef]
        pass

TERMINAL_STATUSES = {
    RunStatus.COMPLETED.value,
    RunStatus.REJECTED.value,
    RunStatus.FAILED.value,
}


class OrchestrationError(RuntimeError):
    pass


class RunNotFound(OrchestrationError):
    pass


class InvalidRunState(OrchestrationError):
    pass


class AgentTimeout(TimeoutError):
    pass


NodeBody = Callable[[AgentState], tuple[dict[str, Any], str]]


def _get(state: AgentState | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _call_with_timeout(fn: Callable[[], Any], timeout: float) -> Any:
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-call")
    future = pool.submit(fn)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeout as exc:
        future.cancel()
        raise AgentTimeout(f"agent exceeded its {timeout:g}s timeout") from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


class Orchestrator:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        tools: ToolRegistry | None = None,
        agents: Agents | None = None,
        llm: LLMClient | None = None,
        checkpointer: Any | None = None,
        run_store: RunStore | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self.tools = tools or ToolRegistry(self.settings)
        self.agents = agents or build_agents(self.settings, self.tools)
        self.llm = llm if llm is not None else build_llm(self.settings)
        if checkpointer is None:
            self.checkpointer, self._close_checkpointer = build_checkpointer(self.settings)
            self._owns_checkpointer = True
        else:
            self.checkpointer = checkpointer
            self._close_checkpointer = lambda: None
            self._owns_checkpointer = False
        self.run_store = run_store or build_run_store(self.settings)
        self._owns_run_store = run_store is None
        self.graph = self._build_graph()
        self._records: dict[str, RunRecord] = {}
        self._registry_lock = threading.RLock()
        self._graph_lock = threading.RLock()

    def close(self) -> None:
        """Release SQLite connections. Safe to call more than once."""
        if self._owns_checkpointer:
            self._close_checkpointer()
        if self._owns_run_store:
            self.run_store.close()

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("reader", self._wrap_agent("reader", self._reader_body))
        builder.add_node("planner", self._wrap_agent("planner", self._planner_body))
        builder.add_node("executor", self._wrap_agent("executor", self._executor_body))
        builder.add_node("validator", self._wrap_agent("validator", self._validator_body))
        builder.add_node("approval", self._approval_node)
        builder.add_node("output", self._wrap_agent("output", self._output_body))
        builder.add_node("give_up", self._give_up_node)

        builder.add_edge(START, "reader")
        builder.add_conditional_edges(
            "reader", self._route_after_reader, {"planner": "planner", "end": END}
        )
        builder.add_conditional_edges(
            "planner", self._route_after_planner, {"executor": "executor", "end": END}
        )
        builder.add_conditional_edges(
            "executor", self._route_after_executor, {"validator": "validator", "end": END}
        )
        builder.add_conditional_edges(
            "validator",
            self._route_after_validator,
            {
                "executor": "executor",
                "approval": "approval",
                "output": "output",
                "give_up": "give_up",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "approval", self._route_after_approval, {"output": "output", "end": END}
        )
        builder.add_edge("give_up", END)
        builder.add_edge("output", END)
        return builder.compile(checkpointer=self.checkpointer)

    # --------------------------------------------------------------- wrapping
    def _wrap_agent(self, name: str, body: NodeBody):
        def node(state: AgentState) -> dict[str, Any]:
            return self._execute_node(name, body, state)

        node.__name__ = f"{name}_node"
        return node

    def _execute_node(self, name: str, body: NodeBody, state: AgentState) -> dict[str, Any]:
        logs: list[dict[str, Any]] = [
            make_log(name, "started", f"{name.title()} started.", attempt=1)
        ]
        retry_budget = self.settings.agent_max_retries
        attempt = 0

        for attempt in range(1, retry_budget + 2):
            started = time.perf_counter()
            try:
                updates, message = _call_with_timeout(
                    lambda: body(state), self.settings.agent_timeout_seconds
                )
                elapsed = (time.perf_counter() - started) * 1000
                data = updates.pop("_log_data", {})
                logs.append(
                    make_log(
                        name,
                        "completed",
                        message,
                        attempt=attempt,
                        duration_ms=elapsed,
                        **data,
                    )
                )
                merged = dict(updates)
                merged["logs"] = logs
                return merged
            except GraphInterrupt:
                raise
            except Exception as exc:
                elapsed = (time.perf_counter() - started) * 1000
                logs.append(
                    make_log(
                        name,
                        "failed",
                        f"{type(exc).__name__}: {exc}",
                        attempt=attempt,
                        duration_ms=elapsed,
                    )
                )
                logger.warning("%s attempt %s failed: %s", name, attempt, exc)
                if attempt <= retry_budget:
                    delay = min(2.0, 0.3 * (2 ** (attempt - 1)))
                    logs.append(
                        make_log(
                            name,
                            "retry",
                            f"Retrying after {delay:.1f}s (attempt {attempt + 1}/{retry_budget + 1}).",
                            attempt=attempt + 1,
                        )
                    )
                    time.sleep(delay)
                    continue
                return {
                    "status": RunStatus.FAILED.value,
                    "error": f"{name} failed after {attempt} attempt(s): {exc}",
                    "completed_at": utc_now_iso(),
                    "logs": logs,
                }
        # Unreachable, retained for type checkers.
        return {"status": RunStatus.FAILED.value, "error": f"{name} failed", "logs": logs}

    # ------------------------------------------------------------------ nodes
    def _reader_body(self, state: AgentState) -> tuple[dict[str, Any], str]:
        output = self.agents.reader.run(state, self.llm)
        updates = {
            "context": output.model_dump(mode="json"),
            "task_type": output.task_type.value,
            "approval_required": output.needs_approval,
            "status": RunStatus.RUNNING.value,
            "_log_data": {
                "task_type": output.task_type.value,
                "risk_level": output.risk_level.value,
                "needs_approval": output.needs_approval,
                "constraints": output.constraints,
            },
        }
        message = (
            f"Classified as {output.task_type.value} ({output.risk_level.value} risk); "
            f"{len(output.entities)} entity groups, {len(output.constraints)} constraints, "
            f"approval {'required' if output.needs_approval else 'not required'}."
        )
        return updates, message

    def _planner_body(self, state: AgentState) -> tuple[dict[str, Any], str]:
        output = self.agents.planner.run(state, self.llm)
        approval_required = bool(state.approval_required or output.requires_approval)
        updates = {
            "plan": output.model_dump(mode="json"),
            "strategy": output.strategy.value,
            "approval_required": approval_required,
            "status": RunStatus.RUNNING.value,
            "_log_data": {
                "strategy": output.strategy.value,
                "steps": [
                    {"id": step.id, "tool": step.tool, "description": step.description}
                    for step in output.steps
                ],
                "requires_approval": approval_required,
            },
        }
        message = (
            f"{output.strategy.value.title()} strategy with {len(output.steps)} step(s): "
            + " -> ".join(step.tool for step in output.steps)
        )
        return updates, message

    def _executor_body(self, state: AgentState) -> tuple[dict[str, Any], str]:
        feedback = ""
        if state.validation and not state.validation.get("valid", False):
            feedback = str(state.validation.get("feedback", ""))
        output = self.agents.executor.run(state, self.llm, feedback=feedback)
        updates = {
            "action": output.action,
            "result": output.model_dump(mode="json"),
            "status": RunStatus.RUNNING.value,
            "_log_data": {
                "tool": output.tool,
                "success": output.success,
                "steps": [
                    {"id": step.step_id, "tool": step.tool, "summary": step.summary}
                    for step in output.steps
                ],
                "revision": bool(feedback),
            },
        }
        message = (
            f"Executed {len(output.steps)} step(s) via '{output.tool}'"
            + (" using validator feedback" if feedback else "")
            + f". {output.summary[:300]}"
        )
        return updates, message

    def _validator_body(self, state: AgentState) -> tuple[dict[str, Any], str]:
        output = self.agents.validator.run(state, self.llm)
        attempts = state.attempts + 1
        updates = {
            "validation": output.model_dump(mode="json"),
            "attempts": attempts,
            "status": RunStatus.RUNNING.value,
            "_log_data": {
                "valid": output.valid,
                "score": output.score,
                "issues": output.issues,
                "correction_pass": attempts,
            },
        }
        verdict = "passed" if output.valid else "failed"
        message = (
            f"Validation {verdict} on attempt {attempts}/{state.max_attempts} "
            f"(score {output.score:.2f})."
            + (f" Issues: {'; '.join(output.issues[:4])}" if output.issues else "")
        )
        return updates, message

    def _output_body(self, state: AgentState) -> tuple[dict[str, Any], str]:
        output = self.agents.output.run(state, self.llm)
        duration_ms = _elapsed_ms(state.started_at)
        metrics = dict(state.metrics)
        metrics.update(
            {
                "duration_ms": duration_ms,
                "attempts": state.attempts,
                "validation_score": state.validation.get("score"),
                "strategy": state.strategy,
            }
        )
        updates = {
            "output": output.model_dump(mode="json"),
            "status": RunStatus.COMPLETED.value,
            "completed_at": utc_now_iso(),
            "metrics": metrics,
            "_log_data": {
                "kind": output.kind,
                "links": [link.url for link in output.links][:3],
                "confidence": output.confidence,
            },
        }
        message = f"Delivered {output.kind} '{output.title}' with confidence {output.confidence:.2f}."
        return updates, message

    def _give_up_node(self, state: AgentState) -> dict[str, Any]:
        logs = [
            make_log(
                "validator",
                "completed",
                f"Retry budget exhausted after {state.attempts} attempt(s).",
                attempt=state.attempts,
                issues=state.validation.get("issues", []),
            )
        ]
        issues = state.validation.get("issues", [])
        return {
            "status": RunStatus.FAILED.value,
            "error": (
                f"Validation failed after {state.attempts} attempt(s): "
                + ("; ".join(str(issue) for issue in issues[:5]) or "acceptance criteria not met")
            ),
            "completed_at": utc_now_iso(),
            "logs": logs,
        }

    # --------------------------------------------------------------- approval
    def _approval_node(self, state: AgentState) -> dict[str, Any]:
        logs: list[dict[str, Any]] = [
            make_log("approval", "started", "Human approval gate reached.")
        ]
        if not state.approval_required:
            return {
                "approval_status": ApprovalStatus.NOT_REQUIRED.value,
                "status": RunStatus.RUNNING.value,
                "logs": logs,
            }
        if state.auto_approve:
            record = {
                "approved": True,
                "reviewer": "auto-approve",
                "comment": "Requested with auto_approve=true.",
                "decided_at": utc_now_iso(),
            }
            logs.append(
                make_log("approval", "completed", "Auto-approved by request setting.")
            )
            return {
                "approval_status": ApprovalStatus.AUTO_APPROVED.value,
                "approval": record,
                "status": RunStatus.RUNNING.value,
                "logs": logs,
            }

        payload = self._approval_payload(state)
        decision = interrupt(payload)
        if isinstance(decision, dict):
            approved = bool(decision.get("approved"))
            reviewer = str(decision.get("reviewer") or "dashboard-user")
            comment = str(decision.get("comment") or "")
        else:  # defensive: a bare boolean resume value
            approved = bool(decision)
            reviewer = "dashboard-user"
            comment = ""

        record = {
            "approved": approved,
            "reviewer": reviewer,
            "comment": comment,
            "decided_at": utc_now_iso(),
        }
        logs.append(
            make_log(
                "approval",
                "completed",
                f"{'Approved' if approved else 'Rejected'} by {reviewer}."
                + (f" Comment: {comment}" if comment else ""),
                approved=approved,
                reviewer=reviewer,
            )
        )
        return {
            "approval_status": (
                ApprovalStatus.APPROVED.value if approved else ApprovalStatus.REJECTED.value
            ),
            "approval": record,
            "status": RunStatus.RUNNING.value if approved else RunStatus.REJECTED.value,
            "completed_at": None if approved else utc_now_iso(),
            "logs": logs,
        }

    def _approval_payload(self, state: AgentState) -> dict[str, Any]:
        artifact: dict[str, Any] = {}
        try:
            artifact = ActionResult.model_validate(state.result).artifact
        except Exception:
            artifact = state.result.get("artifact", {}) if isinstance(state.result, dict) else {}
        preview = (
            artifact.get("diff")
            or artifact.get("body")
            or state.result.get("summary")
            or state.context.get("summary")
            or state.task
        )
        title = (
            artifact.get("pr_title")
            or artifact.get("subject")
            or state.context.get("summary")
            or state.task[:120]
        )
        return {
            "type": "approval_request",
            "run_id": state.run_id,
            "task_type": state.task_type,
            "title": str(title)[:200],
            "summary": str(
                state.result.get("summary")
                or artifact.get("root_cause")
                or state.context.get("summary")
                or state.task
            )[:500],
            "risk_level": str(state.context.get("risk_level") or "medium"),
            "diff_preview": str(preview)[:1500],
            "validation_score": state.validation.get("score"),
            "requested_at": utc_now_iso(),
        }

    # ---------------------------------------------------------------- routers
    def _route_after_reader(self, state: AgentState) -> str:
        return "end" if self._failed(state) else "planner"

    def _route_after_planner(self, state: AgentState) -> str:
        return "end" if self._failed(state) else "executor"

    def _route_after_executor(self, state: AgentState) -> str:
        return "end" if self._failed(state) else "validator"

    def _route_after_validator(self, state: AgentState) -> str:
        if self._failed(state):
            return "end"
        validation = _get(state, "validation", {}) or {}
        if validation.get("valid"):
            if _get(state, "approval_required", False):
                return "approval"
            return "output"
        attempts = int(_get(state, "attempts", 0) or 0)
        max_attempts = int(_get(state, "max_attempts", 1) or 1)
        if attempts < max_attempts:
            return "executor"
        return "give_up"

    def _route_after_approval(self, state: AgentState) -> str:
        status = _get(state, "status", "")
        if status == RunStatus.REJECTED.value:
            return "end"
        return "output"

    @staticmethod
    def _failed(state: AgentState | dict[str, Any]) -> bool:
        return (
            _get(state, "status") == RunStatus.FAILED.value
            or bool(_get(state, "error"))
        )

    # ------------------------------------------------------------------- runs
    def _config(self, run_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": run_id},
            "recursion_limit": max(50, self.settings.max_attempts * 8 + 30),
        }

    def start(self, request: RunRequest) -> RunResponse:
        auto_approve = (
            self.settings.default_auto_approve
            if request.auto_approve is None
            else request.auto_approve
        )
        max_attempts = request.max_attempts or self.settings.max_attempts
        state = initial_state(
            request.task,
            context=request.context,
            auto_approve=auto_approve,
            max_attempts=max_attempts,
            metadata=request.metadata,
        )
        now = utc_now_iso()
        record = RunRecord(state=state, created_at=now, updated_at=now)
        with self._registry_lock:
            self._save_record(record)

        with self._graph_lock:
            try:
                self.graph.invoke(state, config=self._config(state.run_id))
                self._refresh(state.run_id)
            except GraphInterrupt:
                self._refresh(state.run_id)
            except Exception as exc:  # graph-level failure (not a node failure)
                logger.exception("Run %s failed at the graph level", state.run_id)
                self._mark_failed(state.run_id, str(exc))
        return self.get(state.run_id)

    def approve(self, decision: ApprovalDecision) -> RunResponse:
        with self._registry_lock:
            record = self._load_record(decision.run_id)
            if record is None:
                raise RunNotFound(f"Run '{decision.run_id}' was not found.")
            if record.state.status != RunStatus.AWAITING_APPROVAL.value:
                raise InvalidRunState(
                    f"Run '{decision.run_id}' is '{record.state.status}', not awaiting approval."
                )
        payload = {
            "approved": decision.approved,
            "reviewer": decision.reviewer,
            "comment": decision.comment,
            "decided_at": utc_now_iso(),
        }
        with self._graph_lock:
            try:
                self.graph.invoke(
                    Command(resume=payload), config=self._config(decision.run_id)
                )
                self._refresh(decision.run_id)
            except GraphInterrupt:
                self._refresh(decision.run_id)
            except Exception as exc:
                logger.exception("Run %s failed while resuming", decision.run_id)
                self._mark_failed(decision.run_id, str(exc))
        return self.get(decision.run_id)

    def _refresh(self, run_id: str) -> None:
        try:
            snapshot = self.graph.get_state(self._config(run_id))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not read checkpoint for %s: %s", run_id, exc)
            return
        values = getattr(snapshot, "values", None) or {}
        record = self._load_record(run_id)
        if record is None:
            return
        try:
            state = AgentState.from_dict(values) if values else record.state
        except TypeError:
            state = record.state
        if not state.run_id:
            state.run_id = run_id

        if getattr(snapshot, "next", ()):  # paused on an interrupt
            state.status = RunStatus.AWAITING_APPROVAL.value
            state.approval_status = ApprovalStatus.PENDING.value
            payload = self._interrupt_payload(snapshot) or self._approval_payload(state)
            record.approval = payload
            already_logged = any(
                log.get("event") == "awaiting_approval"
                for log in state.logs
                if isinstance(log, dict)
            )
            if not already_logged:
                state.logs = list(state.logs) + [
                    make_log(
                        "approval",
                        "awaiting_approval",
                        "Workflow paused; waiting for a human decision.",
                        risk_level=payload.get("risk_level"),
                    )
                ]
        else:
            if state.status in {RunStatus.PENDING.value, RunStatus.RUNNING.value}:
                state.status = RunStatus.FAILED.value
                state.error = state.error or "Workflow ended without producing a final state."
            record.approval = None

        state.completed_at = state.completed_at or (
            utc_now_iso() if state.status in TERMINAL_STATUSES else None
        )
        record.state = state
        record.updated_at = utc_now_iso()
        record.duration_ms = _elapsed_ms(record.created_at)
        self._save_record(record)

    def _interrupt_payload(self, snapshot: Any) -> dict[str, Any] | None:
        for task in getattr(snapshot, "tasks", ()) or ():
            for item in getattr(task, "interrupts", ()) or ():
                value = getattr(item, "value", None)
                if isinstance(value, dict):
                    return value
        values = getattr(snapshot, "values", None) or {}
        raw = values.get("__interrupt__") if isinstance(values, dict) else None
        if raw:
            first = raw[0] if isinstance(raw, (list, tuple)) else raw
            return getattr(first, "value", None) or None
        return None

    def _mark_failed(self, run_id: str, error: str) -> None:
        record = self._load_record(run_id)
        if record is None:
            return
        state = record.state
        state.status = RunStatus.FAILED.value
        state.error = error[:2000]
        state.completed_at = utc_now_iso()
        state.logs = list(state.logs) + [
            make_log("workflow", "failed", f"Run failed: {error}")
        ]
        record.state = state
        record.updated_at = utc_now_iso()
        record.duration_ms = _elapsed_ms(record.created_at)
        self._save_record(record)

    # -------------------------------------------------------------- registry
    def _save_record(self, record: RunRecord) -> None:
        self._records[record.state.run_id] = record
        try:
            self.run_store.put(record)
        except Exception:  # pragma: no cover - disk failure is environment-specific
            logger.exception("Could not persist run %s", record.state.run_id)

    def _load_record(self, run_id: str) -> RunRecord | None:
        stored = self.run_store.get(run_id)
        if stored is not None:
            self._records[run_id] = stored
            return stored
        return self._records.get(run_id)

    def run_count(self) -> int:
        count = getattr(self.run_store, "count", None)
        if callable(count):
            return int(count())
        return len(self._records)

    # --------------------------------------------------------------- read api
    def _response_from_record(self, record: RunRecord) -> RunResponse:
        state = record.state
        status = state.status
        approval: ApprovalPrompt | None = None
        if status == RunStatus.AWAITING_APPROVAL.value and record.approval:
            payload = record.approval
            approval = ApprovalPrompt(
                run_id=state.run_id,
                task_type=str(payload.get("task_type") or state.task_type),
                title=str(payload.get("title") or state.task[:120]),
                summary=str(payload.get("summary") or ""),
                risk_level=str(payload.get("risk_level") or "medium"),
                diff_preview=payload.get("diff_preview"),
                requested_at=str(payload.get("requested_at") or record.updated_at),
            )
        state_dict = state_json_safe(state)
        if status == RunStatus.AWAITING_APPROVAL.value:
            state_dict["status"] = status
        return RunResponse(
            run_id=state.run_id,
            status=RunStatus(status),
            task=state.task,
            task_type=state.task_type,
            strategy=state.strategy,
            state=state_dict,
            approval=approval,
            error=state.error,
            created_at=record.created_at,
            updated_at=record.updated_at,
            duration_ms=round(record.duration_ms, 2) if record.duration_ms is not None else None,
        )

    def get(self, run_id: str) -> RunResponse:
        with self._registry_lock:
            record = self._load_record(run_id)
            if record is None:
                raise RunNotFound(f"Run '{run_id}' was not found.")
            return self._response_from_record(record)

    def list_runs(self, limit: int = 20) -> list[RunResponse]:
        bounded = max(1, min(limit, 100))
        with self._registry_lock:
            records = self.run_store.list(bounded)
        return [self._response_from_record(record) for record in records]

    # ------------------------------------------------------------ graph spec
    def graph_spec(self) -> dict[str, Any]:
        nodes = [
            {"id": "start", "label": "Start", "type": "terminal", "description": "Task received"},
            {"id": "reader", "label": "Reader", "type": "agent", "description": "Extract context, classify, assess risk"},
            {"id": "planner", "label": "Planner", "type": "agent", "description": "Simple vs complex strategy and plan"},
            {"id": "executor", "label": "Executor", "type": "agent", "description": "Run tools: code fix, CRM, research, email"},
            {"id": "validator", "label": "Validator", "type": "agent", "description": "Check criteria, score, drive retries"},
            {"id": "approval", "label": "Human approval", "type": "human", "description": "Pause for approve/reject on risky actions"},
            {"id": "output", "label": "Output", "type": "agent", "description": "Compose PR, email, report or triage"},
            {"id": "give_up", "label": "Give up", "type": "terminal", "description": "Retry budget exhausted"},
            {"id": "end", "label": "End", "type": "terminal", "description": "Run finished"},
        ]
        edges = [
            {"source": "start", "target": "reader", "label": "", "conditional": False},
            {"source": "reader", "target": "planner", "label": "parsed", "conditional": True},
            {"source": "reader", "target": "end", "label": "error", "conditional": True},
            {"source": "planner", "target": "executor", "label": "planned", "conditional": True},
            {"source": "planner", "target": "end", "label": "error", "conditional": True},
            {"source": "executor", "target": "validator", "label": "result", "conditional": True},
            {"source": "executor", "target": "end", "label": "error", "conditional": True},
            {"source": "validator", "target": "executor", "label": "invalid + retries left", "conditional": True},
            {"source": "validator", "target": "approval", "label": "valid + approval", "conditional": True},
            {"source": "validator", "target": "output", "label": "valid", "conditional": True},
            {"source": "validator", "target": "give_up", "label": "retries exhausted", "conditional": True},
            {"source": "approval", "target": "output", "label": "approved", "conditional": True},
            {"source": "approval", "target": "end", "label": "rejected", "conditional": True},
            {"source": "give_up", "target": "end", "label": "", "conditional": False},
            {"source": "output", "target": "end", "label": "", "conditional": False},
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "mermaid": _mermaid(nodes, edges),
            "dot": _dot(nodes, edges),
            "tools": self.tools.describe(),
            "agents": self.agents.describe(),
        }


# ---------------------------------------------------------------------------
# Graph rendering helpers
# ---------------------------------------------------------------------------
def _mermaid(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    lines = ["flowchart TD"]
    for node in nodes:
        node_id = node["id"]
        if node["type"] == "human":
            lines.append(f'    {node_id}{{{{"{node["label"]}"}}}}')
        elif node["type"] == "terminal":
            lines.append(f'    {node_id}(["{node["label"]}"])')
        else:
            lines.append(f'    {node_id}["{node["label"]}"]')
    for edge in edges:
        label = f'|"{edge["label"]}"|' if edge.get("label") else ""
        lines.append(f'    {edge["source"]} -->{label} {edge["target"]}')
    return "\n".join(lines)


def _dot(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    lines = [
        "digraph workflow {",
        "  rankdir=LR;",
        '  graph [fontname="Helvetica", bgcolor="transparent"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#eef2ff", color="#6366f1"];',
        '  edge [fontname="Helvetica", color="#64748b", fontcolor="#475569"];',
    ]
    for node in nodes:
        shape = "hexagon" if node["type"] == "human" else "box"
        if node["type"] == "terminal":
            shape = "oval"
        lines.append(
            f'  {node["id"]} [label="{node["label"]}", shape={shape}];'
        )
    for edge in edges:
        label = f' [label="{edge["label"]}"]' if edge.get("label") else ""
        lines.append(f'  {edge["source"]} -> {edge["target"]}{label};')
    lines.append("}")
    return "\n".join(lines)


def _elapsed_ms(started_at: str) -> float:
    try:
        started = datetime.fromisoformat(started_at)
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        return round((datetime.now(timezone.utc) - started).total_seconds() * 1000, 2)
    except Exception:
        return 0.0


__all__ = [
    "Orchestrator",
    "OrchestrationError",
    "RunNotFound",
    "InvalidRunState",
    "AgentTimeout",
]
