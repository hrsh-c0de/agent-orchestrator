"""Typed contracts and the universal orchestration state.

Every boundary in the system -- LLM output, tool output, API payload -- is
validated with Pydantic. The in-graph state is a plain dataclass so it stays
cheap to copy, easy to serialise, and framework agnostic.
"""

from __future__ import annotations

import operator
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enums and helpers
# ---------------------------------------------------------------------------
class TaskType(str, Enum):
    GITHUB_ISSUE = "github_issue"
    EMAIL = "email"
    RESEARCH = "research"
    ALERT = "alert"
    CRM = "crm"
    GENERIC = "generic"


class Strategy(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


class ApprovalStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_run_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Pydantic contracts (validated LLM / tool / API payloads)
# ---------------------------------------------------------------------------
class StrictModel(BaseModel):
    """Reject unknown keys so malformed LLM output fails fast and is retried."""

    model_config = ConfigDict(extra="forbid")


class ReaderOutput(StrictModel):
    summary: str = Field(min_length=1, max_length=4000)
    task_type: TaskType
    risk_level: RiskLevel = RiskLevel.LOW
    entities: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    needs_approval: bool = False
    reasoning: str = ""


class PlanStep(StrictModel):
    id: int = Field(ge=1)
    description: str = Field(min_length=1, max_length=1000)
    tool: str = Field(min_length=1, max_length=64)
    rationale: str = ""


class Plan(StrictModel):
    strategy: Strategy
    steps: list[PlanStep] = Field(min_length=1, max_length=12)
    validation_criteria: list[str] = Field(default_factory=list)
    requires_approval: bool = False
    reasoning: str = ""

    @field_validator("steps")
    @classmethod
    def unique_step_ids(cls, steps: list[PlanStep]) -> list[PlanStep]:
        ids = [step.id for step in steps]
        if len(ids) != len(set(ids)):
            raise ValueError("step ids must be unique")
        return steps


class StepResult(StrictModel):
    step_id: int
    tool: str
    success: bool
    summary: str
    artifact: dict[str, Any] = Field(default_factory=dict)


class ActionResult(StrictModel):
    action: str
    tool: str
    success: bool
    summary: str
    steps: list[StepResult] = Field(default_factory=list)
    artifact: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ValidationResult(StrictModel):
    valid: bool
    score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    feedback: str = ""
    criteria_checked: list[str] = Field(default_factory=list)


class OutputLink(StrictModel):
    label: str = Field(min_length=1, max_length=120)
    url: str = Field(min_length=1, max_length=2000)


class FinalOutput(StrictModel):
    kind: Literal["pull_request", "email", "report", "triage", "answer", "crm"]
    title: str = Field(min_length=1, max_length=300)
    summary: str = Field(min_length=1, max_length=4000)
    body: str = Field(min_length=1)
    links: list[OutputLink] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LogEntry(BaseModel):
    timestamp: str
    agent: str
    event: str
    message: str
    attempt: int | None = None
    duration_ms: float | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ApprovalPrompt(BaseModel):
    run_id: str
    task_type: str
    title: str
    summary: str
    risk_level: str
    diff_preview: str | None = None
    requested_at: str


class RunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=20000)
    context: dict[str, Any] = Field(default_factory=dict)
    auto_approve: bool | None = None
    max_attempts: int | None = Field(default=None, ge=1, le=10)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task")
    @classmethod
    def task_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task must not be blank")
        return value.strip()


class ApprovalDecision(BaseModel):
    run_id: str
    approved: bool
    reviewer: str = Field(default="dashboard-user", max_length=200)
    comment: str = Field(default="", max_length=4000)


class RunResponse(BaseModel):
    run_id: str
    status: RunStatus
    task: str
    task_type: str
    strategy: str
    state: dict[str, Any]
    approval: ApprovalPrompt | None = None
    error: str | None = None
    created_at: str
    updated_at: str
    duration_ms: float | None = None


# ---------------------------------------------------------------------------
# Universal state
# ---------------------------------------------------------------------------
@dataclass
class AgentState:
    """Universal state carried through every agent node.

    The six contract fields are ``task`` -> ``context`` -> ``plan`` ->
    ``action`` -> ``result`` -> ``output``. Operational fields (logs,
    attempts, approval, status) support reliability and observability.
    """

    # --- contract fields ---------------------------------------------------
    task: str
    context: dict[str, Any] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    action: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)

    # --- operational fields ------------------------------------------------
    run_id: str = field(default_factory=new_run_id)
    task_type: str = TaskType.GENERIC.value
    strategy: str = ""
    status: str = RunStatus.PENDING.value
    logs: Annotated[list[dict[str, Any]], operator.add] = field(default_factory=list)
    attempts: int = 0
    max_attempts: int = 3
    validation: dict[str, Any] = field(default_factory=dict)
    approval_required: bool = False
    approval_status: str = ApprovalStatus.NOT_REQUIRED.value
    approval: dict[str, Any] = field(default_factory=dict)
    auto_approve: bool = False
    error: str | None = None
    user_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=utc_now_iso)
    completed_at: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    # --- serialisation -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | AgentState) -> "AgentState":
        if isinstance(data, AgentState):
            return data
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})

    def copy_with(self, **updates: Any) -> "AgentState":
        payload = self.to_dict()
        payload.update(updates)
        return AgentState.from_dict(payload)


STATE_FIELD_NAMES: frozenset[str] = frozenset(f.name for f in fields(AgentState))


def initial_state(
    task: str,
    *,
    context: dict[str, Any] | None = None,
    auto_approve: bool = False,
    max_attempts: int = 3,
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentState:
    return AgentState(
        task=task.strip(),
        user_context=context or {},
        auto_approve=auto_approve,
        max_attempts=max_attempts,
        run_id=run_id or new_run_id(),
        metadata=metadata or {},
    )


def make_log(
    agent: str,
    event: str,
    message: str,
    *,
    attempt: int | None = None,
    duration_ms: float | None = None,
    **data: Any,
) -> dict[str, Any]:
    return LogEntry(
        timestamp=utc_now_iso(),
        agent=agent,
        event=event,
        message=message,
        attempt=attempt,
        duration_ms=round(duration_ms, 2) if duration_ms is not None else None,
        data=data,
    ).model_dump(mode="json")


def state_json_safe(state: AgentState | dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serialisable copy, coercing enums and nested models."""
    if isinstance(state, AgentState):
        payload = state.to_dict()
    else:
        payload = dict(state)
    payload.setdefault("run_id", "")
    payload.setdefault("status", RunStatus.FAILED.value)
    payload.setdefault("logs", [])
    payload.setdefault("context", {})
    payload.setdefault("plan", {})
    payload.setdefault("result", {})
    payload.setdefault("output", {})
    payload.setdefault("validation", {})
    payload.setdefault("metrics", {})
    return payload
