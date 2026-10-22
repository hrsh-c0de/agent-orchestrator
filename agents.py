"""Modular agents.

Each agent is a small class with one public ``run`` method. Every agent has a
deterministic engine (used offline and in tests) and can delegate to an
OpenAI-backed LLM that must return schema-valid JSON. The deterministic engine
is also the fallback when a provider call fails, so the workflow degrades
gracefully instead of crashing.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from config import Settings
from llm import LLMClient
from state import (
    ActionResult,
    FinalOutput,
    Plan,
    PlanStep,
    ReaderOutput,
    RiskLevel,
    Strategy,
    StepResult,
    TaskType,
    ValidationResult,
)
from tools import StepResult as ToolStepResult
from tools import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _brief(state: Any, *, max_chars: int = 4000) -> str:
    payload = {
        "task": state.task,
        "task_type": state.task_type,
        "context": state.context,
        "plan": state.plan,
        "action": state.action,
        "result": {
            "summary": state.result.get("summary", ""),
            "tool": state.result.get("tool", ""),
            "artifact": state.result.get("artifact", {}),
        },
        "validation": state.validation,
        "attempts": state.attempts,
        "user_context": state.user_context,
    }
    text = json.dumps(payload, default=str, ensure_ascii=False)
    if len(text) > max_chars:
        text = text[:max_chars] + "...[truncated]"
    return text


def _json(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), indent=2)


def _summarise(text: str, limit: int = 420) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", compact)
    summary = " ".join(sentences[:2]) if sentences else compact
    if len(summary) > limit:
        summary = summary[: limit - 3].rstrip() + "..."
    return summary


def _keywords(text: str, limit: int = 8) -> list[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "have", "will", "your",
        "about", "into", "when", "what", "which", "there", "their", "would", "could",
        "should", "please", "help", "need", "want", "using", "make", "just", "very",
        "also", "than", "then", "them", "they", "been", "were", "was", "are", "our",
    }
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text.lower())
    counts: dict[str, int] = {}
    for word in words:
        if word not in stop:
            counts[word] = counts.get(word, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in ordered[:limit]]


class BaseAgent(ABC):
    name = "agent"
    output_model: type[BaseModel]
    system_prompt = ""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, state: Any, llm: LLMClient | None = None, feedback: str = "") -> BaseModel:
        if llm is not None:
            try:
                return llm.complete_json(
                    system=self.system_prompt,
                    prompt=self.prompt(state, feedback),
                    schema=self.output_model,
                )
            except Exception as exc:  # provider failures must not kill the run
                logger.warning("%s agent: LLM unavailable (%s); using offline engine", self.name, exc)
        return self.offline(state, feedback)

    def prompt(self, state: Any, feedback: str = "") -> str:
        return f"State:\n{_brief(state)}\n\nFeedback: {feedback or 'none'}"

    @abstractmethod
    def offline(self, state: Any, feedback: str = "") -> BaseModel:
        ...


# ---------------------------------------------------------------------------
# 1. Reader
# ---------------------------------------------------------------------------
class ReaderAgent(BaseAgent):
    name = "reader"
    output_model = ReaderOutput
    system_prompt = (
        "You are the Reader agent in a multi-agent workflow. Extract the user's intent, "
        "classify the task as one of github_issue, email, research, alert, crm or generic, "
        "pull out entities (repos, files, error types, emails, services, severity, company, "
        "deadlines), constraints, and a risk level. Set needs_approval for actions that "
        "create pull requests, send external email, or modify CRM records. Never invent "
        "entities that are not supported by the input."
    )

    SIGNALS: dict[TaskType, list[tuple[str, int]]] = {
        TaskType.EMAIL: [
            (r"\bsubject\s*:", 6),
            (r"\bdear\s+\w+", 3),
            (r"\bemails?\b", 2),
            (r"\breply\b", 3),
            (r"\brespond\b", 2),
            (r"\binbox\b", 3),
            (r"\bcustomer (?:wrote|says|asked)\b", 3),
        ],
        TaskType.GITHUB_ISSUE: [
            (r"\btraceback\b", 5),
            (r"\bstack trace\b", 5),
            (r"\bbug\b", 3),
            (r"\bgithub\b", 4),
            (r"\b(?:pull request|PR)\b", 3),
            (r"\brepo(?:sitory)?\b", 4),
            (r"\bregression\b", 3),
            (r"\b[A-Z]\w*(?:Error|Exception)\b", 4),
            (r"\.(?:py|js|ts|tsx|go|java|rb)\b", 3),
            (r"\breproduce\b", 3),
        ],
        TaskType.RESEARCH: [
            (r"\bresearch\b", 5),
            (r"\bcompare\b", 4),
            (r"\b(?:vs\.?|versus)\b", 3),
            (r"\bbest practices?\b", 4),
            (r"\b(?:landscape|state of)\b", 4),
            (r"\b(?:evaluate|investigate|explore)\b", 3),
            (r"\btrade-?offs?\b", 3),
            (r"\bwhat (?:are|is) the\b", 2),
            (r"\bsummar(?:y|ise|ize)\b", 3),
        ],
        TaskType.ALERT: [
            (r"\balert\b", 6),
            (r"\b(?:sev|severity)[\s-]?[12]\b", 6),
            (r"\b(?:p0|p1)\b", 4),
            (r"\bincident\b", 5),
            (r"\b(?:5xx|4xx)\b", 4),
            (r"\berror rate\b", 4),
            (r"\b(?:latency|timeout|outage|degraded)\b", 3),
            (r"\b(?:pagerduty|opsgenie|datadog)\b", 4),
            (r"\b(?:cpu|memory|disk) (?:usage|pressure|spike)\b", 3),
        ],
        TaskType.CRM: [
            (r"\bcrm\b", 6),
            (r"\b(?:deal|pipeline|opportunit(?:y|ies))\b", 4),
            (r"\b(?:contact|lead|account)s?\b", 3),
            (r"\b(?:salesforce|hubspot|pipedrive)\b", 5),
            (r"\b(?:renewal|upsell|churn)\b", 3),
        ],
    }

    def offline(self, state: Any, feedback: str = "") -> ReaderOutput:
        text = state.task
        task_type = _classify(text)
        entities = _extract_entities(text, task_type, state.user_context)
        risk = _assess_risk(task_type, text)
        constraints = _extract_constraints(text)
        needs_approval = task_type in {TaskType.GITHUB_ISSUE, TaskType.EMAIL}
        if task_type == TaskType.CRM and re.search(
            r"\b(?:update|change|set|delete|merge|create)\b", text, re.IGNORECASE
        ):
            needs_approval = True
        if task_type == TaskType.ALERT and risk == RiskLevel.HIGH:
            needs_approval = True
        if str(state.user_context.get("require_approval", "")).lower() in {"true", "1", "yes"}:
            needs_approval = True

        summary = _summarise(text)
        entities["summary"] = summary
        entities["keywords"] = _keywords(text)
        reasoning = (
            f"Classified as {task_type.value} with {risk.value} risk from keyword and "
            f"entity signals. Extracted {len(entities)} entity groups and "
            f"{len(constraints)} explicit constraints."
        )
        return ReaderOutput(
            summary=summary,
            task_type=task_type,
            risk_level=risk,
            entities=entities,
            constraints=constraints,
            needs_approval=needs_approval,
            reasoning=reasoning,
        )


def _classify(text: str) -> TaskType:
    scores = {task_type: 0 for task_type in TaskType}
    for task_type, patterns in ReaderAgent.SIGNALS.items():
        for pattern, weight in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                scores[task_type] += weight
    # Explicit GitHub URLs are unambiguous.
    if re.search(r"github\.com/[\w.-]+/[\w.-]+", text, re.IGNORECASE):
        scores[TaskType.GITHUB_ISSUE] += 6
    best = max(scores, key=lambda item: scores[item])
    if scores[best] < 3:
        return TaskType.GENERIC
    # Deterministic tie-break: operational signals before generic requests.
    priority = [
        TaskType.ALERT,
        TaskType.EMAIL,
        TaskType.GITHUB_ISSUE,
        TaskType.CRM,
        TaskType.RESEARCH,
        TaskType.GENERIC,
    ]
    if list(scores.values()).count(scores[best]) > 1:
        for candidate in priority:
            if scores[candidate] == scores[best]:
                return candidate
    return best


def _extract_entities(text: str, task_type: TaskType, user_context: dict[str, Any]) -> dict[str, Any]:
    entities: dict[str, Any] = {}
    repos = re.findall(r"github\.com/([\w.-]+/[\w.-]+)", text)
    if not repos:
        repos = re.findall(
            r"\b(?:repo(?:sitory)?|project)\s*[:=]?\s*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
            text,
            re.I,
        )
    if repos:
        entities["repos"] = list(dict.fromkeys(repo.rstrip(".,") for repo in repos))

    emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    if emails:
        entities["emails"] = list(dict.fromkeys(emails))
    if user_context.get("from"):
        entities.setdefault("emails", [str(user_context["from"])])

    urls = re.findall(r"https?://[^\s<>\"')]+", text)
    if urls:
        entities["urls"] = list(dict.fromkeys(urls))[:5]

    files = re.findall(r"\b[\w./-]+\.(?:py|js|jsx|ts|tsx|go|java|rb|rs|sql|yaml|yml)\b", text)
    trace_files = re.findall(r'File "([^"]+)"', text)
    files = list(dict.fromkeys(trace_files + files))
    if files:
        entities["files"] = files[:5]

    error = re.search(r"\b([A-Z]\w*(?:Error|Exception)|KeyError|ValueError|TypeError)\b", text)
    if error:
        entities["error_type"] = error.group(1)
    line = re.search(r"line\s+(\d+)", text, re.IGNORECASE)
    if line:
        entities["line"] = int(line.group(1))

    service = re.search(r"\b(?:service|host|pod)\s*[`:\-]\s*([\w.-]+)", text, re.IGNORECASE)
    host = re.search(r"\b([a-z0-9-]+(?:\.[a-z0-9-]+){2,})\b", text, re.IGNORECASE)
    if service:
        entities["services"] = [service.group(1).strip("`.")]
    elif host:
        entities["services"] = [host.group(1)]

    severity = re.search(r"\b(sev[\s-]?[1-4]|p[0-3]|critical|high|medium|low)\b", text, re.IGNORECASE)
    if severity and task_type in {TaskType.ALERT, TaskType.GITHUB_ISSUE}:
        entities["severity"] = severity.group(1).lower()

    company = re.search(
        r"(?:company|account|customer)\s*[:\-]?\s*([A-Z][\w&.\-]+(?:\s+[A-Z][\w&.\-]+){0,3})", text
    )
    if company:
        entities["company"] = company.group(1).strip()
    elif task_type == TaskType.CRM:
        for candidate in re.findall(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b", text):
            if candidate.lower() not in {"find", "the", "deal", "status", "crm"}:
                entities["company"] = candidate
                break

    subject = re.search(r"subject\s*:\s*(.+)", text, re.IGNORECASE)
    if subject:
        entities["subject"] = subject.group(1).strip().splitlines()[0][:200]
    sender = re.search(r"from\s*:\s*(.+)", text, re.IGNORECASE)
    if sender:
        sender_line = sender.group(1).strip().splitlines()[0][:200]
        entities["sender"] = sender_line
        name_match = re.match(r"([^<]+?)\s*<[^>]+>", sender_line)
        if name_match:
            entities["sender_name"] = name_match.group(1).strip().strip('"')
        else:
            entities["sender_name"] = sender_line.split("<")[0].strip()

    questions = re.findall(r"([^?.!\n]{12,180}\?)", text)
    if questions:
        entities["questions"] = questions[:5]
    deadline = re.search(
        r"\b(?:by|before|deadline[:\s]+)\s*((?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|"
        r"eod|end of (?:day|week)|next week|\d{4}-\d{2}-\d{2}))\b",
        text,
        re.IGNORECASE,
    )
    if deadline:
        entities["deadline"] = deadline.group(1)
    return entities


def _assess_risk(task_type: TaskType, text: str) -> RiskLevel:
    high = re.search(
        r"\b(production|prod\b|outage|critical|sev[\s-]?1|p0|security|credential|data loss|"
        r"legal|lawsuit|refund|chargeback|cancel(?:lation)?)\b",
        text,
        re.IGNORECASE,
    )
    if high:
        return RiskLevel.HIGH
    if task_type in {TaskType.GITHUB_ISSUE, TaskType.EMAIL, TaskType.ALERT}:
        return RiskLevel.MEDIUM
    if task_type == TaskType.CRM:
        return RiskLevel.LOW
    return RiskLevel.LOW


def _extract_constraints(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    markers = re.compile(
        r"\b(must|should|required|do not|don't|avoid|deadline|asap|by (?:today|tomorrow|friday|eod)|"
        r"within \d+ (?:hours?|days?)|no breaking|backwards? compat)\b",
        re.IGNORECASE,
    )
    found = [re.sub(r"\s+", " ", sentence).strip() for sentence in sentences if markers.search(sentence)]
    return list(dict.fromkeys(found))[:5]


# ---------------------------------------------------------------------------
# 2. Planner
# ---------------------------------------------------------------------------
class PlannerAgent(BaseAgent):
    name = "planner"
    output_model = Plan
    system_prompt = (
        "You are the Planner agent. Choose a 'simple' or 'complex' strategy, then produce "
        "an ordered plan whose steps use only these tools: code_fix, test_plan, email_reply, "
        "research, crm_query, alert_triage, generic_answer, review. Complex plans add test "
        "or review steps. Provide concrete validation criteria and set requires_approval for "
        "actions that create PRs, send email, or modify CRM records."
    )

    PRIMARY_TOOL: dict[TaskType, str] = {
        TaskType.GITHUB_ISSUE: "code_fix",
        TaskType.EMAIL: "email_reply",
        TaskType.RESEARCH: "research",
        TaskType.ALERT: "alert_triage",
        TaskType.CRM: "crm_query",
        TaskType.GENERIC: "generic_answer",
    }
    ALLOWED_TOOLS = {
        "code_fix",
        "test_plan",
        "email_reply",
        "research",
        "crm_query",
        "alert_triage",
        "generic_answer",
        "review",
    }

    def run(self, state: Any, llm: LLMClient | None = None, feedback: str = "") -> Plan:
        plan = super().run(state, llm, feedback)
        if not isinstance(plan, Plan):  # pragma: no cover - defensive
            plan = Plan.model_validate(plan)
        unknown = [step.tool for step in plan.steps if step.tool not in self.ALLOWED_TOOLS]
        if unknown:
            logger.warning("Planner proposed unknown tools %s; using deterministic plan", unknown)
            return self.offline(state, feedback)
        return plan

    def offline(self, state: Any, feedback: str = "") -> Plan:
        reader = ReaderOutput.model_validate(state.context)
        task_type = reader.task_type
        complex_task = _is_complex(state.task, reader)
        strategy = Strategy.COMPLEX if complex_task else Strategy.SIMPLE
        primary = self.PRIMARY_TOOL[task_type]

        descriptions = {
            TaskType.GITHUB_ISSUE: "Reproduce the failure, patch the root cause and open a pull request",
            TaskType.EMAIL: "Draft a reply that addresses every question in the thread",
            TaskType.RESEARCH: "Research the question and write a structured brief",
            TaskType.ALERT: "Triage the alert and produce an action plan",
            TaskType.CRM: "Query the CRM for matching contacts, deals and pipeline context",
            TaskType.GENERIC: "Produce a structured, actionable answer",
        }
        steps = [
            PlanStep(
                id=1,
                description=descriptions[task_type],
                tool=primary,
                rationale=reader.reasoning or "The reader identified this as the primary action.",
            )
        ]
        if task_type == TaskType.GITHUB_ISSUE:
            steps.append(
                PlanStep(
                    id=2,
                    description="Write a regression-first test plan for the patch",
                    tool="test_plan",
                    rationale="A fix without a regression test can silently regress.",
                )
            )
        if complex_task:
            steps.append(
                PlanStep(
                    id=len(steps) + 1,
                    description="Cross-check the deliverable against the acceptance criteria",
                    tool="review",
                    rationale="Complex tasks get an explicit review gate before delivery.",
                )
            )
        plan = Plan(
            strategy=strategy,
            steps=steps,
            validation_criteria=_criteria(task_type),
            requires_approval=reader.needs_approval,
            reasoning=(
                f"{strategy.value.title()} strategy: {len(steps)} step(s), primary tool "
                f"'{primary}'. Complexity signals: {'present' if complex_task else 'none'}."
            ),
        )
        return plan


def _is_complex(task: str, reader: ReaderOutput) -> bool:
    signals = 0
    if len(task) > 320:
        signals += 1
    if task.count("\n") >= 3:
        signals += 1
    if task.count("?") >= 2:
        signals += 1
    if reader.task_type in {TaskType.GITHUB_ISSUE, TaskType.CRM}:
        signals += 1
    if reader.risk_level == RiskLevel.HIGH:
        signals += 1
    if re.search(r"\b(and then|after that|also|plus|as well as|end-to-end|multiple)\b", task, re.I):
        signals += 1
    if reader.task_type == TaskType.RESEARCH and re.search(r"\b(compare|vs\.?|trade-?offs?)\b", task, re.I):
        signals += 1
    return signals >= 2


def _criteria(task_type: TaskType) -> list[str]:
    return {
        TaskType.GITHUB_ISSUE: [
            "A root-cause explanation is present",
            "A unified diff touches the reported file",
            "A pull request URL is returned",
            "Regression tests are proposed",
        ],
        TaskType.EMAIL: [
            "Recipient and subject are present",
            "The reply addresses the customer's questions",
            "Tone is professional and there is a clear next step",
            "No placeholders remain",
        ],
        TaskType.RESEARCH: [
            "The brief has structured headings",
            "At least three findings are stated",
            "A recommendation and open questions are included",
            "Limitations are disclosed",
        ],
        TaskType.ALERT: [
            "Severity and affected service are identified",
            "Likely causes are listed",
            "Immediate mitigations are actionable",
            "An escalation path is defined",
        ],
        TaskType.CRM: [
            "The CRM was queried read-only",
            "Matches or an explicit no-match result are shown",
            "Pipeline context is included",
        ],
        TaskType.GENERIC: [
            "The answer is actionable",
            "Assumptions and risks are stated",
        ],
    }[task_type]


# ---------------------------------------------------------------------------
# 3. Executor
# ---------------------------------------------------------------------------
class ExecutorAgent:
    """Runs the plan's steps through the tool registry."""

    name = "executor"

    def __init__(self, settings: Settings, tools: ToolRegistry) -> None:
        self.settings = settings
        self.tools = tools

    def run(
        self, state: Any, llm: LLMClient | None = None, feedback: str = ""
    ) -> ActionResult:
        plan = Plan.model_validate(state.plan)
        completed: list[ToolStepResult] = []
        artifacts: list[dict[str, Any]] = []
        notes: list[str] = []

        for step in plan.steps:
            result = self.tools.execute(
                step.tool,
                state=state,
                step=step.model_dump(),
                feedback=feedback,
                prior=list(completed),
            )
            completed.append(result)
            artifacts.append(result.artifact)
            notes.extend(str(note) for note in result.artifact.get("notes", []))

        primary = completed[0]
        merged: dict[str, Any] = dict(primary.artifact)
        merged["_steps"] = [
            {
                "step_id": item.step_id,
                "tool": item.tool,
                "summary": item.summary,
                "artifact": item.artifact,
            }
            for item in completed
        ]
        if feedback:
            merged["revision_feedback"] = feedback[:1000]
        return ActionResult(
            action=primary.tool,
            tool=primary.tool,
            success=all(item.success for item in completed),
            summary=" ".join(item.summary for item in completed),
            steps=[
                StepResult(
                    step_id=item.step_id,
                    tool=item.tool,
                    success=item.success,
                    summary=item.summary,
                    artifact=item.artifact,
                )
                for item in completed
            ],
            artifact=merged,
            notes=notes[:10],
        )


# ---------------------------------------------------------------------------
# 4. Validator
# ---------------------------------------------------------------------------
class ValidatorAgent(BaseAgent):
    name = "validator"
    output_model = ValidationResult
    system_prompt = (
        "You are the Validator agent. Judge whether the executor's artifact satisfies the "
        "plan's acceptance criteria. Be strict and concrete. Return valid=false with actionable "
        "feedback if anything required is missing or looks invented. Do not reward verbosity."
    )

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.threshold = settings.validation_threshold

    def run(self, state: Any, llm: LLMClient | None = None, feedback: str = "") -> ValidationResult:
        rules = self.offline(state)
        if llm is not None:
            try:
                judged = llm.complete_json(
                    system=self.system_prompt,
                    prompt=(
                        f"Plan:\n{json.dumps(state.plan, indent=2, default=str)[:3000]}\n\n"
                        f"Executor result:\n{json.dumps(state.result, indent=2, default=str)[:8000]}\n\n"
                        f"Deterministic checks:\n{rules.model_dump_json(indent=2)}"
                    ),
                    schema=ValidationResult,
                )
                merged_issues = list(dict.fromkeys(rules.issues + judged.issues))
                merged = ValidationResult(
                    valid=rules.valid and judged.valid,
                    score=min(rules.score, judged.score),
                    issues=merged_issues,
                    feedback=(judged.feedback or rules.feedback),
                    criteria_checked=list(
                        dict.fromkeys(rules.criteria_checked + judged.criteria_checked)
                    ),
                )
                return merged
            except Exception as exc:
                logger.warning("validator agent: LLM judge failed (%s); using rules only", exc)
        return rules

    def offline(self, state: Any, feedback: str = "") -> ValidationResult:
        checks: list[tuple[str, bool, bool]] = []  # (name, passed, critical)
        checks.append(("executor completed without error", not state.error, True))
        checks.append(("a non-empty result exists", bool(state.result), True))

        if state.error:
            checks.append(("artifact is present", False, True))
            return self._verdict(checks, ["Resolve the upstream executor error and retry."])

        try:
            action_result = ActionResult.model_validate(state.result)
        except ValidationError as exc:
            checks.append(("result matches the ActionResult schema", False, True))
            return self._verdict(checks, [f"Result schema invalid: {str(exc)[:300]}"])
        checks.append(("result matches the ActionResult schema", True, True))
        checks.append(("result reports success", action_result.success, True))

        artifact = action_result.artifact
        tool = action_result.tool or action_result.action
        if tool == "code_fix":
            checks.extend(
                [
                    ("diff has unified headers", "--- a/" in str(artifact.get("diff")) and "+++ b/" in str(artifact.get("diff")), True),
                    ("pull request URL is well formed", bool(re.match(r"^https://github\.com/[\w.-]+/[\w.-]+/pull/\d+$", str(artifact.get("pr_url", "")))), True),
                    ("root cause is explained", len(str(artifact.get("root_cause", ""))) >= 30, True),
                    ("regression tests proposed", bool(artifact.get("tests")), False),
                ]
            )
        elif tool == "email_reply":
            body = str(artifact.get("body", ""))
            checks.extend(
                [
                    ("recipient is an email", bool(re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", str(artifact.get("to", "")))), True),
                    ("subject is present", len(str(artifact.get("subject", ""))) >= 3, True),
                    ("body is substantive", len(body) >= 80, True),
                    ("body has a greeting and next step", body.lower().startswith(("hi", "hello", "dear")) and "will" in body.lower(), False),
                ]
            )
        elif tool == "research":
            body = str(artifact.get("body", ""))
            checks.extend(
                [
                    ("brief has sections", body.count("## ") >= 3, True),
                    ("brief is substantive", len(body) >= 300, True),
                    ("at least three findings", len(artifact.get("findings", [])) >= 3, True),
                    ("recommendation is present", len(str(artifact.get("recommendation", ""))) >= 20, True),
                    ("limitations are disclosed", bool(artifact.get("limitations")), False),
                ]
            )
        elif tool == "alert_triage":
            actions = artifact.get("immediate_actions", [])
            checks.extend(
                [
                    ("severity is classified", artifact.get("severity") in {"critical", "high", "medium", "low"}, True),
                    ("service is identified", bool(artifact.get("service")), True),
                    ("at least three immediate actions", len(actions) >= 3, True),
                    ("escalation path is defined", len(str(artifact.get("escalation", ""))) >= 10, False),
                ]
            )
        elif tool == "crm_query":
            checks.extend(
                [
                    ("pipeline context is present", bool(artifact.get("pipeline")), True),
                    ("query is recorded", isinstance(artifact.get("query"), dict), True),
                    ("read-only note is present", "read-only" in str(artifact.get("note", "")).lower(), False),
                ]
            )
        else:
            body = str(artifact.get("body", ""))
            checks.extend(
                [
                    ("answer body is substantive", len(body) >= 80, True),
                    ("key points are present", len(artifact.get("key_points", [])) >= 2, False),
                ]
            )

        placeholders = re.search(
            r"\b(TODO|TBD|FIXME|lorem ipsum)\b|<\s*insert\b",
            json.dumps(artifact, default=str),
            re.IGNORECASE,
        )
        checks.append(("no placeholders remain", placeholders is None, True))

        failed = [name for name, passed, _ in checks if not passed]
        issues = [f"Failed check: {name}" for name in failed]
        critical_failed = [name for name, passed, critical in checks if critical and not passed]
        criteria = [
            str(item) for item in state.plan.get("validation_criteria", [])
        ] or [name for name, _, _ in checks]
        feedback_text = (
            "Address every failed check before retrying: " + "; ".join(failed)
            if failed
            else "All automated checks passed."
        )
        return self._verdict(checks, issues, criteria=criteria, feedback_text=feedback_text)

    def _verdict(
        self,
        checks: list[tuple[str, bool, bool]],
        issues: list[str],
        *,
        criteria: list[str] | None = None,
        feedback_text: str | None = None,
    ) -> ValidationResult:
        passed = sum(1 for _, ok, _ in checks if ok)
        score = round(passed / max(1, len(checks)), 4)
        critical_failed = [name for name, ok, critical in checks if critical and not ok]
        valid = not critical_failed and score >= self.threshold
        if critical_failed:
            score = min(score, 0.5)
        feedback = feedback_text or (
            "; ".join(issues) if issues else "All automated checks passed."
        )
        return ValidationResult(
            valid=valid,
            score=score,
            issues=issues,
            feedback=feedback,
            criteria_checked=criteria or [name for name, _, _ in checks],
        )


# ---------------------------------------------------------------------------
# 5. Output
# ---------------------------------------------------------------------------
class OutputAgent(BaseAgent):
    name = "output"
    output_model = FinalOutput
    system_prompt = (
        "You are the Output agent. Turn the validated executor artifact into the final "
        "deliverable for the user. Preserve real URLs exactly, never invent citations, and "
        "format the body as GitHub-flavoured Markdown. Choose kind from pull_request, email, "
        "report, triage, answer or crm."
    )

    def offline(self, state: Any, feedback: str = "") -> FinalOutput:
        result = ActionResult.model_validate(state.result)
        artifact = result.artifact
        confidence = float(state.validation.get("score", 0.0))
        tool = result.tool or result.action

        if tool == "code_fix":
            pr_url = str(artifact.get("pr_url", ""))
            step_artifacts = {
                str(step.get("tool")): step.get("artifact", {})
                for step in artifact.get("_steps", [])
                if isinstance(step, dict)
            }
            tests = list(artifact.get("tests", []))
            tests += [
                str(item)
                for item in step_artifacts.get("test_plan", {}).get("unit_tests", [])
                if item not in tests
            ]
            commands = step_artifacts.get("test_plan", {}).get("commands", [])
            body_lines = [
                "## Pull request",
                f"[{artifact.get('pr_title', 'Fix')}]({pr_url}) — branch "
                f"`{artifact.get('branch', '')}`",
                "",
                "## Root cause",
                str(artifact.get("root_cause", "")),
                "",
                "## Proposed patch",
                "```diff",
                str(artifact.get("diff", "")).rstrip(),
                "```",
                "",
                "## Regression tests",
                *(f"- {test}" for test in tests),
            ]
            if commands:
                body_lines.extend(
                    ["", "```bash", *(str(command) for command in commands), "```"]
                )
            body_lines.extend(
                [
                    "",
                    f"> {artifact.get('mode', 'simulated').title()} PR created without GitHub credentials.",
                ]
            )
            body = "\n".join(body_lines)
            return FinalOutput(
                kind="pull_request",
                title=str(artifact.get("pr_title") or "Code fix"),
                summary=str(artifact.get("root_cause") or result.summary),
                body=body,
                links=[{"label": f"Open PR #{pr_url.rstrip('/').split('/')[-1]}", "url": pr_url}]
                if pr_url
                else [],
                next_steps=[
                    "Review the diff for unintended changes.",
                    "Run the regression tests in CI.",
                    "Merge after approval.",
                ],
                confidence=confidence,
                metadata={
                    "tool": tool,
                    "repo": artifact.get("repo"),
                    "branch": artifact.get("branch"),
                    "mode": artifact.get("mode"),
                    "attempts": state.attempts,
                },
            )

        if tool == "email_reply":
            to = str(artifact.get("to", ""))
            subject = str(artifact.get("subject", ""))
            email_body = str(artifact.get("body", ""))
            mailto = "mailto:" + urllib.parse.quote(to) + "?" + urllib.parse.urlencode(
                {"subject": subject, "body": email_body}
            )
            body = f"**To:** {to}  \n**Subject:** {subject}\n\n---\n\n{email_body}"
            return FinalOutput(
                kind="email",
                title=subject,
                summary=f"Draft reply ready for review before sending to {to}.",
                body=body,
                links=[{"label": "Open in mail client", "url": mailto}],
                next_steps=["Review the draft and adjust tone.", "Approve in the dashboard.", "Send the email."],
                confidence=confidence,
                metadata={"tool": tool, "to": to, "subject": subject, "attempts": state.attempts},
            )

        if tool == "research":
            body = str(artifact.get("body", "")).strip()
            if "## Recommendation" not in body:
                findings = "\n".join(f"- {item}" for item in artifact.get("findings", []))
                questions = "\n".join(
                    f"- {item}" for item in artifact.get("open_questions", [])
                )
                body = (
                    f"{body}\n\n## Findings\n{findings}\n\n"
                    f"## Recommendation\n{artifact.get('recommendation', '')}\n\n"
                    f"## Open questions\n{questions}\n\n"
                    f"## Limitations\n{artifact.get('limitations', '')}"
                )
            links = [
                {"label": str(item.get("label", "Source")), "url": str(item.get("url"))}
                for item in artifact.get("citations", [])
                if isinstance(item, dict) and item.get("url")
            ]
            return FinalOutput(
                kind="report",
                title=f"Research brief: {artifact.get('topic', state.task[:80])}",
                summary=str(artifact.get("recommendation") or result.summary),
                body=body,
                links=links,
                next_steps=["Validate the recommendation with one bounded pilot.", "Re-run with live sources before a high-stakes decision."],
                confidence=confidence,
                metadata={"tool": tool, "sources_retrieved": artifact.get("sources_retrieved", 0), "attempts": state.attempts},
            )

        if tool == "alert_triage":
            causes = "\n".join(f"- {item}" for item in artifact.get("likely_causes", []))
            actions = "\n".join(
                f"{index}. {item}"
                for index, item in enumerate(artifact.get("immediate_actions", []), start=1)
            )
            body = "\n".join(
                [
                    "## Triage summary",
                    f"**Severity:** {str(artifact.get('severity', '')).upper()}",
                    f"**Service:** `{artifact.get('service', '')}`",
                    "",
                    "## Likely causes",
                    causes,
                    "",
                    "## Immediate actions",
                    actions,
                    "",
                    "## Escalation",
                    str(artifact.get("escalation", "")),
                    "",
                    "## Status update template",
                    "```text",
                    str(artifact.get("comms_template", "")),
                    "```",
                    "",
                    f"> {artifact.get('note', '')}",
                ]
            )
            return FinalOutput(
                kind="triage",
                title=f"Triage: {str(artifact.get('severity', '')).upper()} — {artifact.get('service', '')}",
                summary=f"{len(artifact.get('likely_causes', []))} likely causes and {len(artifact.get('immediate_actions', []))} first actions prepared.",
                body=body,
                links=[],
                next_steps=["Confirm against live telemetry.", "Post the status update in the incident channel."],
                confidence=confidence,
                metadata={"tool": tool, "severity": artifact.get("severity"), "attempts": state.attempts},
            )

        if tool == "crm_query":
            contacts = artifact.get("contacts", [])
            deals = artifact.get("deals", [])
            contact_lines = "\n".join(
                f"| {item.get('name')} | {item.get('company')} | {item.get('email')} | {item.get('owner')} |"
                for item in contacts
            ) or "| No contacts matched | — | — | — |"
            deal_lines = "\n".join(
                f"| {item.get('name')} | {item.get('stage')} | ${int(item.get('amount', 0)):,} | {item.get('close_in_days')}d |"
                for item in deals
            ) or "| No deals matched | — | — | — |"
            body = "\n".join(
                [
                    "## CRM result",
                    f"**Query:** `{json.dumps(artifact.get('query', {}))}`",
                    "",
                    "### Contacts",
                    "| Name | Company | Email | Owner |",
                    "| --- | --- | --- | --- |",
                    contact_lines,
                    "",
                    "### Deals",
                    "| Deal | Stage | Amount | Close in |",
                    "| --- | --- | --- | --- |",
                    deal_lines,
                    "",
                    f"**Total pipeline value:** ${int(artifact.get('total_pipeline_value', 0)):,}",
                    "",
                    f"> {artifact.get('note', '')}",
                ]
            )
            return FinalOutput(
                kind="crm",
                title="CRM query result",
                summary=str(result.summary),
                body=body,
                links=[],
                next_steps=["Verify ownership and next steps with the account owner."],
                confidence=confidence,
                metadata={"tool": tool, "attempts": state.attempts},
            )

        return FinalOutput(
            kind="answer",
            title=f"Answer: {state.task[:80]}",
            summary=str(result.summary),
            body=str(artifact.get("body") or result.summary),
            links=[],
            next_steps=["Turn the first recommendation into a measurable task."],
            confidence=confidence,
            metadata={"tool": tool, "attempts": state.attempts},
        )

    def run(self, state: Any, llm: LLMClient | None = None, feedback: str = "") -> FinalOutput:
        if llm is not None:
            try:
                output = llm.complete_json(
                    system=self.system_prompt,
                    prompt=self.prompt(state, feedback),
                    schema=FinalOutput,
                )
                return self._ensure_contract(output, state)
            except Exception as exc:
                logger.warning("output agent: LLM unavailable (%s); using offline engine", exc)
        return self._ensure_contract(self.offline(state, feedback), state)

    def _ensure_contract(self, output: FinalOutput, state: Any) -> FinalOutput:
        artifact = (state.result or {}).get("artifact", {})
        links = list(output.links)
        if artifact.get("pr_url") and not any(link.url == artifact["pr_url"] for link in links):
            links.append({"label": "Open pull request", "url": str(artifact["pr_url"])})
        if not output.summary.strip():
            output = output.model_copy(update={"summary": state.task[:300]})
        expected_kind = {
            "code_fix": "pull_request",
            "email_reply": "email",
            "research": "report",
            "alert_triage": "triage",
            "crm_query": "crm",
            "generic_answer": "answer",
        }.get(str((state.result or {}).get("tool") or ""))
        updates: dict[str, Any] = {"links": links}
        if expected_kind and output.kind != expected_kind:
            updates["kind"] = expected_kind
        metadata = dict(output.metadata)
        metadata.setdefault("attempts", state.attempts)
        metadata.setdefault("strategy", state.strategy)
        updates["metadata"] = metadata
        return output.model_copy(update=updates)


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------
@dataclass
class Agents:
    reader: ReaderAgent
    planner: PlannerAgent
    executor: ExecutorAgent
    validator: ValidatorAgent
    output: OutputAgent

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": "reader", "role": "Extracts context, classifies task, flags risk"},
            {"name": "planner", "role": "Chooses simple vs complex strategy and plans steps"},
            {"name": "executor", "role": "Runs plan steps through the tool registry"},
            {"name": "validator", "role": "Checks acceptance criteria and drives retries"},
            {"name": "output", "role": "Composes the final deliverable"},
        ]


def build_agents(settings: Settings, tools: ToolRegistry) -> Agents:
    return Agents(
        reader=ReaderAgent(settings),
        planner=PlannerAgent(settings),
        executor=ExecutorAgent(settings, tools),
        validator=ValidatorAgent(settings),
        output=OutputAgent(settings),
    )
