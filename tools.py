"""Tools invoked by the Executor agent.

Every tool returns a validated :class:`StepResult`. Tools are deliberately
self-contained so new capabilities (Jira, Slack, Sentry, a vector store, ...)
can be registered without touching the graph.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import textwrap
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Protocol

from config import Settings
from state import AgentState, StepResult


class ToolError(RuntimeError):
    """A tool could not produce a usable artifact."""


class Tool(Protocol):
    name: str
    description: str

    def run(
        self,
        *,
        state: AgentState,
        step: dict[str, Any],
        feedback: str = "",
        prior: list[StepResult] | None = None,
    ) -> StepResult: ...


def _entities(state: AgentState) -> dict[str, Any]:
    entities = state.context.get("entities", {})
    return entities if isinstance(entities, dict) else {}


def _first(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str):
        return value
    return ""


def _clean_summary(text: str, limit: int = 400) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:limit] + ("..." if len(compact) > limit else "")


class BaseTool:
    name = "base"
    description = ""

    def _result(
        self,
        step: dict[str, Any],
        *,
        summary: str,
        artifact: dict[str, Any],
        success: bool = True,
    ) -> StepResult:
        return StepResult(
            step_id=int(step.get("id", 1)),
            tool=self.name,
            success=success,
            summary=summary,
            artifact=artifact,
        )


# ---------------------------------------------------------------------------
# Code fix / pull request
# ---------------------------------------------------------------------------
class CodeFixTool(BaseTool):
    name = "code_fix"
    description = "Diagnose a reported failure, produce a unified diff, and open a PR."

    def run(
        self,
        *,
        state: AgentState,
        step: dict[str, Any],
        feedback: str = "",
        prior: list[StepResult] | None = None,
    ) -> StepResult:
        entities = _entities(state)
        repo = _first(entities.get("repos")) or "acme/service"
        file_path = _first(entities.get("files")) or "src/service.py"
        error_type = _first(entities.get("error_type")) or "RuntimeError"
        line = entities.get("line") or 42
        diff, root_cause, tests = _build_patch(file_path, error_type, int(line))
        slug = re.sub(r"[^a-z0-9]+", "-", error_type.lower()).strip("-") or "fix"
        suffix = hashlib.sha1(f"{state.run_id}:{repo}:{file_path}".encode()).hexdigest()[:8]
        branch = f"fix/{slug}-{suffix}"
        pr_number = 100 + int(suffix[:5], 16) % 900
        pr_url = f"https://github.com/{repo}/pull/{pr_number}"

        artifact: dict[str, Any] = {
            "mode": "simulated",
            "repo": repo,
            "branch": branch,
            "base": "main",
            "file": file_path,
            "diff": diff,
            "root_cause": root_cause,
            "tests": tests,
            "pr_url": pr_url,
            "pr_title": f"fix: guard {file_path} against {error_type}",
            "files_changed": 1,
            "additions": diff.count("\n+"),
            "deletions": diff.count("\n-"),
        }
        if feedback:
            artifact["revision_notes"] = _clean_summary(feedback, 600)
        summary = (
            f"Prepared a one-file patch for {error_type} in {file_path} and opened "
            f"simulated PR #{pr_number} on {repo}. No GitHub credentials were used."
        )
        return self._result(step, summary=summary, artifact=artifact)


def _build_patch(file_path: str, error_type: str, line: int) -> tuple[str, str, list[str]]:
    normalized = error_type.lower()
    if "key" in normalized:
        replacement = (
            '    user_id = payload.get("user_id")\n'
            '    if not user_id:\n'
            '        raise ValueError("user_id is required")\n'
        )
        removed = '    user_id = payload["user_id"]\n'
        root = (
            "The handler indexes the payload directly, so a request without "
            "`user_id` raises KeyError. The patch reads the field defensively and "
            "returns a clear validation error instead of a 500."
        )
        tests = [
            "test_missing_user_id_returns_422",
            "test_valid_payload_still_resolves_user",
        ]
    elif "none" in normalized or "type" in normalized:
        replacement = (
            "    if record is None:\n"
            '        raise LookupError("record not found")\n'
            "    return record.get(\"status\", \"unknown\")\n"
        )
        removed = '    return record.get("status", "unknown")\n'
        root = (
            "The code dereferences a value that can be None when the upstream "
            "lookup misses. An explicit guard turns an accidental AttributeError "
            "into a predictable domain error."
        )
        tests = ["test_missing_record_raises_lookup_error", "test_record_status_is_returned"]
    elif "index" in normalized:
        replacement = (
            "    if not items:\n"
            '        return []\n'
            "    return items[0]\n"
        )
        removed = "    return items[0]\n"
        root = (
            "The empty collection path reaches an unchecked index access. The "
            "guard returns the documented empty result and removes the crash."
        )
        tests = ["test_empty_collection_returns_empty", "test_first_item_is_returned"]
    elif "import" in normalized or "module" in normalized:
        replacement = "from service.client import Client\n"
        removed = "from service.clients import Client\n"
        root = (
            "The module path moved during the package split and the stale import "
            "fails at startup. Pointing the import at the new module restores it."
        )
        tests = ["test_client_import_path", "test_client_can_be_constructed"]
    else:
        replacement = (
            "    try:\n"
            "        return process(payload)\n"
            "    except (KeyError, TypeError) as exc:\n"
            '        raise ValueError(f"invalid payload: {exc}") from exc\n'
        )
        removed = "    return process(payload)\n"
        root = (
            "The failing path receives malformed input and lets a low-level "
            "exception escape. The patch converts it into an explicit validation "
            "error while preserving the original cause."
        )
        tests = ["test_malformed_payload_returns_validation_error", "test_valid_payload_is_processed"]

    hunk_start = max(1, line - 6)
    removed_lines = removed.splitlines()
    replacement_lines = replacement.splitlines()
    old_count = 4 + len(removed_lines)
    new_count = 4 + len(replacement_lines)
    diff_lines = [
        f"--- a/{file_path}",
        f"+++ b/{file_path}",
        f"@@ -{hunk_start},{old_count} +{hunk_start},{new_count} @@",
        " def handle_request(payload: dict) -> dict:",
        *(f"-{item}" for item in removed_lines),
        *(f"+{item}" for item in replacement_lines),
        '     metadata = payload.get("metadata", {})',
        "     audit(payload, metadata)",
        "     return result",
    ]
    return "\n".join(diff_lines) + "\n", root, tests


class TestPlanTool(BaseTool):
    name = "test_plan"
    description = "Turn an issue into a regression test plan."

    def run(
        self,
        *,
        state: AgentState,
        step: dict[str, Any],
        feedback: str = "",
        prior: list[StepResult] | None = None,
    ) -> StepResult:
        entities = _entities(state)
        error_type = _first(entities.get("error_type")) or "the reported failure"
        file_path = _first(entities.get("files")) or "src/service.py"
        unit = f"Given {error_type}, the handler returns a typed error instead of crashing."
        regression = f"Regression test reproduces the original input against {file_path}."
        contract = "Unit and contract tests stay green; response codes remain backwards compatible."
        artifact = {
            "unit_tests": [regression, unit],
            "commands": ["pytest -q", "pytest -q tests/test_service.py -k regression"],
            "acceptance": [contract, "No new warnings in static analysis."],
        }
        summary = "Drafted a regression-first test plan with two unit tests and the pytest command."
        return self._result(step, summary=summary, artifact=artifact)


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------
class EmailReplyTool(BaseTool):
    name = "email_reply"
    description = "Draft a context-aware email reply."

    def run(
        self,
        *,
        state: AgentState,
        step: dict[str, Any],
        feedback: str = "",
        prior: list[StepResult] | None = None,
    ) -> StepResult:
        entities = _entities(state)
        supplied = state.user_context or {}
        to_addr = (
            _first(entities.get("emails"))
            or str(supplied.get("from") or supplied.get("to") or "customer@example.com")
        )
        sender_name = str(
            supplied.get("sender_name") or entities.get("sender_name") or ""
        ).strip()
        subject = str(supplied.get("subject") or entities.get("subject") or "").strip()
        if not subject:
            subject = f"Re: {_clean_summary(state.task, 70)}"
        elif not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"

        topic = str(entities.get("subject") or "your request").strip().rstrip(".")
        topic = topic[:1].lower() + topic[1:] if topic else "your request"
        if not topic.startswith(("your ", "our ", "the ")):
            topic = f"your {topic}"
        intent = state.task.lower()
        if any(word in intent for word in ("refund", "billing", "charge", "invoice")):
            bullets = [
                "I have shared the billing details with our finance team for review.",
                "You will receive a confirmation within one business day.",
                "No further action is needed from your side right now.",
            ]
            closing_action = "review the charge and confirm the outcome"
        elif any(word in intent for word in ("bug", "error", "broken", "not working", "crash")):
            bullets = [
                "I reproduced the issue and documented the steps you shared.",
                "Our engineering team is investigating; I will update you by the next business day.",
                "A temporary workaround is to retry after clearing the current session.",
            ]
            closing_action = "share the fix timeline with you"
        elif any(word in intent for word in ("demo", "pricing", "plan", "quote")):
            bullets = [
                "I have attached the plan details that match what you described.",
                "We can schedule a 30-minute walkthrough at a time that suits you.",
                "Let me know which date works and I will send an invite.",
            ]
            closing_action = "find a time for a short walkthrough"
        else:
            bullets = [
                "Thanks for the context; I have captured the key details.",
                "I am checking the open questions with the right team.",
                "I will follow up with a concrete answer shortly.",
            ]
            closing_action = "come back to you with a concrete answer"

        first_name = sender_name.split()[0] if sender_name else ""
        greeting = f"Hi {first_name}," if first_name else "Hi there,"
        body = (
            f"{greeting}\n\n"
            f"Thanks for reaching out about {topic}. "
            f"I want to make sure this is handled properly, so here is where things stand:\n\n"
            + "\n".join(f"- {item}" for item in bullets)
            + f"\n\nI will {closing_action}. If anything changes on your side, "
            "just reply to this email and I will pick it up.\n\n"
            "Best regards,\nCustomer Success Team"
        )
        if feedback:
            body += f"\n\nRevision note: { _clean_summary(feedback, 200) }"
        artifact = {
            "to": to_addr,
            "subject": subject,
            "body": body,
            "tone": "empathetic" if "frustrat" in intent or "angry" in intent else "professional",
            "draft": True,
        }
        summary = f"Drafted a {artifact['tone']} reply to {to_addr} with subject '{subject}'."
        return self._result(step, summary=summary, artifact=artifact)


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------
class ResearchTool(BaseTool):
    name = "research"
    description = "Produce a structured research brief from the request."

    def run(
        self,
        *,
        state: AgentState,
        step: dict[str, Any],
        feedback: str = "",
        prior: list[StepResult] | None = None,
    ) -> StepResult:
        topic = _research_topic(state.task)
        lowered = state.task.lower()
        questions = re.findall(r"([^?.!\n]{12,160}\?)", state.task)[:3]

        if re.search(r"\b(compare|comparison|versus|vs\.?|trade-?offs?)\b", lowered):
            options = _comparison_options(state.task)
            body, findings, recommendation, open_questions = _comparison_brief(
                topic, options, questions, state.task
            )
        elif "best practice" in lowered or "how do i" in lowered:
            body, findings, recommendation, open_questions = _practices_brief(topic, questions)
        else:
            body, findings, recommendation, open_questions = _landscape_brief(topic, questions)

        artifact = {
            "topic": topic,
            "body": body,
            "findings": findings,
            "recommendation": recommendation,
            "open_questions": open_questions,
            "citations": [],
            "limitations": _RESEARCH_LIMITATIONS,
            "sources_retrieved": 0,
        }
        summary = (
            f"Produced a structured research brief on '{topic}' with "
            f"{len(findings)} findings and a recommendation."
        )
        return self._result(step, summary=summary, artifact=artifact)


_RESEARCH_LIMITATIONS = (
    "Offline synthesis: no external sources were retrieved and no URLs are fabricated. "
    "Set LLM_PROVIDER=openai for model-written analysis, or add a search provider to "
    "ground claims in live sources."
)


def _research_topic(task: str) -> str:
    text = re.sub(r"\s+", " ", task.strip())
    first = re.split(r"(?<=[.!?])\s+", text)[0]
    first = re.sub(
        r"^(?:please\s+)?(?:research|investigate|analyse|analyze|explore|find out about|summari[sz]e)\s+(?:the\s+)?",
        "",
        first,
        flags=re.IGNORECASE,
    )
    first = re.sub(r"^(?:the\s+)", "", first, flags=re.IGNORECASE)
    cleaned = first.strip(" .?!")
    return (cleaned[:200] if cleaned else _clean_summary(task, 160)) or "the requested topic"


def _comparison_options(task: str) -> tuple[str, str]:
    cleaned = re.sub(r"\s+", " ", task)
    match = re.search(
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?=\s+(?:for|in|on|with|regarding|to)\s+|[.,;]|$)",
        cleaned,
        re.IGNORECASE,
    )
    if match:
        return _clean_option(match.group(1)), _clean_option(match.group(2))
    match = re.search(
        r"([A-Za-z0-9 .+#_-]{2,50}?)\s+(?:vs\.?|versus)\s+([A-Za-z0-9 .+#_-]{2,50})",
        cleaned,
        re.IGNORECASE,
    )
    if match:
        return _clean_option(match.group(1)), _clean_option(match.group(2))
    return "Option A", "Option B"


def _clean_option(value: str) -> str:
    option = value.strip(" ,.;:-")
    option = re.sub(r"^(?:the|a|an)\s+", "", option, flags=re.IGNORECASE)
    option = re.sub(
        r"\s+(?:architectures?|approaches?|models?|systems?|frameworks?|solutions?)$",
        "",
        option,
        flags=re.IGNORECASE,
    )
    return option.strip(" ,.;:-") or "Option"


def _comparison_brief(
    topic: str, options: tuple[str, str], questions: list[str], task: str
) -> tuple[str, list[str], str, list[str]]:
    left, right = options
    left_is_event = bool(re.search(r"event|async|stream", left, re.IGNORECASE))
    right_is_event = bool(re.search(r"event|async|stream", right, re.IGNORECASE))
    left_is_sync = bool(re.search(r"request|sync|rest|http|rpc", left, re.IGNORECASE))
    right_is_sync = bool(re.search(r"request|sync|rest|http|rpc", right, re.IGNORECASE))
    is_event_sync_pair = (left_is_event and right_is_sync) or (right_is_event and left_is_sync)
    payments = "payment" in task.lower()

    def profile(is_event: bool, is_sync: bool) -> dict[str, str]:
        if is_event:
            return {
                "Reliability": "Retries, replay and buffering; eventual consistency",
                "Latency": "Queueing can add tail latency",
                "Operational complexity": "Higher: broker, DLQ, tracing, idempotency",
                "Cost": "Broker capacity plus duplicated storage",
                "Best fit": "Fan-out, spiky or long-running work",
            }
        if is_sync:
            return {
                "Reliability": "Failures surface immediately; limited replay",
                "Latency": "Predictable per-call latency",
                "Operational complexity": "Lower: fewer moving parts",
                "Cost": "Scales directly with request volume",
                "Best fit": "User-facing calls and payment authorization",
            }
        return {
            "Reliability": "Depends on delivery guarantees",
            "Latency": "Measure p95/p99 under peak load",
            "Operational complexity": "Depends on the existing platform",
            "Cost": "Model at peak and average load",
            "Best fit": "Depends on workflow shape",
        }

    left_profile = profile(left_is_event, left_is_sync)
    right_profile = profile(right_is_event, right_is_sync)
    rows = [
        (dimension, left_profile[dimension], right_profile[dimension])
        for dimension in [
            "Reliability",
            "Latency",
            "Operational complexity",
            "Cost",
            "Best fit",
        ]
    ]

    if is_event_sync_pair:
        findings = [
            "Event-driven delivery decouples producers from consumers but requires "
            "idempotent handlers, a dead-letter path and replay tooling.",
            "Request-response keeps failure visible to the caller and is simpler to "
            "operate, but couples availability and latency.",
            (
                "For payments, keep authorization synchronous and move settlement, "
                "notifications and analytics to events."
                if payments
                else "A hybrid split usually beats a platform-wide binary choice."
            ),
        ]
        recommendation = (
            "Keep the payment authorization path request-response for immediate "
            "confirmation, and use event-driven delivery for settlement, notifications "
            "and analytics. Require idempotency keys and replay tests before launch."
            if payments
            else "Split by workflow: keep user-facing calls synchronous and move fan-out, "
            "retries and long-running work to events. Pilot the highest-risk workflow first."
        )
        defaults = [
            "What are the p95 latency and peak-throughput targets?",
            "Which consumers require replay or exactly-once delivery?",
            "What is the recovery point objective during a broker or downstream failure?",
        ]
        summary = (
            f"The choice between {left} and {right} is a trade-off between failure "
            "isolation and operational simplicity. "
            f"**Recommendation:** {recommendation}"
        )
    else:
        findings = [
            f"{left} and {right} overlap on the baseline; operational fit is the differentiator.",
            "Failure recovery and observability usually dominate long-term maintenance cost.",
            "A one-week spike on the highest-risk workflow produces better evidence than a feature matrix.",
        ]
        recommendation = (
            "Run a two-week, single-workflow pilot with explicit success metrics, then "
            "compare reliability, operator effort and cost before committing."
        )
        defaults = [
            "Which constraint cannot change (latency, compliance, budget)?",
            "What does success look like numerically?",
            "What is the rollback path if the choice proves wrong?",
        ]
        summary = (
            f"{left} and {right} can both cover the baseline; the decision should be "
            "driven by operational fit and measured pilot results. "
            f"**Recommendation:** {recommendation}"
        )

    open_questions = list(dict.fromkeys(questions + defaults))[:5]
    lines = [
        "## Executive summary",
        summary,
        "",
        "## Comparison",
        f"| Dimension | {left} | {right} |",
        "| --- | --- | --- |",
        *(f"| {name} | {a} | {b} |" for name, a, b in rows),
        "",
        "## Key findings",
        *(f"- {finding}" for finding in findings),
        "",
        "## Recommendation",
        recommendation,
        "",
        "## Open questions",
        *(f"- {question}" for question in open_questions),
        "",
        "## Evidence & limitations",
        _RESEARCH_LIMITATIONS,
    ]
    return "\n".join(lines), findings, recommendation, open_questions


def _practices_brief(
    topic: str, questions: list[str]
) -> tuple[str, list[str], str, list[str]]:
    findings = [
        "Define acceptance criteria before choosing tools; tooling decisions age faster than outcomes.",
        "Instrument a baseline so improvements and regressions are visible.",
        "Keep changes small enough to review and roll back within one sitting.",
        "Assign a single owner for each workflow and its runbook.",
    ]
    recommendation = (
        "Define a measurable target, implement the smallest reversible slice behind a "
        "flag, and automate the check that proves it worked before scaling."
    )
    open_questions = list(
        dict.fromkeys(
            questions
            + [
                "Which constraint is non-negotiable?",
                "What two metrics prove success?",
                "Who owns the workflow after launch?",
            ]
        )
    )[:5]
    lines = [
        "## Executive summary",
        f"For {topic}, the reliable pattern is to define a measurable target, make the "
        "smallest reversible change, and automate the check that proves it worked.",
        "",
        "## Key findings",
        *(f"- {finding}" for finding in findings),
        "",
        "## Recommended approach",
        "1. Write a one-page outcome statement with two metrics.",
        "2. Implement a thin vertical slice behind a feature flag.",
        "3. Add automated checks for the happy path and one failure path.",
        "4. Review results after one week; scale only if the metrics moved.",
        "",
        "## Risks",
        "- Scope drift when success is subjective.",
        "- Hidden manual steps that do not survive handover.",
        "- Premature standardisation before the workflow is understood.",
        "",
        "## Open questions",
        *(f"- {question}" for question in open_questions),
        "",
        "## Evidence & limitations",
        _RESEARCH_LIMITATIONS,
    ]
    return "\n".join(lines), findings, recommendation, open_questions


def _landscape_brief(
    topic: str, questions: list[str]
) -> tuple[str, list[str], str, list[str]]:
    findings = [
        f"The request centres on {topic}.",
        "Constraints (latency, compliance, budget) eliminate more options than features do.",
        "A time-boxed pilot produces better evidence than extended desk research.",
    ]
    recommendation = (
        "Frame the decision as one workflow, two success metrics and a four-week horizon. "
        "Run the cheapest option that satisfies the hard constraint, then decide."
    )
    open_questions = list(
        dict.fromkeys(
            questions
            + [
                "What decision does this research unblock?",
                "What is the binding constraint?",
                "What would make the recommendation wrong?",
            ]
        )
    )[:5]
    lines = [
        "## Executive summary",
        f"The question is best framed as a decision about {topic}, not a request for a "
        "catalogue of options. Start with the constraint that cannot change, then compare "
        "only the approaches that respect it.",
        "",
        "## Landscape",
        "- **Incremental approach** - lowest coordination cost, fastest to validate.",
        "- **Platform approach** - higher upfront cost, better reuse across teams.",
        "- **Buy / adopt approach** - fastest coverage, ongoing vendor and integration cost.",
        "",
        "## Key findings",
        *(f"- {finding}" for finding in findings),
        "",
        "## Recommendation",
        recommendation,
        "",
        "## Open questions",
        *(f"- {question}" for question in open_questions),
        "",
        "## Evidence & limitations",
        _RESEARCH_LIMITATIONS,
    ]
    return "\n".join(lines), findings, recommendation, open_questions


# ---------------------------------------------------------------------------
# CRM
# ---------------------------------------------------------------------------
class CRMQueryTool(BaseTool):
    name = "crm_query"
    description = "Query the demo CRM for contacts, deals and pipeline context."

    CONTACTS = [
        ("Avery Chen", "avery.chen@northwind.example", "Northwind Labs", "VP Engineering", "maya"),
        ("Ravi Patel", "ravi@acme.example", "Acme Corp", "CTO", "maya"),
        ("Sofia Ruiz", "sofia.ruiz@lumen.example", "Lumen Health", "Director of Ops", "jon"),
        ("Noah Williams", "noah@helios.example", "Helios Energy", "Head of Data", "jon"),
        ("Priya Nair", "priya@atlas.example", "Atlas Freight", "COO", "maya"),
    ]
    DEALS = [
        ("Northwind platform expansion", "avery.chen@northwind.example", "Negotiation", 84000, 21),
        ("Acme enterprise rollout", "ravi@acme.example", "Proposal", 120000, 45),
        ("Lumen analytics pilot", "sofia.ruiz@lumen.example", "Discovery", 32000, 60),
        ("Helios data platform", "noah@helios.example", "Qualified", 56000, 38),
        ("Atlas fleet telemetry", "priya@atlas.example", "Negotiation", 97000, 14),
        ("Acme renewals", "ravi@acme.example", "Closed Won", 45000, -30),
    ]

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db_path = Path(settings.crm_db_path)
        self._ensure_seed()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_seed(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS contacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    company TEXT NOT NULL,
                    title TEXT NOT NULL,
                    owner TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS deals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    contact_email TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    close_in_days INTEGER NOT NULL,
                    owner TEXT NOT NULL
                )"""
            )
            count = conn.execute("SELECT COUNT(*) AS n FROM contacts").fetchone()["n"]
            if count == 0:
                conn.executemany(
                    "INSERT INTO contacts(name, email, company, title, owner) VALUES (?,?,?,?,?)",
                    self.CONTACTS,
                )
                conn.executemany(
                    """INSERT INTO deals(name, contact_email, stage, amount, close_in_days, owner)
                       VALUES (?,?,?,?,?,?)""",
                    [(name, email, stage, amount, days, "maya" if idx % 2 == 0 else "jon")
                     for idx, (name, email, stage, amount, days) in enumerate(self.DEALS)],
                )

    def run(
        self,
        *,
        state: AgentState,
        step: dict[str, Any],
        feedback: str = "",
        prior: list[StepResult] | None = None,
    ) -> StepResult:
        entities = _entities(state)
        supplied = state.user_context or {}
        email = _first(entities.get("emails")) or str(supplied.get("email") or "").strip()
        company = str(
            entities.get("company")
            or supplied.get("company")
            or _guess_company(state.task, self._known_companies())
            or ""
        ).strip()

        contact_clauses: list[str] = []
        deal_clauses: list[str] = []
        contact_params: list[Any] = []
        deal_params: list[Any] = []
        if email:
            contact_clauses.append("email = ?")
            contact_params.append(email)
            deal_clauses.append("contact_email = ?")
            deal_params.append(email)
        if company:
            contact_clauses.append("company LIKE ?")
            contact_params.append(f"%{company}%")
            deal_clauses.append(
                "contact_email IN (SELECT email FROM contacts WHERE company LIKE ?)"
            )
            deal_params.append(f"%{company}%")

        contact_where = f"WHERE {' OR '.join(contact_clauses)}" if contact_clauses else ""
        deal_where = f"WHERE {' OR '.join(deal_clauses)}" if deal_clauses else ""
        with self._connect() as conn:
            contacts = [dict(row) for row in conn.execute(
                f"SELECT name, email, company, title, owner FROM contacts {contact_where} ORDER BY name",
                contact_params,
            ).fetchall()]
            if deal_clauses:
                deals = [dict(row) for row in conn.execute(
                    f"""SELECT name, contact_email, stage, amount, close_in_days, owner
                        FROM deals {deal_where} ORDER BY amount DESC""",
                    deal_params,
                ).fetchall()]
            else:
                deals = [dict(row) for row in conn.execute(
                    "SELECT name, contact_email, stage, amount, close_in_days, owner "
                    "FROM deals ORDER BY amount DESC LIMIT 5"
                ).fetchall()]
            pipeline_rows = conn.execute(
                "SELECT stage, COUNT(*) AS count, SUM(amount) AS amount FROM deals GROUP BY stage"
            ).fetchall()

        pipeline = {
            row["stage"]: {"count": row["count"], "amount": row["amount"]}
            for row in pipeline_rows
        }
        total_pipeline = sum(item["amount"] for item in pipeline.values())
        target = email or company or "the full pipeline"
        artifact = {
            "query": {"email": email, "company": company},
            "contacts": contacts,
            "deals": deals,
            "pipeline": pipeline,
            "total_pipeline_value": total_pipeline,
            "matched": bool(contacts or (contact_clauses and deals)),
            "note": (
                "Read-only query against the bundled demo CRM (SQLite). "
                "No records were modified."
            ),
        }
        if not contact_clauses:
            summary = (
                f"No filter supplied; summarised the demo pipeline: {len(deals)} largest deals "
                f"and ${total_pipeline:,.0f} total across {len(pipeline)} stages."
            )
        else:
            summary = (
                f"Queried the demo CRM for {target}: {len(contacts)} contact(s), "
                f"{len(deals)} deal(s), ${sum(d.get('amount', 0) for d in deals):,.0f} in matching value."
            )
        return self._result(step, summary=summary, artifact=artifact)

    def _known_companies(self) -> list[str]:
        with self._connect() as conn:
            return [row["company"] for row in conn.execute("SELECT DISTINCT company FROM contacts")]


def _guess_company(task: str, known: list[str]) -> str:
    lowered = task.lower()
    for company in known:
        if company.lower() in lowered:
            return company
    match = re.search(r"(?:company|account|customer)\s*[:\-]?\s*([A-Z][\w&.\- ]{2,40})", task)
    return match.group(1).strip() if match else ""


# ---------------------------------------------------------------------------
# Alert triage
# ---------------------------------------------------------------------------
class AlertTriageTool(BaseTool):
    name = "alert_triage"
    description = "Triage an operational alert into severity, causes and actions."

    def run(
        self,
        *,
        state: AgentState,
        step: dict[str, Any],
        feedback: str = "",
        prior: list[StepResult] | None = None,
    ) -> StepResult:
        entities = _entities(state)
        text = f"{state.task} {state.context.get('summary', '')}".lower()
        service = (
            _first(entities.get("services"))
            or str((state.user_context or {}).get("service") or "")
            or _service_from_text(state.task)
        )
        severity = _severity(text)
        causes = _likely_causes(text)
        actions = [
            f"Open the {service} dashboard and confirm the alert is not a monitoring artefact.",
            "Check the deploy timeline for changes in the last 60 minutes; roll back if correlated.",
            "Inspect error logs and saturation metrics (CPU, memory, connection pool, queue depth).",
            "Apply the documented mitigation and record the timestamp in the incident channel.",
            "Escalate to the on-call owner if the error budget burn continues for 15 minutes.",
        ]
        escalation = (
            "Page the service owner and incident commander now (Sev1)."
            if severity == "critical"
            else "Notify the service owner in the on-call channel; escalate after 30 minutes."
        )
        artifact = {
            "severity": severity,
            "service": service,
            "likely_causes": causes,
            "immediate_actions": actions,
            "escalation": escalation,
            "comms_template": (
                f"[{severity.upper()}] {service}: investigating elevated errors. "
                "Owner assigned; next update in 15 minutes."
            ),
            "runbooks": [],
            "note": "Offline triage from alert text; verify against live telemetry.",
        }
        summary = f"Triaged a {severity} alert for {service} with {len(causes)} likely causes and five immediate actions."
        return self._result(step, summary=summary, artifact=artifact)


def _severity(text: str) -> str:
    if re.search(r"\b(sev[\s-]?1|p0|critical|outage|down)\b", text):
        return "critical"
    if re.search(r"\b(sev[\s-]?2|p1|high|5xx|error rate|latency|timeout)\b", text):
        return "high"
    return "medium"


def _service_from_text(task: str) -> str:
    match = re.search(r"\b(?:service|host|pod|api)[\s:`]+([\w.-]+)", task, re.IGNORECASE)
    if match:
        return match.group(1).strip("`.")
    host = re.search(r"\b([a-z0-9-]+(?:\.[a-z0-9-]+){2,})\b", task, re.IGNORECASE)
    return host.group(1) if host else "unknown-service"


def _likely_causes(text: str) -> list[str]:
    causes: list[str] = []
    if "5xx" in text or "error" in text:
        causes.append("A recent deploy or dependency change is returning server errors.")
    if "latency" in text or "timeout" in text or "slow" in text:
        causes.append("Saturation or a slow downstream dependency is increasing response time.")
    if "cpu" in text:
        causes.append("Traffic growth or a hot loop is driving CPU above capacity.")
    if "memory" in text or "oom" in text:
        causes.append("A leak or oversized payloads are exhausting memory.")
    if "disk" in text:
        causes.append("Log or data growth is filling the disk.")
    if not causes:
        causes.append("Insufficient signal in the alert text; inspect the primary metric first.")
    causes.append("A configuration or feature-flag change altered runtime behaviour.")
    return causes[:4]


# ---------------------------------------------------------------------------
# Generic answer
# ---------------------------------------------------------------------------
class GenericAnswerTool(BaseTool):
    name = "generic_answer"
    description = "Answer a general task with assumptions, steps and risks."

    def run(
        self,
        *,
        state: AgentState,
        step: dict[str, Any],
        feedback: str = "",
        prior: list[StepResult] | None = None,
    ) -> StepResult:
        summary = _clean_summary(state.task, 240)
        body = textwrap.dedent(
            f"""\
            ## Answer
            {summary}

            ## How to approach it
            1. Clarify the desired outcome and the constraint that cannot change.
            2. Choose the smallest reversible step that produces evidence.
            3. Define how you will know it worked before implementing it.

            ## Assumptions
            - The request is self-contained; no private systems were accessed.
            - A human owner will review the result before external use.

            ## Risks
            - Missing context can make a plausible answer confidently wrong.
            - Operational cost and maintenance are often underestimated.

            ## Next step
            Convert the first action above into a one-week task with a measurable success check.
            """
        ).strip()
        artifact = {
            "body": body,
            "key_points": [
                "Outcome and hard constraint first.",
                "Smallest reversible step.",
                "Measurable success check.",
            ],
            "assumptions": ["Self-contained request", "Human review before external use"],
        }
        return self._result(step, summary=f"Produced a structured answer for: {summary}", artifact=artifact)


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------
class ReviewTool(BaseTool):
    name = "review"
    description = "Cross-check artifacts against the plan's acceptance criteria."

    def run(
        self,
        *,
        state: AgentState,
        step: dict[str, Any],
        feedback: str = "",
        prior: list[StepResult] | None = None,
    ) -> StepResult:
        criteria = list(state.plan.get("validation_criteria", [])) or [
            "Artifact exists",
            "Result is actionable",
        ]
        artifacts = [item.artifact for item in prior or [] if item.success]
        joined = repr(artifacts)
        checklist = [
            {
                "criterion": criterion,
                "status": "pass" if artifacts else "fail",
                "note": "Artifact present in this run." if artifacts else "No upstream artifact.",
            }
            for criterion in criteria
        ]
        verdict = "pass" if artifacts and "TODO" not in joined else "needs attention"
        artifact = {"checklist": checklist, "verdict": verdict}
        summary = f"Reviewed {len(artifacts)} artifact(s) against {len(criteria)} acceptance criteria: {verdict}."
        return self._result(step, summary=summary, artifact=artifact)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class ToolRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._tools: dict[str, Tool] = {
            tool.name: tool
            for tool in (
                CodeFixTool(),
                TestPlanTool(),
                EmailReplyTool(),
                ResearchTool(),
                CRMQueryTool(settings),
                AlertTriageTool(),
                GenericAnswerTool(),
                ReviewTool(),
            )
        }

    def names(self) -> list[str]:
        return sorted(self._tools)

    def register(self, tool: Tool) -> None:
        """Add or replace a tool without changing the graph or agents."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(
                f"Unknown tool '{name}'. Available: {', '.join(self.names())}"
            ) from exc

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": tool.name, "description": tool.description}
            for tool in sorted(self._tools.values(), key=lambda item: item.name)
        ]

    def execute(
        self,
        name: str,
        *,
        state: AgentState,
        step: dict[str, Any],
        feedback: str = "",
        prior: list[StepResult] | None = None,
    ) -> StepResult:
        tool = self.get(name)
        result = tool.run(state=state, step=step, feedback=feedback, prior=prior or [])
        if not isinstance(result, StepResult):
            result = StepResult.model_validate(result)
        if not result.success:
            raise ToolError(f"Tool '{name}' reported failure: {result.summary}")
        return result
