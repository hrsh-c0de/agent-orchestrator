from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Settings
from workflow import Orchestrator

GITHUB_TASK = (
    "Title: TypeError when loading a user profile\n"
    "Repo: acme/webapp\n"
    "File: app/users.py line 84\n\n"
    "Traceback (most recent call last):\n"
    '  File "app/users.py", line 84, in get_profile\n'
    "    return user.profile.name\n"
    "TypeError: 'NoneType' object has no attribute 'profile'\n\n"
    "A deleted user still has a session. Expected behaviour is a clean 404. "
    "Please fix the root cause and open a PR."
)

EMAIL_TASK = (
    "Draft a reply to this email.\n\n"
    "From: Priya Nair <priya@atlas.example>\n"
    "Subject: Refund request for duplicate charge\n\n"
    "Hi team,\n"
    "We were charged twice for the September invoice. Could you refund the duplicate "
    "charge and confirm when it will appear? Our finance close is Friday."
)

RESEARCH_TASK = (
    "Research the trade-offs between event-driven and request-response architectures for "
    "a payments platform. Compare reliability, latency, operational complexity and cost. "
    "Include a recommendation and open questions."
)

ALERT_TASK = (
    "ALERT: checkout-api 5xx error rate is 12% for the last 10 minutes; p95 latency 2.4s. "
    "Service: checkout-api. A deploy went out 25 minutes ago. Draft an incident triage summary."
)

CRM_TASK = "Find the deal status for Acme Corp in the CRM."


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        crm_db_path=tmp_path / "crm.db",
        agent_max_retries=0,
        agent_timeout_seconds=20.0,
        max_attempts=3,
        validation_threshold=0.75,
        default_auto_approve=False,
    )


@pytest.fixture()
def orchestrator(settings: Settings) -> Orchestrator:
    return Orchestrator(settings=settings, llm=None)
