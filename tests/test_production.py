"""Production-gate behaviour: bearer auth, key parsing, durable persistence."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app import create_app
from config import Settings, _parse_api_keys
from persistence import MemoryRunStore, SqliteRunStore, build_checkpointer, build_run_store
from tests.conftest import EMAIL_TASK, RESEARCH_TASK
from workflow import Orchestrator


# --------------------------------------------------------------------- keys
def test_parse_api_keys_variants() -> None:
    assert _parse_api_keys("") == ()
    assert _parse_api_keys("  ") == ()
    assert _parse_api_keys("solo-key") == (("solo-key", "api-client"),)
    assert _parse_api_keys("k1:alice,k2:bob") == (("k1", "alice"), ("k2", "bob"))
    assert _parse_api_keys("k1:alice, ,k2") == (("k1", "alice"), ("k2", "api-client"))


# --------------------------------------------------------------------- auth
def _auth_settings(tmp_path: Path) -> Settings:
    return Settings(
        crm_db_path=tmp_path / "crm.db",
        agent_max_retries=0,
        api_auth_enabled=True,
        api_key_identities=(("test-secret", "ci-reviewer"),),
    )


def test_auth_rejects_missing_and_bad_tokens(tmp_path: Path) -> None:
    settings = _auth_settings(tmp_path)
    app = create_app(settings, Orchestrator(settings=settings, llm=None))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200  # liveness stays open

        denied = client.post("/run", json={"task": RESEARCH_TASK})
        assert denied.status_code == 401

        bad = client.post(
            "/run",
            json={"task": RESEARCH_TASK},
            headers={"Authorization": "Bearer wrong"},
        )
        assert bad.status_code == 401

        listing = client.get("/runs")
        assert listing.status_code == 401


def test_auth_accepts_valid_key_and_overrides_reviewer(tmp_path: Path) -> None:
    settings = _auth_settings(tmp_path)
    app = create_app(settings, Orchestrator(settings=settings, llm=None))
    headers = {"Authorization": "Bearer test-secret"}
    with TestClient(app) as client:
        created = client.post(
            "/run",
            json={"task": EMAIL_TASK, "auto_approve": False},
            headers=headers,
        )
        assert created.status_code == 200
        body = created.json()
        assert body["status"] == "awaiting_approval"
        # The authenticated identity is stamped into run metadata.
        assert body["state"]["metadata"]["submitted_by"] == "ci-reviewer"

        approved = client.post(
            "/approve",
            json={
                "run_id": body["run_id"],
                "approved": True,
                "reviewer": "spoofed-name",
            },
            headers=headers,
        )
        assert approved.status_code == 200
        assert approved.json()["state"]["approval"]["reviewer"] == "ci-reviewer"


def test_auth_enabled_without_keys_fails_closed(tmp_path: Path) -> None:
    settings = Settings(
        crm_db_path=tmp_path / "crm.db",
        api_auth_enabled=True,
        api_key_identities=(),
    )
    app = create_app(settings, Orchestrator(settings=settings, llm=None))
    with TestClient(app) as client:
        denied = client.post("/run", json={"task": RESEARCH_TASK})
        assert denied.status_code == 503  # enabled but unconfigured: reject, never allow


# -------------------------------------------------------------- persistence
def test_sqlite_run_store_roundtrip(tmp_path: Path) -> None:
    settings = Settings(run_store_backend="sqlite", run_db_path=tmp_path / "runs.db")
    store = build_run_store(settings)
    assert isinstance(store, SqliteRunStore)
    app_settings = Settings(crm_db_path=tmp_path / "crm.db", agent_max_retries=0)
    orch = Orchestrator(settings=app_settings, llm=None, run_store=store)
    from state import RunRequest

    response = orch.start(RunRequest(task=RESEARCH_TASK, auto_approve=True))
    run_id = response.run_id
    orch.close()

    # Fresh store instance on the same file: record still there.
    store2 = SqliteRunStore(tmp_path / "runs.db")
    record = store2.get(run_id)
    assert record is not None
    assert record.state.status == "completed"
    assert store2.count() >= 1
    store2.close()


def test_memory_run_store_default() -> None:
    settings = Settings()
    store = build_run_store(settings)
    assert isinstance(store, MemoryRunStore)


def test_checkpointer_builds_for_both_backends(tmp_path: Path) -> None:
    saver, close = build_checkpointer(Settings(checkpoint_backend="memory"))
    assert saver is not None
    close()

    sqlite_saver, close_sqlite = build_checkpointer(
        Settings(checkpoint_backend="sqlite", checkpoint_db_path=tmp_path / "ckpt.db")
    )
    assert sqlite_saver is not None
    close_sqlite()


def test_approval_survives_orchestrator_restart(tmp_path: Path) -> None:
    """The deployment-critical guarantee: pause, restart the API, still approve."""
    persistence_settings = Settings(
        crm_db_path=tmp_path / "crm.db",
        agent_max_retries=0,
        checkpoint_backend="sqlite",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        run_store_backend="sqlite",
        run_db_path=tmp_path / "runs.db",
    )

    # First "process": start a run that pauses at the approval gate.
    first = Orchestrator(settings=persistence_settings, llm=None)
    from state import RunRequest

    created = first.start(RunRequest(task=EMAIL_TASK, auto_approve=False))
    assert created.status == "awaiting_approval"
    run_id = created.run_id
    first.close()  # simulate API shutdown

    # Second "process": same settings, fresh in-memory registries.
    second = Orchestrator(settings=persistence_settings, llm=None)
    try:
        restored = second.get(run_id)
        assert restored.status == "awaiting_approval"
        assert restored.approval is not None

        from state import ApprovalDecision

        resumed = second.approve(
            ApprovalDecision(run_id=run_id, approved=True, reviewer="night-ops")
        )
        assert resumed.status == "completed"
        assert resumed.state["output"]["kind"] == "email"
        assert resumed.state["approval"]["reviewer"] == "night-ops"
    finally:
        second.close()
