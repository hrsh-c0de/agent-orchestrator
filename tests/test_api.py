from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from app import create_app
from config import Settings
from tests.conftest import EMAIL_TASK, RESEARCH_TASK
from workflow import Orchestrator


@pytest.fixture()
def client(settings: Settings):
    app = create_app(settings, Orchestrator(settings=settings, llm=None))
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_graph(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["llm_enabled"] is False

    graph = client.get("/graph")
    assert graph.status_code == 200
    spec = graph.json()
    node_ids = {node["id"] for node in spec["nodes"]}
    assert {"reader", "planner", "executor", "validator", "approval", "output"} <= node_ids
    assert "flowchart" in spec["mermaid"]
    assert "digraph" in spec["dot"]
    assert any(edge["source"] == "validator" and edge["target"] == "executor" for edge in spec["edges"])


def test_run_research(client: TestClient) -> None:
    response = client.post(
        "/run",
        json={"task": RESEARCH_TASK, "auto_approve": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["state"]["output"]["kind"] == "report"

    fetched = client.get(f"/runs/{body['run_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["run_id"] == body["run_id"]

    listing = client.get("/runs?limit=5")
    assert listing.status_code == 200
    assert listing.json()


def test_email_approval_flow_and_conflicts(client: TestClient) -> None:
    created = client.post(
        "/run",
        json={"task": EMAIL_TASK, "auto_approve": False},
    )
    assert created.status_code == 200
    payload = created.json()
    assert payload["status"] == "awaiting_approval"
    assert payload["approval"]["task_type"] == "email"
    run_id = payload["run_id"]

    approved = client.post(
        "/approve",
        json={"run_id": run_id, "approved": True, "reviewer": "pytest", "comment": "ok"},
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "completed"
    assert approved.json()["state"]["output"]["kind"] == "email"

    again = client.post(
        "/approve",
        json={"run_id": run_id, "approved": True},
    )
    assert again.status_code == 409

    unknown = client.post(
        "/approve",
        json={"run_id": "missing", "approved": True},
    )
    assert unknown.status_code == 404


def test_validation_errors(client: TestClient) -> None:
    blank = client.post("/run", json={"task": "   "})
    assert blank.status_code == 422

    missing = client.get("/runs/nope")
    assert missing.status_code == 404

    bad_limit = client.get("/runs?limit=500")
    assert bad_limit.status_code == 422
