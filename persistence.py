"""Persistence adapters.

The demo defaults to in-memory storage, but deployment uses SQLite for both the
LangGraph checkpointer and the run registry. That makes paused approvals
survive an API restart and lets multiple API workers share state on one host.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from langgraph.checkpoint.memory import InMemorySaver

from config import Settings
from state import AgentState

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    state: AgentState
    created_at: str
    updated_at: str
    approval: dict[str, Any] | None = None
    duration_ms: float | None = None
    started_monotonic: float = field(default_factory=time.perf_counter)

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.state.run_id,
            "state": self.state.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "approval": self.approval,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RunRecord":
        state = AgentState.from_dict(payload["state"])
        duration = payload.get("duration_ms")
        return cls(
            state=state,
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            approval=payload.get("approval"),
            duration_ms=float(duration) if duration is not None else None,
            started_monotonic=time.perf_counter()
            - ((float(duration) if duration is not None else 0.0) / 1000.0),
        )


class RunStore(Protocol):
    def put(self, record: RunRecord) -> None: ...

    def get(self, run_id: str) -> RunRecord | None: ...

    def list(self, limit: int = 20) -> list[RunRecord]: ...

    def close(self) -> None: ...


class MemoryRunStore:
    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def put(self, record: RunRecord) -> None:
        with self._lock:
            self._records[record.state.run_id] = record.to_payload()

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            payload = self._records.get(run_id)
            return RunRecord.from_payload(payload) if payload else None

    def list(self, limit: int = 20) -> list[RunRecord]:
        with self._lock:
            payloads = sorted(
                self._records.values(),
                key=lambda item: item["created_at"],
                reverse=True,
            )[: max(1, limit)]
            return [RunRecord.from_payload(payload) for payload in payloads]

    def close(self) -> None:
        return None


class SqliteRunStore:
    """Small durable run registry keyed by run id.

    Records are isolated serialised snapshots, so callers never share mutable
    state with another worker or request.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    duration_ms REAL,
                    payload TEXT NOT NULL
                )"""
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC)"
            )
            self._conn.commit()

    def put(self, record: RunRecord) -> None:
        payload = json.dumps(record.to_payload(), ensure_ascii=False, default=str)
        with self._lock:
            self._conn.execute(
                """INSERT INTO runs(run_id, created_at, updated_at, duration_ms, payload)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(run_id) DO UPDATE SET
                     updated_at=excluded.updated_at,
                     duration_ms=excluded.duration_ms,
                     payload=excluded.payload""",
                (
                    record.state.run_id,
                    record.created_at,
                    record.updated_at,
                    record.duration_ms,
                    payload,
                ),
            )
            self._conn.commit()

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return RunRecord.from_payload(json.loads(row["payload"]))

    def list(self, limit: int = 20) -> list[RunRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM runs ORDER BY created_at DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        return [RunRecord.from_payload(json.loads(row["payload"])) for row in rows]

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()
        return int(row["n"]) if row else 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def build_run_store(settings: Settings) -> RunStore:
    backend = settings.run_store_backend.strip().lower()
    if backend == "sqlite":
        logger.info("Run registry: SQLite (%s)", settings.run_db_path)
        return SqliteRunStore(settings.run_db_path)
    if backend != "memory":
        logger.warning("Unknown RUN_STORE_BACKEND=%r; using memory", backend)
    return MemoryRunStore()


def build_checkpointer(settings: Settings) -> tuple[Any, Callable[[], None]]:
    """Return ``(checkpointer, close_callback)``."""
    backend = settings.checkpoint_backend.strip().lower()
    if backend != "sqlite":
        if backend != "memory":
            logger.warning("Unknown CHECKPOINT_BACKEND=%r; using memory", backend)
        return InMemorySaver(), lambda: None

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise RuntimeError(
            "CHECKPOINT_BACKEND=sqlite requires langgraph-checkpoint-sqlite"
        ) from exc

    path = Path(settings.checkpoint_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    saver = SqliteSaver(connection)
    saver.setup()
    logger.info("Checkpointer: SQLite (%s)", path)
    closed = False

    def close() -> None:
        nonlocal closed
        if not closed:
            connection.close()
            closed = True

    return saver, close
