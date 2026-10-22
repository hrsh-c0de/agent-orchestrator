"""Environment-driven configuration."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1, maximum: int = 600) -> int:
    try:
        return max(minimum, min(maximum, int(os.getenv(name, str(default)))))
    except ValueError:
        return default


def _env_float(name: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        return max(minimum, min(maximum, float(os.getenv(name, str(default)))))
    except ValueError:
        return default


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_api_keys(raw: str) -> tuple[tuple[str, str], ...]:
    """Parse ``key:identity,key2:identity2`` without logging the secret."""
    entries: list[tuple[str, str]] = []
    for item in raw.split(","):
        candidate = item.strip()
        if not candidate:
            continue
        key, separator, identity = candidate.partition(":")
        key = key.strip()
        if not key:
            continue
        entries.append((key, identity.strip() if separator and identity.strip() else "api-client"))
    return tuple(entries)


@dataclass(frozen=True)
class Settings:
    app_name: str = "Multi-Agent Orchestrator"
    version: str = "1.0.0"

    llm_provider: str = "offline"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2

    agent_timeout_seconds: float = 120.0
    agent_max_retries: int = 2
    max_attempts: int = 3
    validation_threshold: float = 0.75
    default_auto_approve: bool = False

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: tuple[str, ...] = (
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    )
    log_level: str = "INFO"
    crm_db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "crm.db")

    # Production persistence: set both backends to "sqlite" to survive restarts.
    checkpoint_backend: str = "memory"
    checkpoint_db_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "checkpoints.db"
    )
    run_store_backend: str = "memory"
    run_db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "runs.db")

    # Optional bearer-token auth for workflow endpoints.
    api_auth_enabled: bool = False
    api_key_identities: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_env(cls) -> "Settings":
        origins = tuple(
            origin.strip()
            for origin in os.getenv(
                "CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
            ).split(",")
            if origin.strip()
        )
        provider = os.getenv("LLM_PROVIDER", "offline").strip().lower()
        return cls(
            llm_provider=provider,
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "").strip(),
            llm_timeout_seconds=float(_env_int("LLM_TIMEOUT_SECONDS", 60, 5, 600)),
            llm_max_retries=_env_int("LLM_MAX_RETRIES", 2, 0, 5),
            agent_timeout_seconds=float(_env_int("AGENT_TIMEOUT_SECONDS", 120, 1, 600)),
            agent_max_retries=_env_int("AGENT_MAX_RETRIES", 2, 0, 5),
            max_attempts=_env_int("MAX_ATTEMPTS", 3, 1, 10),
            validation_threshold=_env_float("VALIDATION_THRESHOLD", 0.75, 0.0, 1.0),
            default_auto_approve=_env_bool("DEFAULT_AUTO_APPROVE", False),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=_env_int("API_PORT", 8000, 1, 65535),
            cors_origins=origins or ("http://localhost:8501",),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            crm_db_path=_resolve_path(os.getenv("CRM_DB_PATH", "data/crm.db")),
            checkpoint_backend=os.getenv("CHECKPOINT_BACKEND", "memory").strip().lower(),
            checkpoint_db_path=_resolve_path(
                os.getenv("CHECKPOINT_DB_PATH", "data/checkpoints.db")
            ),
            run_store_backend=os.getenv("RUN_STORE_BACKEND", "memory").strip().lower(),
            run_db_path=_resolve_path(os.getenv("RUN_DB_PATH", "data/runs.db")),
            api_auth_enabled=_env_bool("API_AUTH_ENABLED", False),
            api_key_identities=_parse_api_keys(os.getenv("API_KEYS", "")),
        )

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider == "openai" and bool(self.openai_api_key)


def configure_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=getattr(logging, (level or "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


settings = Settings.from_env()
