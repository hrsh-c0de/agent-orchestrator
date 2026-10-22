"""FastAPI backend for the multi-agent orchestrator.

Run:  uvicorn app:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import Settings
from config import configure_logging, settings as default_settings
from state import ApprovalDecision, RunRequest, RunResponse
from workflow import InvalidRunState, Orchestrator, RunNotFound

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Principal:
    identity: str
    authenticated: bool


def create_app(
    app_settings: Settings | None = None,
    orchestrator: Orchestrator | None = None,
) -> FastAPI:
    """Application factory; injectable for tests and alternate deployments."""
    resolved = app_settings or default_settings
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.orchestrator = orchestrator or Orchestrator(resolved)
        logger.info(
            "API ready: provider=%s approval_default=%s crm=%s auth=%s persistence=%s/%s",
            "openai" if resolved.llm_enabled else "offline",
            resolved.default_auto_approve,
            resolved.crm_db_path,
            "enabled" if resolved.api_auth_enabled else "disabled",
            resolved.checkpoint_backend,
            resolved.run_store_backend,
        )
        if resolved.api_auth_enabled and not resolved.api_key_identities:
            logger.error("API_AUTH_ENABLED=true but API_KEYS is empty; workflow calls will be rejected")
        if not resolved.api_auth_enabled:
            logger.warning("API authentication is disabled; do not expose this API publicly")
        try:
            yield
        finally:
            app.state.orchestrator.close()

    app = FastAPI(
        title="Multi-Agent Orchestrator",
        version=resolved.version,
        description=(
            "A general-purpose LangGraph workflow: Reader -> Planner -> Executor -> "
            "Validator -> Human approval -> Output. Use POST /run and resume paused "
            "runs with POST /approve."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    def get_orchestrator(request: Request) -> Orchestrator:
        instance = getattr(request.app.state, "orchestrator", None)
        if instance is None:  # pragma: no cover - lifespan always runs in practice
            raise HTTPException(status_code=503, detail="Orchestrator is still starting.")
        return instance

    bearer = HTTPBearer(auto_error=False, description="Bearer token from API_KEYS")

    def require_principal(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> Principal:
        if not resolved.api_auth_enabled:
            return Principal(identity="anonymous", authenticated=False)
        if not resolved.api_key_identities:
            raise HTTPException(
                status_code=503,
                detail="API authentication is enabled but no API_KEYS are configured.",
            )
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=401,
                detail="Missing bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        for key, identity in resolved.api_key_identities:
            if secrets.compare_digest(credentials.credentials, key):
                return Principal(identity=identity, authenticated=True)
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.get("/", tags=["meta"])
    def root() -> dict[str, Any]:
        return {
            "name": resolved.app_name,
            "version": resolved.version,
            "docs": "/docs",
            "endpoints": ["/run", "/approve", "/runs", "/runs/{run_id}", "/graph", "/health"],
        }

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, Any]:
        instance = getattr(app.state, "orchestrator", None)
        try:
            runs = instance.run_count() if instance is not None else 0
        except Exception:
            runs = 0
        return {
            "status": "ok",
            "version": resolved.version,
            "llm_provider": resolved.llm_provider,
            "llm_enabled": resolved.llm_enabled,
            "approval_default": resolved.default_auto_approve,
            "auth_enabled": resolved.api_auth_enabled,
            "checkpoint_backend": resolved.checkpoint_backend,
            "run_store_backend": resolved.run_store_backend,
            "active_runs": runs,
        }

    @app.get("/graph", tags=["meta"])
    def graph(request: Request) -> dict[str, Any]:
        return get_orchestrator(request).graph_spec()

    @app.post("/run", response_model=RunResponse, tags=["workflow"])
    def run_task(
        payload: RunRequest,
        request: Request,
        principal: Principal = Depends(require_principal),
    ) -> RunResponse:
        orchestrator = get_orchestrator(request)
        metadata = dict(payload.metadata)
        metadata.setdefault("submitted_by", principal.identity)
        payload = payload.model_copy(update={"metadata": metadata})
        try:
            return orchestrator.start(payload)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Failed to start run")
            raise HTTPException(status_code=500, detail=f"Could not start run: {exc}") from exc

    @app.post("/approve", response_model=RunResponse, tags=["workflow"])
    def approve(
        payload: ApprovalDecision,
        request: Request,
        principal: Principal = Depends(require_principal),
    ) -> RunResponse:
        orchestrator = get_orchestrator(request)
        if principal.authenticated:
            # The authenticated key is authoritative, not a client-supplied name.
            payload = payload.model_copy(update={"reviewer": principal.identity})
        try:
            return orchestrator.approve(payload)
        except RunNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InvalidRunState as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Failed to resume run %s", payload.run_id)
            raise HTTPException(status_code=500, detail=f"Could not resume run: {exc}") from exc

    @app.get("/runs", response_model=list[RunResponse], tags=["workflow"])
    def list_runs(
        request: Request,
        limit: int = 20,
        _principal: Principal = Depends(require_principal),
    ) -> list[RunResponse]:
        if not 1 <= limit <= 100:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
        return get_orchestrator(request).list_runs(limit)

    @app.get("/runs/{run_id}", response_model=RunResponse, tags=["workflow"])
    def get_run(
        run_id: str,
        request: Request,
        _principal: Principal = Depends(require_principal),
    ) -> RunResponse:
        try:
            return get_orchestrator(request).get(run_id)
        except RunNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "app:app",
        host=default_settings.api_host,
        port=default_settings.api_port,
        reload=False,
    )
