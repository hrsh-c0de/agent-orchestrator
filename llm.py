"""LLM transport with JSON-mode, schema validation, retries and timeouts.

The offline provider is the deterministic agent engine in ``agents.py`` and
needs no network. This module is only used when ``LLM_PROVIDER=openai``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the provider cannot produce a schema-valid response."""


class LLMClient:
    """Thin JSON-mode wrapper around the OpenAI chat completions API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise LLMError("openai package is not installed") from exc

        kwargs: dict[str, Any] = {
            "api_key": settings.openai_api_key,
            "timeout": settings.llm_timeout_seconds,
            "max_retries": 0,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self._client = OpenAI(**kwargs)

    def complete_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        max_retries: int | None = None,
        temperature: float = 0.2,
    ) -> T:
        attempts = (self.settings.llm_max_retries if max_retries is None else max_retries) + 1
        last_error: Exception | None = None
        schema_json = json.dumps(schema.model_json_schema(), default=str)

        for attempt in range(1, attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.settings.openai_model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                f"{system}\n\nRespond with a single JSON object that strictly "
                                f"matches this JSON Schema. No markdown fences.\n{schema_json}"
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                content = response.choices[0].message.content or ""
                payload = _loads_json(content)
                return schema.model_validate(payload)
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "LLM output failed schema validation (attempt %s/%s): %s",
                    attempt,
                    attempts,
                    str(exc)[:300],
                )
            except Exception as exc:  # network / provider errors
                last_error = exc
                logger.warning(
                    "LLM request failed (attempt %s/%s): %s", attempt, attempts, str(exc)[:300]
                )
            if attempt < attempts:
                time.sleep(min(2.0, 0.35 * (2 ** (attempt - 1))))

        raise LLMError(f"LLM failed after {attempts} attempts: {last_error}")


def _loads_json(content: str) -> dict[str, Any]:
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    return payload


def build_llm(settings: Settings) -> LLMClient | None:
    if not settings.llm_enabled:
        if settings.llm_provider == "openai":
            logger.warning("LLM_PROVIDER=openai but OPENAI_API_KEY is missing; using offline engine")
        return None
    logger.info("Using OpenAI LLM provider (model=%s)", settings.openai_model)
    return LLMClient(settings)
