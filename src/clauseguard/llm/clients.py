"""NVIDIA NIM chat client routing for grounded-query generation."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from clauseguard.config import Settings, get_settings

GROUNDED_QUERY_MODELS = (
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "openai/gpt-oss-20b",
    "deepseek-ai/deepseek-v4-flash-0731",
    "nvidia/nemotron-3-super-120b-a12b",
)


class GenerationUnavailable(RuntimeError):
    """Raised when generation cannot be served without exposing provider details."""


class MalformedGenerationOutput(RuntimeError):
    """Raised when the provider returns no usable message content."""


ClientFactory = Callable[[Settings], Any]


def _build_nvidia_client(settings: Settings) -> OpenAI:
    if not settings.nvidia_api_key.strip():
        raise GenerationUnavailable("generation service is not configured")
    return OpenAI(
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
        timeout=settings.nvidia_timeout_seconds,
        max_retries=0,
    )


def _retry_after_seconds(error: APIStatusError) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    raw_value = headers.get("retry-after") or headers.get("Retry-After")
    if raw_value is None:
        return None

    try:
        seconds = float(raw_value)
        return max(0.0, seconds)
    except (TypeError, ValueError):
        pass

    try:
        retry_at = parsedate_to_datetime(str(raw_value))
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


class LLMRouter:
    """Try configured NVIDIA NIM models in priority order with bounded failover."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client_factory: ClientFactory = _build_nvidia_client,
        models: tuple[str, ...] = GROUNDED_QUERY_MODELS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.client_factory = client_factory
        self.models = models
        self.clock = clock
        self._client: Any | None = None
        self._cooldown_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def generate(self, prompt: str, *, task: str = "grounded_query") -> str:
        """Generate once per eligible model, failing over only on transient errors."""
        if task != "grounded_query":
            raise GenerationUnavailable("unsupported generation task")
        settings = self.settings or get_settings()
        if not settings.nvidia_api_key.strip() and self.client_factory is _build_nvidia_client:
            raise GenerationUnavailable("generation service is not configured")

        if self._client is None:
            try:
                self._client = self.client_factory(settings)
            except GenerationUnavailable:
                raise
            except Exception as exc:
                raise GenerationUnavailable("generation service is not configured") from exc

        for model in self.models:
            if self._is_cooling_down(model):
                continue
            try:
                completion = self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
            except (APITimeoutError, APIConnectionError):
                self._cool_down(model, settings.nvidia_cooldown_seconds)
                continue
            except APIStatusError as exc:
                if exc.status_code not in {429, 502, 503, 504}:
                    raise GenerationUnavailable("generation service is unavailable") from exc
                delay = _retry_after_seconds(exc)
                self._cool_down(
                    model,
                    settings.nvidia_cooldown_seconds if delay is None else delay,
                )
                continue
            except Exception as exc:
                raise GenerationUnavailable("generation service is unavailable") from exc

            try:
                content = completion.choices[0].message.content
            except (AttributeError, IndexError, TypeError) as exc:
                raise MalformedGenerationOutput from exc
            if not isinstance(content, str) or not content.strip():
                raise MalformedGenerationOutput
            return content

        # Both exhausted models and models still in cooldown are unavailable.
        raise GenerationUnavailable("all configured generation models are unavailable")

    def _is_cooling_down(self, model: str) -> bool:
        with self._lock:
            return self._cooldown_until.get(model, 0.0) > self.clock()

    def _cool_down(self, model: str, seconds: float) -> None:
        with self._lock:
            self._cooldown_until[model] = self.clock() + seconds
