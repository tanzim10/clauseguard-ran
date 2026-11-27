"""NVIDIA NIM chat client routing for grounded-query and RCA generation."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Literal

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import BaseModel, ConfigDict, Field

from clauseguard.config import Settings, get_settings, nvidia_api_keys

GROUNDED_QUERY_MODELS = (
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "openai/gpt-oss-20b",
    "meta/muse-glimmer-30b",
    "google/diffusiongemma-26b-a4b-it",
)
MODELS_WITHOUT_JSON_MODE = frozenset({"google/diffusiongemma-26b-a4b-it"})
COMPLETION_TASKS = ("rca",)


class GenerationAttempt(BaseModel):
    """One call to one model; error classes only, never provider messages or credentials."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    outcome: Literal["ok", "transient_error", "error", "cooling_down", "empty_output"]
    error_class: str | None = None
    seconds: float = Field(ge=0)
    key_slot: int | None = Field(default=None, ge=0)


class GenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str
    requested_model: str
    served_model: str | None = None
    attempts: list[GenerationAttempt]
    usage: dict[str, Any] | None = None


class GenerationUnavailable(RuntimeError):
    """Raised when generation cannot be served without exposing provider details."""

    attempts: tuple[GenerationAttempt, ...] = ()


class MalformedGenerationOutput(RuntimeError):
    """Raised when the provider returns no usable message content."""

    attempts: tuple[GenerationAttempt, ...] = ()


def _tag(error: RuntimeError, attempts: list[GenerationAttempt]) -> RuntimeError:
    error.attempts = tuple(attempts)
    return error


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
        self._clients: dict[int, Any] = {}
        self._cooldown_until: dict[tuple[str, int], float] = {}
        self._lock = threading.Lock()

    def generate(self, prompt: str, *, task: str = "grounded_query") -> str:
        """Generate once per eligible model, failing over only on transient errors."""
        if task != "grounded_query":
            raise GenerationUnavailable("unsupported generation task")
        messages = [{"role": "user", "content": prompt}]
        return self._complete(messages, self.models, failover=True, params={}).content

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        task: str = "rca",
        models: tuple[str, ...] | None = None,
        failover: bool = True,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        timeout: float | None = None,
    ) -> GenerationResult:
        """Chat completion that reports every attempt and the model that answered.

        With ``failover=False`` exactly one model is called and no other model may substitute
        for it, which scored comparisons require. Unset generation settings are not sent.
        """
        if task not in COMPLETION_TASKS:
            raise GenerationUnavailable("unsupported generation task")
        models = self.models if models is None else models
        if not failover and len(models) != 1:
            raise ValueError("a fixed-model call needs exactly one model")
        params = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "seed": seed,
                  "timeout": timeout}
        return self._complete(messages, models, failover=failover,
                              params={key: value for key, value in params.items() if value is not None})

    def _complete(
        self,
        messages: list[dict[str, str]],
        models: tuple[str, ...],
        *,
        failover: bool,
        params: dict[str, Any],
    ) -> GenerationResult:
        settings = self.settings or get_settings()
        if not settings.nvidia_api_key.strip() and self.client_factory is _build_nvidia_client:
            raise GenerationUnavailable("generation service is not configured")

        slots = range(max(1, len(nvidia_api_keys(settings))))
        attempts: list[GenerationAttempt] = []

        def record(model: str, slot: int, outcome: str, started: float, error: Exception | None = None) -> None:
            attempts.append(GenerationAttempt(
                model=model, outcome=outcome, error_class=type(error).__name__ if error else None,
                seconds=max(0.0, time.perf_counter() - started), key_slot=slot))

        for model in models:
            completion = None
            for slot in slots:
                started = time.perf_counter()
                if self._is_cooling_down(model, slot):
                    record(model, slot, "cooling_down", started)
                    continue
                try:
                    client = self._client_for(settings, slot)
                    request = {"model": model, "messages": messages, **params}
                    if model not in MODELS_WITHOUT_JSON_MODE:
                        request["response_format"] = {"type": "json_object"}
                    completion = client.chat.completions.create(**request)
                    break
                except GenerationUnavailable as exc:
                    raise _tag(exc, attempts) from exc
                except (APITimeoutError, APIConnectionError) as exc:
                    record(model, slot, "transient_error", started, exc)
                    self._cool_down(model, slot, settings.nvidia_cooldown_seconds)
                except APIStatusError as exc:
                    if exc.status_code not in {429, 502, 503, 504}:
                        record(model, slot, "error", started, exc)
                        raise _tag(GenerationUnavailable("generation service is unavailable"), attempts) from exc
                    record(model, slot, "transient_error", started, exc)
                    delay = _retry_after_seconds(exc)
                    self._cool_down(model, slot, settings.nvidia_cooldown_seconds if delay is None else delay)
                except Exception as exc:
                    record(model, slot, "error", started, exc)
                    raise _tag(GenerationUnavailable("generation service is unavailable"), attempts) from exc
            if completion is None:
                if not failover:
                    raise _tag(GenerationUnavailable("fixed generation model is unavailable"), attempts)
                continue

            try:
                content = completion.choices[0].message.content
            except (AttributeError, IndexError, TypeError) as exc:
                record(model, slot, "empty_output", started, exc)
                raise _tag(MalformedGenerationOutput(), attempts) from exc
            if not isinstance(content, str) or not content.strip():
                record(model, slot, "empty_output", started)
                raise _tag(MalformedGenerationOutput(), attempts)
            record(model, slot, "ok", started)
            served = getattr(completion, "model", None)
            usage = getattr(completion, "usage", None)
            return GenerationResult(
                content=content, requested_model=model,
                served_model=served if isinstance(served, str) else None, attempts=attempts,
                usage=usage.model_dump() if hasattr(usage, "model_dump") else None)

        # Both exhausted models and models still in cooldown are unavailable.
        raise _tag(GenerationUnavailable("all configured generation models are unavailable"), attempts)

    def _client_for(self, settings: Settings, slot: int) -> Any:
        with self._lock:
            if slot not in self._clients:
                keys = nvidia_api_keys(settings)
                scoped = settings.model_copy(update={"nvidia_api_key": keys[slot]}) if slot else settings
                try:
                    self._clients[slot] = self.client_factory(scoped)
                except GenerationUnavailable:
                    raise
                except Exception as exc:
                    raise GenerationUnavailable("generation service is not configured") from exc
            return self._clients[slot]

    def _is_cooling_down(self, model: str, slot: int = 0) -> bool:
        with self._lock:
            return self._cooldown_until.get((model, slot), 0.0) > self.clock()

    def _cool_down(self, model: str, slot: int, seconds: float) -> None:
        with self._lock:
            self._cooldown_until[(model, slot)] = self.clock() + seconds
