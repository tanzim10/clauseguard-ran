from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError, APITimeoutError

from clauseguard.config import Settings
from clauseguard.llm.clients import (
    GROUNDED_QUERY_MODELS,
    GenerationUnavailable,
    LLMRouter,
    MalformedGenerationOutput,
    _retry_after_seconds,
)


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _response(content: str = '{"answer":"ok","evidence_ids":["E1"]}'):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _status_error(status: int, retry_after: str | None = None) -> APIStatusError:
    headers = {"Retry-After": retry_after} if retry_after else {}
    response = httpx.Response(
        status,
        headers=headers,
        request=httpx.Request("POST", "https://example.test"),
    )
    return APIStatusError("private provider detail", response=response, body={"secret": "value"})


def _router(completions: FakeCompletions, *, models=("model-a", "model-b"), clock=None):
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return LLMRouter(
        Settings(nvidia_api_key="test-key", nvidia_cooldown_seconds=30),
        client_factory=lambda settings: client,
        models=models,
        **({"clock": clock} if clock else {}),
    )


def test_grounded_query_model_order_is_explicit() -> None:
    assert GROUNDED_QUERY_MODELS == (
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "openai/gpt-oss-20b",
        "deepseek-ai/deepseek-v4-flash-0731",
        "nvidia/nemotron-3-super-120b-a12b",
    )


def test_router_fails_over_on_rate_limit_and_honors_retry_after() -> None:
    now = [100.0]
    completions = FakeCompletions([_status_error(429, "12"), _response()])
    router = _router(completions, clock=lambda: now[0])

    assert router.generate("grounded prompt") == '{"answer":"ok","evidence_ids":["E1"]}'
    assert [call["model"] for call in completions.calls] == ["model-a", "model-b"]

    completions.outcomes.append(_response())
    router.generate("next prompt")
    assert [call["model"] for call in completions.calls][-1] == "model-b"

    now[0] += 13
    completions.outcomes.extend([_response()])
    router.generate("after cooldown")
    assert [call["model"] for call in completions.calls][-1] == "model-a"


def test_router_fails_over_on_retryable_gateway_status() -> None:
    completions = FakeCompletions([_status_error(503), _response()])

    assert _router(completions).generate("prompt").startswith("{")
    assert [call["model"] for call in completions.calls] == ["model-a", "model-b"]


def test_router_fails_over_on_timeout() -> None:
    timeout = APITimeoutError(request=httpx.Request("POST", "https://example.test"))
    completions = FakeCompletions([timeout, _response()])

    assert _router(completions).generate("prompt").startswith("{")
    assert [call["model"] for call in completions.calls] == ["model-a", "model-b"]


def test_router_does_not_fail_over_on_authentication_error() -> None:
    completions = FakeCompletions([_status_error(401), _response()])

    with pytest.raises(GenerationUnavailable):
        _router(completions).generate("prompt")

    assert [call["model"] for call in completions.calls] == ["model-a"]


def test_router_exhaustion_does_not_expose_provider_details() -> None:
    completions = FakeCompletions([_status_error(429), _status_error(504)])

    with pytest.raises(GenerationUnavailable) as error:
        _router(completions).generate("prompt")

    assert "private provider detail" not in str(error.value)
    assert len(completions.calls) == 2


def test_router_rejects_empty_response_without_failing_over() -> None:
    completions = FakeCompletions([_response(" "), _response()])

    with pytest.raises(MalformedGenerationOutput):
        _router(completions).generate("prompt")

    assert len(completions.calls) == 1


def test_retry_after_parser_accepts_http_date() -> None:
    future = datetime.now(UTC) + timedelta(seconds=25)
    error = _status_error(429, future.strftime("%a, %d %b %Y %H:%M:%S GMT"))

    delay = _retry_after_seconds(error)

    assert delay is not None
    assert 0 <= delay <= 25


def test_retry_after_parser_uses_default_for_invalid_header() -> None:
    error = _status_error(429, "not-a-date")

    assert _retry_after_seconds(error) is None


def test_invalid_retry_after_uses_configured_cooldown() -> None:
    now = [50.0]
    completions = FakeCompletions([_status_error(429, "invalid"), _response()])
    router = _router(completions, clock=lambda: now[0])

    router.generate("prompt")
    now[0] += 1
    completions.outcomes.append(_response())
    router.generate("next prompt")

    assert [call["model"] for call in completions.calls] == [
        "model-a",
        "model-b",
        "model-b",
    ]


def test_router_supports_only_grounded_query_profile() -> None:
    completions = FakeCompletions([_response()])

    with pytest.raises(GenerationUnavailable):
        _router(completions).generate("prompt", task="other")

    assert completions.calls == []
