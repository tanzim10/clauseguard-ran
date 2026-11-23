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
        "meta/muse-glimmer-30b",
        "google/diffusiongemma-26b-a4b-it",
    )


def test_diffusiongemma_uses_prompt_constrained_json_instead_of_json_mode() -> None:
    completions = FakeCompletions([_response()])
    router = _router(
        completions,
        models=("google/diffusiongemma-26b-a4b-it",),
    )

    router.generate("Return the required JSON object")

    assert "response_format" not in completions.calls[0]
    assert completions.calls[0]["model"] == "google/diffusiongemma-26b-a4b-it"


def test_other_models_keep_provider_json_mode() -> None:
    completions = FakeCompletions([_response()])
    router = _router(completions, models=("meta/muse-glimmer-30b",))

    router.generate("Return the required JSON object")

    assert completions.calls[0]["response_format"] == {"type": "json_object"}


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


MESSAGES = [{"role": "system", "content": "rules"}, {"role": "user", "content": "packet"}]


def _served(content: str = '{"status":"abstained"}', model: str | None = None, usage=None):
    response = _response(content)
    response.model = model
    response.usage = usage
    return response


def test_rca_fixed_model_never_substitutes_and_records_the_attempt() -> None:
    completions = FakeCompletions([_status_error(503), _served()])
    router = _router(completions)

    with pytest.raises(GenerationUnavailable) as error:
        router.complete(MESSAGES, task="rca", models=("model-a",), failover=False)

    assert [call["model"] for call in completions.calls] == ["model-a"]
    assert [(a.model, a.outcome, a.error_class) for a in error.value.attempts] == [
        ("model-a", "transient_error", "APIStatusError")]


def test_rca_fixed_model_rejects_more_than_one_model() -> None:
    with pytest.raises(ValueError, match="one model"):
        _router(FakeCompletions([])).complete(MESSAGES, models=("a", "b"), failover=False)


def test_rca_passes_messages_and_generation_settings_and_reports_identity() -> None:
    usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": 10, "completion_tokens": 3})
    completions = FakeCompletions([_served(model="model-a-2026", usage=usage)])
    result = _router(completions).complete(
        MESSAGES, task="rca", models=("model-a",), failover=False,
        temperature=0.0, top_p=1.0, max_tokens=512, seed=7)

    call = completions.calls[0]
    assert call["messages"] == MESSAGES and call["response_format"] == {"type": "json_object"}
    assert (call["temperature"], call["top_p"], call["max_tokens"], call["seed"]) == (0.0, 1.0, 512, 7)
    assert result.content == '{"status":"abstained"}'
    assert (result.requested_model, result.served_model) == ("model-a", "model-a-2026")
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 3}
    assert [(a.model, a.outcome) for a in result.attempts] == [("model-a", "ok")]


def test_rca_exploratory_failover_records_every_attempt() -> None:
    timeout = APITimeoutError(request=httpx.Request("POST", "https://example.test"))
    completions = FakeCompletions([timeout, _served(model="model-b")])
    result = _router(completions).complete(MESSAGES, task="rca", failover=True)

    assert result.requested_model == "model-b"
    assert [(a.model, a.outcome, a.error_class) for a in result.attempts] == [
        ("model-a", "transient_error", "APITimeoutError"), ("model-b", "ok", None)]
    assert "temperature" not in completions.calls[0]


def test_rca_empty_output_keeps_attempts_and_hides_provider_details() -> None:
    completions = FakeCompletions([_served(" ")])
    with pytest.raises(MalformedGenerationOutput) as error:
        _router(completions).complete(MESSAGES, models=("model-a",), failover=False)
    assert [(a.model, a.outcome) for a in error.value.attempts] == [("model-a", "empty_output")]

    completions = FakeCompletions([_status_error(401)])
    with pytest.raises(GenerationUnavailable) as error:
        _router(completions).complete(MESSAGES, models=("model-a",), failover=False)
    assert "private provider detail" not in str(error.value)
    assert "test-key" not in repr([a.model_dump() for a in error.value.attempts])


def test_rca_task_is_not_available_through_grounded_generate_and_vice_versa() -> None:
    completions = FakeCompletions([])
    with pytest.raises(GenerationUnavailable):
        _router(completions).generate("prompt", task="rca")
    with pytest.raises(GenerationUnavailable):
        _router(completions).complete(MESSAGES, task="other")
    assert completions.calls == []


def test_rca_timeout_is_per_request_and_grounded_query_keeps_the_client_timeout() -> None:
    completions = FakeCompletions([_served(), _response()])
    router = _router(completions)
    router.complete(MESSAGES, task="rca", models=("model-a",), failover=False, timeout=240.0)
    router.generate("grounded prompt")

    assert completions.calls[0]["timeout"] == 240.0
    assert "timeout" not in completions.calls[1]
