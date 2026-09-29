"""Opt-in live smoke: simulate the primary NIM model's 429, then call its fallback."""

from __future__ import annotations

import json

import httpx
from openai import OpenAI

from clauseguard.config import get_settings
from clauseguard.llm.clients import GROUNDED_QUERY_MODELS, LLMRouter


def main() -> None:
    settings = get_settings()
    if not settings.nvidia_api_key.strip():
        raise SystemExit("Set NVIDIA_API_KEY before running this live smoke test.")

    requests: list[str] = []
    live_transport = httpx.HTTPTransport()

    def simulate_primary_rate_limit(request: httpx.Request) -> httpx.Response:
        request_body = json.loads(request.content)
        model = request_body.get("model", "")
        requests.append(model)
        if len(requests) == 1:
            if model != GROUNDED_QUERY_MODELS[0]:
                raise RuntimeError("the first request was not sent to the configured primary model")
            return httpx.Response(
                429,
                headers={"Retry-After": "0"},
                json={"error": {"message": "local smoke-test rate limit"}},
                request=request,
            )
        return live_transport.handle_request(request)

    http_client = httpx.Client(transport=httpx.MockTransport(simulate_primary_rate_limit))

    def client_factory(current_settings):
        return OpenAI(
            api_key=current_settings.nvidia_api_key,
            base_url=current_settings.nvidia_base_url,
            timeout=current_settings.nvidia_timeout_seconds,
            max_retries=0,
            http_client=http_client,
        )

    fallback_model = "meta/muse-glimmer-30b"
    router = LLMRouter(
        settings=settings,
        client_factory=client_factory,
        models=(GROUNDED_QUERY_MODELS[0], fallback_model),
    )
    prompt = (
        "Answer using only the retrieved specification passage below. Return a JSON object "
        'with exactly two keys: "answer" (string) and "evidence_ids" (array of strings). '
        'Question: What connects the near-real-time RIC to an E2 node? Evidence E1: '
        '"The E2 interface connects the near-real-time RIC and an E2 node."'
    )
    try:
        raw_output = router.generate(prompt)
        result = json.loads(raw_output)
        if set(result) != {"answer", "evidence_ids"}:
            raise RuntimeError("fallback model returned an unexpected JSON shape")
        if requests != [GROUNDED_QUERY_MODELS[0], fallback_model]:
            raise RuntimeError("the expected primary-to-Muse failover did not occur")
        if not isinstance(result.get("answer"), str) or result.get("evidence_ids") != ["E1"]:
            raise RuntimeError("fallback model returned an invalid grounded response")
        print(f"Simulated 429 for: {requests[0]}")
        print(f"Live NVIDIA fallback model: {requests[1]}")
        print(json.dumps(result, indent=2))
    finally:
        http_client.close()
        live_transport.close()


if __name__ == "__main__":
    main()
