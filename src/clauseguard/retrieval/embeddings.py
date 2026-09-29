"""Batched NVIDIA NIM embedding adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openai import OpenAI

from clauseguard.config import Settings, get_settings


def embed_texts(
    texts: Sequence[str],
    *,
    client: Any | None = None,
    settings: Settings | None = None,
    input_type: str = "query",
) -> list[list[float]]:
    """Embed texts in bounded batches while preserving input order.

    ``client`` is injectable so tests never need network access. Production
    callers use the NVIDIA OpenAI-compatible embeddings endpoint. ``input_type``
    must be ``query`` for retrieval queries and ``passage`` for corpus indexing.
    """
    if not texts:
        return []

    resolved_settings = settings or get_settings()
    if input_type not in {"query", "passage"}:
        raise ValueError("input_type must be 'query' or 'passage'")
    batch_size = resolved_settings.embedding_batch_size
    if batch_size < 1:
        raise ValueError("embedding_batch_size must be positive")

    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("cannot embed empty text")

    if client is None:
        if not resolved_settings.nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY is required for embeddings")
        client = OpenAI(
            api_key=resolved_settings.nvidia_api_key,
            base_url=resolved_settings.nvidia_base_url,
            timeout=resolved_settings.nvidia_timeout_seconds,
            max_retries=0,
        )

    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        response = client.embeddings.create(
            input=batch,
            model=resolved_settings.nvidia_embedding_model,
            encoding_format="float",
            extra_body={"input_type": input_type},
        )
        data = list(getattr(response, "data", []))
        if len(data) != len(batch):
            raise ValueError(
                f"expected {len(batch)} embeddings, received {len(data)}"
            )

        indexed_embeddings: dict[int, list[float]] = {}
        for item in data:
            index = getattr(item, "index", None)
            vector = getattr(item, "embedding", None)
            if not isinstance(index, int) or not isinstance(vector, list):
                raise TypeError("embedding response contains malformed data")
            if index in indexed_embeddings or not 0 <= index < len(batch):
                raise ValueError("embedding response contains invalid indexes")
            if len(vector) != resolved_settings.embedding_dimensions:
                raise ValueError(
                    "embedding response has wrong dimension; "
                    f"expected dimension {resolved_settings.embedding_dimensions}, "
                    f"received {len(vector)}"
                )
            indexed_embeddings[index] = vector

        if len(indexed_embeddings) != len(batch):
            raise ValueError("embedding response does not contain every input index")
        embeddings.extend(indexed_embeddings[index] for index in range(len(batch)))

    return embeddings
