"""Batched OpenAI text-embedding adapter."""

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
) -> list[list[float]]:
    """Embed texts in bounded batches while preserving input order.

    ``client`` is injectable so tests never need network access. Production
    callers use an OpenAI client configured from ``OPENAI_API_KEY``.
    """
    if not texts:
        return []

    resolved_settings = settings or get_settings()
    batch_size = resolved_settings.openai_embedding_batch_size
    if batch_size < 1:
        raise ValueError("openai_embedding_batch_size must be positive")

    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("cannot embed empty text")

    if client is None:
        if not resolved_settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for embeddings")
        client = OpenAI(api_key=resolved_settings.openai_api_key)

    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        response = client.embeddings.create(
            input=batch,
            model=resolved_settings.openai_embedding_model,
            dimensions=resolved_settings.openai_embedding_dimensions,
            encoding_format="float",
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
            if len(vector) != resolved_settings.openai_embedding_dimensions:
                raise ValueError(
                    "embedding response has wrong dimension; "
                    f"expected dimension {resolved_settings.openai_embedding_dimensions}, "
                    f"received {len(vector)}"
                )
            indexed_embeddings[index] = vector

        if len(indexed_embeddings) != len(batch):
            raise ValueError("embedding response does not contain every input index")
        embeddings.extend(indexed_embeddings[index] for index in range(len(batch)))

    return embeddings
