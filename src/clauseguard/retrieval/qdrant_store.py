"""Qdrant collection and point helpers for specification retrieval."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from clauseguard.config import Settings, get_settings


class CollectionCompatibilityError(RuntimeError):
    """Raised when an existing collection cannot store this index's vectors."""


def _resolve_client(client: Any | None, settings: Settings) -> Any:
    return client if client is not None else QdrantClient(url=settings.qdrant_url)


def _distance(settings: Settings) -> models.Distance:
    if settings.qdrant_distance.casefold() != "cosine":
        raise ValueError(
            "only cosine distance is supported for the text-embedding index"
        )
    return models.Distance.COSINE


def _vector_params(vectors: Any) -> Any:
    if isinstance(vectors, dict):
        if len(vectors) != 1:
            raise CollectionCompatibilityError(
                "named Qdrant collections must contain exactly one vector"
            )
        return next(iter(vectors.values()))
    return vectors


def _validate_collection(collection: str, info: Any, settings: Settings) -> None:
    configured_distance = _distance(settings)
    vectors = _vector_params(info.config.params.vectors)
    actual_size = getattr(vectors, "size", None)
    actual_distance = getattr(vectors, "distance", None)
    if actual_size != settings.openai_embedding_dimensions:
        raise CollectionCompatibilityError(
            f"collection {collection!r} has dimension {actual_size}; "
            f"expected {settings.openai_embedding_dimensions}"
        )
    if actual_distance != configured_distance:
        raise CollectionCompatibilityError(
            f"collection {collection!r} has incompatible distance "
            f"{actual_distance}; expected {configured_distance}"
        )


def ensure_collection(
    collection: str | None = None,
    *,
    client: Any | None = None,
    settings: Settings | None = None,
) -> None:
    """Create or validate the configured Qdrant collection idempotently."""
    resolved_settings = settings or get_settings()
    name = collection or resolved_settings.qdrant_collection
    resolved_client = _resolve_client(client, resolved_settings)
    if not resolved_client.collection_exists(collection_name=name):
        resolved_client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=resolved_settings.openai_embedding_dimensions,
                distance=_distance(resolved_settings),
            ),
        )
        return

    _validate_collection(name, resolved_client.get_collection(collection_name=name), resolved_settings)


def upsert_chunks(collection: str, chunks: list[dict[str, Any]]) -> int:
    """Upsert chunk vectors. Not implemented in scaffold."""
    raise NotImplementedError("Week 2 — upsert chunks into Qdrant")


def search(collection: str, query_vector: list[float], top_k: int = 8) -> list[dict[str, Any]]:
    """Vector search. Not implemented in scaffold."""
    raise NotImplementedError("Week 2 — Qdrant search")
