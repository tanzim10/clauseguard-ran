"""Qdrant collection and point helpers for specification retrieval."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
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
    if actual_size != settings.embedding_dimensions:
        raise CollectionCompatibilityError(
            f"collection {collection!r} has dimension {actual_size}; "
            f"expected {settings.embedding_dimensions}"
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
                size=resolved_settings.embedding_dimensions,
                distance=_distance(resolved_settings),
            ),
        )
        return

    _validate_collection(name, resolved_client.get_collection(collection_name=name), resolved_settings)


def _point_id(chunk_id: str) -> str:
    """Map a stable chunk ID to a Qdrant-compatible deterministic UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def upsert_chunks(
    collection: str,
    chunks: list[dict[str, Any]],
    *,
    embeddings: Sequence[Sequence[float]] | None = None,
    client: Any | None = None,
    settings: Settings | None = None,
) -> int:
    """Upsert embedded chunks while preserving citation metadata."""
    if not chunks:
        return 0
    if embeddings is None or len(embeddings) != len(chunks):
        raise ValueError("chunks and embeddings must contain the same number of items")

    resolved_settings = settings or get_settings()
    point_ids: set[str] = set()
    points: list[models.PointStruct] = []
    required_metadata = ("spec_id", "section", "source_file", "chunk_id")
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk_id = chunk.get("id")
        text = chunk.get("text")
        metadata = chunk.get("metadata")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("each chunk requires a non-empty id")
        if chunk_id in point_ids:
            raise ValueError(f"duplicate chunk id: {chunk_id}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"chunk {chunk_id!r} requires non-empty text")
        if not isinstance(metadata, dict):
            raise TypeError(f"chunk {chunk_id!r} requires metadata")
        if any(not metadata.get(field) for field in required_metadata):
            raise ValueError(f"chunk {chunk_id!r} is missing required metadata")
        if metadata["chunk_id"] != chunk_id:
            raise ValueError(f"chunk {chunk_id!r} has inconsistent chunk_id metadata")
        vector = list(embedding)
        if len(vector) != resolved_settings.embedding_dimensions:
            raise ValueError(
                f"chunk {chunk_id!r} has vector dimension {len(vector)}; "
                f"expected {resolved_settings.embedding_dimensions}"
            )
        point_ids.add(chunk_id)
        points.append(
            models.PointStruct(
                id=_point_id(chunk_id),
                vector=vector,
                payload={
                    "text": text,
                    "spec_id": metadata["spec_id"],
                    "section": metadata["section"],
                    "source_file": metadata["source_file"],
                    "chunk_id": metadata["chunk_id"],
                },
            )
        )

    resolved_client = _resolve_client(client, resolved_settings)
    batch_size = resolved_settings.qdrant_upsert_batch_size
    if batch_size < 1:
        raise ValueError("qdrant_upsert_batch_size must be positive")
    for start in range(0, len(points), batch_size):
        resolved_client.upsert(
            collection_name=collection,
            points=points[start : start + batch_size],
        )
    return len(points)


def search(
    collection: str,
    query_vector: list[float],
    top_k: int = 8,
    *,
    client: Any | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Return scored Qdrant points with their citation payloads."""
    resolved_settings = settings or get_settings()
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if len(query_vector) != resolved_settings.embedding_dimensions:
        raise ValueError(
            f"query vector has dimension {len(query_vector)}; "
            f"expected {resolved_settings.embedding_dimensions}"
        )
    resolved_client = _resolve_client(client, resolved_settings)
    response = resolved_client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "id": str(point.id),
            "score": point.score,
            "payload": dict(point.payload or {}),
        }
        for point in response.points
    ]
