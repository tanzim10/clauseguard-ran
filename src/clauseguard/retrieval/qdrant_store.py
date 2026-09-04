"""Qdrant client helpers (Week 2). Placeholder only."""

from typing import Any


def ensure_collection(collection: str) -> None:
    """Create collection if missing. Not implemented in scaffold."""
    raise NotImplementedError("Week 2 — ensure Qdrant collection")


def upsert_chunks(collection: str, chunks: list[dict[str, Any]]) -> int:
    """Upsert chunk vectors. Not implemented in scaffold."""
    raise NotImplementedError("Week 2 — upsert chunks into Qdrant")


def search(collection: str, query_vector: list[float], top_k: int = 8) -> list[dict[str, Any]]:
    """Vector search. Not implemented in scaffold."""
    raise NotImplementedError("Week 2 — Qdrant search")
