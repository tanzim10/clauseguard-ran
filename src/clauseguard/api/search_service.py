"""Shared retrieval service used by the search API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Any

from clauseguard.api.schemas import SearchHit
from clauseguard.config import Settings, get_settings
from clauseguard.retrieval.embeddings import embed_texts
from clauseguard.retrieval.qdrant_store import search as search_qdrant


class SearchDependencyError(RuntimeError):
    """Raised when an external retrieval dependency is unavailable."""


class SearchConfigurationError(RuntimeError):
    """Raised when local retrieval configuration or data is invalid."""


EmbeddingFunction = Callable[..., list[list[float]]]
QdrantSearchFunction = Callable[..., list[dict[str, Any]]]


@dataclass(slots=True)
class SearchService:
    """Compose query embedding, vector search, and public result projection."""

    settings: Settings | None = None
    embedder: EmbeddingFunction = embed_texts
    qdrant_search: QdrantSearchFunction = search_qdrant

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        settings = self.settings or get_settings()
        try:
            embeddings = self.embedder([query], settings=settings, input_type="query")
        except ValueError as exc:
            raise SearchConfigurationError(str(exc)) from exc
        except Exception as exc:
            raise SearchDependencyError from exc

        if len(embeddings) != 1:
            raise SearchConfigurationError(
                f"expected one query embedding, received {len(embeddings)}"
            )

        try:
            points = self.qdrant_search(
                settings.qdrant_collection,
                embeddings[0],
                top_k,
                settings=settings,
            )
        except ValueError as exc:
            raise SearchConfigurationError(str(exc)) from exc
        except Exception as exc:
            raise SearchDependencyError from exc

        return [self._project_hit(point) for point in points]

    @staticmethod
    def _project_hit(point: dict[str, Any]) -> SearchHit:
        if not isinstance(point, dict):
            raise SearchConfigurationError("retrieval result is not an object")

        payload = point.get("payload")
        if not isinstance(payload, dict):
            raise SearchConfigurationError("retrieval result has no payload")

        required_fields = ("text", "spec_id", "section", "source_file", "chunk_id")
        if any(not isinstance(payload.get(field), str) or not payload[field].strip() for field in required_fields):
            raise SearchConfigurationError("retrieval result is missing required metadata")

        point_id = point.get("id")
        score = point.get("score")
        if not isinstance(point_id, str) or not point_id:
            raise SearchConfigurationError("retrieval result is missing an id")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not isfinite(score):
            raise SearchConfigurationError("retrieval result has an invalid score")

        return SearchHit(
            id=point_id,
            score=float(score),
            text=payload["text"],
            spec_id=payload["spec_id"],
            section=payload["section"],
            source_file=payload["source_file"],
            chunk_id=payload["chunk_id"],
        )
