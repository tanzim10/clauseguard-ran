import pytest
from pydantic import ValidationError

from clauseguard.api.schemas import SearchRequest, SearchResponse
from clauseguard.api.search_service import (
    SearchConfigurationError,
    SearchDependencyError,
    SearchService,
)
from clauseguard.config import Settings


def test_search_request_normalizes_query_and_defaults_top_k() -> None:
    request = SearchRequest(query="  A1 policy  ")

    assert request.query == "A1 policy"
    assert request.top_k == 8


@pytest.mark.parametrize("query", ["", " ", "\t\n"])
def test_search_request_rejects_blank_query(query: str) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query=query)


@pytest.mark.parametrize("top_k", [0, -1, 51])
def test_search_request_rejects_invalid_top_k(top_k: int) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="A1 policy", top_k=top_k)


def test_search_response_has_public_result_shape() -> None:
    response = SearchResponse(
        results=[
            {
                "id": "point-1",
                "score": 0.9,
                "text": "A policy passage",
                "spec_id": "spec-1",
                "section": "1.2",
                "source_file": "spec.pdf",
                "chunk_id": "chunk-1",
            }
        ]
    )

    assert response.model_dump() == {
        "results": [
            {
                "id": "point-1",
                "score": 0.9,
                "text": "A policy passage",
                "spec_id": "spec-1",
                "section": "1.2",
                "source_file": "spec.pdf",
                "chunk_id": "chunk-1",
            }
        ]
    }


def test_search_service_embeds_and_projects_qdrant_results() -> None:
    settings = Settings(openai_embedding_dimensions=3, qdrant_collection="specs")
    calls: dict[str, object] = {}

    def fake_embedder(texts: list[str], *, settings: Settings) -> list[list[float]]:
        calls["texts"] = texts
        return [[0.1, 0.2, 0.3]]

    def fake_qdrant_search(
        collection: str,
        vector: list[float],
        top_k: int,
        *,
        settings: Settings,
    ) -> list[dict[str, object]]:
        calls["search"] = (collection, vector, top_k)
        return [
            {
                "id": "point-1",
                "score": 0.9,
                "payload": {
                    "text": "A policy passage",
                    "spec_id": "spec-1",
                    "section": "1.2",
                    "source_file": "spec.pdf",
                    "chunk_id": "chunk-1",
                    "internal": "not public",
                },
            }
        ]

    service = SearchService(
        settings=settings,
        embedder=fake_embedder,
        qdrant_search=fake_qdrant_search,
    )

    results = service.search("A1 policy", 3)

    assert results[0].model_dump() == {
        "id": "point-1",
        "score": 0.9,
        "text": "A policy passage",
        "spec_id": "spec-1",
        "section": "1.2",
        "source_file": "spec.pdf",
        "chunk_id": "chunk-1",
    }
    assert calls == {"texts": ["A1 policy"], "search": ("specs", [0.1, 0.2, 0.3], 3)}


def test_search_service_preserves_empty_results() -> None:
    service = SearchService(
        settings=Settings(openai_embedding_dimensions=1),
        embedder=lambda texts, *, settings: [[1.0]],
        qdrant_search=lambda collection, vector, top_k, *, settings: [],
    )

    assert service.search("unknown", 8) == []


def test_search_service_rejects_incomplete_payload() -> None:
    service = SearchService(
        settings=Settings(openai_embedding_dimensions=1),
        embedder=lambda texts, *, settings: [[1.0]],
        qdrant_search=lambda collection, vector, top_k, *, settings: [
            {"id": "point-1", "score": 0.9, "payload": {"text": "missing metadata"}}
        ],
    )

    with pytest.raises(SearchConfigurationError):
        service.search("query", 8)


def test_search_service_classifies_dependency_failures() -> None:
    def failing_embedder(texts, *, settings):
        raise TimeoutError("provider unavailable")

    service = SearchService(settings=Settings(), embedder=failing_embedder)

    with pytest.raises(SearchDependencyError):
        service.search("query", 8)
