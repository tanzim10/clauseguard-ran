import pytest
from pydantic import ValidationError

from clauseguard.api.schemas import SearchRequest, SearchResponse


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
