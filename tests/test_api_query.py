import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from clauseguard.api.main import create_app
from clauseguard.api.query_service import QueryOutputError, QueryService
from clauseguard.api.routes.query import get_llm_router
from clauseguard.api.routes.search import get_search_service
from clauseguard.api.schemas import QueryRequest, QueryResponse, SearchHit
from clauseguard.api.search_service import SearchDependencyError, SearchService
from clauseguard.config import Settings
from clauseguard.llm.clients import GenerationUnavailable, MalformedGenerationOutput


def test_query_request_normalizes_question_and_defaults_top_k() -> None:
    request = QueryRequest(question="  What is an O-RAN E2 node?  ")

    assert request.question == "What is an O-RAN E2 node?"
    assert request.top_k == 8


@pytest.mark.parametrize("question", ["", " ", "\t\n"])
def test_query_request_rejects_blank_question(question: str) -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question=question)


@pytest.mark.parametrize("top_k", [0, -1, 51])
def test_query_request_rejects_top_k_outside_contract(top_k: int) -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="What is an O-RAN E2 node?", top_k=top_k)


def test_query_request_accepts_top_k_bounds() -> None:
    assert QueryRequest(question="Question", top_k=1).top_k == 1
    assert QueryRequest(question="Question", top_k=50).top_k == 50


def test_query_response_has_stable_public_shape() -> None:
    response = QueryResponse(
        answer="An E2 node connects an E2 termination point.",
        citations=[
            {
                "spec_id": "O-RAN.WG3.E2AP-R003-v04.00",
                "section": "5.2.2",
                "source_file": "e2ap.pdf",
                "chunk_id": "chunk-17",
            }
        ],
    )

    assert response.model_dump() == {
        "answer": "An E2 node connects an E2 termination point.",
        "citations": [
            {
                "spec_id": "O-RAN.WG3.E2AP-R003-v04.00",
                "section": "5.2.2",
                "source_file": "e2ap.pdf",
                "chunk_id": "chunk-17",
            }
        ],
    }


def _hit(chunk_id: str, section: str) -> SearchHit:
    return SearchHit(
        id=f"point-{chunk_id}",
        score=0.9,
        text=f"Passage for {section}",
        spec_id="spec-1",
        section=section,
        source_file="spec.pdf",
        chunk_id=chunk_id,
    )


class FakeSearchService:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, question, top_k):
        self.calls.append((question, top_k))
        return self.hits


class FakeRouter:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def generate(self, prompt, *, task):
        self.calls.append((prompt, task))
        return self.output


def test_query_service_abstains_without_calling_model_when_retrieval_empty() -> None:
    search = FakeSearchService([])
    router = FakeRouter('{"answer":"unexpected","evidence_ids":[]}')

    result = QueryService(search, router).query("question", 8)

    assert result.model_dump() == {"answer": "not found", "citations": []}
    assert search.calls == [("question", 8)]
    assert router.calls == []


def test_query_service_projects_only_citations_from_retrieved_evidence() -> None:
    search = FakeSearchService([_hit("chunk-a", "1.1"), _hit("chunk-b", "1.2")])
    router = FakeRouter('{"answer":"Grounded answer","evidence_ids":["E2","E1"]}')

    result = QueryService(search, router).query("question", 2)

    assert result.model_dump() == {
        "answer": "Grounded answer",
        "citations": [
            {
                "spec_id": "spec-1",
                "section": "1.2",
                "source_file": "spec.pdf",
                "chunk_id": "chunk-b",
            },
            {
                "spec_id": "spec-1",
                "section": "1.1",
                "source_file": "spec.pdf",
                "chunk_id": "chunk-a",
            },
        ],
    }
    prompt, task = router.calls[0]
    assert "Passage for 1.1" in prompt
    assert "Passage for 1.2" in prompt
    assert task == "grounded_query"


def test_query_service_accepts_model_abstention_with_no_citations() -> None:
    search = FakeSearchService([_hit("chunk-a", "1.1")])
    router = FakeRouter('{"answer":"not found","evidence_ids":[]}')

    result = QueryService(search, router).query("question", 1)

    assert result.model_dump() == {"answer": "not found", "citations": []}


@pytest.mark.parametrize(
    "output",
    [
        "not json",
        '{"answer":"answer","evidence_ids":["E9"]}',
        '{"answer":"answer","evidence_ids":[]}',
        '{"answer":"not found","evidence_ids":["E1"]}',
        '{"answer":"answer","evidence_ids":["E1","E1"]}',
        '{"answer":"answer","evidence_ids":[1]}',
        '{"answer":"answer","evidence_ids":[],"extra":true}',
    ],
)
def test_query_service_rejects_malformed_or_ungrounded_output(output: str) -> None:
    service = QueryService(FakeSearchService([_hit("chunk-a", "1.1")]), FakeRouter(output))

    with pytest.raises(QueryOutputError):
        service.query("question", 1)


def _api_client(search_service, router) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_search_service] = lambda: search_service
    app.dependency_overrides[get_llm_router] = lambda: router
    return TestClient(app)


def test_query_route_returns_grounded_response_without_internal_fields() -> None:
    search = FakeSearchService([_hit("chunk-a", "1.1")])
    router = FakeRouter('{"answer":"Grounded answer","evidence_ids":["E1"]}')

    response = _api_client(search, router).post(
        "/query",
        json={"question": "  What does the clause say?  ", "top_k": 3},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Grounded answer",
        "citations": [
            {
                "spec_id": "spec-1",
                "section": "1.1",
                "source_file": "spec.pdf",
                "chunk_id": "chunk-a",
            }
        ],
    }
    assert search.calls == [("What does the clause say?", 3)]


def test_query_route_abstains_with_http_200_without_model_call() -> None:
    search = FakeSearchService([])
    router = FakeRouter('{"answer":"unused","evidence_ids":[]}')

    response = _api_client(search, router).post("/query", json={"question": "Unknown"})

    assert response.status_code == 200
    assert response.json() == {"answer": "not found", "citations": []}
    assert router.calls == []


@pytest.mark.parametrize(
    "payload",
    [{}, {"question": ""}, {"question": "  "}, {"question": "q", "top_k": 0}, {"question": "q", "top_k": 51}],
)
def test_query_route_rejects_invalid_input_without_service_calls(payload: dict) -> None:
    search = FakeSearchService([_hit("chunk-a", "1.1")])
    router = FakeRouter('{"answer":"answer","evidence_ids":["E1"]}')

    response = _api_client(search, router).post("/query", json=payload)

    assert response.status_code == 422
    assert search.calls == []
    assert router.calls == []


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (GenerationUnavailable("provider secret"), {"error": "generation_unavailable"}),
        (MalformedGenerationOutput("raw output"), {"error": "generation_unavailable"}),
        (QueryOutputError("model detail"), {"error": "generation_unavailable"}),
    ],
)
def test_query_route_hides_generation_details(error, expected: dict) -> None:
    search = FakeSearchService([_hit("chunk-a", "1.1")])

    class FailingRouter:
        def generate(self, prompt, *, task):
            raise error

    response = _api_client(search, FailingRouter()).post("/query", json={"question": "q"})

    assert response.status_code == 503
    assert response.json() == expected
    assert "secret" not in response.text
    assert "raw output" not in response.text


def test_query_route_maps_retrieval_failures() -> None:
    def unavailable_embedder(texts, *, settings):
        raise SearchDependencyError("provider secret")

    service = SearchService(
        settings=Settings(),
        embedder=unavailable_embedder,
    )
    response = _api_client(service, FakeRouter("unused")).post(
        "/query", json={"question": "q"}
    )

    assert response.status_code == 503
    assert response.json() == {"error": "retrieval_unavailable"}
    assert "provider secret" not in response.text


def test_query_route_maps_retrieval_configuration_failures() -> None:
    def invalid_embedder(texts, *, settings):
        raise ValueError("configuration secret")

    service = SearchService(settings=Settings(), embedder=invalid_embedder)
    response = _api_client(service, FakeRouter("unused")).post(
        "/query", json={"question": "q"}
    )

    assert response.status_code == 500
    assert response.json() == {"error": "retrieval_configuration_error"}
    assert "configuration secret" not in response.text


def test_query_openapi_documents_request_and_response_contract() -> None:
    schema = _api_client(FakeSearchService([]), FakeRouter("unused")).get("/openapi.json").json()
    operation = schema["paths"]["/query"]["post"]
    request_schema = schema["components"]["schemas"]["QueryRequest"]

    assert operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"] == (
        "#/components/schemas/QueryResponse"
    )
    assert request_schema["required"] == ["question"]
    assert request_schema["properties"]["top_k"]["default"] == 8
    assert request_schema["properties"]["top_k"]["minimum"] == 1
    assert request_schema["properties"]["top_k"]["maximum"] == 50
