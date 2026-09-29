import pytest
from pydantic import ValidationError

from clauseguard.api.schemas import QueryRequest, QueryResponse


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
