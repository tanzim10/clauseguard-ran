import pytest
from pydantic import ValidationError

from clauseguard.api.schemas import RcaModelOutput, RcaRequest


def _output():
    return {
        "status": "answered",
        "final_fault_categories": [{"category": "congestion", "claim_ids": ["C1"]}],
        "claims": [
            {
                "claim_id": "C1",
                "text": "The evidence supports congestion.",
                "claim_type": "diagnostic_inference",
                "evidence_ids": ["K1", "S1"],
            }
        ],
    }


def test_rca_request_has_bounded_retrieval_and_no_extra_fields():
    assert RcaRequest(scenario_id="case-1", question="Why?", top_k=8).top_k == 8
    with pytest.raises(ValidationError):
        RcaRequest(scenario_id="case-1", question="Why?", answer="leaked")
    with pytest.raises(ValidationError):
        RcaRequest(question="Why?")


def test_rca_output_allows_single_or_multiple_cause_links():
    data = _output()
    data["final_fault_categories"].append(
        {"category": "interference", "claim_ids": ["C1"]}
    )
    assert len(RcaModelOutput.model_validate(data).final_fault_categories) == 2


def test_rca_output_requires_evidence_linked_diagnostic_claims():
    data = _output()
    data["claims"][0]["claim_type"] = "observation"
    with pytest.raises(ValidationError, match="diagnostic inference"):
        RcaModelOutput.model_validate(data)


def test_abstention_requires_empty_causes_and_claims():
    data = {"status": "abstained", "final_fault_categories": [], "claims": [],
            "abstention_reason": "no_spec_evidence"}
    assert RcaModelOutput.model_validate(data).status == "abstained"
    data["final_fault_categories"] = [{"category": "x", "claim_ids": ["C1"]}]
    with pytest.raises(ValidationError, match="abstention needs"):
        RcaModelOutput.model_validate(data)
