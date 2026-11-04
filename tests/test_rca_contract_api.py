import pytest
from pydantic import ValidationError

from clauseguard.api.schemas import RcaModelOutput, RcaRequest
from clauseguard.rca.contracts import (
    DATASET_SOURCE_RULES,
    EvidencePacket,
    KpiStatisticEvidence,
    SourceIdentity,
    SpecificationEvidence,
    TextEvidence,
    validate_output_for_packet,
)


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


def _packet():
    return EvidencePacket(
        source=SourceIdentity(
            dataset="TelecomTS", revision="r1", case_id="case-1",
            artifact_sha256="a" * 64, recording_family_status="unresolved",
        ),
        question="What caused the service issue?",
        kpi_state="included",
        text_state="missing",
        kpi_evidence=[KpiStatisticEvidence(
            evidence_id="K1", source_window_id="W1", channel="UL_BLER",
            statistic="mean", value=0.1, unit=None, valid_count=5, sample_count=5,
            missing_fraction=0.0, calculation_version="v1",
        )],
        specification_evidence=[SpecificationEvidence(
            evidence_id="S1", chunk_id="C1", document_id="3GPP TS 38.300",
            version="18.1", clause="10.1", text="Mechanism text.", content_sha256="b" * 64,
        )],
        retrieval_status="completed",
    )


def test_packet_rejects_reference_fields_and_has_all_source_rules():
    _packet()
    assert "expected_answer" not in EvidencePacket.model_fields
    assert set(DATASET_SOURCE_RULES) == {
        "TelecomTS", "BubbleRAN", "SpotLight", "TeleLogs", "5G core faults"
    }


def test_citation_validation_uses_exact_packet_and_claim_types():
    output = RcaModelOutput.model_validate(_output())
    validate_output_for_packet(output, _packet())

    output.claims[0].evidence_ids = ["missing"]
    with pytest.raises(ValueError, match="unknown evidence IDs"):
        validate_output_for_packet(output, _packet())


def test_text_evidence_requires_approval_for_rca():
    with pytest.raises(ValidationError):
        TextEvidence(
            evidence_id="T1", record_id="R1", text="Alarm text",
            origin="independent_observation", approval_status="pending",
            source="alarm-feed",
        )
