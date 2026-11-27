"""Task 4.9: label and KPI-support checks on teacher drafts."""

from test_rca_checks import SEVERE, _answer, _packet

from clauseguard.api.schemas import RcaModelOutput
from clauseguard.rca.candidates import CandidateReference
from clauseguard.rca.training_checks import kpi_support_check, label_check

ABSTAINED = RcaModelOutput.model_validate(
    {"status": "abstained", "final_fault_categories": [], "claims": [], "abstention_reason": "insufficient_mechanism"})


def ref(status: str, causes: list[str] | None = None) -> CandidateReference:
    return CandidateReference(candidate_id="c1", taxonomy_id="telecomts-v1", label_status=status, causes=causes or [])


def answer(category: str = SEVERE) -> RcaModelOutput:
    return RcaModelOutput.model_validate(
        _answer(final_fault_categories=[{"category": category, "claim_ids": ["C1", "C2", "C3"]}]))


def test_fault_label_passes_on_matching_cause_and_fails_otherwise():
    fault = ref("labelled_fault", [SEVERE, "Jamming"])
    assert label_check(answer(), fault).status == "passed"
    assert label_check(answer("Antenna Failure"), fault).status == "failed"
    assert label_check(answer("normal"), fault).status == "failed"
    result = label_check(ABSTAINED, fault)
    assert result.status == "failed" and result.detail == "abstained on a labelled fault"
    assert label_check(None, fault).status == "failed"


def test_normal_label_needs_exactly_normal():
    normal = ref("labelled_normal")
    assert label_check(answer("normal"), normal).status == "passed"
    assert label_check(answer(), normal).status == "failed"
    assert label_check(ABSTAINED, normal).detail == "abstained on a labelled normal"


def test_unlabelled_reference_never_passes():
    assert label_check(answer(), ref("unlabelled")).status == "failed"


def test_kpi_support_requires_kpi_evidence_in_each_diagnostic_claim():
    assert kpi_support_check(answer(), _packet()).status == "passed"
    claims = _answer()["claims"]
    claims[2] = {**claims[2], "evidence_ids": ["S1", "S2"]}
    weak = RcaModelOutput.model_validate(_answer(claims=claims))
    result = kpi_support_check(weak, _packet())
    assert result.check_id == "kpi_support.v1" and result.status == "failed" and "C3" in result.detail
    unknown = [*claims[:2], {**claims[2], "evidence_ids": ["K99", "S1"]}]
    assert kpi_support_check(RcaModelOutput.model_validate(_answer(claims=unknown)), _packet()).status == "failed"


def test_kpi_support_passes_for_abstention_and_ignores_normal():
    assert kpi_support_check(ABSTAINED, _packet()).status == "passed"
    claims = [{"claim_id": "C1", "text": "No fault shown.", "claim_type": "diagnostic_inference", "evidence_ids": ["S1"]}]
    normal = RcaModelOutput.model_validate(_answer(
        final_fault_categories=[{"category": "normal", "claim_ids": ["C1"]}], claims=claims))
    assert kpi_support_check(normal, _packet()).status == "passed"
