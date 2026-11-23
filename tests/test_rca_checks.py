"""Task 4.7: strict parsing and versioned structural checks on RCA output."""

import json

import pytest

from clauseguard.api.schemas import RcaModelOutput
from clauseguard.rca import checks
from clauseguard.rca.contracts import EvidencePacket, SourceIdentity
from clauseguard.rca.evidence import EvidenceRegistry
from clauseguard.rca.summarizer import summarize_channel

SOURCE = SourceIdentity(dataset="TelecomTS", revision="r", case_id="c1", artifact_sha256="a" * 64,
                        recording_family_status="unresolved")
SEVERE, MILD = "Co-Channel Interference (Severe)", "Antenna Failure"


def _packet() -> EvidencePacket:
    registry = EvidenceRegistry()
    for channel, values in (("UL_SNR", [21.4, 6.4]), ("RSRP", [-87.8, -98.1])):
        for stat in summarize_channel(channel, values, semantics="reported_value")[0]:
            registry.add_kpi(stat, source=SOURCE, location={"row": 1})  # UL_SNR K1-K3, RSRP K4-K6
    registry.add_text(record_id="log", text="UE reports poor uplink.", origin="independent_observation",
                      source="log", permission="approved")
    for index, text in enumerate(("Interference lowers SINR.", "Antenna faults lower RSRP.")):
        registry.add_specification(chunk_id=f"c{index}", document_id="TS 38.300", version="18.0.0",
                                   clause=f"{index + 1} Clause", chunk_text=text)
    return registry.packet(source=SOURCE, question="Why?", kpi_state="included", text_state="included",
                           retrieval_status="completed")


def _answer(**changes) -> dict:
    output = {
        "status": "answered",
        "final_fault_categories": [{"category": SEVERE, "claim_ids": ["C1", "C2", "C3"]}],
        "claims": [
            {"claim_id": "C1", "text": "UL_SNR minimum is 6.4.", "claim_type": "observation", "evidence_ids": ["K2"]},
            {"claim_id": "C2", "text": "Interference lowers SINR.", "claim_type": "mechanism", "evidence_ids": ["S1"]},
            {"claim_id": "C3", "text": "Low UL_SNR fits interference.", "claim_type": "diagnostic_inference",
             "evidence_ids": ["K2", "S1"]},
        ],
        "uncertainty_reasons": ["units unknown"],
        "abstention_reason": None,
    }
    return output | changes


def _failed(output: dict, taxonomy=checks.TELECOMTS_TAXONOMY) -> dict[str, list[str | None]]:
    results = checks.structural_checks(RcaModelOutput.model_validate(output), _packet(), taxonomy)
    found: dict[str, list[str | None]] = {}
    for result in results:
        if result.status == "failed":
            found.setdefault(result.check_id, []).append(result.subject)
    return found


def test_valid_single_cause_passes_every_check_once():
    results = checks.structural_checks(RcaModelOutput.model_validate(_answer()), _packet(),
                                       checks.TELECOMTS_TAXONOMY)
    assert [r.check_id for r in results] == list(checks.STRUCTURAL_CHECK_IDS)
    assert {r.status for r in results} == {"passed"}


def test_parse_is_strict_and_never_repairs():
    assert checks.parse_model_output(json.dumps(_answer())).status == "answered"
    for raw, message in (("```json\n{}\n```", "not valid JSON"), ("[]", "not a JSON object"),
                         (json.dumps(_answer(extra=1)), "extra"),
                         (json.dumps(_answer(claims=_answer()["claims"] * 2)), "unique"),
                         ("Sure! " + json.dumps(_answer()), "not valid JSON")):
        with pytest.raises(checks.OutputParseError, match=message):
            checks.parse_model_output(raw)


def test_every_citation_failure_is_reported_not_only_the_first():
    claims = _answer()["claims"]
    claims[0] = {**claims[0], "evidence_ids": ["K2", "S1", "K99"]}
    claims[1] = {**claims[1], "evidence_ids": ["S1", "K1", "S1"]}
    claims[2] = {**claims[2], "evidence_ids": ["K2"]}
    found = _failed(_answer(claims=claims))
    assert found["evidence_ids_known.v1"] == ["C1"]
    assert found["evidence_ids_unique_in_claim.v1"] == ["C2"]
    assert found["observation_cites_incident.v1"] == ["C1"]
    assert found["mechanism_cites_specification.v1"] == ["C2"]
    assert found["diagnostic_cites_incident_and_specification.v1"] == ["C3"]
    assert found["diagnostic_links_mechanism.v1"] == [SEVERE]
    assert "packet_contract.v1" not in found


def test_categories_follow_the_declared_taxonomy_and_normal_flag():
    assert _failed(_answer(final_fault_categories=[{"category": "Jamming", "claim_ids": ["C3", "C2"]}])) == {
        "category_in_taxonomy.v1": ["Jamming"]}
    normal = _answer(final_fault_categories=[{"category": "normal", "claim_ids": ["C2", "C3"]}])
    assert _failed(normal) == {}
    strict = checks.Taxonomy(taxonomy_id="t", labels=(SEVERE,), allows_normal=False)
    assert _failed(normal, strict) == {"category_in_taxonomy.v1": ["normal"]}
    with pytest.raises(ValueError):
        checks.Taxonomy(taxonomy_id="t", labels=("normal",), allows_normal=True)


def _two_causes(second_diagnostic_ids, second_link="C6"):
    claims = _answer()["claims"] + [
        {"claim_id": "C5", "text": "Antenna faults lower RSRP.", "claim_type": "mechanism", "evidence_ids": ["S2"]},
        {"claim_id": "C6", "text": "Low RSRP fits an antenna fault.", "claim_type": "diagnostic_inference",
         "evidence_ids": second_diagnostic_ids},
    ]
    return _answer(claims=claims, final_fault_categories=[
        {"category": SEVERE, "claim_ids": ["C1", "C2", "C3"]},
        {"category": MILD, "claim_ids": ["C5", second_link]}])


def test_coexisting_causes_with_separate_support_pass():
    assert _failed(_two_causes(["K5", "S2"])) == {}


def test_two_causes_sharing_one_diagnostic_claim_fail_separate_support():
    output = _two_causes(["K5", "S2"], second_link="C3")
    found = _failed(output)
    assert found["cause_separate_support.v1"] == [SEVERE, MILD]


def test_competing_causes_over_the_same_incident_evidence_must_abstain():
    found = _failed(_two_causes(["K2", "S2"]))
    assert found == {"competing_causes_abstain.v1": [SEVERE, MILD]}


def test_normal_cannot_be_combined_with_a_fault():
    output = _two_causes(["K5", "S2"])
    output["final_fault_categories"][1]["category"] = "normal"
    assert _failed(output) == {"normal_exclusive.v1": ["normal"]}


def test_abstention_reason_must_be_enumerated():
    abstain = {"status": "abstained", "final_fault_categories": [], "claims": [],
               "uncertainty_reasons": [], "abstention_reason": "conflicting_evidence"}
    assert _failed(abstain) == {}
    assert _failed(abstain | {"abstention_reason": "model unsure"}) == {
        "abstention_reason_known.v1": ["abstention_reason"]}


def _numeric(claims, packet=None):
    claims = [*claims, _claim("D", "Low UL_SNR fits interference.", ["K2", "S1"], "diagnostic_inference")]
    output = RcaModelOutput.model_validate(_answer(claims=claims, final_fault_categories=[
        {"category": SEVERE, "claim_ids": [c["claim_id"] for c in claims]}]))
    return {r.claim_id: r for r in checks.numeric_checks(output, packet or _packet())}


def _claim(claim_id, text, ids, kind="observation"):
    return {"claim_id": claim_id, "text": text, "claim_type": kind, "evidence_ids": ids}


def test_numbers_match_cited_records_exactly_or_rounded():
    results = _numeric([
        _claim("C1", "UL_SNR mean is 13.9 and minimum 6.4 over 2 samples.", ["K1", "K2"]),
        _claim("C2", "UL_SNR mean is about 14 (1.39e1).", ["K1"]),
        _claim("C3", "Interference per clause 2 lowers UL_SNR to 6.4.", ["K2", "S1"], "diagnostic_inference"),
        _claim("C4", "Interference lowers SINR.", ["S1"], "mechanism"),
    ])
    assert results["C1"].status == "passed"
    assert [(n.token, n.evidence_id, n.field) for n in results["C1"].numbers] == [
        ("13.9", "K1", "value"), ("6.4", "K2", "value"), ("2", "K1", "valid_count")]
    assert results["C2"].status == "passed"
    assert results["C3"].status == "passed" and "C4" not in results


def test_wrong_or_uncited_numbers_fail():
    results = _numeric([
        _claim("C1", "UL_SNR minimum is 6.5.", ["K2"]),
        _claim("C2", "RSRP minimum is -98.1.", ["K2"]),
        _claim("C3", "UL_SNR fell by 70%.", ["K2"]),
        _claim("C4", "UL_SNR is low.", ["K2"]),
    ])
    assert results["C1"].status == "failed" and results["C1"].numbers[0].status == "unmatched"
    assert results["C2"].status == "failed"
    assert results["C3"].status == "failed"
    assert results["C4"].status == "no_numbers" and results["C4"].numbers == []


def test_specification_numbers_ids_and_percent_missing_fraction():
    registry = EvidenceRegistry()
    registry.add_text(record_id="log", text="Alarm 4012 raised twice.", origin="independent_observation",
                      source="log", permission="approved")
    registry.add_specification(chunk_id="c", document_id="TS 38.300", version="18.0.0", clause="10.1 BLER",
                               chunk_text="BLER target")
    for stat in summarize_channel("UL_SNR", [21.4, 6.4], semantics="reported_value")[0]:
        registry.add_kpi(stat, source=SOURCE, location={"row": 1})
    packet = registry.packet(source=SOURCE, question="Why?", kpi_state="included", text_state="included",
                             retrieval_status="completed")
    results = _numeric([
        _claim("C1", "Per TS 38.300 clause 10.1, alarm 4012 and 0% missing data in K1 matter.",
               ["S1", "T1", "K1"], "diagnostic_inference"),
    ], packet)
    assert [(n.token, n.status, n.field) for n in results["C1"].numbers] == [
        ("38.300", "ignored", "specification_reference"), ("10.1", "ignored", "specification_reference"),
        ("4012", "matched", "text"), ("0%", "matched", "missing_fraction")]
    assert results["C1"].status == "passed"


def test_numeric_v2_accepts_numbers_quoted_from_the_question_but_not_invented_or_truncated_ones():
    packet = _packet().model_copy(update={"question": "In a 12.8-second window, why did UL_SNR fall?"})
    assert checks.NUMERIC_CHECK_ID == "numeric.v2"
    results = _numeric([
        _claim("C1", "Over the 12.8-second window UL_SNR minimum is 6.4.", ["K2"]),
        _claim("C2", "Over the 12.9-second window UL_SNR minimum is 6.4.", ["K2"]),
        _claim("C3", "UL_SNR mean is 13.89999.", ["K1"]),
        _claim("C4", "UL_SNR mean is 13.", ["K1"]),
    ], packet)
    assert results["C1"].status == "passed"
    assert [(n.token, n.status, n.field, n.evidence_id) for n in results["C1"].numbers] == [
        ("12.8", "matched", "question", None), ("6.4", "matched", "value", "K2")]
    assert results["C2"].status == "failed" and results["C2"].numbers[0].token == "12.9"
    assert results["C3"].status == "failed"  # invented precision not in the record
    assert results["C4"].status == "failed"  # truncated: 13.9 rounds to 14, not 13
