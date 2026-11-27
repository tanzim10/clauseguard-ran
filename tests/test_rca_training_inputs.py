"""Task 4.9 inputs: one window per eligible event, normal share, pilot subset and hash-checked rebuild."""

from types import SimpleNamespace

import pytest

from clauseguard.rca.candidates import (
    CandidateIdentity,
    CandidateRecord,
    CandidateReference,
    CandidateVersions,
    Check,
    Eligibility,
    Grouping,
    ModalityStates,
    Observations,
    PredictionRef,
    Taxonomy,
)
from clauseguard.rca.training_inputs import normal_count, rebuild_inputs, select_events


def _pair(cid, event, *, cause=None, status="eligible_pilot", source="TelecomTS", split="train"):
    eligible = status == "eligible_pilot"
    candidate = CandidateRecord(
        identity=CandidateIdentity(candidate_id=cid, source_dataset=source, source_revision="r",
                                   source_record_id=cid, source_location={}, source_sha256="a" * 64),
        modalities=ModalityStates(kpi="included", text="missing", settings="missing"),
        observations=Observations(packet_status="built", packet_sha256="b" * 64, observation_sha256="c" * 64),
        prediction_ref=PredictionRef(packet_head="diagnosis", status="absent", absence_reason="none"),
        grouping=Grouping(group_id=f"fam-{event}", group_status="unresolved", grouping_version="g", event_id=event),
        taxonomy=Taxonomy(taxonomy_id="t", cause_count=1 if cause else 0,
                          label_status="labelled_fault" if cause else "labelled_normal"),
        versions=CandidateVersions(adapter="a"),
        eligibility=Eligibility(status=status, split=split if eligible else None,
                                checks=[Check(check_id="x", version="1", status="passed" if eligible else "failed",
                                              reasons=[] if eligible else ["blocked"])],
                                blockers=[] if eligible else ["blocked"]))
    reference = CandidateReference(candidate_id=cid, taxonomy_id="t", causes=[cause] if cause else [],
                                   label_status="labelled_fault" if cause else "labelled_normal")
    return candidate, reference


def _pool():
    pairs = [_pair(f"f{i}w{w}", f"F{i}", cause=f"cause{i % 4}") for i in range(22) for w in range(2)]
    pairs += [_pair(f"n{i}", f"N{i}", source="TelecomTS" if i % 2 else "BubbleRAN") for i in range(10)]
    pairs += [_pair("blocked", "B0", cause="cause0", status="ineligible")]
    return [c for c, _ in pairs], {r.candidate_id: r for _, r in pairs}


def test_normal_count_keeps_three_in_twenty_five():
    assert normal_count(22, 0.12) == 3 and normal_count(67, 0.12) == 9 and normal_count(10, 0) == 0


def test_selection_keeps_every_fault_event_once_adds_normals_and_skips_ineligible():
    candidates, references = _pool()
    selected = select_events(candidates, references, seed=42, normal_share=0.12, pilot_size=10)
    faults = [s for s in selected if s.stratum == "fault"]
    assert sorted(s.event_id for s in faults) == sorted(f"F{i}" for i in range(22))
    assert len({s.event_id for s in selected}) == len(selected)  # one window per event
    assert sum(s.stratum == "normal" for s in selected) == 3
    assert {s.source_dataset for s in selected if s.stratum == "normal"} == {"TelecomTS", "BubbleRAN"}
    assert "B0" not in {s.event_id for s in selected}
    pilot = [s for s in selected if s.pilot]
    assert len(pilot) == 10 and sum(s.stratum == "normal" for s in pilot) == 1
    assert {s.causes[0] for s in pilot if s.causes} == {"cause0", "cause1", "cause2", "cause3"}
    again = select_events(candidates, references, seed=42, normal_share=0.12, pilot_size=10)
    assert [s.candidate_id for s in again] == [s.candidate_id for s in selected]


def test_selection_rejects_an_invalid_normal_share():
    candidates, references = _pool()
    with pytest.raises(ValueError, match="normal_share"):
        select_events(candidates, references, seed=42, normal_share=1.0, pilot_size=5)


def test_rebuild_refuses_a_packet_that_differs_from_its_candidate_hash():
    candidates, references = _pool()
    selected = select_events(candidates, references, seed=42, normal_share=0.0, pilot_size=1)[:1]
    packet = SimpleNamespace(model_dump=lambda mode=None: {"x": 1})
    with pytest.raises(ValueError, match="differs"):
        rebuild_inputs(selected, {c.identity.candidate_id: c for c in candidates},
                       lambda candidate: (packet, SimpleNamespace()))


# --- Baselines ---------------------------------------------------------------------------------

def _located(cid, source, location, *, split="train"):
    candidate, _ = _pair(cid, f"E-{cid}", cause="x", source=source, split=split)
    return candidate.model_copy(update={"identity": candidate.identity.model_copy(
        update={"source_location": location})})


def test_zone_application_is_read_from_the_location_only():
    from clauseguard.rca.training_baselines import zone_application

    assert zone_application("anomalous/synthetic/Zone_B/Twitch/processed/chunked.jsonl") == ("Zone_B", "Twitch")
    assert zone_application("normal/stationary/Zone_C/no_congestion/YouTube/processed/chunked.jsonl") == (
        "Zone_C", "YouTube")
    assert zone_application("anomalous/jammer/Twitch/processed/chunked.jsonl") is None


def test_baseline_plans_follow_fixed_label_blind_rules():
    from clauseguard.rca.training_baselines import plan_baselines

    tail = "/processed/chunked.jsonl"
    candidates = {c.identity.candidate_id: c for c in (
        _located("fault", "TelecomTS", {"file": "anomalous/synthetic/Zone_B/File" + tail, "row": 3}),
        _located("congested", "TelecomTS", {"file": "normal/stationary/Zone_B/congestion/File" + tail, "row": 1}),
        _located("calm", "TelecomTS", {"file": "normal/stationary/Zone_B/no_congestion/File" + tail, "row": 1}),
        _located("calm_val", "TelecomTS", {"file": "normal/stationary/Zone_B/no_congestion/File" + tail, "row": 2},
                 split="validation"),
        _located("jam", "TelecomTS", {"file": "anomalous/jammer/Twitch" + tail, "row": 1}),
        _located("br", "BubbleRAN", {"file": "a.csv", "first_row": 120, "last_row": 179}),
        _located("br0", "BubbleRAN", {"file": "a.csv", "first_row": 0, "last_row": 59}),
    )}
    plans = {p.candidate_id: p for p in plan_baselines(["fault", "jam", "br", "br0"], candidates, seed=42)}
    assert plans["fault"].rule == "same_zone_application_normal"
    assert plans["fault"].baseline_candidate_id == "calm"  # no-congestion first, train split only
    assert plans["jam"].rule == "none" and "no zone" in plans["jam"].reason
    assert plans["br"].baseline_location == {"file": "a.csv", "first_row": 60, "last_row": 119}
    assert plans["br0"].rule == "none"


def test_preceding_window_baseline_must_end_before_the_incident():
    from clauseguard.rca.contracts import SourceIdentity
    from clauseguard.rca.evidence import BaselineVerification, EvidenceError, EvidenceRegistry
    from clauseguard.rca.summarizer import ChannelStatistic

    registry = EvidenceRegistry()
    stat = ChannelStatistic(channel="snr", statistic="mean", value=1.0, sample_start=0, sample_end=1, valid_count=2,
                            sample_count=2, missing_fraction=0.0, non_finite_count=0)

    def source(case):
        return SourceIdentity(dataset="BubbleRAN", revision="r", case_id=case, artifact_sha256="a" * 64,
                              recording_family_status="unresolved")

    incident = registry.add_kpi(stat, source=source("w2"), location={}, times=["2026-01-01 00:01:00"] * 2)
    earlier = registry.add_kpi(stat, source=source("w1"), location={}, times=["2026-01-01 00:00:00"] * 2)
    later = registry.add_kpi(stat, source=source("w3"), location={}, times=["2026-01-01 00:02:00"] * 2)
    rule = BaselineVerification(kind="earlier_interval", interval_rule="preceding_window", status="verified")
    with pytest.raises(EvidenceError, match="preceding window"):
        registry.attach_baseline(incident.evidence_id, later.evidence_id, rule)
    linked = registry.attach_baseline(incident.evidence_id, earlier.evidence_id, rule)
    assert linked.baseline_evidence_id == earlier.evidence_id
