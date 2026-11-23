"""Task 4.6: candidate records, references kept apart, eligibility, reservations and the release."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from clauseguard.datasets.release import prepare_release
from clauseguard.datasets.scenarios import ScenarioDataset
from clauseguard.rca import candidate_pool
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
    SourceInventory,
    Taxonomy,
)
from clauseguard.rca.contracts import DetectorPrediction, EvidencePacket


def _candidate(**updates) -> CandidateRecord:
    base = {
        "identity": CandidateIdentity(candidate_id="TelecomTS:s1", source_dataset="TelecomTS", source_revision="r",
                                   source_record_id="s1", source_location={"file": "f", "row": 1},
                                   source_sha256="a" * 64),
        "modalities": ModalityStates(kpi="included", text="missing", settings="missing"),
        "observations": Observations(packet_status="not_built", not_built_reason="fixture"),
        "prediction_ref": PredictionRef(packet_head="diagnosis", status="absent", absence_reason="none"),
        "grouping": Grouping(group_id="fam", group_status="unresolved", grouping_version="g1"),
        "taxonomy": Taxonomy(taxonomy_id="telecomts-v1", cause_count=1, label_status="labelled_fault"),
        "versions": CandidateVersions(adapter="rca-evidence-v1"),
    }
    return CandidateRecord(**{**base, **updates})


def test_candidate_rejects_unknown_fields_and_has_no_reference_fields():
    candidate = _candidate()
    assert candidate.eligibility.status == "ineligible"  # safe default until checks run
    with pytest.raises(ValidationError):
        CandidateRecord.model_validate({**candidate.model_dump(), "source_labels": {"rca_label": ["x"]}})
    for field in ("source_labels", "causes", "source_response", "annotation"):
        assert field not in CandidateRecord.model_fields


def test_references_cannot_be_serialized_into_a_packet():
    reference = CandidateReference(candidate_id="c", taxonomy_id="t", source_labels={"rca_label": ["x"]},
                                   causes=["x"], label_status="labelled_fault")
    packet = {"source": {"dataset": "TelecomTS", "revision": "r", "case_id": "c", "artifact_sha256": "a" * 64,
                         "recording_family_status": "unresolved"},
              "question": "q", "kpi_state": "included", "text_state": "missing", "retrieval_status": "not_run"}
    EvidencePacket.model_validate(packet)
    with pytest.raises(ValidationError):
        EvidencePacket.model_validate({**packet, **reference.model_dump(include={"source_labels", "causes"})})


@pytest.mark.parametrize("status, causes", [
    ("labelled_fault", []), ("labelled_normal", ["x"]), ("unlabelled", ["x"]), ("ambiguous", ["x"]),
])
def test_label_status_and_cause_list_must_agree(status, causes):
    with pytest.raises(ValidationError, match="labelled fault"):
        CandidateReference(candidate_id="c", taxonomy_id="t", causes=causes, label_status=status)


def test_multiple_causes_are_preserved_and_normal_differs_from_unlabelled():
    multi = CandidateReference(candidate_id="c", taxonomy_id="t", source_labels={"causes": ["C1", "C3"]},
                               causes=["C1", "C3"], label_status="labelled_fault")
    assert multi.cause_count == 2 and multi.source_labels["causes"] == ["C1", "C3"]
    normal = CandidateReference(candidate_id="n", taxonomy_id="t", source_labels={"rca_label": ["normal"]},
                                label_status="labelled_normal")
    assert normal.cause_count == 0 and normal.label_status != "unlabelled"
    with pytest.raises(ValidationError, match="unlabelled"):
        CandidateReference(candidate_id="u", taxonomy_id="t", source_labels={"x": ["y"]}, label_status="unlabelled")
    with pytest.raises(ValidationError, match="distinct"):
        CandidateReference(candidate_id="d", taxonomy_id="t", causes=["a", "a"], label_status="labelled_fault")


def test_hashes_must_be_lowercase_hex():
    with pytest.raises(ValidationError):
        CandidateIdentity(candidate_id="c", source_dataset="TelecomTS", source_revision="r", source_record_id="s",
                          source_location={}, source_sha256="A" * 64)
    with pytest.raises(ValidationError):
        Observations(packet_status="built", packet_sha256="a" * 63, observation_sha256="b" * 64)


def test_source_names_follow_the_five_source_inventory():
    with pytest.raises(ValidationError):
        CandidateIdentity(candidate_id="c", source_dataset="Other", source_revision="r", source_record_id="s",
                          source_location={}, source_sha256="a" * 64)


def test_eligibility_states_need_consistent_checks_and_reasons():
    passed = Check(check_id="not_reserved", version="1", status="passed")
    assert Eligibility(status="eligible_pilot", checks=[passed]).status == "eligible_pilot"
    with pytest.raises(ValidationError, match="eligible_pilot"):
        Eligibility(status="eligible_pilot", checks=[passed.model_copy(update={"status": "failed"})])
    with pytest.raises(ValidationError, match="eligible_pilot"):
        Eligibility(status="eligible_pilot", checks=[passed], blockers=["x"])
    with pytest.raises(ValidationError, match="reason"):
        Eligibility(status="quarantined")
    with pytest.raises(ValidationError, match="split"):
        Eligibility(status="ineligible", blockers=["x"], split="train")


def test_observations_and_prediction_references_are_explicit():
    with pytest.raises(ValidationError, match="hashes"):
        Observations(packet_status="built")
    with pytest.raises(ValidationError, match="reason"):
        Observations(packet_status="not_built")
    with pytest.raises(ValidationError, match="reason"):
        PredictionRef(packet_head="diagnosis", status="absent")
    head = DetectorPrediction(task="diagnosis", status="ok", predicted_label_id=0, predicted_label="normal")
    with pytest.raises(ValidationError, match="packet head"):
        PredictionRef(packet_head="category", status="available", file_sha256="a" * 64, record_id="s",
                      heads={"diagnosis": head})
    with pytest.raises(ValidationError, match="own task"):
        PredictionRef(packet_head="diagnosis", status="available", file_sha256="a" * 64, record_id="s",
                      heads={"category": head, "diagnosis": head})


def test_inventory_requires_blockers_for_uncleared_sources():
    common = {"identifier": "hf:x", "access": "public", "redistribution": "unknown", "local_status": "absent",
                  "verified_by": "review", "verified_on": "2026-11-17"}
    with pytest.raises(ValidationError, match="blockers"):
        SourceInventory(dataset="TeleLogs", training_use="blocked", **common)
    with pytest.raises(ValidationError, match="local files"):
        SourceInventory(dataset="TeleLogs", training_use="cleared", **{**common, "local_status": "present"})
    assert SourceInventory(dataset="TeleLogs", training_use="blocked", blockers=["gated"], **common)


# --- Reservations (fixture release: one family per split) -------------------------------------

MINIMAL_QUESTION = "Explain the observed KPI changes using the supplied specification evidence."


@pytest.fixture
def dataset(release_sources, recording_evidence):
    source = release_sources
    prepare_release(source["scenarios"], source["output"], source["output"], trials=128)
    return ScenarioDataset(source["output"])


def _write_golden(directory, dataset) -> None:
    membership = dataset.splits["membership"]
    dev = min(sid for sid, row in membership.items() if row["split"] == "dev" and row["origin"] == "synthetic")
    rows = {
        "rca.jsonl": [{"id": "rca_001", "scenario_id": dev, "question": "Why did uplink SNR fall in this window?",
                       "expected_conclusion": "Interference remains possible.",
                       "acceptable_conclusions": ["Radio-quality degradation with an unresolved cause."]}],
        "spec_qa.jsonl": [{"id": "spec_qa_001", "question": "What does RRU.PrbUsedDl measure?",
                           "expected_answer": "Average PRBs used for DL data traffic."}],
    }
    for name, content in rows.items():
        (directory / name).write_text("".join(json.dumps(r) + "\n" for r in content))
    (directory / "rca_questions.candidate.json").write_text(json.dumps([
        {"id": "rca_001_questions", "scenario_id": dev,
         "qualitative_question": "Compared with the earlier interval, UL_SNR decreased. Which causes fit?",
         "numeric_question": "Compared with the earlier interval, UL_SNR changed from 21 to 6. Which causes fit?"}]))
    (directory / "rca_policy.json").write_text(json.dumps({
        "source_release_id": dataset.release_id, "minimal_question": MINIMAL_QUESTION,
        "cases": {"rca_001": {"scenario_id": dev}}}))
    files = {name: {"sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest()}
             for name in ("rca.jsonl", "spec_qa.jsonl", "rca_policy.json")}
    (directory / "manifest.json").write_text(json.dumps({"release_id": "golden-fixture", "files": files}))


@pytest.fixture
def golden_dir(tmp_path, dataset):
    directory = tmp_path / "golden-candidates"
    directory.mkdir()
    _write_golden(directory, dataset)
    return directory


def test_reservations_cover_golden_cases_families_and_texts(dataset, golden_dir):
    reservations = candidate_pool.build_reservations(dataset, golden_dir)
    membership = dataset.splits["membership"]
    dev_test = {row["group_id"] for row in membership.values() if row["split"] in ("dev", "test")}
    [case] = reservations.scenario_ids
    assert set(reservations.families) == dev_test and membership[case]["split"] == "dev"
    assert {t.text_id for t in reservations.texts} == {
        "rca_001:question", "rca_001:expected_conclusion", "rca_001:acceptable_conclusion:0",
        "spec_qa_001:question", "spec_qa_001:expected_answer",
        "rca_001_questions:qualitative_question", "rca_001_questions:numeric_question"}
    assert {e["dataset"] for e in reservations.external} == {"TeleLogs"}
    # A golden case, its family and its paraphrases are all refused.
    assert len(reservations.reasons(family=membership[case]["group_id"], record_id=case)) == 2
    train = next(row for row in membership.values() if row["split"] == "train")
    assert reservations.reasons(family=train["group_id"], record_id=train["scenario_id"]) == []
    paraphrase = "compared with the earlier interval,  UL_SNR decreased. Which causes FIT?"
    assert reservations.text_matches(paraphrase, threshold=0.8, size=5) == ["rca_001_questions:qualitative_question"]
    near = "Compared with the earlier interval, UL_SNR changed from 22 to 6. Which causes fit?"
    assert "rca_001_questions:numeric_question" in reservations.text_matches(near, threshold=0.8, size=5)
    assert reservations.text_matches(MINIMAL_QUESTION, threshold=0.8, size=5) == []


def test_reservations_refuse_changed_golden_files_and_are_hash_checked(dataset, golden_dir, tmp_path):
    reservations = candidate_pool.build_reservations(dataset, golden_dir)
    path = tmp_path / "reservations.json"
    digest = candidate_pool.write_json(path, reservations.model_dump(mode="json"))
    assert candidate_pool.load_reservations(path, digest) == reservations
    path.write_text(path.read_text().replace("rca_001:question", "rca_009:question"))
    with pytest.raises(ValueError, match="changed"):
        candidate_pool.load_reservations(path, digest)
    (golden_dir / "rca.jsonl").write_text("")
    with pytest.raises(ValueError, match="differs from its release manifest"):
        candidate_pool.build_reservations(dataset, golden_dir)


def test_minimal_question_matches_the_tracked_golden_policy():
    policy = json.loads(Path("data/golden/rca_policy.json").read_text(encoding="utf-8"))
    assert policy["minimal_question"] == MINIMAL_QUESTION


def test_inventory_lists_all_five_sources_with_terms_and_local_hashes(tmp_path):
    from clauseguard.rca.contracts import DATASET_SOURCE_RULES

    bubbleran = tmp_path / "bubbleran"
    bubbleran.mkdir()
    (bubbleran / "anomalous_data.csv").write_text("timestamp\n1\n")
    acquisition = tmp_path / "acquisition.json"
    acquisition.write_text(json.dumps({"status": "complete", "revision": "a" * 40,
                                       "acquired_at": "2026-10-14T05:55:35+00:00",
                                       "files": [{"source_file": "x/processed/chunked.jsonl", "sha256": "c" * 64}]}))
    inventory = {s.dataset: s for s in candidate_pool.build_inventory(
        telecomts_acquisition=acquisition, bubbleran_dir=bubbleran, spotlight_dir=tmp_path / "missing")}
    assert set(inventory) == set(DATASET_SOURCE_RULES)
    tele = inventory["TelecomTS"]
    assert (tele.training_use, tele.revision, tele.verified_on, tele.blockers) == ("cleared", "a" * 40, "2026-10-14", [])
    bubble = inventory["BubbleRAN"]
    assert bubble.training_use == "pending" and bubble.redistribution == "share_alike"
    assert bubble.local_files == {"anomalous_data.csv": hashlib.sha256(b"timestamp\n1\n").hexdigest()}
    assert inventory["SpotLight"].local_status == "absent" and inventory["SpotLight"].training_use == "blocked"
    assert "source files are not available locally" in inventory["SpotLight"].blockers
    for name in ("TeleLogs", "5G core faults"):
        assert inventory[name].training_use == "blocked" and inventory[name].adapter is None
        assert inventory[name].local_status == "absent" and inventory[name].blockers
    assert inventory["TeleLogs"].redistribution == "forbidden_by_request"
    absent = {s.dataset: s for s in candidate_pool.build_inventory()}["TelecomTS"]
    assert absent.local_status == "absent" and "source files are not available locally" in absent.blockers


# --- TelecomTS candidates -----------------------------------------------------------------------

def _case(sid, release_id, diagnosis_label):
    from clauseguard.rca.detector import CaseRecord, PredictionRecord, TextEligibility, label_names

    def record(task, label):
        if label is None:
            return PredictionRecord(prediction=DetectorPrediction(task=task, status="absent", absence_reason="normal"))
        return PredictionRecord(prediction=DetectorPrediction(
            task=task, status="ok", predicted_label=label, predicted_label_id=label_names(task).index(label),
            label_mapping_version="1", model_run_id=f"fixture/{task}", source_modalities=["kpi"]))

    fault = diagnosis_label != "normal"
    return CaseRecord(release_id=release_id, scenario_id=sid, purpose="slm_training_candidate",
                      selection_sha256="f" * 64, fold=0,
                      detection=record("detection", "abnormal" if fault else "normal"),
                      category=record("category", diagnosis_label if fault else None),
                      diagnosis=record("diagnosis", diagnosis_label),
                      text=TextEligibility(state="missing", origin="unreviewed", reason="no_approved_text"),
                      readiness="provisional_exploratory")


@pytest.fixture
def predictions(dataset, tmp_path):
    """Out-of-fold style predictions: every fault is called Antenna Failure, every normal is normal."""
    from clauseguard.rca.detector import write_jsonl

    membership = dataset.splits["membership"]
    records = [_case(sid, dataset.release_id, "Antenna Failure" if row["fault_label"] == "abnormal" else "normal")
               for sid, row in sorted(membership.items()) if row["split"] == "train"]
    path = tmp_path / "cross_fit" / "predictions.jsonl"
    digest = write_jsonl(path, records)
    (path.parent / "folds.json").write_text(json.dumps({"predictions_sha256": digest,
                                                        "release_id": dataset.release_id}))
    return path


def _built(dataset, predictions, config=None):
    config = config or candidate_pool.CandidateConfig()
    cases, digest = candidate_pool.load_predictions(predictions, predictions.parent / "folds.json",
                                                    release_id=dataset.release_id)
    return list(candidate_pool.telecomts_candidates(dataset, cases, predictions_sha256=digest, config=config))


def test_telecomts_candidates_cover_training_windows_only(dataset, predictions):
    membership = dataset.splits["membership"]
    built = _built(dataset, predictions)
    train = {sid for sid, row in membership.items() if row["split"] == "train"}
    assert {b.candidate.identity.source_record_id for b in built} == train
    for item in built:
        candidate, reference, texts = item
        assert texts == ()
        record = dataset.load_scenario(candidate.identity.source_record_id)
        assert candidate.identity.candidate_id == reference.candidate_id
        assert candidate.identity.source_sha256 == record.provenance.source_sha256
        assert candidate.observations.packet_status == "built" and candidate.observations.kpi_records == 48
        assert candidate.observations.rejected_channels == ["DL_Protocol", "UL_Protocol"]
        assert (candidate.modalities.kpi, candidate.modalities.text, candidate.modalities.settings) == (
            "included", "missing", "missing")
        assert candidate.prediction_ref.packet_head == "diagnosis" and candidate.prediction_ref.status == "available"
        assert candidate.taxonomy.cause_count == reference.cause_count
        if record.fault_label == "abnormal":
            assert reference.causes == [record.rca_label] and reference.label_status == "labelled_fault"
        else:
            assert reference.causes == [] and reference.label_status == "labelled_normal"
        assert reference.annotation["anomaly_start_index"] == record.anomaly.start_index
        assert "reference.description" in reference.source_text_refs and reference.source_response is None
    with pytest.raises(ValueError, match="training-split"):
        next(candidate_pool.telecomts_candidates(dataset, {}, predictions_sha256=None,
                                                 config=candidate_pool.CandidateConfig(), split="dev"))


def test_rebuilt_packet_matches_its_hash_and_holds_no_reference_fields(dataset, predictions):
    config = candidate_pool.CandidateConfig()
    item = next(b for b in _built(dataset, predictions) if b.reference.label_status == "labelled_fault")
    sid = item.candidate.identity.source_record_id
    record = dataset.load_scenario(sid)
    cases, _ = candidate_pool.load_predictions(predictions, predictions.parent / "folds.json",
                                               release_id=dataset.release_id)
    packet, _ = candidate_pool.telecomts_packet(record, dataset.splits["membership"][sid],
                                                cases[sid].diagnosis.prediction, config)
    assert candidate_pool.packet_digest(packet) == item.candidate.observations.packet_sha256
    assert packet.question == MINIMAL_QUESTION and packet.retrieval_status == "not_run"
    assert packet.detector_prediction.task == "diagnosis" and packet.detector_prediction.predicted_label == "Antenna Failure"
    serialized = packet.model_dump_json()
    for leaked in (record.rca_label, "Revealing source text", "anomaly", "QnA", "source_labels"):
        assert leaked not in serialized


def test_prediction_head_is_configurable_and_prediction_files_are_hash_checked(dataset, predictions):
    built = _built(dataset, predictions, candidate_pool.CandidateConfig(packet_head="category"))
    assert {b.candidate.prediction_ref.packet_head for b in built} == {"category"}
    predictions.write_text(predictions.read_text().replace("Antenna Failure", "Faulty RF Filters", 1))
    with pytest.raises(ValueError, match="cross-fit record"):
        _built(dataset, predictions)


# --- Grouping and duplicate checks -------------------------------------------------------------

def _span(sid, source, start, category=1, detection=1, group="fam", split="train"):
    from clauseguard.datasets.window_overlap import WindowSpan

    return WindowSpan(sid, source, start, start + 127, detection, category, group, split)


def test_overlapping_windows_share_an_event_and_cross_label_overlap_is_reported():
    from clauseguard.rca.candidate_release import telecomts_events

    spans = [_span("a", "f1", 1), _span("b", "f1", 97), _span("c", "f1", 300), _span("d", "f2", 1),
             _span("e", "f3", 1, category=2), _span("f", "f3", 100, category=3), _span("g", "f3", 180, category=2)]
    events, shared = telecomts_events(spans)
    assert events["a"] == events["b"] != events["c"]
    assert events["d"] != events["a"]  # same rows, different source file
    assert events["e"] != events["f"] and events["e"] != events["g"]  # different labels, or no overlap
    assert shared == [sorted({events["e"], events["f"], events["g"]})]


def _with(item, *, event, group=None, observation=None, texts=()):
    candidate = item.candidate
    grouping = candidate.grouping.model_copy(update={"event_id": event, "group_id": group or candidate.grouping.group_id})
    observations = candidate.observations.model_copy(
        update={"observation_sha256": observation or candidate.observations.observation_sha256})
    return item._replace(candidate=candidate.model_copy(update={"grouping": grouping, "observations": observations}),
                         texts=tuple(texts))


PARAPHRASE = "Compared with the earlier interval, UL_SNR decreased. Which causes fit?"


def test_duplicates_are_grouped_or_quarantined_and_reserved_text_is_found(dataset, predictions, golden_dir):
    from clauseguard.rca.candidate_release import duplicate_checks

    reservations = candidate_pool.build_reservations(dataset, golden_dir)
    config = candidate_pool.CandidateConfig()
    a, b, c, d, e = _built(dataset, predictions)[:5]
    items = [
        _with(a, event="e1", observation="1" * 64), _with(b, event="e1", observation="1" * 64),  # same event
        _with(c, event="e2", observation="2" * 64),
        _with(d, event="e3", group="other-family", observation="2" * 64,  # cross-group exact duplicate
              texts=[PARAPHRASE]),
        _with(e, event="e4", group="third-family", texts=[PARAPHRASE]),
    ]
    report = duplicate_checks(items, reservations, config, shared_row_events=[["e1", "e2"]])
    ids = [i.candidate.identity.candidate_id for i in items]
    resolutions = {tuple(g["candidates"]): g["resolution"] for g in report["exact_duplicate_groups"]}
    assert resolutions[tuple(sorted(ids[:2]))] == "same_event"
    assert resolutions[tuple(sorted(ids[2:4]))] == "unresolved_cross_group"
    assert set(report["quarantined"]) == set(ids[2:5])
    assert report["near_duplicate_text"] == [{"candidates": sorted(ids[3:5]), "resolution": "unresolved_cross_group"}]
    assert report["reserved_text_matches"][ids[3]] == ["rca_001_questions:qualitative_question"]
    assert report["question_template_matches"] == {"kpi": [], "text": [], "kpi+text": []}
    assert report["events_sharing_raw_rows"] == [["e1", "e2"]]


# --- Eligibility and grouped split -------------------------------------------------------------

def _pool(dataset, predictions, golden_dir, config=None, inventory_updates=None):
    from clauseguard.datasets.window_overlap import load_window_spans
    from clauseguard.rca.candidate_release import assign_events, duplicate_checks, telecomts_events

    config = config or candidate_pool.CandidateConfig()
    reservations = candidate_pool.build_reservations(dataset, golden_dir)
    events, shared = telecomts_events(load_window_spans(dataset))
    built = assign_events(_built(dataset, predictions, config), events)
    dedup = duplicate_checks(built, reservations, config, shared_row_events=shared)
    inventory = {s.dataset: s for s in candidate_pool.build_inventory()}
    for name, update in (inventory_updates or {}).items():
        inventory[name] = inventory[name].model_copy(update=update)
    return built, inventory, reservations, dedup, config


def test_eligibility_follows_family_terms_reservations_and_quarantine(dataset, predictions, golden_dir):
    from clauseguard.rca.candidate_release import evaluate_eligibility

    built, inventory, reservations, dedup, config = _pool(dataset, predictions, golden_dir)
    assert {b.candidate.grouping.group_status for b in built} == {"verified"}  # the fixture families are reviewed
    result = evaluate_eligibility(built, inventory=inventory, reservations=reservations, dedup=dedup, config=config)
    assert {r.candidate.eligibility.status for r in result} == {"eligible_pilot"}
    assert {c.check_id for c in result[0].candidate.eligibility.checks} == {
        "not_reserved", "terms_cleared", "observations_built", "family_resolved", "event_grouped",
        "duplicates_resolved", "prediction_reference"}

    unresolved = [b._replace(candidate=b.candidate.model_copy(update={
        "grouping": b.candidate.grouping.model_copy(update={"group_status": "unresolved"})})) for b in built]
    result = evaluate_eligibility(unresolved, inventory=inventory, reservations=reservations, dedup=dedup,
                                  config=config)
    first = result[0].candidate.eligibility
    assert {r.candidate.eligibility.status for r in result} == {"exploratory"}
    assert first.blockers == ["recording_family_unresolved"] and first.exclusion_reasons == []

    for training_use, expected in (("pending", "ineligible"), ("blocked", "ineligible")):
        terms = {"TelecomTS": {"training_use": training_use, "blockers": ["terms under review"]}}
        _, inventory, *_ = _pool(dataset, predictions, golden_dir, inventory_updates=terms)
        result = evaluate_eligibility(built, inventory=inventory, reservations=reservations, dedup=dedup,
                                      config=config)
        assert {r.candidate.eligibility.status for r in result} == {expected}
        assert f"training_terms_{training_use}" in result[0].candidate.eligibility.blockers
    lenient = config.model_copy(update={"pending_terms_status": "exploratory"})
    terms = {"TelecomTS": {"training_use": "pending", "blockers": ["terms under review"]}}
    _, inventory, *_ = _pool(dataset, predictions, golden_dir, inventory_updates=terms)
    result = evaluate_eligibility(built, inventory=inventory, reservations=reservations, dedup=dedup, config=lenient)
    assert {r.candidate.eligibility.status for r in result} == {"exploratory"}

    first_id = built[0].candidate.identity.candidate_id
    family = built[0].candidate.grouping.group_id
    _, inventory, *_ = _pool(dataset, predictions, golden_dir)
    reserved = reservations.model_copy(update={"families": [*reservations.families, family]})
    flagged = {**dedup, "quarantined": {built[1].candidate.identity.candidate_id: ["exact duplicate"]}}
    result = evaluate_eligibility(built, inventory=inventory, reservations=reserved, dedup=flagged, config=config)
    by_id = {r.candidate.identity.candidate_id: r.candidate.eligibility for r in result}
    assert by_id[first_id].status == "ineligible" and "reserved evaluation family" in by_id[first_id].exclusion_reasons[0]
    assert by_id[built[1].candidate.identity.candidate_id].status == "quarantined"


def test_splits_are_grouped_by_family_and_skip_ineligible_candidates(dataset, predictions, golden_dir):
    from clauseguard.rca.candidate_release import assign_splits, evaluate_eligibility

    built, inventory, reservations, dedup, config = _pool(dataset, predictions, golden_dir)
    result = evaluate_eligibility(built, inventory=inventory, reservations=reservations, dedup=dedup, config=config)
    # Give each origin several families so a stratum can lose one to validation.
    regrouped = [r._replace(candidate=r.candidate.model_copy(update={"grouping": r.candidate.grouping.model_copy(
        update={"group_id": f"{r.reference.annotation['origin']}-{i % 5}"})})) for i, r in enumerate(result)]
    excluded = regrouped[0]._replace(candidate=regrouped[0].candidate.model_copy(update={
        "eligibility": Eligibility(status="ineligible", exclusion_reasons=["fixture"])}))
    split, summary = assign_splits([excluded, *regrouped[1:]], config)
    families = {}
    for item in split:
        group, chosen = item.candidate.grouping.group_id, item.candidate.eligibility.split
        if item.candidate.eligibility.status == "ineligible":
            assert chosen is None
            continue
        assert families.setdefault(group, chosen) == chosen  # no family in both splits
    assert "validation" in families.values() and "train" in families.values()
    again, _ = assign_splits([excluded, *regrouped[1:]], config)
    assert [i.candidate for i in again] == [i.candidate for i in split]  # deterministic
    single, summary = assign_splits(result, config)
    assert {i.candidate.eligibility.split for i in single} == {"train"}  # one family per stratum stays in train
    assert all(v["validation_families"] == 0 for v in summary["strata"].values())


# --- BubbleRAN candidates ----------------------------------------------------------------------

def _bubbleran_csv(path, labels, stamps=None):
    stamps = stamps or [100, 101, 102, 103, 104, 110, 111, 112, 113, 114]
    header = ["timestamp", "ran_ue_id", "e2node_nb_id", "mac_pusch_snr", "mac_dl_aggr_prb", "timestamp_iso",
              "persistent_anomaly"]
    lines = [",".join(header)] + [
        f"{t},2,50,{10 + i},{i * 5},2026-01-01T00:00:{i:02d},{label}" for i, (t, label) in enumerate(zip(stamps, labels))]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_bubbleran_grid_is_label_blind_and_never_crosses_a_gap(tmp_path):
    labels = ["False"] * 6 + ["True", "True"] + ["False"] * 2
    path = _bubbleran_csv(tmp_path / "anomalous_data.csv", labels)
    assert candidate_pool.bubbleran_windows(path, rows=3, stride=3) == [(0, 2), (5, 7)]
    flipped = _bubbleran_csv(tmp_path / "flipped.csv", ["True"] * 10)
    assert candidate_pool.bubbleran_windows(flipped, rows=3, stride=3) == [(0, 2), (5, 7)]
    assert candidate_pool.bubbleran_windows(path, rows=2, stride=1) == [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6),
                                                                        (6, 7), (7, 8), (8, 9)]


def test_bubbleran_candidates_keep_labels_in_references_only(tmp_path):
    labels = ["False"] * 6 + ["True", "True"] + ["False"] * 2
    path = _bubbleran_csv(tmp_path / "anomalous_data.csv", labels)
    config = candidate_pool.CandidateConfig(bubbleran_window_rows=3, bubbleran_stride=3)
    normal, fault = candidate_pool.bubbleran_candidates([path], config=config, family_id="BubbleRAN:ue-2:gnb-50")
    assert normal.candidate.identity.candidate_id == "BubbleRAN:anomalous_data.csv:rows-0-2"
    assert fault.candidate.identity.source_location == {"file": "anomalous_data.csv", "first_row": 5, "last_row": 7}
    assert (normal.reference.label_status, fault.reference.label_status) == ("labelled_normal", "labelled_fault")
    assert fault.reference.causes == ["persistent_interference"] and fault.reference.annotation["anomalous_rows"] == 2
    assert fault.reference.source_labels == {"persistent_anomaly": ["False", "True"]}
    assert fault.candidate.prediction_ref.status == "absent" and "MAC-layer" in fault.candidate.prediction_ref.absence_reason
    assert fault.candidate.grouping.group_id == normal.candidate.grouping.group_id == "BubbleRAN:ue-2:gnb-50"
    assert "persistent_anomaly" in fault.candidate.observations.rejected_channels
    assert "mac_dl_aggr_prb" in fault.candidate.observations.rejected_channels  # counter, not differenced
    packet, _ = candidate_pool.bubbleran_packet(path, 5, 7, config)
    assert candidate_pool.packet_digest(packet) == fault.candidate.observations.packet_sha256
    assert packet.detector_prediction.status == "absent" and "persistent_anomaly" not in packet.model_dump_json()
    again = candidate_pool.bubbleran_candidates([path], config=config, family_id="BubbleRAN:ue-2:gnb-50")
    assert [b.candidate for b in again] == [normal.candidate, fault.candidate]  # stable IDs and hashes


# --- Batch, manifest, gap report and verification ----------------------------------------------

def _admitted(dataset, predictions, golden_dir, config=None):
    from clauseguard.rca.candidate_release import assign_splits, evaluate_eligibility

    built, inventory, reservations, dedup, config = _pool(dataset, predictions, golden_dir, config)
    built = evaluate_eligibility(built, inventory=inventory, reservations=reservations, dedup=dedup, config=config)
    built, split_summary = assign_splits(built, config)
    return built, inventory, reservations, dedup, split_summary, config


def test_batch_is_deterministic_and_one_per_event(dataset, predictions, golden_dir):
    from clauseguard.rca.candidate_release import select_batch

    built, *_, config = _admitted(dataset, predictions, golden_dir)
    rows, shortfalls = select_batch(built, config)
    assert rows == select_batch(list(reversed(built)), config)[0]  # input order does not matter
    by_id = {b.candidate.identity.candidate_id: b for b in built}
    events = [by_id[r["candidate_id"]].candidate.grouping.event_id for r in rows]
    assert len(events) == len(set(events)) and len(rows) <= config.batch_size
    synthetic = [r for r in rows if r["stratum"].startswith("TelecomTS/synthetic/")]
    assert synthetic and all(r["detector_correction"] for r in synthetic
                             if "Antenna Failure" not in r["stratum"])  # every fixture fault is called Antenna Failure
    assert not next(r for r in rows if r["stratum"] == "TelecomTS/synthetic/Antenna Failure")["detector_correction"]
    assert [r for r in rows if r["stratum"] == "labelled_normal"]  # normals enter only as labelled_normal
    assert shortfalls["BubbleRAN"] == {"quota": 3, "selected": 0, "available_events": 0}
    small = config.model_copy(update={"batch_size": 4})
    assert len(select_batch(built, small)[0]) == 4
    blocked = [b._replace(candidate=b.candidate.model_copy(update={"eligibility": Eligibility(
        status="ineligible", blockers=["x"])})) for b in built]
    assert select_batch(blocked, config)[0] == []


def _rebuild(dataset, predictions, config):
    cases, _ = candidate_pool.load_predictions(predictions, predictions.parent / "folds.json",
                                               release_id=dataset.release_id)
    membership = dataset.splits["membership"]

    def rebuild(candidate):
        sid = candidate.identity.source_record_id
        return candidate_pool.telecomts_packet(dataset.load_scenario(sid), membership[sid],
                                               getattr(cases[sid], config.packet_head).prediction, config)[0]
    return rebuild


def test_release_export_counts_and_verification_detect_tampering(dataset, predictions, golden_dir, tmp_path):
    from clauseguard.rca.candidate_release import export_release, select_batch, verify_release

    built, inventory, reservations, dedup, split_summary, config = _admitted(dataset, predictions, golden_dir)
    rows, shortfalls = select_batch(built, config)
    out = tmp_path / "candidates"
    manifest = export_release(out, built=built, inventory=list(inventory.values()), reservations=reservations,
                              dedup=dedup, split_summary=split_summary, batch_rows=rows, shortfalls=shortfalls,
                              config=config, rebuild_packet=_rebuild(dataset, predictions, config),
                              prediction_files={"cross_fit": "a" * 64}, created_on="2026-11-17")
    result = verify_release(out)
    assert result["candidates"] == len(built) and result["packets_checked"] == len(rows) == result["batch"]
    counts = manifest.counts
    assert counts["by_source"]["TelecomTS"]["rows"] == len(built) and counts["by_source"]["TelecomTS"]["families"] == 1
    assert counts["total"]["events"] < counts["total"]["rows"]  # overlapping normal windows share an event
    assert counts["by_prediction"]["diagnosis:ok"]["rows"] == len(built)
    assert set(counts["by_cause_count"]) == {"0", "1"} and manifest.batch.candidate_ids == [r["candidate_id"] for r in rows]
    assert [s.dataset for s in manifest.sources] == list(inventory)
    report = (out / "gap_report.md").read_text()
    assert "No multiple-cause candidates" in report and "No `text` candidates" in report and "TeleLogs" in report
    references = (out / "references.jsonl").read_text()
    packets = (out / "packets.jsonl").read_text()
    assert "rca_label" in references and "rca_label" not in packets and "causes" not in packets
    with pytest.raises(FileExistsError):
        export_release(out, built=built, inventory=list(inventory.values()), reservations=reservations, dedup=dedup,
                       split_summary=split_summary, batch_rows=rows, shortfalls=shortfalls, config=config,
                       rebuild_packet=_rebuild(dataset, predictions, config), prediction_files={},
                       created_on="2026-11-17")
    (out / "candidates.jsonl").write_text((out / "candidates.jsonl").read_text().replace('"train"', '"validation"', 1))
    with pytest.raises(ValueError, match="changed: candidates.jsonl"):
        verify_release(out)


def test_export_refuses_a_packet_that_no_longer_matches_its_hash(dataset, predictions, golden_dir, tmp_path):
    from clauseguard.rca.candidate_release import export_release, select_batch

    built, inventory, reservations, dedup, split_summary, config = _admitted(dataset, predictions, golden_dir)
    rows, shortfalls = select_batch(built, config)
    changed = config.model_copy(update={"question_templates": {**config.question_templates, "kpi": "Other question."}})
    with pytest.raises(ValueError, match="rebuilt packet differs"):
        export_release(tmp_path / "c", built=built, inventory=list(inventory.values()), reservations=reservations,
                       dedup=dedup, split_summary=split_summary, batch_rows=rows, shortfalls=shortfalls, config=config,
                       rebuild_packet=_rebuild(dataset, predictions, changed), prediction_files={},
                       created_on="2026-11-17")


# --- CLI ----------------------------------------------------------------------------------------

def test_cli_builds_and_verifies_a_release_and_detects_an_altered_file(dataset, predictions, golden_dir, tmp_path,
                                                                        capsys):
    from clauseguard.cli import build_candidates as cli

    bubbleran = tmp_path / "bubbleran"
    bubbleran.mkdir()
    _bubbleran_csv(bubbleran / "anomalous_data.csv", ["False"] * 6 + ["True", "True"] + ["False"] * 2)
    spotlight = tmp_path / "spotlight"
    spotlight.mkdir()
    _spotlight_csv(spotlight / "radio.csv")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"bubbleran_window_rows": 3, "bubbleran_stride": 3, "spotlight_window_rows": 5,
                                  "spotlight_stride": 5}))
    out = tmp_path / "candidate-release"
    common = ["--release-dir", str(dataset.release_dir), "--golden-dir", str(golden_dir)]
    paths = ["--acquisition", str(tmp_path / "none.json"), "--bubbleran-dir", str(bubbleran),
             "--spotlight-dir", str(spotlight)]
    assert cli.main(["build", *common, *paths, "--predictions", str(predictions), "--config", str(config),
                     "--created-on", "2026-11-17", "--output-dir", str(out)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert set(result["by_source"]) == {"TelecomTS", "BubbleRAN", "SpotLight"}
    assert result["by_source"]["BubbleRAN"]["rows"] == 2 and result["by_source"]["SpotLight"]["rows"] == 2
    candidates = [json.loads(line) for line in (out / "candidates.jsonl").read_text().splitlines()]
    bubble = [c for c in candidates if c["identity"]["source_dataset"] == "BubbleRAN"]
    assert {c["eligibility"]["status"] for c in bubble} == {"ineligible"}  # CC BY-SA terms still pending
    assert all("training_terms_pending" in c["eligibility"]["blockers"] for c in bubble)
    spot = [c for c in candidates if c["identity"]["source_dataset"] == "SpotLight"]
    assert {c["eligibility"]["status"] for c in spot} == {"ineligible"}  # no licence
    assert all("training_terms_blocked" in c["eligibility"]["blockers"] for c in spot)
    assert cli.main(["verify", "--output-dir", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["candidates"] == len(candidates)
    assert cli.main(["reserve", *common, "--output", str(tmp_path / "reservations.json")]) == 0
    assert json.loads(capsys.readouterr().out)["scenario_ids"] == 1
    assert cli.main(["inventory", *paths]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 5
    (out / "references.jsonl").write_text((out / "references.jsonl").read_text() + "\n")
    assert cli.main(["verify", "--output-dir", str(out)]) == 1
    assert "changed: references.jsonl" in capsys.readouterr().err
    assert cli.main(["build", *common, *paths, "--predictions", str(predictions), "--created-on", "2026-11-17",
                     "--output-dir", str(out)]) == 1


# --- SpotLight candidates ----------------------------------------------------------------------

def _spotlight_csv(path, *, index_column=False):
    """Two captures of 5 rows at 0.15 s, separated by a repeated header, like the SpotLight sample."""
    header = (["", "timestamp"] if index_column else ["timestamp"]) + ["mac_sinr_update_mean", "anomaly"]
    lines = [",".join(header)]
    for capture, base in enumerate((0, 10)):
        if capture:
            lines.append(",".join(header))
        for i in range(5):
            stamp = f"2024-02-15 05:28:{base + 33 + i * 0.15:06.3f}000000"
            values = [str(i)] if index_column else []
            lines.append(",".join([*values, stamp, str(20 + i), "0"]))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_spotlight_grid_never_crosses_a_repeated_header(tmp_path):
    path = _spotlight_csv(tmp_path / "radio.csv")
    assert candidate_pool.spotlight_windows(path, rows=2, stride=2) == [(0, 0, 1), (0, 2, 3), (1, 6, 7), (1, 8, 9)]
    assert candidate_pool.spotlight_windows(path, rows=5, stride=5) == [(0, 0, 4), (1, 6, 10)]
    assert candidate_pool.spotlight_windows(path, rows=6, stride=1) == []


def test_spotlight_candidates_are_ambiguous_and_keep_labels_in_references(tmp_path):
    radio = _spotlight_csv(tmp_path / "radio.csv")
    platform = _spotlight_csv(tmp_path / "platform.csv", index_column=True)
    start = candidate_pool._naive_ns("2024-02-15 05:28:33.000000000")
    label = tmp_path / "label"
    label.write_text(f"{start},{start + 200_000_000}\n")
    config = candidate_pool.CandidateConfig(spotlight_window_rows=5, spotlight_stride=5)
    built = candidate_pool.spotlight_candidates([radio, platform], label_path=label, config=config,
                                                family_id="SpotLight:experiment")
    assert len(built) == 4 and {b.candidate.grouping.group_id for b in built} == {"SpotLight:experiment"}
    first = built[0]
    assert first.reference.label_status == "ambiguous" and first.reference.causes == []
    assert first.reference.source_labels == {"anomaly_column": ["0"], "label_file_overlap": ["yes"]}
    assert built[1].reference.source_labels["label_file_overlap"] == ["no"]
    assert first.reference.annotation["stream"] == "radio" and built[2].reference.annotation["stream"] == "platform"
    assert first.candidate.prediction_ref.status == "absent" and first.candidate.observations.kpi_records == 3
    assert "anomaly" in first.candidate.observations.rejected_channels
    location = built[1].candidate.identity.source_location
    assert (location["capture"], location["first_row"], location["last_row"]) == (1, 6, 10)
    packet, _ = candidate_pool.spotlight_packet(radio, location["start_time"], location["end_time"], config)
    assert candidate_pool.packet_digest(packet) == built[1].candidate.observations.packet_sha256
    assert '"anomaly"' not in packet.model_dump_json()
    assert len({b.candidate.grouping.event_id for b in built}) == 4


# --- Multiple causes and label states through export (synthetic fixtures only) -----------------

def _relabel(item, *, taxonomy, causes, status, labels):
    reference = CandidateReference(candidate_id=item.candidate.identity.candidate_id, taxonomy_id=taxonomy,
                                   source_labels=labels, causes=causes, label_status=status)
    taxonomy_record = Taxonomy(taxonomy_id=taxonomy, cause_count=len(causes), label_status=status)
    return item._replace(candidate=item.candidate.model_copy(update={"taxonomy": taxonomy_record}),
                         reference=reference)


def test_cause_lists_taxonomies_and_label_states_survive_export(dataset, predictions, golden_dir, tmp_path):
    from clauseguard.rca.candidate_release import export_release, select_batch, verify_release

    built, inventory, reservations, dedup, split_summary, config = _admitted(dataset, predictions, golden_dir)
    built = list(built)
    built[0] = _relabel(built[0], taxonomy="fixture-multi-v1", causes=["C1", "C3"], status="labelled_fault",
                        labels={"causes": ["C1", "C3"]})
    built[1] = _relabel(built[1], taxonomy="fixture-multi-v1", causes=[], status="unlabelled", labels={})
    built[2] = _relabel(built[2], taxonomy="fixture-multi-v1", causes=[], status="ambiguous",
                        labels={"anomaly_flag": ["0", "1"]})
    rows, shortfalls = select_batch(built, config)
    out = tmp_path / "multi"
    manifest = export_release(out, built=built, inventory=list(inventory.values()), reservations=reservations,
                              dedup=dedup, split_summary=split_summary, batch_rows=rows, shortfalls=shortfalls,
                              config=config, rebuild_packet=_rebuild(dataset, predictions, config),
                              prediction_files={}, created_on="2026-11-17")
    verify_release(out)
    references = {r["candidate_id"]: r for r in map(json.loads, (out / "references.jsonl").read_text().splitlines())}
    multi = references[built[0].candidate.identity.candidate_id]
    assert (multi["causes"], multi["taxonomy_id"], multi["source_labels"]) == (
        ["C1", "C3"], "fixture-multi-v1", {"causes": ["C1", "C3"]})  # order and source taxonomy kept
    assert manifest.counts["by_cause_count"]["2"]["rows"] == 1
    states = manifest.counts["by_label_status"]
    assert {"labelled_fault", "labelled_normal", "unlabelled", "ambiguous"} <= set(states)
    # Every empty-cause state stays distinct; none of them is an abstention, which belongs to the target.
    for item in built[1:3]:
        assert references[item.candidate.identity.candidate_id]["causes"] == []
    assert "abstention_reason" not in CandidateReference.model_fields
    with pytest.raises(ValidationError):
        CandidateReference(candidate_id="x", taxonomy_id="t", label_status="abstained")
    with pytest.raises(ValidationError):
        CandidateReference(candidate_id="x", taxonomy_id="t", label_status="labelled_normal",
                           abstention_reason="no_spec_evidence")
    assert "No multiple-cause candidates" not in (out / "gap_report.md").read_text()


def test_verify_refuses_a_candidate_whose_cause_count_differs_from_its_reference(
        dataset, predictions, golden_dir, tmp_path):
    from clauseguard.rca.candidate_release import export_release, select_batch, verify_release

    built, inventory, reservations, dedup, split_summary, config = _admitted(dataset, predictions, golden_dir)
    built = list(built)
    item = _relabel(built[0], taxonomy="fixture-multi-v1", causes=["C1", "C3"], status="labelled_fault",
                    labels={"causes": ["C1", "C3"]})
    wrong = Taxonomy(taxonomy_id="fixture-multi-v1", cause_count=1, label_status="labelled_fault")
    built[0] = item._replace(candidate=item.candidate.model_copy(update={"taxonomy": wrong}))
    rows, shortfalls = select_batch(built, config)
    out = tmp_path / "mismatch"
    export_release(out, built=built, inventory=list(inventory.values()), reservations=reservations, dedup=dedup,
                   split_summary=split_summary, batch_rows=rows, shortfalls=shortfalls, config=config,
                   rebuild_packet=_rebuild(dataset, predictions, config), prediction_files={}, created_on="2026-11-17")
    with pytest.raises(ValueError, match="taxonomy differs"):
        verify_release(out)


def test_batch_mixes_detector_corrections_and_agreements_by_configured_share(dataset, predictions, golden_dir):
    from clauseguard.rca.candidate_release import _pick_events, detector_disagrees

    built, *_ = _admitted(dataset, predictions, golden_dir)
    # Fixture predictions call every fault Antenna Failure: other faults disagree; normals and Antenna agree.
    disagreeing = [b for b in built if detector_disagrees(b)]
    agreeing = [b for b in built if not detector_disagrees(b)]
    assert len(disagreeing) >= 3 and len(agreeing) >= 3
    pool = [_with(item, event=f"event-{i}") for i, item in enumerate([*disagreeing[:3], *agreeing[:3]])]
    for share, corrections in ((0.5, 1), (1.0, 2), (0.0, 0)):
        picked = _pick_events(pool, 2, 42, correction_share=share)
        assert len(picked) == 2 and sum(map(detector_disagrees, picked)) == corrections
    # A short side is filled from the other, still one window per event.
    short = pool[:4]  # three disagreeing, one agreeing
    picked = _pick_events(short, 3, 42, correction_share=0.0)
    assert sum(map(detector_disagrees, picked)) == 2 and len({p.candidate.grouping.event_id for p in picked}) == 3
    same_event = [_with(item, event="shared") for item in pool]
    assert len(_pick_events(same_event, 4, 42, correction_share=0.5)) == 1
