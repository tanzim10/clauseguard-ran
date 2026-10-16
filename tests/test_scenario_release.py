import json
from copy import deepcopy

import pytest

from clauseguard.datasets.evidence import build_recording_evidence
from clauseguard.datasets.release import _identity, prepare_release, project_inputs
from clauseguard.datasets.scenarios import ScenarioDataset
from clauseguard.datasets.splits import assign_splits, validate_membership
from clauseguard.datasets.telecomts import file_sha256


def test_full_release_loads_both_targets_without_leaking_answers(release_sources, recording_evidence):
    s = release_sources
    result = prepare_release(s["scenarios"], s["output"], s["output"], trials=256)
    dataset = ScenarioDataset(s["output"], expected_release_id=result["release_id"])
    detection = list(dataset.iter_examples(split="train", task="detection"))
    category = list(dataset.iter_examples(split="train", task="category"))
    assert {r.target for r in detection} == {0, 1}
    assert {r.target for r in category} == set(range(10))
    assert {r.scenario_id for r in category} < {r.scenario_id for r in detection}
    for example in detection:
        payload = example.inputs.model_dump(exclude_none=True)
        assert set(payload) == {"kpi"}
        assert set(payload["kpi"]) == {"channel_names", "sampling_rate_hz", "values"}
        assert len(payload["kpi"]["values"]) == 128
        assert all(len(row) == 16 for row in payload["kpi"]["values"])
        assert "jammer" not in json.dumps(payload)
        assert "QnA" not in json.dumps(payload)
    assert (s["output"] / "inputs.text.jsonl").read_text() == ""
    with pytest.raises(KeyError, match="unknown scenario"):
        dataset.load_scenario("missing")
    with pytest.raises(ValueError, match="invalid split"):
        list(dataset.iter_examples(split="excluded", task="detection"))
    with pytest.raises(ValueError, match="text is not approved"):
        project_inputs(dataset.load_scenario(detection[0].scenario_id), modality="text", task="detection")
    assert dataset.load_scenario(detection[0].scenario_id).reference["QnA"]


def test_exploratory_requires_opt_in_and_never_claims_recording_readiness(release_sources):
    s = release_sources
    build_recording_evidence(s["scenarios"], s["acquired"], s["output"])
    result = prepare_release(s["scenarios"], s["output"], s["output"], mode="exploratory", trials=256)
    assert not result["quality"]["tasks"]["detection"]["kpi"]["ready"]
    dataset = ScenarioDataset(s["output"])
    with pytest.raises(ValueError, match="explicit allow_exploratory"):
        list(dataset.iter_examples(split="train", task="detection"))
    opted_in = ScenarioDataset(s["output"], allow_exploratory=True)
    assert list(opted_in.iter_examples(split="test", task="detection"))


def test_corrupt_release_and_changed_source_are_rejected(release_sources, recording_evidence):
    s = release_sources
    prepare_release(s["scenarios"], s["output"], s["output"], trials=128)
    dataset = ScenarioDataset(s["output"])
    with s["scenarios"].open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="source changed"):
        dataset.load_scenario(next(iter(dataset.offsets)))
    with pytest.raises(ValueError, match="scenario checksum"):
        ScenarioDataset(s["output"])


def test_membership_missing_id_crossing_family_or_wrong_target_fails(recording_evidence):
    original = assign_splits(recording_evidence, trials=128)
    bad = deepcopy(original)
    bad["membership"].pop(next(iter(bad["membership"])))
    with pytest.raises(ValueError, match="membership"):
        validate_membership(bad, recording_evidence)
    bad = deepcopy(original)
    first = next(iter(bad["membership"].values()))
    first["split"] = "test" if first["split"] == "train" else "train"
    with pytest.raises(ValueError, match="crosses splits"):
        validate_membership(bad, recording_evidence)
    bad = deepcopy(original)
    next(iter(bad["membership"].values()))["detection_target"] = 99
    with pytest.raises(ValueError, match="incorrect target"):
        validate_membership(bad, recording_evidence)


def test_artifact_hashes_and_expected_release_are_enforced(release_sources, recording_evidence):
    s = release_sources
    prepare_release(s["scenarios"], s["output"], s["output"], trials=128)
    with pytest.raises(ValueError, match="unexpected release"):
        ScenarioDataset(s["output"], expected_release_id="different")
    (s["output"] / "membership.csv").write_text("corrupted")
    with pytest.raises(ValueError, match="checksum.*membership.csv"):
        ScenarioDataset(s["output"])


def test_malformed_record_has_file_and_line_even_if_manifest_resealed(release_sources, recording_evidence):
    s = release_sources
    prepare_release(s["scenarios"], s["output"], s["output"], trials=128)
    s["scenarios"].write_text("not json\n")
    manifest_path = s["output"] / "release.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("release_id")
    manifest["scenarios_sha256"] = file_sha256(s["scenarios"])
    # Hash identity mismatch with splits is detected even before reading malformed records.
    manifest["release_id"] = _identity(manifest)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identities differ"):
        ScenarioDataset(s["output"])


def test_approved_text_and_paired_views_inherit_membership(release_sources):
    s = release_sources
    records = [json.loads(line) for line in s["scenarios"].read_text().splitlines()]
    for record in records:
        record["nl_context"] = "Uplink error measurements rose while receive bytes fell."
        record["text_review"] = {
            "status": "approved", "source": "reviewed measurements", "reviewer": "fixture human",
            "reviewed_at": "2026-10-15", "approved_tasks": ["detection", "category"],
            "text_origin": "kpi_derived",
        }
    s["scenarios"].write_text("".join(json.dumps(r) + "\n" for r in records))
    build_recording_evidence(s["scenarios"], s["acquired"], s["output"], reviews=s["reviews"])
    prepare_release(s["scenarios"], s["output"], s["output"], trials=128)
    dataset = ScenarioDataset(s["output"])
    kpi = list(dataset.iter_examples(split="train", task="detection"))
    text = list(dataset.iter_examples(split="train", task="detection", modality="text"))
    paired = list(dataset.iter_examples(split="train", task="category", modality="paired"))
    assert {r.scenario_id for r in text} == {r.scenario_id for r in kpi}
    assert all(r.inputs.kpi is None and r.inputs.text for r in text)
    assert len(paired) == 10
    assert all(r.inputs.kpi and r.inputs.text for r in paired)
    assert all(r.split == "train" for r in paired)


@pytest.mark.parametrize("origin, text, tasks", [
    ("diagnosis_derived", "Error rates rose.", ["detection"]),
    ("independent_observation", "There is a jammer.", ["detection"]),
    ("kpi_derived", "Error rates rose.", ["category"]),
])
def test_text_approval_is_target_and_provenance_specific(release_sources, origin, text, tasks):
    from clauseguard.datasets.evidence import text_eligibility
    from clauseguard.datasets.schema import ScenarioRecord

    row = json.loads(release_sources["scenarios"].open().readline())
    row["nl_context"] = text
    row["text_review"] = {
        "status": "approved", "source": "review", "reviewer": "human", "reviewed_at": "2026-10-15",
        "text_origin": origin, "approved_tasks": tasks,
    }
    record = ScenarioRecord.model_validate(row)
    assert text_eligibility(record, "detection")
    with pytest.raises(ValueError, match="not approved"):
        project_inputs(record, modality="text", task="detection")
