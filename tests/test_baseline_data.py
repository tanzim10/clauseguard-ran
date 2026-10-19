import json

import pytest

from clauseguard.baselines.data import (
    ExperimentCohort,
    combined_diagnosis_ids,
    experiment_cohort,
    iter_experiment_batches,
)
from clauseguard.datasets.contracts import NUMERIC_CHANNELS
from clauseguard.datasets.evidence import build_recording_evidence
from clauseguard.datasets.release import prepare_release, project_inputs
from clauseguard.datasets.scenarios import ScenarioDataset


@pytest.fixture
def experiment_release(release_sources, recording_evidence):
    s = release_sources
    prepare_release(s["scenarios"], s["output"], s["output"], trials=128)
    return ScenarioDataset(s["output"])


def test_cohorts_keep_saved_membership_targets_and_stable_identities(experiment_release):
    data = experiment_release
    detection = experiment_cohort(data, split="train", task="detection")
    category = experiment_cohort(data, split="train", task="category")
    assert detection.ready and category.ready
    assert len(detection.scenario_ids) == 13
    assert len(category.scenario_ids) == 10
    assert set(category.scenario_ids) < set(detection.scenario_ids)
    assert set(category.exclusions.values()) == {"not_faulty", "unsupported_category"}
    assert all(data.splits["membership"][sid]["split"] == "train" for sid in detection.scenario_ids)
    assert detection.split_hash == data.manifest["files"]["splits.json"]
    assert detection == ExperimentCohort.model_validate_json(detection.model_dump_json())
    assert detection.cohort_id == experiment_cohort(data, split="train", task="detection").cohort_id
    assert detection.cohort_id != category.cohort_id
    assert detection.input_hash != category.input_hash
    # Release validation already owns unsupported targets; the experiment layer must retain it.
    assert {data.splits["membership"][sid]["category_target"] for sid in category.scenario_ids} == set(range(10))


def test_text_readiness_is_separate_from_kpi_readiness(experiment_release):
    cohort = experiment_cohort(experiment_release, split="dev", task="category", modality="text")
    assert not cohort.ready and not cohort.coverage_ready
    assert not cohort.scenario_ids
    assert "no_approved_text" in cohort.exclusions.values()
    assert cohort.coverage["dev"]["missing_labels"] == list(range(10))


@pytest.mark.parametrize("mode, expected", [("exploratory", 13), ("scored", 0)])
def test_unresolved_releases_remain_unready(release_sources, mode, expected):
    s = release_sources
    build_recording_evidence(s["scenarios"], s["acquired"], s["output"])
    prepare_release(s["scenarios"], s["output"], s["output"], mode=mode, trials=128)
    data = ScenarioDataset(s["output"])
    cohort = experiment_cohort(data, split="test", task="detection")
    assert not cohort.ready and cohort.mode == mode
    assert len(cohort.scenario_ids) == expected
    if mode == "scored":
        assert len(cohort.exclusions) == 39
        assert set(cohort.exclusions.values()) == {"recording_family_unresolved"}
    with pytest.raises(ValueError):
        list(data.iter_examples(split="test", task="detection"))


def test_invalid_selection_and_changed_sources_fail(experiment_release):
    with pytest.raises(ValueError, match="invalid split, task or modality"):
        experiment_cohort(experiment_release, split="train", task="other")
    with experiment_release.scenarios_path.open("a") as stream:
        stream.write(json.dumps({}) + "\n")
    with pytest.raises(ValueError, match="source changed"):
        experiment_cohort(experiment_release, split="train", task="detection")
    with pytest.raises(ValueError, match="source changed"):
        experiment_cohort(experiment_release, split="train", task="detection", modality="text")


def test_batches_preserve_alignment_and_inputs_without_source_metadata(experiment_release):
    cohort = experiment_cohort(experiment_release, split="train", task="detection")
    batches = list(iter_experiment_batches(experiment_release, cohort, batch_size=5))
    assert [len(b.inputs) for b in batches] == [5, 5, 3]
    assert tuple(sid for batch in batches for sid in batch.scenario_ids) == cohort.scenario_ids
    for batch in batches:
        assert batch.cohort_id == cohort.cohort_id
        for sid, target, group, inputs in zip(batch.scenario_ids, batch.targets,
                                               batch.group_ids, batch.inputs, strict=True):
            member = experiment_release.splits["membership"][sid]
            assert target == member["detection_target"] and group == member["group_id"]
            assert inputs.kpi.channel_names == list(NUMERIC_CHANNELS)
            assert len(inputs.kpi.values) == 128
            assert all(len(row) == 16 for row in inputs.kpi.values)
            assert set(inputs.model_dump(exclude_none=True)) == {"kpi"}
            assert set(inputs.kpi.model_dump()) == {"channel_names", "sampling_rate_hz", "values"}
            assert "QnA" not in inputs.model_dump_json()
    changed = cohort.model_copy(update={"input_hash": "wrong"})
    with pytest.raises(ValueError, match="cohort differs"):
        list(iter_experiment_batches(experiment_release, changed))
    for size in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            list(iter_experiment_batches(experiment_release, cohort, batch_size=size))


def test_combined_diagnosis_membership_excludes_jamming(experiment_release):
    ids = combined_diagnosis_ids(experiment_release, split="test")
    assert len(ids) == 12  # Two normal windows plus ten category windows.
    assert all(experiment_release.splits["membership"][sid]["rca_label"] != "Jamming" for sid in ids)
    labels = {experiment_release.splits["membership"][sid]["category_target"] for sid in ids}
    assert labels == {None, *range(10)}


def test_shuffled_channels_and_native_values_survive_batches(release_sources):
    s = release_sources
    records = [json.loads(line) for line in s["scenarios"].read_text().splitlines()]
    for record in records:
        channels = record["kpi_window"]["channels"]
        record["kpi_window"]["channels"] = dict(reversed(list(channels.items())))
    s["scenarios"].write_text("".join(json.dumps(row) + "\n" for row in records))
    build_recording_evidence(s["scenarios"], s["acquired"], s["output"], reviews=s["reviews"])
    prepare_release(s["scenarios"], s["output"], s["output"], trials=128)
    data = ScenarioDataset(s["output"])
    cohort = experiment_cohort(data, split="train", task="detection")
    batch = next(iter_experiment_batches(data, cohort))
    record = data.load_scenario(batch.scenario_ids[0])
    assert batch.inputs[0].kpi.values[0] == [
        record.kpi_window.channels[name].values[0] for name in NUMERIC_CHANNELS
    ]
    # A fixture source with a missing measurement must match a missing raw sample too.
    # The projection itself never imputes or rescales it.
    record.kpi_window.channels[NUMERIC_CHANNELS[0]].values[0] = None
    projected = project_inputs(record)
    assert projected.kpi.values[0][0] is None
    assert projected.kpi.values[0][1] == record.kpi_window.channels[NUMERIC_CHANNELS[1]].values[0]


def test_batch_iteration_preserves_recording_and_text_readiness_gates(release_sources):
    s = release_sources
    build_recording_evidence(s["scenarios"], s["acquired"], s["output"])
    prepare_release(s["scenarios"], s["output"], s["output"], mode="exploratory", trials=128)
    data = ScenarioDataset(s["output"])
    cohort = experiment_cohort(data, split="train", task="detection")
    with pytest.raises(ValueError, match="explicit allow_exploratory"):
        list(iter_experiment_batches(data, cohort))
    inspection = ScenarioDataset(s["output"], allow_exploratory=True)
    assert sum(len(b.inputs) for b in iter_experiment_batches(inspection, cohort)) == 13
    assert not cohort.ready
    text = experiment_cohort(inspection, split="train", task="detection", modality="text")
    with pytest.raises(ValueError, match="not ready"):
        list(iter_experiment_batches(inspection, text))


@pytest.mark.parametrize("modality", ["text", "paired"])
def test_approved_observation_layers_share_batches_and_membership(release_sources, modality):
    s = release_sources
    records = [json.loads(line) for line in s["scenarios"].read_text().splitlines()]
    for record in records:
        record["nl_context"] = "Error measurements rose while receive bytes fell."
        record["text_review"] = {
            "status": "approved", "source": "fixture observations", "reviewer": "fixture reviewer",
            "reviewed_at": "2026-10-18", "approved_tasks": ["detection", "category"],
            "text_origin": "kpi_derived",
        }
    s["scenarios"].write_text("".join(json.dumps(row) + "\n" for row in records))
    build_recording_evidence(s["scenarios"], s["acquired"], s["output"], reviews=s["reviews"])
    prepare_release(s["scenarios"], s["output"], s["output"], trials=128)
    data = ScenarioDataset(s["output"])
    cohort = experiment_cohort(data, split="test", task="category", modality=modality)
    kpi = experiment_cohort(data, split="test", task="category")
    assert cohort.scenario_ids == kpi.scenario_ids
    assert cohort.input_hash != kpi.input_hash
    batches = list(iter_experiment_batches(data, cohort))
    assert len(batches[0].inputs) == 10
    assert set(batches[0].targets) == set(range(10))
    assert len(combined_diagnosis_ids(data, split="test", modality=modality)) == 12
    for inputs in batches[0].inputs:
        assert inputs.text == "Error measurements rose while receive bytes fell."
        assert (inputs.kpi is not None) == (modality == "paired")
        assert "text_origin" not in inputs.model_dump_json()
        assert "reference" not in inputs.model_dump_json()
