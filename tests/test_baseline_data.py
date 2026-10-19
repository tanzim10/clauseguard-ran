import json

import pytest

from clauseguard.baselines.data import ExperimentCohort, experiment_cohort
from clauseguard.datasets.evidence import build_recording_evidence
from clauseguard.datasets.release import prepare_release
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
