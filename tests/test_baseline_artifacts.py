import pytest

from clauseguard.baselines.artifacts import compare_runs, load_run, write_experiment_run, write_run
from clauseguard.baselines.contracts import CasePrediction, RunMetadata
from clauseguard.baselines.data import experiment_cohort
from clauseguard.cli.baseline_runs import main as run_cli
from clauseguard.datasets.release import prepare_release
from clauseguard.datasets.scenarios import ScenarioDataset


@pytest.fixture
def experiment_release(release_sources, recording_evidence):
    source = release_sources
    prepare_release(source["scenarios"], source["output"], source["output"], trials=128)
    return ScenarioDataset(source["output"])


def test_saved_run_replays_metrics_and_refuses_overwrite(experiment_release, tmp_path):
    cohort = experiment_cohort(experiment_release, split="train", task="detection")
    expected = {sid: experiment_release.splits["membership"][sid]["detection_target"]
                for sid in cohort.scenario_ids}
    groups = {sid: experiment_release.splits["membership"][sid]["group_id"]
              for sid in cohort.scenario_ids}
    predictions = [CasePrediction(scenario_id=sid, task="detection", predicted_label=target)
                   for sid, target in expected.items()]
    path = tmp_path / "rf-run"
    metadata = RunMetadata(model="rf", task="detection", cohort=cohort, seed=42,
                           versions={"python": "test"}, configuration={"trees": 100})

    written = write_experiment_run(path, experiment_release, metadata, predictions,
                                   bootstrap_replicates=10)
    loaded = load_run(path)

    assert written == loaded["metrics"]
    assert loaded["metadata"].configuration == {"trees": 100}
    assert loaded["expected"] == expected
    assert loaded["groups"] == groups
    assert loaded["metrics"]["accuracy"] == 1
    with pytest.raises(FileExistsError):
        write_run(path, metadata, expected, groups, predictions)


def test_run_rejects_corruption_and_comparison_requires_same_cohort(experiment_release, tmp_path,
                                                                   capsys):
    cohort = experiment_cohort(experiment_release, split="test", task="category")
    expected = {sid: experiment_release.splits["membership"][sid]["category_target"]
                for sid in cohort.scenario_ids}
    groups = {sid: experiment_release.splits["membership"][sid]["group_id"]
              for sid in cohort.scenario_ids}
    predictions = [CasePrediction(scenario_id=sid, task="category", predicted_label=label)
                   for sid, label in expected.items()]
    paths = []
    for model in ("rf", "xgb"):
        path = tmp_path / model
        write_run(path, RunMetadata(model=model, task="category", cohort=cohort),
                  expected, groups, predictions, bootstrap_replicates=0)
        paths.append(path)
    comparison = compare_runs(paths)
    assert comparison["cohort_id"] == cohort.cohort_id
    assert set(comparison["runs"]) == {"rf", "xgb"}
    assert run_cli(["score", str(paths[0])]) == 0
    assert '"accuracy": 1.0' in capsys.readouterr().out
    assert run_cli(["compare", *(str(path) for path in paths)]) == 0
    assert '"cohort_id"' in capsys.readouterr().out

    changed = cohort.model_copy(update={"input_hash": "different"})
    third = tmp_path / "different"
    write_run(third, RunMetadata(model="other", task="category", cohort=changed),
              expected, groups, predictions, bootstrap_replicates=0)
    with pytest.raises(ValueError, match="different task or cohort"):
        compare_runs([paths[0], third])

    target = paths[0] / "predictions.jsonl"
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        load_run(paths[0])


def test_run_cli_reports_invalid_artifact(tmp_path, capsys):
    assert run_cli(["score", str(tmp_path / "missing")]) == 1
    assert "baseline score failed" in capsys.readouterr().err
