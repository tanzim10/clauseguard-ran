import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from clauseguard.baselines import random_forest as runner
from clauseguard.baselines.artifacts import load_run
from clauseguard.cli.random_forest import main
from clauseguard.datasets.release import prepare_release
from clauseguard.datasets.scenarios import ScenarioDataset
from clauseguard.kpi.features import ChannelMedianFeatures
from clauseguard.kpi.random_forest import CONFIGURATIONS, RandomForestModel, load_model


@pytest.fixture
def dataset(release_sources, recording_evidence):
    source = release_sources
    prepare_release(source["scenarios"], source["output"], source["output"], trials=128)
    return ScenarioDataset(source["output"])


def test_both_tasks_tune_only_on_dev_save_three_seeds_and_replay(dataset, tmp_path, monkeypatch):
    calls = []
    original = runner._split_data

    def observe_split(data, task, split):
        calls.append(split)
        return original(data, task, split)

    monkeypatch.setattr(runner, "_split_data", observe_split)
    output = tmp_path / "experiment"
    summary = runner.run_random_forest(dataset, output, bootstrap_replicates=0)
    assert set(calls) == {"train", "dev"}
    assert set(summary["tasks"]) == {"detection", "category"}
    assert json.loads((output / "selection.json").read_text()) == summary
    for task, result in summary["tasks"].items():
        assert [row["configuration"] for row in result["trials"]] == list(CONFIGURATIONS)
        assert [row["seed"] for row in result["selected_seeds"]] == [42, 43, 44]
        assert result["selected_configuration"] == max(
            result["trials"], key=lambda row: row["dev_macro_f1"])["configuration"]
        for row in result["selected_seeds"]:
            path = output / row["run"]
            saved = load_run(path)
            metadata = saved["metadata"]
            model = load_model(path / "model.joblib", expected_sha256=metadata.checkpoint)
            assert model.task == task
            assert model.seed == row["seed"]
            assert model.forest.n_estimators == 300
            assert model.forest.class_weight == "balanced"
            assert len(metadata.transformations["feature_order"]) == 224
            assert metadata.resources["fit_seconds"] > 0
            assert saved["metrics"]["macro_f1"] == row["dev_macro_f1"]
            assert metadata.cohort.split == "dev"
            assert (path / "errors.json").exists()
        assert result["train_windows"] == (13 if task == "detection" else 10)
    with pytest.raises(FileExistsError):
        runner.run_random_forest(dataset, output, bootstrap_replicates=0)


def test_cli_train_and_prediction_reload_without_refitting(dataset, tmp_path, monkeypatch, capsys):
    output = tmp_path / "train"
    args = ["train", "--task", "detection", "--release-dir", str(dataset.release_dir),
            "--output-dir", str(output), "--bootstrap-replicates", "0"]
    assert main(args) == 0
    capsys.readouterr()
    summary = json.loads((output / "selection.json").read_text())
    run_dir = output / summary["tasks"]["detection"]["selected_seeds"][0]["run"]
    original = load_run(run_dir)["metadata"]

    def no_fit(*args, **kwargs):
        raise AssertionError("prediction must not fit transforms")

    monkeypatch.setattr(ChannelMedianFeatures, "fit", no_fit)
    prediction_dir = tmp_path / "test"
    assert main(["predict", "--release-dir", str(dataset.release_dir),
                 "--model-run-dir", str(run_dir), "--output-dir", str(prediction_dir),
                 "--bootstrap-replicates", "0"]) == 0
    saved = load_run(prediction_dir)
    assert saved["metadata"].cohort.split == "test"
    assert saved["metadata"].transformations == original.transformations
    assert saved["metrics"]["count"] == 13
    assert len(saved["predictions"]) == 13
    assert main(args) == 1

    def failed_predict(*args, **kwargs):
        raise RuntimeError("fixture prediction failure")

    monkeypatch.setattr(RandomForestModel, "predict", failed_predict)
    failed_dir = tmp_path / "failed-test"
    metrics = runner.predict_saved_model(dataset, run_dir, failed_dir, bootstrap_replicates=0)
    assert metrics["accuracy"] == 0
    assert metrics["prediction_status_counts"]["error"] == 13
    assert len(load_run(failed_dir)["predictions"]) == 13
    checkpoint = run_dir / "model.joblib"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_model(checkpoint, expected_sha256=original.checkpoint)


def test_selection_changes_training_only_and_requires_release_identity(dataset, tmp_path):
    members = dataset.splits["membership"]
    removed = next(sid for sid, row in members.items()
                   if row["split"] == "train" and row["detection_target"] == 0)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"release_id": "wrong", "scenario_ids": list(members)}))
    output = tmp_path / "selected"
    with pytest.raises(ValueError, match="different release"):
        runner.run_random_forest(dataset, output, tasks=("detection",), training_selection=path,
                                 bootstrap_replicates=0)
    assert not output.exists()
    removed_fault = next(sid for sid, row in members.items()
                         if row["split"] == "train" and row["detection_target"] == 1)
    path.write_text(json.dumps({"release_id": dataset.release_id,
                               "scenario_ids": [sid for sid in members if sid != removed_fault]}))
    with pytest.raises(ValueError, match="retain every training fault"):
        runner.run_random_forest(dataset, output, tasks=("detection",), training_selection=path,
                                 bootstrap_replicates=0)
    assert not output.exists()
    path.write_text(json.dumps({"release_id": dataset.release_id,
                                "scenario_ids": [sid for sid in members if sid != removed]}))
    summary = runner.run_random_forest(dataset, output, tasks=("detection",),
                                       training_selection=path, bootstrap_replicates=0)
    result = summary["tasks"]["detection"]
    assert result["train_windows"] == 12 and result["full_train_windows"] == 13
    saved = load_run(output / result["selected_seeds"][0]["run"])
    assert saved["metrics"]["count"] == 13  # Complete dev cohort, regardless of training IDs.


def test_exploratory_release_is_blocked_before_any_training(
        release_sources, recording_evidence, tmp_path, capsys):
    source = release_sources
    prepare_release(source["scenarios"], source["output"], source["output"],
                    mode="exploratory", trials=128)
    output = tmp_path / "blocked"
    assert main(["train", "--release-dir", str(source["output"]),
                 "--output-dir", str(output)]) == 1
    assert "requires a scored release" in capsys.readouterr().err
    assert not output.exists()
