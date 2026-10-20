"""XGBoost experiments must preserve the RF comparison and held-out boundary."""

import hashlib
import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("xgboost")

from clauseguard.baselines import random_forest as rf_runner
from clauseguard.baselines import xgboost as runner
from clauseguard.baselines.artifacts import load_run
from clauseguard.cli.xgboost import main
from clauseguard.datasets.release import prepare_release
from clauseguard.datasets.scenarios import ScenarioDataset
from clauseguard.kpi.features import ChannelMedianFeatures
from clauseguard.kpi.xgboost import CONFIGURATIONS, load_model


@pytest.fixture
def dataset(release_sources, recording_evidence):
    source = release_sources
    prepare_release(source["scenarios"], source["output"], source["output"], trials=128)
    return ScenarioDataset(source["output"])


@pytest.fixture
def rf_reference(dataset, tmp_path):
    path = tmp_path / "rf-reference"
    rf_runner.run_random_forest(dataset, path, bootstrap_replicates=0)
    return path


def test_four_configurations_three_seeds_and_dev_only_selection(
        dataset, rf_reference, tmp_path, monkeypatch):
    calls = []
    original = runner.split_data

    def observe(data, task, split):
        calls.append(split)
        return original(data, task, split)

    monkeypatch.setattr(runner, "split_data", observe)
    output = tmp_path / "xgb"
    summary = runner.run_xgboost(dataset, output, rf_reference=rf_reference,
                                 bootstrap_replicates=0)
    assert set(calls) == {"train", "dev"}
    assert set(summary["tasks"]) == {"detection", "category"}
    assert summary["selection_split"] == "dev"
    assert json.loads((output / "selection.json").read_text()) == summary
    for task, result in summary["tasks"].items():
        assert [row["configuration"] for row in result["trials"]] == list(CONFIGURATIONS)
        assert result["selected_configuration"] == max(
            result["trials"], key=lambda row: row["dev_macro_f1"])["configuration"]
        assert [row["seed"] for row in result["selected_seeds"]] == [42, 43, 44]
        assert result["train_windows"] == (13 if task == "detection" else 10)
        for row in result["selected_seeds"]:
            path = output / row["run"]
            saved = load_run(path)
            metadata = saved["metadata"]
            config = metadata.configuration
            assert config["objective"] == (
                "binary:logistic" if task == "detection" else "multi:softprob")
            assert config["eval_metric"] == ("logloss" if task == "detection" else "mlogloss")
            assert config["n_jobs"] == 1
            assert config["sample_weight_policy"] == "balanced_training_n_over_k_nclass"
            assert config["validation_loss_weighting"] == "unweighted"
            assert len(metadata.transformations["feature_order"]) == 224
            assert metadata.cohort.split == "dev"
            assert metadata.resources["fit_seconds"] > 0
            assert saved["metrics"]["macro_f1"] == row["dev_macro_f1"]
            transformer = ChannelMedianFeatures()
            transformer.medians_ = np.array(metadata.transformations["channel_medians"])
            model = load_model(path / "model.ubj", expected_sha256=metadata.checkpoint,
                               task=task, features=transformer,
                               configuration=result["selected_configuration"], seed=row["seed"])
            assert model.booster.best_iteration < 500
            assert model.booster.get_booster().num_boosted_rounds() == model.booster.best_iteration + 1
            assert (path / "errors.json").exists()
    with pytest.raises(FileExistsError):
        runner.run_xgboost(dataset, output, rf_reference=rf_reference, bootstrap_replicates=0)


def test_cli_reload_predicts_without_refitting_and_preserves_dev_predictions(
        dataset, rf_reference, tmp_path, monkeypatch, capsys):
    output = tmp_path / "xgb"
    train_args = ["train", "--task", "detection", "--release-dir", str(dataset.release_dir),
                  "--rf-reference", str(rf_reference), "--output-dir", str(output),
                  "--bootstrap-replicates", "0"]
    assert main(train_args) == 0
    capsys.readouterr()
    summary = json.loads((output / "selection.json").read_text())
    run_dir = output / summary["tasks"]["detection"]["selected_seeds"][0]["run"]
    original = load_run(run_dir)

    def no_fit(*args, **kwargs):
        raise AssertionError("held-out prediction must not refit preprocessing")

    monkeypatch.setattr(ChannelMedianFeatures, "fit", no_fit)
    replay = tmp_path / "replayed-dev"
    assert main(["predict", "--release-dir", str(dataset.release_dir),
                 "--model-run-dir", str(run_dir), "--output-dir", str(replay),
                 "--split", "dev", "--bootstrap-replicates", "0"]) == 0
    restored = load_run(replay)
    assert restored["predictions"] == original["predictions"]
    assert restored["metadata"].transformations == original["metadata"].transformations
    test_output = tmp_path / "test"
    result = runner.predict_saved_model(dataset, run_dir, test_output, bootstrap_replicates=0)
    assert result["count"] == 13
    assert load_run(test_output)["metadata"].cohort.split == "test"
    assert main(train_args) == 1


def test_changed_training_selection_is_rejected_before_creating_output(
        dataset, rf_reference, tmp_path):
    members = dataset.splits["membership"]
    excluded = next(sid for sid, row in members.items()
                    if row["split"] == "train" and row["detection_target"] == 0)
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"release_id": dataset.release_id,
                                     "scenario_ids": [sid for sid in members if sid != excluded]}))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        runner.run_xgboost(dataset, output, rf_reference=rf_reference, tasks=("detection",),
                          training_selection=selection, bootstrap_replicates=0)
    assert not output.exists()


@pytest.mark.parametrize("corruption", ["release", "feature_order", "channel_medians"])
def test_incompatible_rf_reference_is_rejected_before_creating_output(
        dataset, rf_reference, tmp_path, corruption):
    selection_path = rf_reference / "selection.json"
    selection = json.loads(selection_path.read_text())
    if corruption == "release":
        selection["release_id"] = "different-release"
        selection_path.write_text(json.dumps(selection))
    else:
        path = rf_reference / selection["tasks"]["detection"]["selected_seeds"][0]["run"]
        run_path = path / "run.json"
        metadata = json.loads(run_path.read_text())
        if corruption == "feature_order":
            metadata["transformations"][corruption].reverse()
        else:
            metadata["transformations"][corruption][0] += 123
        run_path.write_text(json.dumps(metadata))
        integrity_path = path / "integrity.json"
        integrity = json.loads(integrity_path.read_text())
        integrity["run.json"] = hashlib.sha256(run_path.read_bytes()).hexdigest()
        integrity_path.write_text(json.dumps(integrity))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        runner.run_xgboost(dataset, output, rf_reference=rf_reference,
                          tasks=("detection",), bootstrap_replicates=0)
    assert not output.exists()


def test_exploratory_release_requires_explicit_opt_in(
        release_sources, recording_evidence, tmp_path):
    source = release_sources
    prepare_release(source["scenarios"], source["output"], source["output"],
                    mode="exploratory", trials=128)
    dataset = ScenarioDataset(source["output"], allow_exploratory=True)
    reference = tmp_path / "exploratory-rf"
    rf_runner.run_random_forest(dataset, reference, tasks=("detection",),
                               allow_exploratory=True, bootstrap_replicates=0)
    blocked = tmp_path / "blocked"
    with pytest.raises(ValueError, match="scored release|explicit"):
        runner.run_xgboost(dataset, blocked, rf_reference=reference,
                          tasks=("detection",), bootstrap_replicates=0)
    assert not blocked.exists()
    output = tmp_path / "exploratory-xgb"
    summary = runner.run_xgboost(dataset, output, rf_reference=reference,
                                 tasks=("detection",), allow_exploratory=True,
                                 bootstrap_replicates=0)
    assert summary["evaluation_status"] == "provisional_exploratory"
    run_dir = output / summary["tasks"]["detection"]["selected_seeds"][0]["run"]
    assert load_run(run_dir)["metadata"].transformations["evaluation_status"] == "provisional_exploratory"
    with pytest.raises(ValueError, match="scored release|explicit"):
        runner.predict_saved_model(dataset, run_dir, tmp_path / "blocked-predict",
                                   bootstrap_replicates=0)
