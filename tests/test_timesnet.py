import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("xgboost")
pytest.importorskip("torch")

from clauseguard.baselines import random_forest as rf_runner
from clauseguard.baselines import timesnet as runner
from clauseguard.baselines import xgboost as xgb_runner
from clauseguard.baselines.artifacts import load_run
from clauseguard.cli.timesnet import main
from clauseguard.datasets.release import prepare_release
from clauseguard.datasets.scenarios import ScenarioDataset
from clauseguard.kpi.timesnet import CONFIGURATIONS

FAST = {"max_epochs": 2, "patience": 1, "bootstrap_replicates": 0}


@pytest.fixture
def dataset(release_sources, recording_evidence):
    source = release_sources
    prepare_release(source["scenarios"], source["output"], source["output"], trials=128)
    return ScenarioDataset(source["output"])


@pytest.fixture
def xgboost_reference(dataset, tmp_path):
    rf = tmp_path / "rf-reference"
    rf_runner.run_random_forest(dataset, rf, bootstrap_replicates=0)
    path = tmp_path / "xgboost-reference"
    xgb_runner.run_xgboost(dataset, path, rf_reference=rf, bootstrap_replicates=0)
    return path


def test_four_configurations_three_seeds_dev_only_selection(
        dataset, xgboost_reference, tmp_path, monkeypatch):
    calls = []
    original = runner.split_data

    def observe(data, task, split):
        calls.append(split)
        return original(data, task, split)

    monkeypatch.setattr(runner, "split_data", observe)
    output = tmp_path / "timesnet"
    summary = runner.run_timesnet(dataset, output, xgboost_reference=xgboost_reference, **FAST)
    assert set(calls) == {"train", "dev"}  # test is never loaded
    assert json.loads((output / "selection.json").read_text()) == summary
    for task, result in summary["tasks"].items():
        assert [row["configuration"] for row in result["trials"]] == list(CONFIGURATIONS)
        assert result["selected_configuration"] == max(
            result["trials"], key=lambda row: row["dev_macro_f1"])["configuration"]
        assert [row["seed"] for row in result["selected_seeds"]] == [42, 43, 44]
        for row in result["selected_seeds"]:
            saved = load_run(output / row["run"])
            config = saved["metadata"].configuration
            assert config["optimizer"] == "RAdam" and config["gradient_clip"] == 4.0
            assert config["inference_batch_size"] == 32
            assert config["validation_loss_weighting"] == "unweighted"
            assert config["parameter_count"] == (698_690 if task == "detection" else 731_466)
            assert config["attempted_epochs"] <= FAST["max_epochs"]
            assert 0 <= config["batch_size_1_vs_32_agreement"] <= 1
            assert saved["metadata"].cohort.split == "dev"
            assert (saved["metadata"].transformations["evaluation_status"]
                    == summary["evaluation_status"])
            assert saved["metrics"]["macro_f1"] == row["dev_macro_f1"]
            assert (output / row["run"] / "model.pt").exists()
    with pytest.raises(FileExistsError):
        runner.run_timesnet(dataset, output, xgboost_reference=xgboost_reference, **FAST)


def test_skipping_seed_repeats_leaves_them_unfinished(dataset, xgboost_reference, tmp_path):
    summary = runner.run_timesnet(dataset, tmp_path / "t", xgboost_reference=xgboost_reference,
                                  repeat_seeds=False, tasks=("detection",), **FAST)
    result = summary["tasks"]["detection"]
    assert [row["seed"] for row in result["selected_seeds"]] == [42]
    assert result["seeds_repeated"] is False


def test_cli_reload_reproduces_dev_predictions_on_cpu(
        dataset, xgboost_reference, tmp_path, capsys):
    output = tmp_path / "timesnet"
    train_args = ["train", "--task", "detection", "--release-dir", str(dataset.release_dir),
                  "--xgboost-reference", str(xgboost_reference), "--output-dir", str(output),
                  "--max-epochs", "2", "--patience", "1", "--bootstrap-replicates", "0"]
    assert main(train_args) == 0
    capsys.readouterr()
    summary = json.loads((output / "selection.json").read_text())
    run_dir = output / summary["tasks"]["detection"]["selected_seeds"][0]["run"]
    original = load_run(run_dir)
    replay = tmp_path / "replayed-dev"
    assert main(["predict", "--release-dir", str(dataset.release_dir),
                 "--model-run-dir", str(run_dir), "--output-dir", str(replay),
                 "--split", "dev", "--bootstrap-replicates", "0"]) == 0
    assert load_run(replay)["predictions"] == original["predictions"]
    test_output = tmp_path / "test"
    result = runner.predict_saved_model(dataset, run_dir, test_output, bootstrap_replicates=0)
    assert load_run(test_output)["metadata"].cohort.split == "test"
    assert result["count"] == 13
    assert main(train_args) == 1  # existing output directory is refused


def test_mismatched_xgboost_reference_is_rejected_before_creating_output(
        dataset, xgboost_reference, tmp_path):
    selection = json.loads((xgboost_reference / "selection.json").read_text())
    selection["release_id"] = "someone-else"
    (xgboost_reference / "selection.json").write_text(json.dumps(selection))
    output = tmp_path / "timesnet"
    with pytest.raises(ValueError, match="different release"):
        runner.run_timesnet(dataset, output, xgboost_reference=xgboost_reference, **FAST)
    assert not output.exists()


def _selection(dataset, path):
    """Keep every fault and every other normal window in each split."""
    members = dataset.splits["membership"]
    ids = {sid for sid, row in members.items() if row["detection_target"] == 1}
    for split in ("train", "dev", "test"):
        normals = sorted(sid for sid, row in members.items()
                         if row["split"] == split and row["detection_target"] == 0)
        ids |= set(normals[::2])
    ids = sorted(ids)
    path.write_text(json.dumps({"purpose": "audit_only", "release_id": dataset.release_id,
                                "policy": "fixture_half_normals", "scenario_ids": ids}))
    return path, set(ids)


def test_window_selection_switch_narrows_train_dev_and_test(
        dataset, xgboost_reference, tmp_path):
    path, ids = _selection(dataset, tmp_path / "selection.json")
    output = tmp_path / "reduced"
    summary = runner.run_timesnet(dataset, output, xgboost_reference=xgboost_reference,
                                  tasks=("detection",), repeat_seeds=False, recipe="v2",
                                  window_selection=path, **FAST)
    assert summary["window_selection"] == "fixture_half_normals" and summary["recipe"] == "v2"
    result = summary["tasks"]["detection"]
    assert result["train_windows"] < result["full_train_windows"]
    assert set(json.loads((output / "detection" / "training_ids.json").read_text())) <= ids
    run_dir = output / result["selected_seeds"][0]["run"]
    saved = load_run(run_dir)
    assert set(saved["expected"]) <= ids
    config = saved["metadata"].configuration
    assert config["num_kernels"] == 3 and config["lr_schedule"] == "cosine"
    assert config["selection_loss"] == "dev_prior_corrected_loss"
    assert saved["metadata"].transformations["window_selection"]["policy"] == "fixture_half_normals"
    replay = tmp_path / "replay"
    runner.predict_saved_model(dataset, run_dir, replay, split="dev", window_selection=path,
                               bootstrap_replicates=0)
    assert load_run(replay)["predictions"] == saved["predictions"]
    scored = tmp_path / "test"
    runner.predict_saved_model(dataset, run_dir, scored, window_selection=path,
                               bootstrap_replicates=0)
    assert set(load_run(scored)["expected"]) <= ids
    with pytest.raises(ValueError, match="same --window-selection"):
        runner.predict_saved_model(dataset, run_dir, tmp_path / "x", bootstrap_replicates=0)


def test_window_selection_must_keep_every_fault(dataset, xgboost_reference, tmp_path):
    path, ids = _selection(dataset, tmp_path / "selection.json")
    members = dataset.splits["membership"]
    fault = next(sid for sid in sorted(ids) if members[sid]["detection_target"] == 1)
    data = json.loads(path.read_text())
    data["scenario_ids"].remove(fault)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="every fault"):
        runner.run_timesnet(dataset, tmp_path / "o", xgboost_reference=xgboost_reference,
                            window_selection=path, **FAST)
    with pytest.raises(ValueError, match="not both"):
        runner.run_timesnet(dataset, tmp_path / "o", xgboost_reference=xgboost_reference,
                            window_selection=path, training_selection=path, **FAST)


def test_pooled_head_and_recording_weights_switches_round_trip(
        dataset, xgboost_reference, tmp_path):
    output = tmp_path / "pooled"
    summary = runner.run_timesnet(dataset, output, xgboost_reference=xgboost_reference,
                                  tasks=("detection",), repeat_seeds=False, head="pooled",
                                  recording_weights=True, **FAST)
    assert summary["head"] == "pooled" and summary["recording_weights"] is True
    run_dir = output / summary["tasks"]["detection"]["selected_seeds"][0]["run"]
    saved = load_run(run_dir)
    config = saved["metadata"].configuration
    assert config["head"] == "pooled" and config["recording_weights"] is True
    assert config["parameter_count"] < 698_690
    replay = tmp_path / "replay"
    runner.predict_saved_model(dataset, run_dir, replay, split="dev", bootstrap_replicates=0)
    assert load_run(replay)["predictions"] == saved["predictions"]
    with pytest.raises(ValueError, match="head must be"):
        runner.run_timesnet(dataset, tmp_path / "bad", xgboost_reference=xgboost_reference,
                            head="nope", **FAST)
