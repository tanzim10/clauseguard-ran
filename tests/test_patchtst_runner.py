import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("xgboost")
pytest.importorskip("torch")

from clauseguard.baselines import patchtst as runner
from clauseguard.baselines import random_forest as rf_runner
from clauseguard.baselines import xgboost as xgb_runner
from clauseguard.baselines.artifacts import load_run
from clauseguard.cli.patchtst import main
from clauseguard.datasets.release import prepare_release
from clauseguard.datasets.scenarios import ScenarioDataset
from clauseguard.kpi.patchtst import CONFIGURATIONS

FAST = {"max_epochs": 2, "patience": 1, "bootstrap_replicates": 0}


@pytest.fixture
def dataset(release_sources, recording_evidence):
    source = release_sources
    prepare_release(source["scenarios"], source["output"], source["output"], trials=128)
    return ScenarioDataset(source["output"])


@pytest.fixture
def references(dataset, tmp_path):
    rf = tmp_path / "rf"
    rf_runner.run_random_forest(dataset, rf, bootstrap_replicates=0)
    xgb = tmp_path / "xgboost"
    xgb_runner.run_xgboost(dataset, xgb, rf_reference=rf, bootstrap_replicates=0)
    return rf, xgb


def test_four_settings_three_seeds_dev_only_and_reload(dataset, references, tmp_path, monkeypatch):
    splits, fitted = [], []
    original_split, original_fit = runner.split_data, runner.WindowScaler.fit

    def observe_split(data, task, split):
        splits.append(split)
        return original_split(data, task, split)

    def observe_fit(self, inputs):
        fitted.append(len(inputs))
        return original_fit(self, inputs)

    monkeypatch.setattr(runner, "split_data", observe_split)
    monkeypatch.setattr(runner.WindowScaler, "fit", observe_fit)
    output = tmp_path / "patchtst"
    summary = runner.run_patchtst(dataset, output, xgboost_reference=references[1], **FAST)
    assert set(splits) == {"train", "dev"}  # test is never loaded
    assert json.loads((output / "selection.json").read_text()) == summary
    for result in summary["tasks"].values():
        assert fitted.pop(0) == result["train_windows"]  # scaling statistics come from train only
        assert [row["configuration"] for row in result["trials"]] == list(CONFIGURATIONS)
        assert result["selected_configuration"] == max(
            result["trials"], key=lambda row: row["dev_macro_f1"])["configuration"]
        assert [row["seed"] for row in result["selected_seeds"]] == [42, 43, 44]
        for row in result["selected_seeds"]:
            saved = load_run(output / row["run"])
            config = saved["metadata"].configuration
            assert config["optimizer"] == "AdamW" and config["weight_decay"] == 1e-4
            assert config["gradient_clip"] == 1.0 and config["norm"] == "LayerNorm"
            assert config["instance_norm"] is False and config["deterministic_algorithms"]
            assert config["reload_max_abs_logit_diff"] <= 1e-6
            assert saved["metadata"].cohort.split == "dev"
            assert (output / row["run"] / "model.pt").exists()

    run_dir = output / summary["tasks"]["detection"]["selected_seeds"][0]["run"]
    replay = tmp_path / "replay"
    runner.predict_saved_model(dataset, run_dir, replay, split="dev", bootstrap_replicates=0)
    original = [(p.scenario_id, p.predicted_label) for p in load_run(run_dir)["predictions"]]
    assert [(p.scenario_id, p.predicted_label) for p in load_run(replay)["predictions"]] == original


def test_comparison_and_cli(dataset, references, tmp_path, capsys):
    rf, xgb = references
    output = tmp_path / "patchtst"
    code = main(["train", "--release-dir", str(dataset.release_dir), "--output-dir", str(output),
                 "--xgboost-reference", str(xgb), "--task", "detection", "--max-epochs", "1",
                 "--patience", "1", "--bootstrap-replicates", "0"])
    assert code == 0
    capsys.readouterr()
    code = main(["compare", "--experiment", f"random_forest={rf}", "--experiment", f"xgboost={xgb}",
                 "--experiment", f"patchtst={output}", "--output-dir", str(tmp_path / "cmp")])
    assert code == 1  # category has no PatchTST run, so the comparison refuses
    assert "no category selection" in capsys.readouterr().err
    reports = runner.compare_with_baselines(
        {"random_forest": rf, "xgboost": xgb, "patchtst": output}, tmp_path / "cmp2",
        tasks=("detection",))
    assert set(reports[0]["models"]) == {"random_forest", "xgboost", "patchtst"}
    assert "patchtst" in (tmp_path / "cmp2" / "comparison.md").read_text()
    assert main(["train", "--release-dir", str(dataset.release_dir), "--output-dir", str(output),
                 "--xgboost-reference", str(xgb)]) == 1
    assert "output exists" in capsys.readouterr().err
