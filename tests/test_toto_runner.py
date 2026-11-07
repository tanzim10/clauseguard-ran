import json

import numpy as np
import pytest

from clauseguard.baselines import toto as runner
from clauseguard.baselines.artifacts import load_run, write_run
from clauseguard.baselines.contracts import CasePrediction, RunMetadata
from clauseguard.baselines.data import experiment_cohort, experiment_targets
from clauseguard.cli import toto as cli
from clauseguard.datasets.release import prepare_release
from clauseguard.datasets.scenarios import ScenarioDataset

torch = pytest.importorskip("torch")
from test_toto import FakeToto


@pytest.fixture
def dataset(release_sources, recording_evidence):
    s = release_sources
    prepare_release(s["scenarios"], s["output"], s["output"], trials=128)
    return ScenarioDataset(s["output"])


@pytest.fixture
def xgboost_reference(dataset, tmp_path):
    root = tmp_path / "xgboost"
    cohort = experiment_cohort(dataset, split="dev", task="detection")
    expected, groups = experiment_targets(dataset, cohort)
    write_run(root / "detection" / "seed-42",
              RunMetadata(model="xgboost/detection/seed-42", task="detection", cohort=cohort, seed=42),
              expected, groups, [CasePrediction(scenario_id=sid, task="detection", predicted_label=1)
                                 for sid in expected], bootstrap_replicates=0)
    (root / "selection.json").write_text(json.dumps({
        "release_id": dataset.release_id,
        "tasks": {"detection": {"selected_seeds": [{"seed": 42, "run": "detection/seed-42"}]}}}))
    return root


def test_dev_selection_saves_replayable_detectors_and_comparisons(dataset, xgboost_reference,
                                                                   tmp_path, monkeypatch):
    model = FakeToto()
    summary = runner.run_toto(dataset, tmp_path / "out", xgboost_reference=xgboost_reference,
                              cache_dir=tmp_path / "cache", bootstrap_replicates=0, forecaster=model)
    assert set(summary["detectors"]) == {"toto", "last_value"}
    assert summary["model_checks"]["repeat_identical"]
    assert summary["model_checks"]["batch_isolation_max_abs_diff"] == 0.0
    assert set(summary["comparison"]["runs"]) == {"xgboost/detection/seed-42", "toto/detection/dev",
                                                  "last_value/detection/dev"}
    assert not model.weight.requires_grad
    for name in ("toto", "last_value"):
        run = load_run(tmp_path / "out" / name)
        assert run["metadata"].cohort.split == "dev"
        assert len(run["predictions"]) == summary["dev_windows"]
        detector = json.loads((tmp_path / "out" / name / "detector.json").read_text())
        assert len(detector["channel_scales"]) == 16
        assert run["metadata"].checkpoint == runner._digest(
            (tmp_path / "out" / name / "detector.json").read_bytes())
        thresholds = json.loads((tmp_path / "out" / f"{name}_thresholds.json").read_text())
        assert thresholds["candidates"] <= 102
        with np.load(tmp_path / "out" / name / "scores.npz") as saved:
            assert saved["context_mask"].shape == (summary["dev_windows"], 16, 96)
            assert saved["target_mask"].shape == (summary["dev_windows"], 16, 32)
    # The fake forecasts the last value, so Toto and the last-value detector agree exactly.
    assert summary["detectors"]["toto"]["macro_f1"] == summary["detectors"]["last_value"]["macro_f1"]
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 1

    # A cached replay needs no model at all.
    def unavailable(**_):
        raise AssertionError("cached forecasts must not reload Toto")

    monkeypatch.setattr(runner, "load_forecaster", unavailable)
    again = runner.run_toto(dataset, tmp_path / "out2", xgboost_reference=xgboost_reference,
                            cache_dir=tmp_path / "cache", bootstrap_replicates=0)
    assert again["detectors"]["toto"]["macro_f1"] == summary["detectors"]["toto"]["macro_f1"]
    assert again["model_checks"] is None


def test_saved_detector_replays_on_test_and_rejects_tampering(dataset, xgboost_reference, tmp_path):
    runner.run_toto(dataset, tmp_path / "out", xgboost_reference=xgboost_reference,
                    cache_dir=tmp_path / "cache", bootstrap_replicates=0, forecaster=FakeToto())
    result = runner.predict_saved_detector(dataset, tmp_path / "out" / "toto", tmp_path / "test",
                                           cache_dir=tmp_path / "cache", bootstrap_replicates=0,
                                           forecaster=FakeToto())
    run = load_run(tmp_path / "test")
    assert run["metadata"].cohort.split == "test"
    assert run["metadata"].configuration["threshold"] == json.loads(
        (tmp_path / "out" / "toto" / "detector.json").read_text())["threshold"]
    assert result["failed_predictions"] == 0
    detector = tmp_path / "out" / "last_value" / "detector.json"
    detector.write_text(detector.read_text().replace('"detector"', '"detector" ', 1))
    with pytest.raises(ValueError, match="detector checksum mismatch"):
        runner.predict_saved_detector(dataset, tmp_path / "out" / "last_value", tmp_path / "test2",
                                      bootstrap_replicates=0)


def test_failed_forecasts_stay_in_the_run(dataset, xgboost_reference, tmp_path, monkeypatch):
    original = runner.window_scores

    def one_failure(forecast, target, target_mask, context_mask, scales):
        scores, failures = original(forecast, target, target_mask, context_mask, scales)
        scores[0] = np.nan
        return scores, {**failures, 0: "RSRP has no observed context"}

    monkeypatch.setattr(runner, "window_scores", one_failure)
    summary = runner.run_toto(dataset, tmp_path / "out", xgboost_reference=xgboost_reference,
                              cache_dir=tmp_path / "cache", bootstrap_replicates=0, forecaster=FakeToto())
    assert summary["detectors"]["toto"]["failed_predictions"] == 1
    statuses = [p.status for p in load_run(tmp_path / "out" / "toto")["predictions"]]
    assert statuses.count("error") == 1 and len(statuses) == summary["dev_windows"]


def test_changed_weights_and_cli_errors_are_reported(dataset, xgboost_reference, tmp_path, capsys):
    class Drifting(FakeToto):
        def forecast(self, *args, **kwargs):
            with torch.no_grad():
                self.weight.add_(1.0)
            return super().forecast(*args, **kwargs)

    with pytest.raises(ValueError, match="weights changed"):
        runner.run_toto(dataset, tmp_path / "out", xgboost_reference=xgboost_reference,
                        cache_dir=tmp_path / "cache", bootstrap_replicates=0, forecaster=Drifting())
    (tmp_path / "exists").mkdir()
    code = cli.main(["train", "--release-dir", str(dataset.release_dir), "--output-dir",
                     str(tmp_path / "exists"), "--xgboost-reference", str(xgboost_reference)])
    assert code == 1 and "output exists" in capsys.readouterr().err
