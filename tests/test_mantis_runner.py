import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from clauseguard.baselines import mantis as runner
from clauseguard.baselines.artifacts import load_run, write_run
from clauseguard.baselines.contracts import CasePrediction, RunMetadata
from clauseguard.baselines.tree_data import split_data
from clauseguard.cli.mantis import main
from clauseguard.datasets.release import prepare_release
from clauseguard.datasets.scenarios import ScenarioDataset
from clauseguard.kpi.mantis import fit_head


@pytest.fixture
def dataset(release_sources, recording_evidence):
    source = release_sources
    prepare_release(source["scenarios"], source["output"], source["output"], trials=128)
    return ScenarioDataset(source["output"])


@pytest.fixture
def references(dataset, tmp_path):
    paths = {}
    for name in ("rf", "xgboost"):
        root = tmp_path / f"{name}-reference"
        root.mkdir()
        selected = {}
        for task in ("detection", "category"):
            train_cohort, train = split_data(dataset, task, "train")
            dev_cohort, dev = split_data(dataset, task, "dev")
            run_path = root / task / "seed-42"
            expected = dict(zip(dev.scenario_ids, dev.targets, strict=True))
            groups = dict(zip(dev.scenario_ids, dev.group_ids, strict=True))
            predictions = [CasePrediction(scenario_id=sid, task=task, predicted_label=label)
                           for sid, label in expected.items()]
            metadata = RunMetadata(
                model=f"{name}/{task}", task=task, cohort=dev_cohort,
                train_cohort_id=train_cohort.cohort_id, seed=42, checkpoint="fixture-reference",
            )
            write_run(run_path, metadata, expected, groups, predictions, bootstrap_replicates=0)
            selected[task] = {"selected_seeds": [{"seed": 42, "run": f"{task}/seed-42"}]}
            (root / task).mkdir(exist_ok=True)
            (root / task / "training_ids.json").write_text(json.dumps(list(train.scenario_ids)))
        (root / "selection.json").write_text(json.dumps({"release_id": dataset.release_id,
                                                           "tasks": selected}))
        paths[name] = root
    return paths


class FakeEncoder(torch.nn.Module):
    def __init__(self, *, frozen=True):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()), requires_grad=not frozen)

    def forward(self, values):
        mean = values.mean(dim=(1, 2)) * self.scale
        return mean[:, None].repeat(1, 512)


def test_two_variants_use_only_train_and_dev_and_save_replayable_runs(
        dataset, references, tmp_path, monkeypatch):
    calls = []
    original = runner.split_data

    def track_split(data, task, split):
        calls.append(split)
        return original(data, task, split)

    monkeypatch.setattr(runner, "split_data", track_split)
    output = tmp_path / "mantis"
    cache = tmp_path / "cache"
    result = runner.run_mantis(
        dataset, output, tasks=("detection",), rf_reference=references["rf"],
        xgboost_reference=references["xgboost"], cache_dir=cache,
        bootstrap_replicates=0, encoder=FakeEncoder())

    assert set(calls) == {"train", "dev"}
    assert result["tasks"]["detection"]["status"] == "complete"
    trials = result["tasks"]["detection"]["feature_trials"]
    assert [row["feature_variant"] for row in trials] == ["embeddings", "embeddings+statistics"]
    assert [row["status"] for row in trials] == ["ok", "ok"]
    assert [load_run(output / row["run"])["metadata"].configuration["feature_width"]
            for row in trials] == [8192, 8224]
    assert all((output / row["run"] / "head.joblib").exists() for row in trials)
    assert all(Path(path).exists() for path in result["tasks"]["detection"]["cache_paths"].values())
    assert result["tasks"]["detection"]["comparison"]["cohort_id"] == (
        result["tasks"]["detection"]["dev_cohort_id"])
    assert load_run(output / result["tasks"]["detection"]["selected_run"])["metadata"].seed == 42


def test_saved_head_replays_dev_and_predicts_test(dataset, references, tmp_path):
    encoder = FakeEncoder()
    output = tmp_path / "train"
    result = runner.run_mantis(
        dataset, output, tasks=("detection",), rf_reference=references["rf"],
        xgboost_reference=references["xgboost"], cache_dir=tmp_path / "cache",
        bootstrap_replicates=0, encoder=encoder)
    run_path = output / result["tasks"]["detection"]["selected_run"]
    original = load_run(run_path)

    replay = runner.predict_saved_model(
        dataset, run_path, tmp_path / "replay", split="dev", cache_dir=tmp_path / "cache",
        bootstrap_replicates=0)
    restored = load_run(Path(replay["run_dir"]))
    assert restored["predictions"] == original["predictions"]

    scored = runner.predict_saved_model(
        dataset, run_path, tmp_path / "test", split="test", cache_dir=tmp_path / "cache",
        bootstrap_replicates=0, encoder=encoder)
    assert load_run(Path(scored["run_dir"]))["metadata"].cohort.split == "test"


def test_feature_cache_reuses_only_matching_identities(dataset, references, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    args = {"tasks": ("detection",), "rf_reference": references["rf"],
            "xgboost_reference": references["xgboost"], "cache_dir": cache,
            "bootstrap_replicates": 0, "encoder": FakeEncoder()}
    runner.run_mantis(dataset, tmp_path / "first", **args)

    def unexpected_encoding(*_args, **_kwargs):
        raise AssertionError("matching cached features should be reused")

    monkeypatch.setattr(runner, "encode_windows", unexpected_encoding)
    runner.run_mantis(dataset, tmp_path / "second", **args)

    cache_path = next(cache.glob("*.npz"))
    with np.load(cache_path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"].item()))
        corrupted = saved["embeddings"].copy()
        statistics = saved["statistics"].copy()
    corrupted[0, 0] += 1
    with cache_path.open("wb") as stream:
        np.savez(stream, embeddings=corrupted, statistics=statistics,
                 metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    with pytest.raises(ValueError, match="cache checksum"):
        runner.run_mantis(dataset, tmp_path / "third", **args)


def test_mismatched_reference_fails_before_output_creation(dataset, references, tmp_path):
    path = references["xgboost"] / "selection.json"
    selection = json.loads(path.read_text())
    selection["release_id"] = "different"
    path.write_text(json.dumps(selection))
    output = tmp_path / "blocked"

    with pytest.raises(ValueError, match="different release"):
        runner.run_mantis(
            dataset, output, tasks=("detection",), rf_reference=references["rf"],
            xgboost_reference=references["xgboost"], encoder=FakeEncoder())
    assert not output.exists()


def test_encoder_must_be_frozen(dataset, references, tmp_path):
    output = tmp_path / "unfrozen"

    with pytest.raises(ValueError, match="must be frozen"):
        runner.run_mantis(
            dataset, output, tasks=("detection",), rf_reference=references["rf"],
            xgboost_reference=references["xgboost"], encoder=FakeEncoder(frozen=False))
    assert not output.exists()


def test_exact_balanced_weights_and_fixed_logistic_settings():
    targets = np.repeat(np.arange(2), [8, 2])
    features = np.eye(10, 20, dtype=np.float32)

    scaler, classifier, weights = fit_head(features, targets, task="detection")

    assert weights == {0: 0.625, 1: 2.5}
    assert classifier.class_weight == weights
    assert classifier.C == 1 and classifier.solver == "lbfgs" and classifier.max_iter == 2000
    assert scaler.mean_.shape == (features.shape[1],)


def test_cli_rejects_existing_output(dataset, references, tmp_path, capsys):
    output = tmp_path / "exists"
    output.mkdir()
    code = main(["train", "--task", "detection", "--release-dir", str(dataset.release_dir),
                 "--rf-reference", str(references["rf"]), "--xgboost-reference",
                 str(references["xgboost"]), "--output-dir", str(output)])
    assert code == 1
    assert "output exists" in capsys.readouterr().err
