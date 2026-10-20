import json
import statistics

import pytest

from clauseguard.baselines.artifacts import write_run
from clauseguard.baselines.comparison import compare_experiments, render_markdown
from clauseguard.baselines.contracts import CasePrediction, RunMetadata
from clauseguard.baselines.data import experiment_cohort
from clauseguard.cli.compare_models import main as compare_cli
from clauseguard.datasets.release import prepare_release
from clauseguard.datasets.scenarios import ScenarioDataset


@pytest.fixture
def dataset(release_sources, recording_evidence):
    source = release_sources
    prepare_release(source["scenarios"], source["output"], source["output"], trials=128)
    return ScenarioDataset(source["output"])


def _write_experiment(root, dataset, name, *, flips=0, split="dev", cohort_update=None,
                      score_shift=0.0, task="detection"):
    """Save a fake experiment: two seeds of one dev run plus the matching selection.json."""
    cohort = experiment_cohort(dataset, split=split, task=task)
    key = f"{task}_target"
    expected = {sid: dataset.splits["membership"][sid][key] for sid in cohort.scenario_ids}
    groups = {sid: dataset.splits["membership"][sid]["group_id"] for sid in cohort.scenario_ids}
    wrong = set([sid for sid, target in expected.items() if target == 0 or task == "category"]
                [:flips])
    classes = 2 if task == "detection" else 10
    predictions = [CasePrediction(
        scenario_id=sid, task=task,
        predicted_label=(target + 1) % classes if sid in wrong else target)
        for sid, target in expected.items()]
    if cohort_update:
        cohort = cohort.model_copy(update=cohort_update)
    seeds = []
    for seed in (42, 43):
        run = f"{task}/seed-{seed}"
        metadata = RunMetadata(
            model=f"{name}/{task}/seed-{seed}", task=task, cohort=cohort, seed=seed,
            versions={"platform": "test-box"},
            resources={"fit_seconds": 2.0 + seed, "prediction_seconds": 0.5,
                       "process_peak_rss_bytes": 1e9})
        metrics = write_run(root / name / run, metadata, expected, groups, predictions,
                            bootstrap_replicates=0)
        (root / name / run / "model.pt").write_bytes(b"x" * 1000)
        seeds.append({"run": run, "seed": seed, "dev_macro_f1": metrics["macro_f1"] + score_shift})
    selection = {
        "evaluation_status": "provisional_exploratory", "release_id": cohort.release_id,
        "tasks": {task: {
            "dev_cohort_id": cohort.cohort_id, "train_ids_sha256": "abc", "train_windows": 7,
            "selected_configuration": {"depth": 3}, "trials": [{}, {}], "selected_seeds": seeds}},
    }
    (root / name / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
    return root / name


def test_compare_reports_scores_errors_and_cost(dataset, tmp_path):
    good = _write_experiment(tmp_path, dataset, "good")
    weak = _write_experiment(tmp_path, dataset, "weak", flips=2)

    report = compare_experiments({"good": good, "weak": weak}, "detection")

    assert report["split"] == "dev"
    assert report["models"]["good"]["dev_macro_f1_mean"] == 1.0
    assert report["models"]["good"]["dev_macro_f1_std"] == 0.0
    assert report["models"]["weak"]["dev_macro_f1_mean"] < 1.0
    weak_seed = report["models"]["weak"]["seeds"][0]
    assert weak_seed["false_alarms"] + weak_seed["missed_faults"] == 2
    assert weak_seed["prediction_status_counts"]["error"] == 0
    cost = report["models"]["good"]["cost"]
    assert cost["fit_seconds_mean"] == pytest.approx(2.0 + 42.5)
    assert cost["model_size_bytes"] == 1000
    assert cost["platforms"] == ["test-box"]


def test_compare_rejects_changed_cohort_identity(dataset, tmp_path):
    a = _write_experiment(tmp_path, dataset, "a")
    b = _write_experiment(tmp_path, dataset, "b", cohort_update={"input_hash": "other"})

    with pytest.raises(ValueError, match="dev_cohort_id"):
        compare_experiments({"a": a, "b": b}, "detection")


def test_compare_rejects_non_development_runs(dataset, tmp_path):
    a = _write_experiment(tmp_path, dataset, "a")
    b = _write_experiment(tmp_path, dataset, "b", split="test")

    with pytest.raises(ValueError, match="development run"):
        compare_experiments({"a": a, "b": b}, "detection")


def test_compare_rejects_selection_score_mismatch(dataset, tmp_path):
    a = _write_experiment(tmp_path, dataset, "a")
    b = _write_experiment(tmp_path, dataset, "b", score_shift=0.05)

    with pytest.raises(ValueError, match="selection record"):
        compare_experiments({"a": a, "b": b}, "detection")


def test_compare_rejects_corrupted_run(dataset, tmp_path):
    a = _write_experiment(tmp_path, dataset, "a")
    b = _write_experiment(tmp_path, dataset, "b")
    with (b / "detection/seed-42/predictions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(ValueError, match="checksum"):
        compare_experiments({"a": a, "b": b}, "detection")


def test_cli_writes_tables_and_refuses_overwrite(dataset, tmp_path, capsys):
    a = _write_experiment(tmp_path, dataset, "a", task="category")
    b = _write_experiment(tmp_path, dataset, "b", task="category", flips=1)
    out = tmp_path / "out"
    args = ["--model", f"a={a}", "--model", f"b={b}", "--task", "category",
            "--output-dir", str(out)]

    assert compare_cli(args) == 0
    saved = json.loads((out / "comparison.json").read_text())
    assert saved[0]["task"] == "category"
    markdown = (out / "comparison.md").read_text()
    assert "development split only" in markdown and "Per-class F1" in markdown
    assert render_markdown(saved) == markdown
    assert compare_cli(args) == 1
    assert "output exists" in capsys.readouterr().err


def _selection_file(tmp_path, dataset, ids, *, release_id=None):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"policy": "test-policy", "purpose": "audit_only",
                                "release_id": release_id or dataset.release_id,
                                "scenario_ids": sorted(ids)}), encoding="utf-8")
    return path


def test_window_selection_rescores_every_model_on_the_same_subset(dataset, tmp_path):
    a = _write_experiment(tmp_path, dataset, "a")
    b = _write_experiment(tmp_path, dataset, "b", flips=1)
    cohort = experiment_cohort(dataset, split="dev", task="detection")
    membership = dataset.splits["membership"]
    normals = [sid for sid in cohort.scenario_ids if membership[sid]["detection_target"] == 0]
    flipped = set(normals[:1])
    keep = [sid for sid in cohort.scenario_ids if sid not in flipped]
    path = _selection_file(tmp_path, dataset, keep)

    full = compare_experiments({"a": a, "b": b}, "detection")
    reduced = compare_experiments({"a": a, "b": b}, "detection", window_selection=path)

    assert full["models"]["b"]["dev_macro_f1_mean"] < full["models"]["a"]["dev_macro_f1_mean"]
    assert (reduced["models"]["b"]["dev_macro_f1_mean"]
            == reduced["models"]["a"]["dev_macro_f1_mean"])
    assert reduced["window_selection"]["windows"] == len(keep)
    assert reduced["window_selection"]["policy"] == "test-policy"
    assert sum(reduced["models"]["a"]["seeds"][0]["prediction_status_counts"].values()) == len(keep)


def test_window_selection_must_match_release_and_keep_faults(dataset, tmp_path):
    a = _write_experiment(tmp_path, dataset, "a")
    b = _write_experiment(tmp_path, dataset, "b")
    cohort = experiment_cohort(dataset, split="dev", task="detection")
    ids = cohort.scenario_ids
    membership = dataset.splits["membership"]
    faults = [sid for sid in ids if membership[sid]["detection_target"] == 1]

    wrong_release = _selection_file(tmp_path, dataset, ids, release_id="other")
    with pytest.raises(ValueError, match="different release"):
        compare_experiments({"a": a, "b": b}, "detection", window_selection=wrong_release)
    dropping = _selection_file(tmp_path, dataset, [sid for sid in ids if sid != faults[0]])
    with pytest.raises(ValueError, match="every fault"):
        compare_experiments({"a": a, "b": b}, "detection", window_selection=dropping)


def test_seed_spread_exposes_a_lucky_seed(dataset, tmp_path):
    a = _write_experiment(tmp_path, dataset, "a")
    b = _write_experiment(tmp_path, dataset, "b", flips=3)

    model = compare_experiments({"a": a, "b": b}, "detection")["models"]["b"]
    scores = [seed["macro_f1"] for seed in model["seeds"]]

    assert model["dev_macro_f1_median"] == pytest.approx(statistics.median(scores))
    assert model["dev_macro_f1_range"] == [min(scores), max(scores)]
