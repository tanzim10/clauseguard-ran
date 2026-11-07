import json

import pytest

from clauseguard.baselines import final_evaluation
from clauseguard.baselines.artifacts import load_run, write_run
from clauseguard.baselines.contracts import CasePrediction, RunMetadata
from clauseguard.baselines.data import experiment_cohort, experiment_targets
from clauseguard.baselines.final_evaluation import (
    MODELS,
    freeze_protocol,
    load_protocol,
    matching_case_counts,
    predict_test,
    write_report,
)
from clauseguard.baselines.tree_data import ids_hash
from clauseguard.cli import final_evaluation as cli
from clauseguard.datasets.release import prepare_release
from clauseguard.datasets.scenarios import ScenarioDataset

TASKS = ("detection", "category")


@pytest.fixture
def dataset(release_sources, recording_evidence):
    s = release_sources
    prepare_release(s["scenarios"], s["output"], s["output"], trials=128)
    return ScenarioDataset(s["output"])


def _label(model, task, target, seed, index):
    """random_forest is perfect, xgboost calls everything normal/class 0, timesnet alarms on all."""
    if model == "timesnet" and seed == 44 and index == 0:
        return None
    if model == "xgboost":
        return 0
    if model == "timesnet" and task == "detection":
        return 1
    return target


def _predictions(model, task, seed, expected):
    rows = []
    for index, (sid, target) in enumerate(expected.items()):
        label = _label(model, task, target, seed, index)
        rows.append(CasePrediction(scenario_id=sid, task=task, predicted_label=label,
                                   status="ok" if label is not None else "error",
                                   error=None if label is not None else "fixture failure"))
    return rows


def _write(dataset, path, model, task, seed, split):
    cohort = experiment_cohort(dataset, split=split, task=task)
    train = experiment_cohort(dataset, split="train", task=task)
    expected, groups = experiment_targets(dataset, cohort)
    metadata = RunMetadata(model=f"{model}/{task}/seed-{seed}", task=task, cohort=cohort,
                           train_cohort_id=train.cohort_id, seed=seed,
                           checkpoint=f"{model}-{task}-{seed}", configuration={"depth": 1},
                           resources={"fit_seconds": 2.0} if split == "dev" else
                           {"prediction_seconds": 0.5})
    metrics = write_run(path, metadata, expected, groups,
                        _predictions(model, task, seed, expected), bootstrap_replicates=0)
    return metrics


def _experiment(dataset, root, model):
    selection = {"release_id": dataset.release_id, "evaluation_status": "scored", "tasks": {}}
    for task in TASKS:
        seeds = []
        for seed in (42, 43, 44):
            path = root / task / f"seed-{seed}"
            metrics = _write(dataset, path, model, task, seed, "dev")
            (path / "model.joblib").write_bytes(b"0" * 2_000)
            seeds.append({"seed": seed, "run": f"{task}/seed-{seed}", "dev_macro_f1": metrics["macro_f1"]})
        train = experiment_cohort(dataset, split="train", task=task)
        selection["tasks"][task] = {
            "selected_seeds": seeds, "dev_cohort_id": experiment_cohort(
                dataset, split="dev", task=task).cohort_id,
            "train_ids_sha256": ids_hash(train.scenario_ids), "train_windows": len(train.scenario_ids),
            "selected_configuration": {"depth": 1}, "trials": [{}, {}]}
    (root / "selection.json").write_text(json.dumps(selection))
    return root


@pytest.fixture
def experiments(dataset, tmp_path):
    return {model: _experiment(dataset, tmp_path / model, model) for model in MODELS}


@pytest.fixture
def fake_predictor(monkeypatch):
    calls = []

    def factory(model):
        def predict(dataset, model_run_dir, output_dir, *, split, bootstrap_replicates,
                    allow_exploratory, **options):
            metadata = load_run(model_run_dir)["metadata"]
            calls.append((model, metadata.task, metadata.seed, options))
            _write(dataset, output_dir, model, metadata.task, metadata.seed, split)
        return predict

    monkeypatch.setattr(final_evaluation, "_predictor", factory)
    return calls


def test_protocol_freezes_dev_choices_and_discloses_test_exposure(dataset, experiments, tmp_path):
    _write(dataset, experiments["xgboost"] / "detection" / "test-seed-42", "xgboost", "detection",
           42, "test")
    protocol = freeze_protocol(dataset, experiments, tmp_path / "eval", frozen_on="2026-11-05",
                               exposure_dirs=[tmp_path])
    assert protocol["dev_comparators"] == {"detection": "random_forest", "category": "random_forest"}
    assert [row["model"] for row in protocol["prior_test_exposure"]] == ["xgboost/detection/seed-42"]
    assert protocol["text"]["status"] == "not supplied"
    assert protocol["models"]["timesnet"]["tasks"]["category"]["seeds"][2]["checkpoint"] == "timesnet-category-44"
    assert load_protocol(tmp_path / "eval") == protocol
    with pytest.raises(FileExistsError):
        freeze_protocol(dataset, experiments, tmp_path / "eval", frozen_on="2026-11-05")
    saved = json.loads((tmp_path / "eval" / "protocol.json").read_text())
    saved["seeds"] = [42]
    (tmp_path / "eval" / "protocol.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="changed after it was frozen"):
        load_protocol(tmp_path / "eval")


def test_report_regenerates_from_saved_runs_with_hand_checked_diagnosis(
        dataset, experiments, fake_predictor, tmp_path):
    out = tmp_path / "eval"
    freeze_protocol(dataset, experiments, out, frozen_on="2026-11-05")
    runs = predict_test(dataset, out, device="mps", bootstrap_replicates=0)
    assert len(fake_predictor) == 18
    assert {call[3].get("device") for call in fake_predictor if call[0] == "timesnet"} == {"mps"}
    tables = write_report(dataset, out, runs, bootstrap_replicates=0)["tables"]

    members = dataset.splits["membership"]
    detection = runs["random_forest", "detection", 42]["expected"]
    normals = sum(target == 0 for target in detection.values())
    synthetic = len(runs["random_forest", "category", 42]["expected"])
    jammers = sum(members[sid]["origin"] == "real" for sid in detection)
    diagnosis = {(row["model"], row["seed"]): row for row in tables["diagnosis"]}
    assert diagnosis["random_forest", 42]["windows"] == normals + synthetic
    assert diagnosis["random_forest", 42]["accuracy"] == 1.0
    # Calling everything normal is right only for normal windows.
    assert diagnosis["xgboost", 42]["accuracy"] == normals / (normals + synthetic)
    # Alarming on every window is right only for synthetic faults with a correct category.
    assert diagnosis["timesnet", 42]["accuracy"] == synthetic / (normals + synthetic)
    assert diagnosis["timesnet", 44]["failed_predictions"] >= 1

    rows = {(row["model"], row["seed"]): row for row in tables["detection"]}
    assert rows["xgboost", 43]["missed_faults"] == synthetic + jammers
    assert rows["timesnet", 43]["false_alarms"] == normals
    reference = tables["detection_reference"][0]
    assert reference["false_alarms"] == 0 and reference["missed_faults"] == synthetic + jammers
    jamming = [row for row in tables["errors_by_origin"]
               if (row["model"], row["seed"], row["origin"]) == ("xgboost", 42, "real")]
    assert sum(row["missed_faults"] for row in jamming) == jammers
    report = (out / "report.md").read_text()
    assert "Earlier test exposure" in report and "not complete" in report
    assert "not proof of the physical cause" in report
    assert (out / "tables" / "per_class.csv").exists()

    first = (out / "results.json").read_text()
    fake_predictor.clear()
    write_report(dataset, out, predict_test(dataset, out, bootstrap_replicates=0),
                 bootstrap_replicates=0)
    assert not fake_predictor and (out / "results.json").read_text() == first


def test_changed_checkpoint_or_cohort_is_refused(dataset, experiments, fake_predictor, tmp_path):
    out = tmp_path / "eval"
    freeze_protocol(dataset, experiments, out, frozen_on="2026-11-05")
    run = experiments["xgboost"] / "category" / "seed-43"
    saved = json.loads((run / "run.json").read_text())
    saved["checkpoint"] = "retrained"
    (run / "run.json").write_text(json.dumps(saved, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError, match="checksum mismatch|frozen checkpoint"):
        predict_test(dataset, out, bootstrap_replicates=0)


def test_matching_case_counts_follow_hand_worked_example():
    expected = {"a": 1, "b": 0, "c": 1, "d": 0, "e": 1}

    def preds(labels):
        return [CasePrediction(scenario_id=sid, task="detection", predicted_label=label,
                               status="ok" if label is not None else "error")
                for sid, label in zip(expected, labels, strict=True)]

    counts = matching_case_counts(expected, preds([1, 0, 0, 1, None]), preds([1, 1, 1, 1, 1]))
    assert counts == {"both_correct": 1, "kpi_only": 1, "text_only": 2, "neither": 1}
    with pytest.raises(ValueError):
        matching_case_counts(expected, preds([1, 0, 0, 1, None])[:4], preds([1, 1, 1, 1, 1]))


def test_cli_requires_exploratory_opt_in(release_sources, tmp_path, capsys):
    s = release_sources
    from clauseguard.datasets.evidence import build_recording_evidence

    build_recording_evidence(s["scenarios"], s["acquired"], s["output"])
    prepare_release(s["scenarios"], s["output"], s["output"], mode="exploratory", trials=128)
    code = cli.main(["evaluate", "--release-dir", str(s["output"]), "--output-dir", str(tmp_path / "x")])
    assert code == 1 and "--allow-exploratory" in capsys.readouterr().err
