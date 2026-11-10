"""Task 4.2: detector selection, contract adapter, pipeline order, failures and family exclusion."""

import json
import shutil
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("xgboost")

from clauseguard.baselines import random_forest as rf_runner
from clauseguard.baselines import xgboost as xgb_runner
from clauseguard.baselines.artifacts import load_run
from clauseguard.baselines.contracts import CasePrediction
from clauseguard.baselines.tree_data import split_data
from clauseguard.cli import detector_predictions as cli
from clauseguard.datasets.contracts import CATEGORY_LABELS
from clauseguard.datasets.release import prepare_release
from clauseguard.datasets.scenarios import ScenarioDataset
from clauseguard.datasets.schema import TextReview
from clauseguard.kpi.features import ChannelMedianFeatures
from clauseguard.rca import cross_fit, detector
from clauseguard.rca.contracts import DATASET_SOURCE_RULES


def _head(task, labels, *, calls=None, fail=None):
    def predict(inputs):
        if calls is not None:
            calls.append(list(inputs))
        if fail:
            raise RuntimeError(fail)
        return labels[:len(inputs)] if isinstance(labels, list) else labels(inputs)

    def proba(inputs):
        width = 2 if task == "detection" else 10
        return np.full((len(inputs), width), 1 / width)

    return detector.Head(task, "xgboost" if task == "detection" else "random_forest",
                         f"{task}-run", "a" * 64, predict, proba)


def test_label_mappings_for_each_task():
    assert detector.label_names("detection") == ("normal", "abnormal")
    assert detector.label_names("category") == tuple(CATEGORY_LABELS)
    assert detector.label_names("diagnosis") == ("normal", *CATEGORY_LABELS)
    assert detector.resolve_label("diagnosis", 3) == CATEGORY_LABELS[2]
    for bad in (True, -1, 2, "1", None):
        assert detector.resolve_label("detection", bad) is None
    with pytest.raises(ValueError, match="unknown"):
        detector.label_names("text")


def test_adapter_maps_case_predictions_and_keeps_failures():
    ok = detector.to_detector_prediction(
        CasePrediction(scenario_id="s", task="category", predicted_label=2), run_id="r", confidence=0.4)
    assert (ok.status, ok.predicted_label_id, ok.predicted_label) == ("ok", 2, CATEGORY_LABELS[2])
    assert (ok.label_mapping_version, ok.model_run_id, ok.source_modalities, ok.confidence) == (
        "1", "r", ["kpi"], 0.4)
    error = detector.to_detector_prediction(
        CasePrediction(scenario_id="s", task="detection", status="error", error="boom"), run_id="r")
    assert error.status == "error" and error.predicted_label is None and error.absence_reason is None
    for case in (CasePrediction(scenario_id="s", task="detection", predicted_label=7),
                 CasePrediction(scenario_id="s", task="detection", status="invalid")):
        assert detector.to_detector_prediction(case, run_id="r").status == "invalid"


def test_category_runs_only_for_abnormal_windows_and_diagnosis_shifts_by_one():
    calls = []
    heads = {"detection": _head("detection", [0, 1, 1]),
             "category": _head("category", [3, 9], calls=calls)}
    results = detector.run_pipeline(heads, ["a", "b", "c"], ["x", "y", "z"])
    assert calls == [["y", "z"]]
    det, cat, diag = zip(*results, strict=True)
    assert [r.prediction.predicted_label for r in det] == ["normal", "abnormal", "abnormal"]
    assert cat[0].prediction.status == "absent" and "not called" in cat[0].prediction.absence_reason
    assert [r.prediction.predicted_label_id for r in diag] == [0, 4, 10]
    assert diag[2].prediction.predicted_label == CATEGORY_LABELS[9]
    assert det[1].prediction.confidence == 0.5 and det[1].calibration_status == "uncalibrated"
    assert diag[1].prediction.confidence is None and diag[1].calibration_status == "not_available"
    assert diag[1].prediction.model_run_id == "detection-then-category-v1:detection-run+category-run"


def test_failed_calls_invalid_outputs_and_absences_stay_distinct():
    failed = detector.run_pipeline({"detection": _head("detection", [], fail="lost"),
                                    "category": _head("category", [1])}, ["a"], ["x"])[0]
    assert [r.prediction.status for r in failed] == ["error", "absent", "error"]
    assert failed[0].error == "RuntimeError: lost" and failed[2].error == "RuntimeError: lost"
    category_failed = detector.run_pipeline({"detection": _head("detection", [1]),
                                             "category": _head("category", [], fail="x")}, ["a"], ["x"])[0]
    assert [r.prediction.status for r in category_failed] == ["ok", "error", "error"]
    short = detector.predict_head(_head("detection", lambda inputs: [1]), ["a", "b"], ["x", "y"])
    assert {r.prediction.status for r in short} == {"error"} and "1 outputs for 2" in short[0].error
    invalid = detector.predict_head(_head("detection", lambda inputs: ["1"]), ["a"], ["x"])
    assert invalid[0].prediction.status == "invalid" and invalid[0].error is None
    missing = detector.run_pipeline({"detection": _head("detection", [1]), "category": None}, ["a"], ["x"],
                                    unavailable={"category": "fold lacks labels"})[0]
    assert [r.prediction.status for r in missing] == ["ok", "absent", "absent"]
    assert missing[2].prediction.absence_reason == "model unavailable: fold lacks labels"


def test_dataset_compatibility_covers_every_contract_source():
    assert set(detector.DATASET_COMPATIBILITY) == set(DATASET_SOURCE_RULES)
    for name, entry in detector.DATASET_COMPATIBILITY.items():
        if entry["status"] == "absent":
            absent = detector.dataset_absence(name)
            assert {p.status for p in absent.values()} == {"absent"}
            assert all(p.absence_reason == entry["reason"] for p in absent.values())
    with pytest.raises(ValueError, match="compatible detector"):
        detector.dataset_absence("TelecomTS")


def test_saved_case_records_have_no_reference_fields():
    fields = set(detector.CaseRecord.model_fields) | set(detector.PredictionRecord.model_fields)
    assert not fields & {"target", "label", "expected", "split", "rca_label", "fault_label",
                         "source_labels", "detection_target", "category_target", "reference"}


def test_folds_are_deterministic_and_spread_faults():
    origins = {f"n{i}": "normal" for i in range(4)} | {"s0": "synthetic", "s1": "synthetic", "j": "real"}
    folds = cross_fit.assign_folds(origins, 3)
    assert folds == cross_fit.assign_folds(dict(reversed(origins.items())), 3)
    assert (folds["s0"], folds["s1"], folds["j"]) == (0, 1, 2)
    assert sorted(folds.values()) == [0, 0, 0, 1, 1, 2, 2]
    with pytest.raises(ValueError, match="folds"):
        cross_fit.assign_folds(origins, 1)


@pytest.fixture
def dataset(release_sources, recording_evidence):
    source = release_sources
    prepare_release(source["scenarios"], source["output"], source["output"], trials=128)
    return ScenarioDataset(source["output"])


@pytest.fixture
def experiments(dataset, tmp_path):
    rf = tmp_path / "rf"
    rf_runner.run_random_forest(dataset, rf, bootstrap_replicates=0)
    xgb = tmp_path / "xgb"
    xgb_runner.run_xgboost(dataset, xgb, rf_reference=rf, bootstrap_replicates=0)
    return {"random_forest": rf, "xgboost": xgb}


@pytest.fixture
def selection(dataset, experiments, tmp_path):
    return detector.freeze_selection(dataset, experiments, tmp_path / "selection.json",
                                     frozen_on="2026-11-09", exposure_dirs=[tmp_path])


def _dev_scores(path, task):
    return json.loads((path / "selection.json").read_text())["tasks"][task]["dev_macro_f1_mean"]


def test_freeze_picks_best_dev_model_per_task_and_seals_it(dataset, experiments, selection, tmp_path):
    for task, entry in selection["models"].items():
        best = max(experiments, key=lambda name: _dev_scores(experiments[name], task))
        assert entry["model"] == best and entry["seed"] == 42
        run = load_run(entry["run"])["metadata"]
        assert entry["checkpoint_sha256"] == run.checkpoint and run.cohort.split == "dev"
        history = entry["training_history"]
        assert set(history["training_families"]).isdisjoint(history["development_families"])
        assert entry["label_mapping"]["names"] == list(detector.label_names(task))
    assert detector.load_selection(tmp_path / "selection.json") == selection
    with pytest.raises(FileExistsError):
        detector.freeze_selection(dataset, experiments, tmp_path / "selection.json", frozen_on="x")
    tampered = dict(selection, frozen_on="2026-01-01")
    (tmp_path / "tampered.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="sealed"):
        detector.load_selection(tmp_path / "tampered.json")
    other = tmp_path / "timesnet"
    other.mkdir()
    (other / "selection.json").write_text(json.dumps({
        "release_id": dataset.release_id, "selection_split": "dev",
        "tasks": {"detection": {"dev_macro_f1_mean": 2.0}}}))
    with pytest.raises(ValueError, match="no Task 4 adapter"):
        detector.freeze_selection(dataset, {**experiments, "timesnet": other}, tmp_path / "s2.json",
                                  frozen_on="x")
    (other / "selection.json").write_text(json.dumps({"release_id": "z", "selection_split": "dev"}))
    with pytest.raises(ValueError, match="different release"):
        detector.freeze_selection(dataset, {"timesnet": other}, tmp_path / "s3.json", frozen_on="x")


def test_case_predictions_reproduce_saved_runs_and_reject_incompatible_heads(
        dataset, selection, tmp_path):
    _, dev = split_data(dataset, "detection", "dev")
    records = detector.predict_cases(dataset, selection, list(dev.scenario_ids))
    agreement = detector.saved_run_agreement(records, selection)
    assert agreement["detection"]["compared"] == len(dev.scenario_ids)
    assert all(row["compared"] == row["identical"] for row in agreement.values())
    assert detector.predict_cases(dataset, selection, list(dev.scenario_ids)) == records
    assert all(r.text.state == "missing" and r.readiness == "scored" for r in records)
    assert "Revealing source text" not in "".join(r.model_dump_json() for r in records)
    entry = selection["models"]["detection"]
    for change, message in (({"data": {**entry["data"], "release_id": "x"}}, "release"),
                            ({"preprocessing": {**entry["preprocessing"],
                                                "channel_order": entry["preprocessing"]["channel_order"][::-1]}},
                             "channel layout"),
                            ({"label_mapping": {**entry["label_mapping"], "names": ["a", "b"]}}, "mapping")):
        with pytest.raises(ValueError, match=message):
            detector.load_head(dataset, {**entry, **change})
    copy = tmp_path / "model.ubj"
    shutil.copy(entry["checkpoint_path"], copy)
    copy.write_bytes(copy.read_bytes() + b"x")
    with pytest.raises(ValueError, match="checkpoint"):
        detector.load_head(dataset, {**entry, "checkpoint_path": str(copy)})
    with pytest.raises(ValueError, match="duplicate"):
        detector.predict_cases(dataset, selection, [dev.scenario_ids[0]] * 2)


def test_text_eligibility_requires_rca_permission(dataset):
    record = dataset.load_scenario(next(iter(dataset.splits["membership"])))
    assert detector.rca_text_eligibility(record) == detector.TextEligibility(
        state="missing", origin="unreviewed", reason="no_approved_text")
    approved = TextReview(status="approved", approved_tasks=["detection"],
                          text_origin="independent_observation", reviewer="r", reviewed_at="2026-11-09")
    clean = record.model_copy(update={"text_review": approved, "nl_context": "Throughput dropped."})
    assert detector.rca_text_eligibility(clean).reason == "no_rca_permission"
    leaking = clean.model_copy(update={"nl_context": f"{CATEGORY_LABELS[0]} observed."})
    assert detector.rca_text_eligibility(leaking).reason == "text_discloses_diagnosis"
    derived = clean.model_copy(update={"text_review": approved.model_copy(
        update={"text_origin": "diagnosis_derived"})})
    assert detector.rca_text_eligibility(derived) == detector.TextEligibility(
        state="excluded", origin="diagnosis_derived", reason="text_origin_not_permitted")


class _Families:
    """A training split with six recording families and one reserved development family."""

    def __init__(self, dataset, release_id):
        rng = np.random.default_rng(0)
        self.release_id, self.real = release_id, dataset
        self.template = next(iter(dataset.splits["membership"]))
        rows, self.batches = {}, {"detection": [], "category": []}
        plan = [("synth-a", "synthetic"), ("synth-b", "synthetic"), ("synth-c", "synthetic"),
                ("jam", "real"), ("norm-a", "normal"), ("norm-b", "normal")]
        for family, origin in plan:
            labels = list(range(10)) if origin == "synthetic" else [None] * (3 if origin == "real" else 10)
            for i, category in enumerate(labels):
                sid = f"{family}-{i}"
                fault = origin != "normal"
                values = rng.normal(size=(128, 16)) + (3.0 if fault else 0.0) + (category or 0) * 0.5
                rows[sid] = {"split": "train", "group_id": family, "origin": origin}
                self.batches["detection"].append((sid, family, int(fault), values))
                if category is not None:
                    self.batches["category"].append((sid, family, category, values))
        rows["dev-0"] = {"split": "dev", "group_id": "dev-family", "origin": "normal"}
        self.splits = {"membership": rows}

    def batch(self, task):
        from clauseguard.baselines.data import ExperimentBatch
        from clauseguard.datasets.contracts import NUMERIC_CHANNELS, ModelInputs, NumericInputs
        items = self.batches[task]
        return ExperimentBatch(
            cohort_id="c", scenario_ids=tuple(i[0] for i in items), group_ids=tuple(i[1] for i in items),
            targets=tuple(i[2] for i in items),
            inputs=tuple(ModelInputs(kpi=NumericInputs(channel_names=list(NUMERIC_CHANNELS),
                                                       sampling_rate_hz=10.0, values=i[3].tolist()))
                         for i in items))

    def load_scenario(self, sid):
        return self.real.load_scenario(self.template)

    def check_source(self):
        return None


def test_cross_fit_refuses_a_single_training_family(dataset, selection, tmp_path):
    with pytest.raises(ValueError, match="number of training families"):
        cross_fit.cross_fit(dataset, selection, tmp_path / "cross", folds=2)


def test_cross_fit_excludes_held_out_and_reserved_families_and_reproduces(
        dataset, selection, tmp_path, monkeypatch):
    families = _Families(dataset, selection["release_id"])
    monkeypatch.setattr(cross_fit, "split_data", lambda data, task, split: (None, families.batch(task)))
    output = tmp_path / "cross"
    summary = cross_fit.cross_fit(families, selection, output, folds=3)
    assert summary["family_folds"] == {"jam": 0, "norm-a": 1, "norm-b": 2,
                                       "synth-a": 0, "synth-b": 1, "synth-c": 2}
    assert summary["reserved_families"] == ["dev-family"]
    records = detector.read_cases(output / "predictions.jsonl")
    train = families.batch("detection")
    assert sorted(r.scenario_id for r in records) == sorted(train.scenario_ids)
    for record in records:
        family = families.splits["membership"][record.scenario_id]["group_id"]
        assert record.fold == summary["family_folds"][family]
        assert record.purpose == "slm_training_candidate"
        assert record.detection.prediction.status == "ok"
        assert record.detection.prediction.model_run_id.startswith(
            f"{selection['models']['detection']['model']}/detection/fold-{record.fold}@")
    for row in summary["fold_models"]:
        for task in ("detection", "category"):
            info = row[task]
            assert info["status"] == "fitted"
            assert set(info["training_families"]).isdisjoint([*row["held_out_families"], "dev-family"])
            batch = families.batch(task)
            fitted = [x for x, g in zip(batch.inputs, batch.group_ids, strict=True)
                      if g in info["training_families"]]
            assert info["training_windows"] == len(fitted)
            assert info["channel_medians"] == ChannelMedianFeatures().fit(fitted).medians_.tolist()
    assert summary["separation"]["predictions_checked"] == len(records)
    assert cross_fit.reproduce_cross_fit(families, selection, output) == {
        "reproduced": len(records), "windows": len(records)}
    with pytest.raises(FileExistsError):
        cross_fit.cross_fit(families, selection, output, folds=3)
    (output / "predictions.jsonl").write_text((output / "predictions.jsonl").read_text() + "\n")
    with pytest.raises(ValueError, match="altered"):
        cross_fit.reproduce_cross_fit(families, selection, output)


def test_cross_fit_records_unavailable_fold_models(dataset, selection, tmp_path, monkeypatch):
    families = _Families(dataset, selection["release_id"])
    families.batches["category"] = [i for i in families.batches["category"] if i[1] == "synth-a"]
    monkeypatch.setattr(cross_fit, "split_data", lambda data, task, split: (None, families.batch(task)))
    summary = cross_fit.cross_fit(families, selection, tmp_path / "cross", folds=3)
    fold0 = summary["fold_models"][0]["category"]
    assert fold0["status"] == "unavailable" and "no training windows" in fold0["reason"]
    records = [r for r in detector.read_cases(tmp_path / "cross" / "predictions.jsonl") if r.fold == 0]
    called = [r for r in records if r.detection.prediction.predicted_label_id == 1]
    assert called and all(r.category.prediction.absence_reason.startswith("model unavailable:")
                          and r.diagnosis.prediction.status == "absent" for r in called)
    with pytest.raises(ValueError, match="evaluation families"):
        families.splits["membership"]["dev-0"]["group_id"] = "jam"
        cross_fit.cross_fit(families, selection, tmp_path / "cross2", folds=3)


def test_cli_freeze_predict_and_reproduce(dataset, experiments, tmp_path, monkeypatch, capsys):
    _, dev = split_data(dataset, "detection", "dev")
    rows = [SimpleNamespace(id=f"rca_{i}", scenario_id=sid) for i, sid in enumerate(dev.scenario_ids[:3])]
    monkeypatch.setattr("clauseguard.eval.golden.load_rca", lambda path: rows)
    release = ["--release-dir", str(dataset.release_dir)]
    selection = tmp_path / "cli-selection.json"
    assert cli.main(["freeze", *release, "--frozen-on", "2026-11-09", "--output", str(selection),
                     *[f"--candidate={k}={v}" for k, v in experiments.items()]]) == 0
    output = tmp_path / "cli-predictions"
    assert cli.main(["predict", *release, "--selection", str(selection), "--skip-cross-fit",
                     "--output-dir", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    assert [row["case_id"] for row in report["rca"]["case_table"]] == ["rca_0", "rca_1", "rca_2"]
    assert "Selected pipeline" in (output / "report.md").read_text()
    capsys.readouterr()
    assert cli.main(["reproduce", *release, "--selection", str(selection), "--output-dir", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["rca_cases_reproduced"] == 3
    assert cli.main(["predict", *release, "--selection", str(selection), "--output-dir", str(output)]) == 1
    assert "output exists" in capsys.readouterr().err
