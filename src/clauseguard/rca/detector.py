"""Frozen Task 3 detector pipeline mapped into the RCA contract (Task 4.2).

Detection runs on every KPI window. The category head runs only when detection returns
``abnormal``. The combined diagnosis is ``0`` for normal and ``category + 1`` otherwise, the
mapping used by ``score_combined_diagnosis``. Reference labels never reach these functions:
heads receive ``ModelInputs`` only, and saved records carry predictions, not targets.
"""

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from clauseguard.baselines.artifacts import load_run
from clauseguard.baselines.contracts import CasePrediction
from clauseguard.baselines.final_evaluation import find_test_exposure
from clauseguard.datasets.contracts import (
    CATEGORY_LABELS,
    LABEL_MAPPINGS,
    NUMERIC_CHANNELS,
    ModelInputs,
)
from clauseguard.datasets.evidence import text_eligibility
from clauseguard.datasets.schema import ScenarioRecord
from clauseguard.kpi import random_forest
from clauseguard.kpi import xgboost as xgb
from clauseguard.kpi.features import ChannelMedianFeatures
from clauseguard.rca.contracts import DetectorPrediction

SELECTION_VERSION = "detector-selection-v1"
RECORD_VERSION = "detector-case-v1"
SELECTION_SEED = 42
SUPPORTED_MODELS = ("xgboost", "random_forest")
PIPELINE = {
    "version": "detection-then-category-v1",
    "detection": "called for every KPI window",
    "category": "called only when detection returns abnormal (1)",
    "diagnosis": "0 (normal) when detection is normal, otherwise category label + 1",
    "unsupported_faults": "the category head covers the ten synthetic categories only; a category "
                          "prediction for any other fault, such as real jamming, is still recorded "
                          "as a prediction and never as a physical cause",
}
CHECKPOINT_FILES = {"xgboost": "model.ubj", "random_forest": "model.joblib"}
DEVELOPMENT_USE = {"xgboost": ["early_stopping", "configuration_selection"],
                   "random_forest": ["configuration_selection"]}
CONFIDENCE_SOURCES = {"xgboost": "xgboost predict_proba of the predicted class",
                      "random_forest": "random forest predict_proba (tree vote share) of the predicted class"}
DATASET_COMPATIBILITY = {
    "TelecomTS": {
        "status": "implemented",
        "detail": "128-step, 16-channel, 10 Hz windows from the checked release; the selected "
                  "checkpoints were fitted on this release",
    },
    "BubbleRAN": {
        "status": "absent",
        "reason": "MAC-layer KPIs at 1 Hz with different channel names, cumulative counters and "
                  "unverified units; no reviewed mapping to the 16 TelecomTS channels or their "
                  "preprocessing exists",
    },
    "SpotLight": {
        "status": "absent",
        "reason": "radio and platform streams with unverified capture alignment and different "
                  "measurements; no compatible 16-channel window exists",
    },
    "TeleLogs": {
        "status": "absent",
        "reason": "no compatible detector or KPI window input; source files are not available locally",
    },
    "5G core faults": {
        "status": "absent",
        "reason": "text-only cases; no KPI input exists for a detector",
    },
}


def label_names(task: str) -> tuple[str, ...]:
    if task == "detection":
        mapping = LABEL_MAPPINGS["detection"]
        return tuple(sorted(mapping, key=mapping.get))
    if task == "category":
        return tuple(CATEGORY_LABELS)
    if task == "diagnosis":
        return ("normal", *CATEGORY_LABELS)
    raise ValueError(f"unknown detector task: {task}")


def resolve_label(task: str, label_id: Any) -> str | None:
    names = label_names(task)
    if isinstance(label_id, bool) or not isinstance(label_id, int) or not 0 <= label_id < len(names):
        return None
    return names[label_id]


def to_detector_prediction(prediction: CasePrediction, *, run_id: str,
                           confidence: float | None = None) -> DetectorPrediction:
    """Map a Task 3 ``CasePrediction`` into the Task 4.1 contract; failures stay failures."""
    common = {"task": prediction.task, "label_mapping_version": LABEL_MAPPINGS["version"],
              "model_run_id": run_id, "source_modalities": ["kpi"]}
    if prediction.status == "error":
        return DetectorPrediction(status="error", **common)
    name = resolve_label(prediction.task, prediction.predicted_label)
    if prediction.status != "ok" or name is None:
        return DetectorPrediction(status="invalid", **common)
    return DetectorPrediction(status="ok", predicted_label_id=prediction.predicted_label,
                              predicted_label=name, confidence=confidence, **common)


def absent_prediction(task: str, reason: str) -> DetectorPrediction:
    return DetectorPrediction(task=task, status="absent", absence_reason=reason)


def dataset_absence(dataset: str) -> dict[str, DetectorPrediction]:
    """Absent predictions for a source without a compatible detector."""
    entry = DATASET_COMPATIBILITY.get(dataset)
    if entry is None:
        raise ValueError(f"unknown dataset: {dataset}")
    if entry["status"] != "absent":
        raise ValueError(f"{dataset} has a compatible detector; run the pipeline instead")
    return {task: absent_prediction(task, entry["reason"])
            for task in ("detection", "category", "diagnosis")}


class PredictionRecord(BaseModel):
    """One head's output with the identity needed to reproduce it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction: DetectorPrediction
    error: str | None = None
    run: str | None = None
    checkpoint_sha256: str | None = None
    confidence_source: str | None = None
    calibration_status: Literal["uncalibrated", "not_available"] = "not_available"


class TextEligibility(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: Literal["eligible", "missing", "excluded"]
    origin: str
    reason: str | None = None


class CaseRecord(BaseModel):
    """Saved detector output for one case; it carries no reference label or split field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_version: Literal["detector-case-v1"] = RECORD_VERSION
    dataset: str = "TelecomTS"
    release_id: str
    scenario_id: str = Field(min_length=1)
    purpose: Literal["rca_evaluation", "slm_training_candidate"]
    selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold: int | None = None
    detection: PredictionRecord
    category: PredictionRecord
    diagnosis: PredictionRecord
    text: TextEligibility
    readiness: Literal["provisional_exploratory", "scored"]
    readiness_reasons: list[str] = Field(default_factory=list)


def rca_text_eligibility(record: ScenarioRecord) -> TextEligibility:
    """Text needs an explicit RCA permission; detection/category approval is not enough."""
    review = record.text_review
    if review.status != "approved" or not record.nl_context.strip():
        return TextEligibility(state="missing", origin=review.text_origin, reason="no_approved_text")
    reason = (text_eligibility(record, review.approved_tasks[0]) if review.approved_tasks
              else "text_not_approved_for_task")
    # TextReview records detection/category approval only; it has no RCA permission yet.
    return TextEligibility(state="excluded", origin=review.text_origin,
                           reason=reason or "no_rca_permission")


@dataclass
class Head:
    """A loaded detector head: labels and class probabilities from observations only."""

    task: str
    model_name: str
    run_id: str
    checkpoint_sha256: str
    predict: Callable[[Sequence[ModelInputs]], list[int]]
    proba: Callable[[Sequence[ModelInputs]], np.ndarray]


def predict_head(head: Head, scenario_ids: Sequence[str],
                 inputs: Sequence[ModelInputs]) -> list[PredictionRecord]:
    """Predict a batch; any model failure is kept per case instead of dropping the cases."""
    common = {"run": head.run_id, "checkpoint_sha256": head.checkpoint_sha256}
    if not scenario_ids:
        return []

    def failed(message):
        return [PredictionRecord(prediction=to_detector_prediction(
                    CasePrediction(scenario_id=sid, task=head.task, status="error", error=message),
                    run_id=head.run_id), error=message, **common)
                for sid in scenario_ids]

    try:
        labels = list(head.predict(list(inputs)))
        probabilities = np.asarray(head.proba(list(inputs)), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - a failed call is recorded, never dropped.
        return failed(f"{type(exc).__name__}: {exc}")
    if len(labels) != len(scenario_ids) or len(probabilities) != len(scenario_ids):
        return failed(f"head returned {len(labels)} outputs for {len(scenario_ids)} cases")
    records = []
    for sid, label, row in zip(scenario_ids, labels, probabilities, strict=True):
        status = "ok" if isinstance(label, int) and not isinstance(label, bool) else "invalid"
        prediction = CasePrediction(scenario_id=sid, task=head.task, predicted_label=label, status=status)
        confidence = None
        if status == "ok" and 0 <= label < len(row) and np.isfinite(row[label]) and 0 <= row[label] <= 1:
            confidence = float(row[label])
        mapped = to_detector_prediction(prediction, run_id=head.run_id, confidence=confidence)
        records.append(PredictionRecord(
            prediction=mapped, **common,
            confidence_source=CONFIDENCE_SOURCES[head.model_name] if mapped.confidence is not None else None,
            calibration_status="uncalibrated" if mapped.confidence is not None else "not_available"))
    return records


def _diagnosis(detection: PredictionRecord, category: PredictionRecord) -> PredictionRecord:
    runs = "+".join(r.prediction.model_run_id for r in (detection, category) if r.prediction.model_run_id)

    def derived(status, label_id=None, source=None):
        return PredictionRecord(prediction=DetectorPrediction(
            task="diagnosis", status=status, predicted_label_id=label_id,
            predicted_label=resolve_label("diagnosis", label_id) if label_id is not None else None,
            label_mapping_version=LABEL_MAPPINGS["version"],
            model_run_id=f"{PIPELINE['version']}:{runs}" if runs else PIPELINE["version"],
            absence_reason=source.absence_reason if source is not None else None,
            source_modalities=["kpi"]), error=detection.error or category.error)

    det, cat = detection.prediction, category.prediction
    if det.status != "ok":
        return derived(det.status, source=det)
    if det.predicted_label_id == 0:
        return derived("ok", 0)
    if cat.status != "ok":
        return derived(cat.status, source=cat)
    return derived("ok", cat.predicted_label_id + 1)


def _predict_or_absent(head: Head | None, task: str, ids, inputs, reason: str | None):
    if head is not None:
        return predict_head(head, ids, inputs)
    return [PredictionRecord(prediction=absent_prediction(task, f"model unavailable: {reason}"))
            for _ in ids]


def run_pipeline(heads: Mapping[str, Head | None], scenario_ids: Sequence[str],
                 inputs: Sequence[ModelInputs], *, unavailable: Mapping[str, str] | None = None,
                 ) -> list[tuple[PredictionRecord, PredictionRecord, PredictionRecord]]:
    """Detection for every window, category for windows detected as abnormal, then diagnosis.

    A head that could not be fitted (``None``) yields absent predictions with the recorded
    reason, which stay distinct from failed calls.
    """
    unavailable = unavailable or {}
    detections = _predict_or_absent(heads["detection"], "detection", scenario_ids, inputs,
                                    unavailable.get("detection"))
    abnormal = [i for i, record in enumerate(detections)
                if record.prediction.status == "ok" and record.prediction.predicted_label_id == 1]
    called = dict(zip(abnormal, _predict_or_absent(
        heads["category"], "category", [scenario_ids[i] for i in abnormal],
        [inputs[i] for i in abnormal], unavailable.get("category")), strict=True))
    results = []
    for i, detection in enumerate(detections):
        if i in called:
            category = called[i]
        else:
            reason = ("detection returned normal; category head not called"
                      if detection.prediction.status == "ok"
                      else f"detection {detection.prediction.status}; category head not called")
            category = PredictionRecord(prediction=absent_prediction("category", reason))
        results.append((detection, category, _diagnosis(detection, category)))
    return results


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _seal(value: dict) -> dict:
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()
    return {**body, "manifest_sha256": digest}


def _dev_score(task_entry: Mapping[str, Any]) -> float | None:
    for key in ("dev_macro_f1_mean", "dev_macro_f1"):
        if isinstance(task_entry.get(key), (int, float)):
            return float(task_entry[key])
    return None


def _candidate_scores(dataset, candidates: Mapping[str, Path], task: str) -> list[dict]:
    rows = []
    for name, directory in candidates.items():
        selection = json.loads((Path(directory) / "selection.json").read_text(encoding="utf-8"))
        if selection.get("release_id") != dataset.release_id:
            raise ValueError(f"{name} belongs to a different release")
        if selection.get("selection_split") != "dev":
            raise ValueError(f"{name} was not selected on the development split")
        entry = selection.get("tasks", {}).get(task)
        score = _dev_score(entry) if entry else None
        if score is not None:
            rows.append({"model": name, "experiment": str(directory), "dev_macro_f1": score})
    return rows


def _family_ids(dataset, scenario_ids: Iterable[str]) -> list[str]:
    membership = dataset.splits["membership"]
    return sorted({membership[sid]["group_id"] for sid in scenario_ids})


def _selected_model(dataset, name: str, directory: Path, task: str) -> dict:
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    seeds = selection["tasks"][task]["selected_seeds"]
    row = next((item for item in seeds if item["seed"] == SELECTION_SEED), None)
    if row is None:
        raise ValueError(f"{name} {task} has no seed-{SELECTION_SEED} run")
    run = directory / row["run"]
    metadata = load_run(run)["metadata"]
    checkpoint = run / CHECKPOINT_FILES[name]
    if (metadata.task != task or metadata.seed != SELECTION_SEED
            or metadata.cohort.release_id != dataset.release_id or metadata.cohort.split != "dev"):
        raise ValueError(f"{run} differs in task, seed, release or development cohort")
    # XGBoost runs record their mapping; Random Forest runs are checked through the forest's classes.
    if metadata.transformations.get("class_mapping", LABEL_MAPPINGS[task]) != LABEL_MAPPINGS[task]:
        raise ValueError(f"{run} label mapping differs from the release")
    if _sha256_file(checkpoint) != metadata.checkpoint:
        raise ValueError(f"{checkpoint} checksum differs from its run metadata")
    training_ids = json.loads((directory / task / "training_ids.json").read_text(encoding="utf-8"))
    if hashlib.sha256(json.dumps(sorted(training_ids), separators=(",", ":")).encode()).hexdigest() \
            != metadata.transformations["training_ids_sha256"]:
        raise ValueError(f"{name} {task} training IDs differ from the run metadata")
    configuration = {key: metadata.configuration[key]
                     for key in (("max_depth", "min_child_weight") if name == "xgboost"
                                 else ("max_depth", "min_samples_leaf"))}
    refit = ({"method": "fixed_rounds", "rounds": metadata.configuration["retained_rounds"],
              "uses_validation_data": False} if name == "xgboost"
             else {"method": "refit", "n_estimators": 300, "uses_validation_data": False})
    return {
        "model": name, "task": task, "experiment": str(directory), "run": str(run),
        "configuration": configuration, "seed": SELECTION_SEED, "dev_macro_f1": row["dev_macro_f1"],
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": metadata.checkpoint,
        "preprocessing": {
            "channel_order": list(NUMERIC_CHANNELS),
            "feature_order": metadata.transformations["feature_order"],
            "channel_medians": metadata.transformations["channel_medians"],
            "missing_values": "training-split channel medians fill missing measurements before "
                              "the 14 window statistics; categorical channels are excluded",
            "versions": dict(metadata.versions),
        },
        "data": {"source_revision": dataset.manifest["revision"], "release_id": dataset.release_id,
                 "train_cohort_id": metadata.train_cohort_id,
                 "dev_cohort_id": metadata.cohort.cohort_id,
                 "training_ids_sha256": metadata.transformations["training_ids_sha256"]},
        "training_history": {
            "training_families": _family_ids(dataset, training_ids),
            "development_families": _family_ids(dataset, metadata.cohort.scenario_ids),
            "development_use": DEVELOPMENT_USE[name],
        },
        "label_mapping": {"task": task, "version": LABEL_MAPPINGS["version"],
                          "names": list(label_names(task))},
        "refit": refit,
    }


def freeze_selection(dataset, candidates: Mapping[str, Path], output_path: Path, *,
                     frozen_on: str, exposure_dirs: Sequence[Path] = ()) -> dict:
    """Pick each task's highest mean dev macro-F1 among candidates and seal the choice."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("selection exists; a frozen selection is never overwritten")
    candidates = {name: Path(path) for name, path in candidates.items()}
    models, scores = {}, {}
    for task in ("detection", "category"):
        rows = _candidate_scores(dataset, candidates, task)
        if not rows:
            raise ValueError(f"no candidate reports a {task} development score")
        best = max(rows, key=lambda row: row["dev_macro_f1"])  # ties keep the candidate order
        if best["model"] not in SUPPORTED_MODELS:
            raise ValueError(f"{best['model']} has the best {task} score but no Task 4 adapter")
        scores[task] = rows
        models[task] = _selected_model(dataset, best["model"], candidates[best["model"]], task)
    membership = dataset.splits["membership"]
    unresolved = sorted({row["group_id"] for row in membership.values()
                         if row["evidence_status"] != "verified"})
    mode = dataset.splits["mode"]
    manifest = _seal({
        "version": SELECTION_VERSION, "frozen_on": frozen_on, "release_id": dataset.release_id,
        "source_revision": dataset.manifest["revision"], "pipeline": PIPELINE, "models": models,
        "selection": {
            "rule": "highest mean development macro-F1 per task over the supplied Task 3 "
                    "candidates; ties keep candidate order; seed 42 (the configuration-search "
                    "seed) is used so no seed is chosen by score; test results are never read",
            "candidates": scores,
        },
        "readiness": {
            "evaluation_status": "provisional_exploratory" if mode == "exploratory" else "scored",
            "scored_eligible": mode == "scored" and not unresolved,
            "unresolved_recording_families": len(unresolved),
            "conditions": [
                f"{len(unresolved)} recording families have unresolved relationships",
                "development families were used for model selection"
                + (" and XGBoost early stopping" if "xgboost" in {m["model"] for m in models.values()}
                   else ""),
            ],
        },
        "prior_test_exposure": find_test_exposure(exposure_dirs, dataset.release_id),
        "dataset_compatibility": DATASET_COMPATIBILITY,
        "text_policy": "text enters RCA only with explicit RCA permission; detection/category "
                       "approval is not RCA permission; diagnosis-revealing text is excluded; the "
                       "text-only configuration receives no KPI-derived detector prediction",
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    return manifest


def load_selection(path: Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("version") != SELECTION_VERSION or _seal(manifest) != manifest:
        raise ValueError("selection manifest is not a sealed detector-selection-v1 record")
    return manifest


def check_compatibility(dataset, entry: Mapping[str, Any]) -> None:
    """Reject a selected head whose release, channels, mapping or checkpoint differ."""
    if entry["data"]["release_id"] != dataset.release_id:
        raise ValueError("incompatible release: selection was frozen for a different release")
    if (entry["preprocessing"]["channel_order"] != list(NUMERIC_CHANNELS)
            or dataset.manifest["numeric_channels"] != list(NUMERIC_CHANNELS)):
        raise ValueError("incompatible channel layout")
    if (entry["label_mapping"]["version"] != LABEL_MAPPINGS["version"]
            or entry["label_mapping"]["names"] != list(label_names(entry["task"]))):
        raise ValueError("incompatible label mapping")
    if _sha256_file(Path(entry["checkpoint_path"])) != entry["checkpoint_sha256"]:
        raise ValueError("incompatible checkpoint: checksum differs from the selection")


def run_id(entry: Mapping[str, Any]) -> str:
    return f"{entry['model']}/{entry['task']}/{Path(entry['run']).name}@{entry['checkpoint_sha256'][:12]}"


def load_head(dataset, entry: Mapping[str, Any]) -> Head:
    check_compatibility(dataset, entry)
    features = ChannelMedianFeatures()
    features.feature_names_ = tuple(entry["preprocessing"]["feature_order"])
    features.medians_ = np.asarray(entry["preprocessing"]["channel_medians"], dtype=np.float64)
    if entry["model"] == "xgboost":
        model = xgb.load_model(Path(entry["checkpoint_path"]), expected_sha256=entry["checkpoint_sha256"],
                               task=entry["task"], features=features, seed=entry["seed"],
                               configuration=entry["configuration"])
        proba = model.predict_proba
    elif entry["model"] == "random_forest":
        model = random_forest.load_model(Path(entry["checkpoint_path"]),
                                         expected_sha256=entry["checkpoint_sha256"])
        if (model.task != entry["task"] or model.features.medians_.tolist() != features.medians_.tolist()
                or list(model.features.feature_names_) != list(features.feature_names_)):
            raise ValueError("incompatible checkpoint: task or preprocessing differs from the selection")
        if model.forest.classes_.tolist() != list(range(len(label_names(entry["task"])))):
            raise ValueError("incompatible label mapping: forest classes differ from the task labels")

        def proba(inputs, model=model):
            return model.forest.predict_proba(model.features.transform(inputs))
    else:
        raise ValueError(f"no Task 4 adapter for {entry['model']}")
    return Head(entry["task"], entry["model"], run_id(entry), entry["checkpoint_sha256"],
                model.predict, proba)


def case_record(scenario_id, results, *, release_id, purpose, selection_sha256, text, readiness,
                fold=None) -> CaseRecord:
    detection, category, diagnosis = results
    return CaseRecord(release_id=release_id, scenario_id=scenario_id, purpose=purpose,
                      selection_sha256=selection_sha256, fold=fold, detection=detection,
                      category=category, diagnosis=diagnosis, text=text,
                      readiness=readiness["evaluation_status"],
                      readiness_reasons=readiness["conditions"])


def kpi_inputs(dataset, scenario_ids: Iterable[str]) -> dict[str, ModelInputs]:
    """KPI observations only, read through the release loader's detection view."""
    wanted = set(scenario_ids)
    membership = dataset.splits["membership"]
    found = {}
    for split in sorted({membership[sid]["split"] for sid in wanted}):
        for example in dataset.iter_examples(split=split, task="detection", modality="kpi"):
            if example.scenario_id in wanted:
                found[example.scenario_id] = example.inputs
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"no KPI input for {len(missing)} scenarios, e.g. {min(missing)}")
    return found


def predict_cases(dataset, selection: Mapping[str, Any], scenario_ids: Sequence[str]) -> list[CaseRecord]:
    """Predict RCA evaluation cases with the selected checkpoints."""
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("duplicate RCA case scenario IDs")
    heads = {task: load_head(dataset, selection["models"][task]) for task in ("detection", "category")}
    inputs = kpi_inputs(dataset, scenario_ids)
    results = run_pipeline(heads, list(scenario_ids), [inputs[sid] for sid in scenario_ids])
    return [case_record(sid, result, release_id=dataset.release_id, purpose="rca_evaluation",
                        selection_sha256=selection["manifest_sha256"],
                        text=rca_text_eligibility(dataset.load_scenario(sid)),
                        readiness=selection["readiness"])
            for sid, result in zip(scenario_ids, results, strict=True)]


def saved_run_agreement(records: Sequence[CaseRecord], selection: Mapping[str, Any]) -> dict:
    """Compare against the selected runs' saved predictions for cases those runs scored."""
    summary = {}
    for task in ("detection", "category"):
        saved = {p.scenario_id: p.predicted_label
                 for p in load_run(Path(selection["models"][task]["run"]))["predictions"]}
        rows = [(r.scenario_id, getattr(r, task).prediction) for r in records
                if r.scenario_id in saved and getattr(r, task).prediction.status == "ok"]
        summary[task] = {"compared": len(rows),
                         "identical": sum(saved[sid] == p.predicted_label_id for sid, p in rows)}
    return summary


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(record.model_dump_json() + "\n" for record in records)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest()


def read_cases(path: Path) -> list[CaseRecord]:
    return [CaseRecord.model_validate_json(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
