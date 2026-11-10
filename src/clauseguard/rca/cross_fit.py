"""Out-of-fold detector predictions for SLM training candidates (Task 4.2).

Every training-split window is predicted by fold models that never saw its recording family.
Each fold refits the frozen selection (same configuration, seed and, for XGBoost, the recorded
round count) and its own channel medians on the permitted families only. Development and test
families are reserved for evaluation and never enter fitting.
"""

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from clauseguard.baselines.tree_data import split_data
from clauseguard.kpi import random_forest
from clauseguard.kpi import xgboost as xgb
from clauseguard.kpi.features import ChannelMedianFeatures
from clauseguard.rca.detector import (
    CONFIDENCE_SOURCES,
    Head,
    case_record,
    rca_text_eligibility,
    read_cases,
    run_pipeline,
    write_jsonl,
)

FOLDS = 5
CROSS_FIT_VERSION = "detector-cross-fit-v1"
ORIGIN_ORDER = {"synthetic": 0, "real": 1, "normal": 2}
FOLD_RULE = ("training families are ordered by origin (synthetic, real, normal) and then by "
             "family ID, and dealt round-robin into the folds, so faults spread across folds")
SUFFIXES = {"xgboost": ".ubj", "random_forest": ".joblib"}
CANDIDATE_NOTE = ("unresolved recording-family relationships keep this candidate exploratory "
                  "and ineligible for the approved training release")


def family_origins(dataset, split: str = "train") -> dict[str, str]:
    origins = defaultdict(set)
    for row in dataset.splits["membership"].values():
        if row["split"] == split:
            origins[row["group_id"]].add(row["origin"])
    return {family: "+".join(sorted(values)) for family, values in origins.items()}


def assign_folds(origins: Mapping[str, str], folds: int = FOLDS) -> dict[str, int]:
    if not 2 <= folds <= len(origins):
        raise ValueError("folds must be between 2 and the number of training families")
    order = sorted(origins, key=lambda family: (ORIGIN_ORDER.get(origins[family], len(ORIGIN_ORDER)),
                                                family))
    return {family: index % folds for index, family in enumerate(order)}


def reserved_families(dataset, case_ids: Iterable[str] = ()) -> set[str]:
    """Development and test families plus the families of every Task 4 evaluation case."""
    membership = dataset.splits["membership"]
    reserved = {row["group_id"] for row in membership.values() if row["split"] in ("dev", "test")}
    return reserved | {membership[sid]["group_id"] for sid in case_ids}


def _fit_fold(task: str, entry: Mapping, inputs, targets):
    if not inputs:
        raise ValueError("no training windows remain in the permitted families")
    features = ChannelMedianFeatures().fit(inputs)
    matrix = features.transform(inputs)
    if entry["model"] == "xgboost":
        booster = xgb.fit_fixed_rounds(matrix, targets, task=task, configuration=entry["configuration"],
                                       seed=entry["seed"], rounds=entry["refit"]["rounds"])
        model = xgb.XGBoostModel(task, features, booster, dict(entry["configuration"]), entry["seed"])
        content, checksum = xgb.serialize_model(model)
        return model, model.predict_proba, content, checksum
    forest = random_forest.fit_forest(matrix, targets, task=task, configuration=entry["configuration"],
                                      seed=entry["seed"])
    model = random_forest.RandomForestModel(task, features, forest, dict(entry["configuration"]), entry["seed"])
    content, checksum = random_forest.serialize_model(model)
    return model, lambda inputs: forest.predict_proba(features.transform(inputs)), content, checksum


def _load_fold(task: str, entry: Mapping, info: Mapping, path: Path):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != info["checkpoint_sha256"]:
        raise ValueError(f"fold checkpoint checksum mismatch: {path}")
    if entry["model"] == "xgboost":
        booster = XGBClassifier(n_jobs=1, device="cpu")
        booster.load_model(bytearray(content))
        native = booster.get_booster()
        if (native.num_boosted_rounds() != entry["refit"]["rounds"]
                or native.feature_names != list(ChannelMedianFeatures.feature_names_)):
            raise ValueError(f"fold checkpoint rounds or feature order differ: {path}")
        features = ChannelMedianFeatures()
        features.medians_ = np.asarray(info["channel_medians"], dtype=np.float64)
        model = xgb.XGBoostModel(task, features, booster, dict(entry["configuration"]), entry["seed"])
        return model, model.predict_proba
    model = random_forest.load_model(path, expected_sha256=info["checkpoint_sha256"])
    if model.features.medians_.tolist() != info["channel_medians"]:
        raise ValueError(f"fold preprocessing differs from its record: {path}")
    return model, lambda inputs: model.forest.predict_proba(model.features.transform(inputs))


def _head(task, entry, fold, model, proba, checksum) -> Head:
    return Head(task, entry["model"], f"{entry['model']}/{task}/fold-{fold}@{checksum[:12]}",
                checksum, model.predict, proba)


def cross_fit(dataset, selection: Mapping, output_dir: Path, *, folds: int = FOLDS,
              case_ids: Iterable[str] = ()) -> dict:
    """Fit fold models on permitted families and predict each held-out fold."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("output exists; choose a new cross-fit directory")
    if selection["release_id"] != dataset.release_id:
        raise ValueError("selection was frozen for a different release")
    _, detection = split_data(dataset, "detection", "train")
    _, category = split_data(dataset, "category", "train")
    origins = family_origins(dataset)
    reserved = reserved_families(dataset, case_ids)
    if reserved & set(origins):
        raise ValueError("evaluation families appear in the training split")
    fold_of = assign_folds(origins, folds)
    if set(detection.group_ids) - set(fold_of) or set(category.group_ids) - set(fold_of):
        raise ValueError("a training window belongs to a family without a fold")
    output_dir.mkdir(parents=True)
    records, fold_rows = [], []
    for fold in range(folds):
        heads, unavailable, row = {}, {}, {
            "fold": fold, "held_out_families": sorted(f for f, k in fold_of.items() if k == fold)}
        for task, batch in (("detection", detection), ("category", category)):
            entry = selection["models"][task]
            keep = [i for i, group in enumerate(batch.group_ids) if fold_of[group] != fold]
            info = {"training_families": sorted({batch.group_ids[i] for i in keep}),
                    "training_windows": len(keep), "refit": entry["refit"]}
            try:
                model, proba, content, checksum = _fit_fold(
                    task, entry, [batch.inputs[i] for i in keep], [batch.targets[i] for i in keep])
            except ValueError as exc:
                heads[task], unavailable[task] = None, str(exc)
                row[task] = {**info, "status": "unavailable", "reason": str(exc)}
                continue
            path = output_dir / f"fold-{fold}" / f"{task}{SUFFIXES[entry['model']]}"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(content)
            heads[task] = _head(task, entry, fold, model, proba, checksum)
            row[task] = {**info, "status": "fitted", "checkpoint": str(path.relative_to(output_dir)),
                         "checkpoint_sha256": checksum,
                         "channel_medians": model.features.medians_.tolist(),
                         "confidence_source": CONFIDENCE_SOURCES[entry["model"]]}
        held = [i for i, group in enumerate(detection.group_ids) if fold_of[group] == fold]
        ids = [detection.scenario_ids[i] for i in held]
        results = run_pipeline(heads, ids, [detection.inputs[i] for i in held], unavailable=unavailable)
        row["predicted_windows"] = len(ids)
        fold_rows.append(row)
        records.extend(case_record(
            sid, result, release_id=dataset.release_id, purpose="slm_training_candidate",
            selection_sha256=selection["manifest_sha256"], fold=fold,
            text=rca_text_eligibility(dataset.load_scenario(sid)),
            readiness={"evaluation_status": selection["readiness"]["evaluation_status"],
                       "conditions": [*selection["readiness"]["conditions"], CANDIDATE_NOTE]})
            for sid, result in zip(ids, results, strict=True))
    predictions_sha256 = write_jsonl(output_dir / "predictions.jsonl", records)
    summary = {
        "version": CROSS_FIT_VERSION, "release_id": dataset.release_id,
        "selection_sha256": selection["manifest_sha256"], "folds": folds, "fold_rule": FOLD_RULE,
        "family_folds": dict(sorted(fold_of.items())), "family_origins": dict(sorted(origins.items())),
        "reserved_families": sorted(reserved), "fold_models": fold_rows,
        "predictions": "predictions.jsonl", "predictions_sha256": predictions_sha256,
        "windows": len(records),
    }
    summary["separation"] = check_separation(summary, records, dataset)
    (output_dir / "folds.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    dataset.check_source()
    return summary


def check_separation(summary: Mapping, records, dataset) -> dict:
    """No family in two folds, no prediction from a model that saw its family, no reserved family fitted."""
    membership = dataset.splits["membership"]
    held = [family for row in summary["fold_models"] for family in row["held_out_families"]]
    if len(held) != len(set(held)) or set(held) != set(summary["family_folds"]):
        raise ValueError("every training family must be held out in exactly one fold")
    reserved = set(summary["reserved_families"])
    by_fold = {row["fold"]: row for row in summary["fold_models"]}
    checked = 0
    for record in records:
        family = membership[record.scenario_id]["group_id"]
        row = by_fold[record.fold]
        if summary["family_folds"][family] != record.fold:
            raise ValueError(f"{record.scenario_id} predicted by the wrong fold")
        for task in ("detection", "category"):
            fitted = set(row[task]["training_families"])
            if family in fitted or fitted & reserved:
                raise ValueError(f"fold {record.fold} {task} model saw a held-out or reserved family")
        checked += 1
    return {"predictions_checked": checked, "families": len(held),
            "reserved_families_fitted": 0, "held_out_family_in_training": 0}


def reproduce_cross_fit(dataset, selection: Mapping, output_dir: Path) -> dict:
    """Reload saved fold checkpoints and medians, re-predict, and compare every saved record."""
    output_dir = Path(output_dir)
    summary = json.loads((output_dir / "folds.json").read_text(encoding="utf-8"))
    if summary["selection_sha256"] != selection["manifest_sha256"]:
        raise ValueError("cross-fit outputs belong to a different selection")
    content = (output_dir / summary["predictions"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != summary["predictions_sha256"]:
        raise ValueError("saved cross-fit predictions were altered")
    saved = read_cases(output_dir / summary["predictions"])
    _, detection = split_data(dataset, "detection", "train")
    inputs = dict(zip(detection.scenario_ids, detection.inputs, strict=True))
    identical = 0
    for row in summary["fold_models"]:
        fold = row["fold"]
        heads, unavailable = {}, {}
        for task in ("detection", "category"):
            entry, info = selection["models"][task], row[task]
            if info["status"] != "fitted":
                heads[task], unavailable[task] = None, info["reason"]
                continue
            model, proba = _load_fold(task, entry, info, output_dir / info["checkpoint"])
            heads[task] = _head(task, entry, fold, model, proba, info["checkpoint_sha256"])
        cases = [record for record in saved if record.fold == fold]
        results = run_pipeline(heads, [r.scenario_id for r in cases],
                               [inputs[r.scenario_id] for r in cases], unavailable=unavailable)
        for record, (det, cat, diag) in zip(cases, results, strict=True):
            if (det, cat, diag) != (record.detection, record.category, record.diagnosis):
                raise ValueError(f"reloaded fold {fold} prediction differs for {record.scenario_id}")
            identical += 1
    return {"reproduced": identical, "windows": len(saved)}
