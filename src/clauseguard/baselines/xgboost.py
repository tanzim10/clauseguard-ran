"""Development-selected XGBoost runs on the exact Random Forest cohorts."""

import hashlib
import json
import platform
import resource
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np

from clauseguard.baselines.artifacts import load_run, write_run
from clauseguard.baselines.contracts import RunMetadata
from clauseguard.baselines.operations import predict_adapter
from clauseguard.baselines.tree_data import (
    ids_hash,
    require_ready,
    select_training,
    split_data,
    training_ids,
)
from clauseguard.datasets.contracts import LABEL_MAPPINGS
from clauseguard.kpi.features import ChannelMedianFeatures
from clauseguard.kpi.random_forest import load_model as load_rf_model
from clauseguard.kpi.xgboost import (
    CONFIGURATIONS,
    XGBoostAdapter,
    XGBoostModel,
    fit_booster,
    load_model,
    serialize_model,
)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _matrix_hash(matrix):
    return hashlib.sha256(np.ascontiguousarray(matrix, dtype="<f8").tobytes()).hexdigest()


def _reference(reference, dataset, task, train_cohort, train, dev_cohort, dev):
    selection = json.loads((reference / "selection.json").read_text())
    if selection["release_id"] != dataset.release_id:
        raise ValueError("RF reference belongs to a different release")
    chosen = selection["tasks"][task]["selected_seeds"][0]
    if chosen["seed"] != 42:
        raise ValueError("RF reference must identify the selected seed-42 run")
    path = reference / chosen["run"]
    saved = load_run(path)
    metadata = saved["metadata"]
    if (metadata.task != task or metadata.seed != 42
            or metadata.train_cohort_id != train_cohort.cohort_id
            or metadata.cohort.cohort_id != dev_cohort.cohort_id):
        raise ValueError("RF reference differs in task, train cohort or development cohort")
    if (metadata.transformations["training_ids_sha256"] != ids_hash(train.scenario_ids)
            or json.loads((reference / task / "training_ids.json").read_text()) != list(train.scenario_ids)):
        raise ValueError("RF reference training IDs differ from XGBoost training selection")
    if saved["expected"] != dict(zip(dev.scenario_ids, dev.targets, strict=True)):
        raise ValueError("RF reference development labels differ")
    model = load_rf_model(path / "model.joblib", expected_sha256=metadata.checkpoint)
    if (model.task != task or model.forest.class_weight != "balanced"
            or model.features.feature_names_ != tuple(metadata.transformations["feature_order"])
            or model.features.medians_.tolist() != metadata.transformations["channel_medians"]):
        raise ValueError("RF reference preprocessing or weighting contract differs")
    return path, metadata, model


def _record_run(path, model, cohort, data, *, train_cohort_id, transformations,
                resources, bootstrap_replicates, training_history=None):
    started = time.perf_counter()
    predictions = predict_adapter(XGBoostAdapter(model.task), model, cohort, [data])
    prediction_seconds = time.perf_counter() - started
    content, checksum = serialize_model(model)
    best_iteration = model.booster.best_iteration
    loss = "logloss" if model.task == "detection" else "mlogloss"
    configuration = {
        **model.configuration, "objective": "binary:logistic" if model.task == "detection" else "multi:softprob",
        "eval_metric": loss, "validation_loss_weighting": "unweighted",
        "sample_weight_policy": "balanced_training_n_over_k_nclass", "learning_rate": 0.05,
        "n_estimators": 500, "early_stopping_rounds": 30, "tree_method": "hist", "device": "cpu",
        "n_jobs": 1, "subsample": 1, "colsample_bytree": 1,
        "best_iteration": best_iteration, "retained_rounds": best_iteration + 1,
        "best_loss": float(model.booster.best_score), "binary_probability_threshold": 0.5,
        "binary_probability_tie_label": 0,
        "resolved_booster_config": json.loads(model.booster.get_booster().save_config()),
    }
    if training_history is not None:
        configuration.update(training_history=training_history,
                             attempted_rounds=len(training_history), hit_round_cap=len(training_history) == 500)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metadata = RunMetadata(
        model=f"xgboost/{model.task}/{path.name}", task=model.task, cohort=cohort,
        train_cohort_id=train_cohort_id, seed=model.seed, checkpoint=checksum,
        transformations=transformations, configuration=configuration,
        versions={"python": platform.python_version(), "platform": platform.platform(),
                  **{name: version(name) for name in ("numpy", "scikit-learn", "xgboost")}},
        resources={**resources, "prediction_seconds": prediction_seconds,
                   "process_peak_rss_bytes": float(peak_rss * (1 if sys.platform == "darwin" else 1024))},
    )
    expected = dict(zip(data.scenario_ids, data.targets, strict=True))
    groups = dict(zip(data.scenario_ids, data.group_ids, strict=True))
    metrics = write_run(path, metadata, expected, groups, predictions,
                        bootstrap_replicates=bootstrap_replicates)
    (path / "model.ubj").write_bytes(content)
    _write_json(path / "errors.json", [
        {"scenario_id": p.scenario_id, "expected": expected[p.scenario_id],
         "predicted": p.predicted_label, "status": p.status, "error": p.error}
        for p in predictions if p.status != "ok" or expected[p.scenario_id] != p.predicted_label
    ])
    return metrics


def _run_task(task, prepared, output_dir, evaluation_status, bootstrap_replicates):
    train_cohort, full_train, train, dev_cohort, dev, reference = prepared
    reference_path, reference_metadata, reference_model = reference
    started = time.perf_counter()
    transformer = ChannelMedianFeatures().fit(train.inputs)
    if (transformer.feature_names_ != reference_model.features.feature_names_
            or not np.array_equal(transformer.medians_, reference_model.features.medians_)):
        raise ValueError("RF and XGBoost fitted preprocessing differs")
    train_features = transformer.transform(train.inputs)
    dev_features = transformer.transform(dev.inputs)
    feature_seconds = time.perf_counter() - started
    started = time.perf_counter()
    if (not np.array_equal(train_features, reference_model.features.transform(train.inputs))
            or not np.array_equal(dev_features, reference_model.features.transform(dev.inputs))):
        raise ValueError("RF and XGBoost feature matrices differ")
    parity_seconds = time.perf_counter() - started
    transformations = {
        "channel_medians": transformer.medians_.tolist(), "feature_order": list(transformer.feature_names_),
        "training_ids_sha256": ids_hash(train.scenario_ids),
        "train_features_sha256": _matrix_hash(train_features), "dev_features_sha256": _matrix_hash(dev_features),
        "reference_rf_run": str(reference_path), "reference_rf_checkpoint": reference_metadata.checkpoint,
        "evaluation_status": evaluation_status,
        "class_mapping": LABEL_MAPPINGS[task],
        "training_class_counts": dict(sorted(Counter(map(str, train.targets)).items())),
    }
    trials, selected = [], []

    def fit_and_record(configuration, seed, path):
        started = time.perf_counter()
        booster = fit_booster(train_features, train.targets, dev_features, dev.targets,
                              task=task, configuration=configuration, seed=seed)
        fit_seconds = time.perf_counter() - started
        model = XGBoostModel(task, transformer, booster, dict(configuration), seed)
        metric = "logloss" if task == "detection" else "mlogloss"
        metrics = _record_run(
            path, model, dev_cohort, dev, train_cohort_id=train_cohort.cohort_id,
            transformations=transformations,
            resources={"fit_seconds": fit_seconds, "feature_seconds": feature_seconds,
                       "feature_parity_seconds": parity_seconds},
            bootstrap_replicates=bootstrap_replicates,
            training_history=booster.evals_result()["validation_0"][metric],
        )
        return {"seed": seed, "dev_macro_f1": metrics["macro_f1"],
                "best_iteration": booster.best_iteration, "run": str(path.relative_to(output_dir))}

    for index, configuration in enumerate(CONFIGURATIONS):
        row = fit_and_record(configuration, 42, output_dir / task / f"config-{index}-seed-42")
        trials.append({**row, "configuration": configuration})
    best = max(trials, key=lambda row: row["dev_macro_f1"])
    selected.append({key: value for key, value in best.items() if key != "configuration"})
    for seed in (43, 44):
        selected.append(fit_and_record(best["configuration"], seed,
                                       output_dir / task / f"selected-seed-{seed}"))
    scores = [row["dev_macro_f1"] for row in selected]
    result = {
        "trials": trials, "selected_configuration": best["configuration"], "selected_seeds": selected,
        "dev_macro_f1_mean": float(np.mean(scores)), "dev_macro_f1_std": float(np.std(scores)),
        "train_windows": len(train.inputs), "full_train_windows": len(full_train.inputs),
        "train_ids_sha256": ids_hash(train.scenario_ids), "dev_cohort_id": dev_cohort.cohort_id,
    }
    _write_json(output_dir / task / "training_ids.json", list(train.scenario_ids))
    return result


def run_xgboost(dataset, output_dir, *, rf_reference, tasks=("detection", "category"),
                training_selection=None, bootstrap_replicates=1000, allow_exploratory=False):
    """Fit four configurations on train, choose on dev, then repeat seeds 43/44."""
    tasks = tuple(tasks)
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("choose one or both distinct tasks")
    require_ready(dataset, tasks, allow_exploratory=allow_exploratory)
    output_dir, rf_reference = Path(output_dir), Path(rf_reference)
    if output_dir.exists():
        raise FileExistsError("output exists; choose a new experiment directory")
    selected_ids = training_ids(dataset, training_selection)
    prepared = {}
    for task in tasks:
        train_cohort, full_train = split_data(dataset, task, "train")
        train = select_training(full_train, selected_ids)
        dev_cohort, dev = split_data(dataset, task, "dev")
        reference = _reference(rf_reference, dataset, task, train_cohort, train, dev_cohort, dev)
        prepared[task] = train_cohort, full_train, train, dev_cohort, dev, reference
    output_dir.mkdir(parents=True)
    summary = {"release_id": dataset.release_id, "selection_split": "dev", "tasks": {},
               "evaluation_status": "provisional_exploratory" if dataset.splits["mode"] == "exploratory" else "scored",
               "rf_reference": str(rf_reference)}
    for task, task_data in prepared.items():
        summary["tasks"][task] = _run_task(task, task_data, output_dir,
                                           summary["evaluation_status"], bootstrap_replicates)
        _write_json(output_dir / "selection.json", summary)
    dataset.check_source()
    return summary


def predict_saved_model(dataset, model_run_dir, output_dir, *, split="test",
                         bootstrap_replicates=1000, allow_exploratory=False):
    """Reload native trees and saved preprocessing without fitting on the scored split."""
    path = Path(model_run_dir)
    metadata = load_run(path)["metadata"]
    require_ready(dataset, (metadata.task,), allow_exploratory=allow_exploratory)
    if metadata.cohort.release_id != dataset.release_id:
        raise ValueError("checkpoint belongs to a different release")
    if metadata.transformations["class_mapping"] != LABEL_MAPPINGS[metadata.task]:
        raise ValueError("checkpoint class mapping differs from release")
    transformer = ChannelMedianFeatures()
    transformer.feature_names_ = tuple(metadata.transformations["feature_order"])
    transformer.medians_ = np.asarray(metadata.transformations["channel_medians"], dtype=np.float64)
    model = load_model(
        path / "model.ubj", expected_sha256=metadata.checkpoint, task=metadata.task,
        features=transformer, seed=metadata.seed,
        configuration={key: metadata.configuration[key] for key in ("max_depth", "min_child_weight")},
    )
    cohort, data = split_data(dataset, metadata.task, split)
    return _record_run(Path(output_dir), model, cohort, data, train_cohort_id=metadata.train_cohort_id,
                       transformations=metadata.transformations, resources={},
                       bootstrap_replicates=bootstrap_replicates,
                       training_history=metadata.configuration.get("training_history"))
