"""Development-selected PatchTST runs on the exact XGBoost/RF/TimesNet cohorts (Task 3.11)."""

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
from clauseguard.baselines.comparison import compare_experiments, render_markdown
from clauseguard.baselines.contracts import RunMetadata
from clauseguard.baselines.operations import predict_adapter
from clauseguard.baselines.timesnet import _reference
from clauseguard.baselines.tree_data import ids_hash, require_ready, split_data
from clauseguard.datasets.contracts import LABEL_MAPPINGS
from clauseguard.kpi.patchtst import (
    ARCHITECTURE,
    CONFIGURATIONS,
    GRADIENT_CLIP,
    INFERENCE_BATCH_SIZE,
    RELOAD_TOLERANCE,
    STRIDES,
    TRAIN_BATCH_SIZE,
    UPSTREAM_COMMIT,
    WEIGHT_DECAY,
    PatchTSTAdapter,
    PatchTSTModel,
    fit_patchtst,
    load_model,
    patch_count,
    predict_logits,
    serialize_model,
)
from clauseguard.kpi.timesnet import WindowScaler


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _record_run(path, model, cohort, data, *, train_cohort_id, transformations, resources,
                bootstrap_replicates, training):
    """Save one run, then reload the weights and require the same predictions."""
    started = time.perf_counter()
    predictions = predict_adapter(PatchTSTAdapter(model.task), model, cohort, [data])
    prediction_seconds = time.perf_counter() - started
    content, checksum = serialize_model(model)
    windows = model.scaler.transform(data.inputs)
    logits = predict_logits(model.net, windows, batch_size=model.batch_size, device=model.device)
    reloaded = load_model(content, expected_sha256=checksum, task=model.task, scaler=model.scaler,
                          configuration=model.configuration, seed=model.seed, device=model.device,
                          batch_size=model.batch_size)
    replay = predict_logits(reloaded.net, windows, batch_size=model.batch_size, device=model.device)
    if not (np.array_equal(logits.argmax(1), replay.argmax(1))
            and np.allclose(logits, replay, **RELOAD_TOLERANCE)):
        raise ValueError("PatchTST predictions changed after reload")
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    patch = model.configuration["patch_length"]
    configuration = {
        **model.configuration, **ARCHITECTURE, "upstream_commit": UPSTREAM_COMMIT,
        "stride": STRIDES[patch], "patches_per_channel": patch_count(patch, STRIDES[patch]),
        "optimizer": "AdamW", "weight_decay": WEIGHT_DECAY, "gradient_clip": GRADIENT_CLIP,
        "train_batch_size": TRAIN_BATCH_SIZE, "inference_batch_size": model.batch_size,
        "loss_weighting": "balanced_training_n_over_k_weighted_mean",
        "validation_loss_weighting": "unweighted", "instance_norm": False,
        "device": model.device, "deterministic_algorithms": model.device == "cpu",
        "prediction_tie_label": "lowest class index",
        "parameter_count": sum(p.numel() for p in model.net.parameters()),
        "reload_tolerance": RELOAD_TOLERANCE,
        "reload_max_abs_logit_diff": float(np.abs(logits - replay).max()) if len(logits) else 0.0,
        **training,
    }
    metadata = RunMetadata(
        model=f"patchtst/{model.task}/{path.name}", task=model.task, cohort=cohort,
        train_cohort_id=train_cohort_id, seed=model.seed, checkpoint=checksum,
        transformations=transformations, configuration=configuration,
        versions={"python": platform.python_version(), "platform": platform.platform(),
                  **{name: version(name) for name in ("numpy", "torch")}},
        resources={**resources, "prediction_seconds": prediction_seconds,
                   "prediction_seconds_per_window": prediction_seconds / max(len(data.inputs), 1),
                   "process_peak_rss_bytes": float(peak_rss * (1 if sys.platform == "darwin" else 1024))},
    )
    expected = dict(zip(data.scenario_ids, data.targets, strict=True))
    groups = dict(zip(data.scenario_ids, data.group_ids, strict=True))
    metrics = write_run(path, metadata, expected, groups, predictions,
                        bootstrap_replicates=bootstrap_replicates)
    (path / "model.pt").write_bytes(content)
    _write_json(path / "errors.json", [
        {"scenario_id": p.scenario_id, "expected": expected[p.scenario_id],
         "predicted": p.predicted_label, "status": p.status, "error": p.error}
        for p in predictions if p.status != "ok" or expected[p.scenario_id] != p.predicted_label])
    return metrics


def _run_task(task, prepared, output_dir, evaluation_status, bootstrap_replicates, *,
              max_epochs, patience, device, repeat_seeds, progress):
    train_cohort, train, dev_cohort, dev, reference = prepared
    reference_path, reference_metadata = reference
    scaler = WindowScaler().fit(train.inputs)
    train_windows, dev_windows = scaler.transform(train.inputs), scaler.transform(dev.inputs)
    transformations = {
        **scaler.state(), "training_ids_sha256": ids_hash(train.scenario_ids),
        "reference_xgboost_run": str(reference_path),
        "reference_xgboost_checkpoint": reference_metadata.checkpoint,
        "evaluation_status": evaluation_status, "class_mapping": LABEL_MAPPINGS[task],
        "training_class_counts": dict(sorted(Counter(map(str, train.targets)).items())),
        "scaling": "train medians impute missing values, then train mean and population SD; "
                   "SD 1 for a constant channel; no per-window normalization",
    }
    train_targets, dev_targets = np.asarray(train.targets), np.asarray(dev.targets)
    trials, selected = [], []

    def fit_and_record(configuration, seed, path):
        started = time.perf_counter()
        net, info = fit_patchtst(
            train_windows, train_targets, dev_windows, dev_targets, task=task,
            configuration=configuration, seed=seed, max_epochs=max_epochs, patience=patience,
            device=device, on_epoch=lambda row: progress(task, path.name, row))
        fit_seconds = time.perf_counter() - started
        model = PatchTSTModel(task, scaler, net.to(device), dict(configuration), seed, device)
        metrics = _record_run(
            path, model, dev_cohort, dev, train_cohort_id=train_cohort.cohort_id,
            transformations=transformations, bootstrap_replicates=bootstrap_replicates,
            resources={"fit_seconds": fit_seconds,
                       "mean_epoch_seconds": fit_seconds / info["attempted_epochs"]},
            training={"max_epochs": max_epochs, "patience": patience, **info})
        return {"seed": seed, "dev_macro_f1": metrics["macro_f1"], "best_epoch": info["best_epoch"],
                "best_dev_loss": info["best_dev_loss"], "hit_epoch_cap": info["hit_epoch_cap"],
                "run": str(path.relative_to(output_dir))}

    for index, configuration in enumerate(CONFIGURATIONS):
        row = fit_and_record(configuration, 42, output_dir / task / f"config-{index}-seed-42")
        trials.append({**row, "configuration": configuration})
    best = max(trials, key=lambda row: row["dev_macro_f1"])  # ties keep grid order
    selected.append({key: value for key, value in best.items() if key != "configuration"})
    if repeat_seeds:
        for seed in (43, 44):
            selected.append(fit_and_record(best["configuration"], seed,
                                           output_dir / task / f"selected-seed-{seed}"))
    scores = [row["dev_macro_f1"] for row in selected]
    _write_json(output_dir / task / "training_ids.json", list(train.scenario_ids))
    return {
        "trials": trials, "selected_configuration": best["configuration"], "selected_seeds": selected,
        "dev_macro_f1_mean": float(np.mean(scores)), "dev_macro_f1_std": float(np.std(scores)),
        "seeds_repeated": repeat_seeds, "train_windows": len(train.inputs),
        "dev_windows": len(dev.inputs), "train_ids_sha256": ids_hash(train.scenario_ids),
        "dev_cohort_id": dev_cohort.cohort_id,
    }


def run_patchtst(dataset, output_dir, *, xgboost_reference, tasks=("detection", "category"),
                 bootstrap_replicates=1000, allow_exploratory=False, max_epochs=100, patience=10,
                 device="cpu", repeat_seeds=True, progress=lambda *_: None):
    """Fit four settings with seed 42, choose on full dev macro-F1, then repeat seeds 43/44."""
    tasks = tuple(tasks)
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("choose one or both distinct tasks")
    require_ready(dataset, tasks, allow_exploratory=allow_exploratory)
    output_dir, xgboost_reference = Path(output_dir), Path(xgboost_reference)
    if output_dir.exists():
        raise FileExistsError("output exists; choose a new experiment directory")
    prepared = {}
    for task in tasks:
        train_cohort, train = split_data(dataset, task, "train")
        dev_cohort, dev = split_data(dataset, task, "dev")
        reference = _reference(xgboost_reference, dataset, task, train_cohort, train,
                               dev_cohort, dev, None)
        prepared[task] = train_cohort, train, dev_cohort, dev, reference
    output_dir.mkdir(parents=True)
    summary = {"release_id": dataset.release_id, "selection_split": "dev", "tasks": {},
               "evaluation_status": "provisional_exploratory" if dataset.splits["mode"] == "exploratory"
               else "scored", "xgboost_reference": str(xgboost_reference), "device": device}
    for task, task_data in prepared.items():
        summary["tasks"][task] = _run_task(
            task, task_data, output_dir, summary["evaluation_status"], bootstrap_replicates,
            max_epochs=max_epochs, patience=patience, device=device, repeat_seeds=repeat_seeds,
            progress=progress)
        _write_json(output_dir / "selection.json", summary)
    dataset.check_source()
    return summary


def compare_with_baselines(experiments, output_dir, tasks=("detection", "category")):
    """Dev-only comparison with earlier baselines; refuses mismatched cohorts or training data."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("output exists; choose a new comparison directory")
    reports = [compare_experiments(experiments, task) for task in tasks]
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "comparison.json", reports)
    (output_dir / "comparison.md").write_text(render_markdown(reports), encoding="utf-8")
    return reports


def predict_saved_model(dataset, model_run_dir, output_dir, *, split="test", device="cpu",
                        bootstrap_replicates=1000, allow_exploratory=False):
    """Reload weights and saved scaler statistics; nothing is fitted on the scored split."""
    path = Path(model_run_dir)
    metadata = load_run(path)["metadata"]
    require_ready(dataset, (metadata.task,), allow_exploratory=allow_exploratory)
    if metadata.cohort.release_id != dataset.release_id:
        raise ValueError("checkpoint belongs to a different release")
    if metadata.transformations["class_mapping"] != LABEL_MAPPINGS[metadata.task]:
        raise ValueError("checkpoint class mapping differs from release")
    config = metadata.configuration
    model = load_model(
        (path / "model.pt").read_bytes(), expected_sha256=metadata.checkpoint, task=metadata.task,
        scaler=WindowScaler.from_state(metadata.transformations), seed=metadata.seed, device=device,
        configuration={key: config[key] for key in ("patch_length", "learning_rate")},
        batch_size=config.get("inference_batch_size", INFERENCE_BATCH_SIZE))
    cohort, data = split_data(dataset, metadata.task, split)
    training = {key: config[key] for key in ("max_epochs", "patience", "history", "best_epoch",
                                             "best_dev_loss", "attempted_epochs", "hit_epoch_cap")}
    return _record_run(Path(output_dir), model, cohort, data, train_cohort_id=metadata.train_cohort_id,
                       transformations=metadata.transformations, resources={},
                       bootstrap_replicates=bootstrap_replicates, training=training)
