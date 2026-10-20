"""Development-selected TimesNet runs on the exact XGBoost/RF cohorts."""

import json
import platform
import resource
import sys
import time
from collections import Counter
from dataclasses import replace
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
from clauseguard.kpi.timesnet import (
    ARCHITECTURE,
    CONFIGURATIONS,
    GRADIENT_CLIP,
    HEADS,
    INFERENCE_BATCH_SIZE,
    UPSTREAM_COMMIT,
    TimesNetAdapter,
    TimesNetModel,
    WindowScaler,
    fit_timesnet,
    load_model,
    predict_logits,
    serialize_model,
)

DIAGNOSTIC_WINDOWS = 512
# "issue33" is the protocol in #33. "v2" applies the 2026-10-19 training diagnosis:
# prior-corrected stopping, cosine learning-rate decay and three inception kernels.
RECIPES = {
    "issue33": {"prior_corrected_stopping": False, "lr_schedule": None, "architecture": {}},
    "v2": {"prior_corrected_stopping": True, "lr_schedule": "cosine",
           "architecture": {"num_kernels": 3}},
}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_window_selection(dataset, path):
    """Read an audit selection (e.g. nonoverlap_normal_keep_faults.json) applied to every split."""
    if path is None:
        return None
    selection = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = selection["scenario_ids"]
    if selection["release_id"] != dataset.release_id:
        raise ValueError("window selection belongs to a different release")
    if len(ids) != len(set(ids)) or set(ids) - set(dataset.splits["membership"]):
        raise ValueError("window selection contains duplicate or unknown IDs")
    kept = set(ids)
    if any(sid not in kept for sid, row in dataset.splits["membership"].items()
           if row["split"] != "excluded" and row["detection_target"] == 1):
        raise ValueError("window selection must retain every fault window")
    return {"policy": selection.get("policy"), "ids": kept, "ids_sha256": ids_hash(ids)}


def _restrict(cohort, batch, selection):
    """Narrow a cohort and its batch to the selected windows; the cohort ID changes with it."""
    if selection is None:
        return cohort, batch
    narrowed = cohort.model_copy(update={
        "scenario_ids": tuple(sid for sid in cohort.scenario_ids if sid in selection["ids"])})
    return narrowed, replace(select_training(batch, selection["ids"]), cohort_id=narrowed.cohort_id)


def _reference(reference, dataset, task, train_cohort, train, dev_cohort, dev, selection):
    """Require the same release, training IDs, dev cohort and dev labels as XGBoost seed 42.

    With a window selection, the reference's training IDs and dev labels are compared after
    applying the same selection.
    """
    selection_json = json.loads((reference / "selection.json").read_text())
    if selection_json["release_id"] != dataset.release_id:
        raise ValueError("XGBoost reference belongs to a different release")
    chosen = selection_json["tasks"][task]["selected_seeds"][0]
    if chosen["seed"] != 42:
        raise ValueError("XGBoost reference must identify the selected seed-42 run")
    saved = load_run(reference / chosen["run"])
    metadata = saved["metadata"]
    if (metadata.task != task or metadata.seed != 42
            or metadata.train_cohort_id != train_cohort.cohort_id
            or metadata.cohort.cohort_id != dev_cohort.cohort_id):
        raise ValueError("XGBoost reference differs in task, train cohort or development cohort")
    reference_ids = json.loads((reference / task / "training_ids.json").read_text())
    expected = saved["expected"]
    if selection is not None:
        reference_ids = [sid for sid in reference_ids if sid in selection["ids"]]
        expected = {sid: target for sid, target in expected.items() if sid in selection["ids"]}
    elif metadata.transformations["training_ids_sha256"] != ids_hash(train.scenario_ids):
        raise ValueError("XGBoost reference training IDs differ from TimesNet training selection")
    if reference_ids != list(train.scenario_ids):
        raise ValueError("XGBoost reference training IDs differ from TimesNet training selection")
    if expected != dict(zip(dev.scenario_ids, dev.targets, strict=True)):
        raise ValueError("XGBoost reference development labels differ")
    return reference / chosen["run"], metadata


def _record_run(path, model, cohort, data, *, train_cohort_id, transformations, resources,
                bootstrap_replicates, training):
    started = time.perf_counter()
    predictions = predict_adapter(TimesNetAdapter(model.task), model, cohort, [data])
    prediction_seconds = time.perf_counter() - started
    # Periods are chosen per batch, so report how much the batch size matters.
    sample = model.scaler.transform(data.inputs[:DIAGNOSTIC_WINDOWS])
    one_by_one = predict_logits(model.net, sample, batch_size=1, device=model.device)
    batched = predict_logits(model.net, sample, batch_size=model.batch_size, device=model.device)
    agreement = float((one_by_one.argmax(1) == batched.argmax(1)).mean()) if len(sample) else 1.0
    content, checksum = serialize_model(model)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    configuration = {
        **model.configuration, **ARCHITECTURE, **training.get("architecture", {}),
        "upstream_commit": UPSTREAM_COMMIT,
        "optimizer": "RAdam", "weight_decay": 0, "gradient_clip": GRADIENT_CLIP,
        "train_batch_size": 32, "inference_batch_size": model.batch_size,
        "loss_weighting": "balanced_training_n_over_k_weighted_mean",
        "validation_loss_weighting": "unweighted", "device": model.device,
        "deterministic_algorithms": model.device == "cpu",
        "binary_probability_tie_label": 0,
        "parameter_count": sum(p.numel() for p in model.net.parameters()),
        "batch_size_1_vs_32_agreement": agreement,
        "batch_size_agreement_windows": len(sample), **training,
    }
    metadata = RunMetadata(
        model=f"timesnet/{model.task}/{path.name}", task=model.task, cohort=cohort,
        train_cohort_id=train_cohort_id, seed=model.seed, checkpoint=checksum,
        transformations=transformations, configuration=configuration,
        versions={"python": platform.python_version(), "platform": platform.platform(),
                  **{name: version(name) for name in ("numpy", "torch")}},
        resources={**resources, "prediction_seconds": prediction_seconds,
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
        for p in predictions if p.status != "ok" or expected[p.scenario_id] != p.predicted_label
    ])
    return metrics


def _run_task(task, prepared, output_dir, evaluation_status, bootstrap_replicates, *,
              max_epochs, patience, device, repeat_seeds, progress, recipe, instance_norm,
              selection, head, recording_weights):
    train_cohort, full_train, train, dev_cohort, dev, reference = prepared
    reference_path, reference_metadata = reference
    scaler = WindowScaler().fit(train.inputs)
    train_windows, dev_windows = scaler.transform(train.inputs), scaler.transform(dev.inputs)
    transformations = {
        **scaler.state(), "training_ids_sha256": ids_hash(train.scenario_ids),
        "reference_xgboost_run": str(reference_path),
        "reference_xgboost_checkpoint": reference_metadata.checkpoint,
        "evaluation_status": evaluation_status, "class_mapping": LABEL_MAPPINGS[task],
        "training_class_counts": dict(sorted(Counter(map(str, train.targets)).items())),
        "window_selection": None if selection is None else {
            "policy": selection["policy"], "ids_sha256": selection["ids_sha256"],
            "windows": len(selection["ids"])},
    }
    train_targets, dev_targets = np.asarray(train.targets), np.asarray(dev.targets)
    settings = RECIPES[recipe]
    trials, selected = [], []

    def fit_and_record(configuration, seed, path):
        started = time.perf_counter()
        net, info = fit_timesnet(
            train_windows, train_targets, dev_windows, dev_targets, task=task,
            configuration=configuration, seed=seed, max_epochs=max_epochs, patience=patience,
            device=device, on_epoch=lambda row: progress(task, path.name, row),
            dev_groups=list(dev.group_ids), instance_norm=instance_norm, head=head,
            train_groups=list(train.group_ids), recording_weights=recording_weights, **settings)
        fit_seconds = time.perf_counter() - started
        model = TimesNetModel(task, scaler, net.to(device), dict(configuration), seed, device)
        training = {"max_epochs": max_epochs, "patience": patience, "recipe": recipe,
                    "instance_norm": instance_norm, "head": head,
                    "recording_weights": recording_weights, **settings, **info}
        metrics = _record_run(
            path, model, dev_cohort, dev, train_cohort_id=train_cohort.cohort_id,
            transformations=transformations, bootstrap_replicates=bootstrap_replicates,
            resources={"fit_seconds": fit_seconds,
                       "mean_epoch_seconds": fit_seconds / info["attempted_epochs"]},
            training=training)
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
    result = {
        "trials": trials, "selected_configuration": best["configuration"], "selected_seeds": selected,
        "dev_macro_f1_mean": float(np.mean(scores)), "dev_macro_f1_std": float(np.std(scores)),
        "seeds_repeated": repeat_seeds, "train_windows": len(train.inputs),
        "full_train_windows": len(full_train.inputs), "dev_windows": len(dev.inputs),
        "train_ids_sha256": ids_hash(train.scenario_ids), "dev_cohort_id": dev_cohort.cohort_id,
    }
    _write_json(output_dir / task / "training_ids.json", list(train.scenario_ids))
    return result


def run_timesnet(dataset, output_dir, *, xgboost_reference, tasks=("detection", "category"),
                 training_selection=None, bootstrap_replicates=1000, allow_exploratory=False,
                 max_epochs=100, patience=10, device="cpu", repeat_seeds=True,
                 progress=lambda *_: None, recipe="issue33", instance_norm=False,
                 window_selection=None, head="flatten", recording_weights=False):
    """Fit four configurations on train, choose on dev, then repeat seeds 43/44.

    ``window_selection`` narrows train, dev and test to the same audit selection; it cannot be
    combined with ``training_selection``, which narrows train only.
    """
    tasks = tuple(tasks)
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("choose one or both distinct tasks")
    if recipe not in RECIPES:
        raise ValueError(f"recipe must be one of {sorted(RECIPES)}")
    if head not in HEADS:
        raise ValueError(f"head must be one of {HEADS}")
    if training_selection is not None and window_selection is not None:
        raise ValueError("use either a training selection or a window selection, not both")
    require_ready(dataset, tasks, allow_exploratory=allow_exploratory)
    output_dir, xgboost_reference = Path(output_dir), Path(xgboost_reference)
    if output_dir.exists():
        raise FileExistsError("output exists; choose a new experiment directory")
    selected_ids = training_ids(dataset, training_selection)
    selection = load_window_selection(dataset, window_selection)
    prepared = {}
    for task in tasks:
        train_cohort, full_train = split_data(dataset, task, "train")
        train = select_training(full_train, selected_ids if selection is None else selection["ids"])
        full_dev_cohort, full_dev = split_data(dataset, task, "dev")
        dev_cohort, dev = _restrict(full_dev_cohort, full_dev, selection)
        reference = _reference(xgboost_reference, dataset, task, train_cohort, train,
                               full_dev_cohort, dev, selection)
        prepared[task] = train_cohort, full_train, train, dev_cohort, dev, reference
    output_dir.mkdir(parents=True)
    summary = {"release_id": dataset.release_id, "selection_split": "dev", "tasks": {},
               "evaluation_status": "provisional_exploratory" if dataset.splits["mode"] == "exploratory" else "scored",
               "xgboost_reference": str(xgboost_reference), "recipe": recipe,
               "instance_norm": instance_norm, "head": head,
               "recording_weights": recording_weights,
               "window_selection": None if selection is None else selection["policy"]}
    for task, task_data in prepared.items():
        summary["tasks"][task] = _run_task(
            task, task_data, output_dir, summary["evaluation_status"], bootstrap_replicates,
            max_epochs=max_epochs, patience=patience, device=device,
            repeat_seeds=repeat_seeds, progress=progress, recipe=recipe,
            instance_norm=instance_norm, selection=selection, head=head,
            recording_weights=recording_weights)
        _write_json(output_dir / "selection.json", summary)
    dataset.check_source()
    return summary


def predict_saved_model(dataset, model_run_dir, output_dir, *, split="test", device="cpu",
                        bootstrap_replicates=1000, allow_exploratory=False,
                        window_selection=None):
    """Reload weights and saved scaler statistics; nothing is fitted on the scored split."""
    path = Path(model_run_dir)
    metadata = load_run(path)["metadata"]
    require_ready(dataset, (metadata.task,), allow_exploratory=allow_exploratory)
    if metadata.cohort.release_id != dataset.release_id:
        raise ValueError("checkpoint belongs to a different release")
    if metadata.transformations["class_mapping"] != LABEL_MAPPINGS[metadata.task]:
        raise ValueError("checkpoint class mapping differs from release")
    selection = load_window_selection(dataset, window_selection)
    trained_with = metadata.transformations.get("window_selection")
    if (trained_with or {}).get("ids_sha256") != (selection or {}).get("ids_sha256"):
        raise ValueError("pass the same --window-selection the checkpoint was trained with")
    config = metadata.configuration
    model = load_model(
        (path / "model.pt").read_bytes(), expected_sha256=metadata.checkpoint, task=metadata.task,
        scaler=WindowScaler.from_state(metadata.transformations), seed=metadata.seed, device=device,
        configuration={key: config[key] for key in ("learning_rate", "dropout")},
        batch_size=config.get("inference_batch_size", INFERENCE_BATCH_SIZE),
        architecture={key: config[key] for key in ARCHITECTURE},
        instance_norm=config.get("instance_norm", False), head=config.get("head", "flatten"))
    cohort, data = _restrict(*split_data(dataset, metadata.task, split), selection)
    training = {key: config[key] for key in
                ("max_epochs", "patience", "history", "best_epoch", "best_dev_loss",
                 "attempted_epochs", "hit_epoch_cap", "recipe", "instance_norm", "head",
                 "recording_weights", "prior_corrected_stopping", "lr_schedule", "selection_loss")
                if key in config}
    training["architecture"] = {key: config[key] for key in ARCHITECTURE}
    return _record_run(Path(output_dir), model, cohort, data, train_cohort_id=metadata.train_cohort_id,
                       transformations=metadata.transformations, resources={},
                       bootstrap_replicates=bootstrap_replicates, training=training)
