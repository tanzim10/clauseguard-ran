"""Frozen Toto 2.0 forecast-error detector with a development-selected threshold (Task 3.10)."""

import hashlib
import json
import os
import platform
import resource
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from clauseguard.baselines.artifacts import compare_runs, load_run, write_run
from clauseguard.baselines.contracts import CasePrediction, RunMetadata
from clauseguard.baselines.tree_data import ids_hash, require_ready, split_data
from clauseguard.datasets.contracts import LABEL_MAPPINGS, NUMERIC_CHANNELS
from clauseguard.kpi.mantis import state_digest
from clauseguard.kpi.toto import (
    CHECKPOINT_PARAMETERS,
    CHECKPOINT_REPOSITORY,
    CHECKPOINT_REVISION,
    CHECKPOINT_SHA256,
    CONTEXT,
    HORIZON,
    PACKAGE_VERSION,
    PATCH_SIZE,
    QUANTILES,
    apply_threshold,
    fit_scales,
    forecast_median,
    last_value_forecast,
    load_forecaster,
    select_threshold,
    split_windows,
    window_arrays,
    window_scores,
)

DETECTORS = ("toto", "last_value")
BATCH_SIZE = 64
ISOLATION_WINDOWS = 8
ISOLATION_TOLERANCE = {"rtol": 1e-4, "atol": 1e-4}
INFERENCE = {"context": CONTEXT, "horizon": HORIZON, "patch_size": PATCH_SIZE,
             "decode_block_size": None, "quantile": 0.5, "has_missing_values": True,
             "series_ids": "zeros (batch, 16)", "placeholder": "last observed value, else next, else 0"}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _array_digest(values):
    return hashlib.sha256(memoryview(np.ascontiguousarray(values)).cast("B")).hexdigest()


def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _optional_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return "not loaded"


def _peak_rss():
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))


def _cached_forecast(cache_dir, dataset, cohort, context, mask, forecaster, *, device, batch_size):
    """Forecasts keyed by checkpoint, release, cohort, context, masks, channel order and settings."""
    identity = {
        "version": 1, "checkpoint_repository": CHECKPOINT_REPOSITORY,
        "checkpoint_revision": CHECKPOINT_REVISION, "checkpoint_sha256": CHECKPOINT_SHA256,
        "package_version": PACKAGE_VERSION, "release_id": dataset.release_id,
        "cohort_id": cohort.cohort_id, "split": cohort.split,
        "context_sha256": _array_digest(np.nan_to_num(context, nan=0.0)),
        "mask_sha256": _array_digest(mask), "channel_order": list(NUMERIC_CHANNELS),
        "inference": INFERENCE,
    }
    path = Path(cache_dir) / f"{_digest(_json_bytes(identity))}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"].item()))
            forecasts = saved["forecasts"]
        if metadata.get("identity") != identity or metadata.get("forecasts_sha256") != _array_digest(forecasts):
            raise ValueError("Toto forecast cache identity or checksum differs")
        return forecasts, str(path), 0.0, forecaster
    if forecaster is None:
        forecaster = load_forecaster(device=device)
    before = state_digest(forecaster)
    started = time.perf_counter()
    forecasts = forecast_median(forecaster, context, mask, batch_size=batch_size, device=device)
    seconds = time.perf_counter() - started
    if state_digest(forecaster) != before:
        raise ValueError("Toto weights changed during forecasting")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, forecasts=forecasts, metadata=np.asarray(json.dumps(
            {"identity": identity, "forecasts_sha256": _array_digest(forecasts)}, sort_keys=True)))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return forecasts, str(path), seconds, forecaster


def check_forecaster(forecaster, context, mask, *, device="cpu"):
    """Repeat predictions must match, and batch neighbours must not change a window's forecast."""
    count = min(ISOLATION_WINDOWS, len(context))
    if not count:
        return {"repeat_identical": True, "batch_isolation_max_abs_diff": 0.0, "windows": 0}
    together = forecast_median(forecaster, context[:count], mask[:count], batch_size=count, device=device)
    repeated = forecast_median(forecaster, context[:count], mask[:count], batch_size=count, device=device)
    alone = np.concatenate([forecast_median(forecaster, context[i:i + 1], mask[i:i + 1], batch_size=1,
                                            device=device) for i in range(count)])
    if not np.array_equal(together, repeated):
        raise ValueError("Toto repeat predictions differ")
    if not np.allclose(together, alone, **ISOLATION_TOLERANCE):
        raise ValueError("Toto forecasts depend on other batch members beyond the tolerance")
    return {"repeat_identical": True, "windows": count,
            "batch_isolation_max_abs_diff": float(np.abs(together - alone).max()),
            "tolerance": ISOLATION_TOLERANCE}


def _forecast_coverage(dataset, ids, targets):
    """For faults with a source interval, whether it reaches the forecast segment (steps 96-127)."""
    result = {}
    for sid, target in zip(ids, targets, strict=True):
        if target != 1:
            continue
        anomaly = dataset.load_scenario(sid).anomaly
        if anomaly.start_index is None:
            result[sid] = None
        else:
            result[sid] = anomaly.end_index >= CONTEXT
    return result


def _reference(reference_dir, dataset, cohort, data):
    reference_dir = Path(reference_dir)
    selection = json.loads((reference_dir / "selection.json").read_text(encoding="utf-8"))
    if selection["release_id"] != dataset.release_id:
        raise ValueError("XGBoost reference belongs to a different release")
    chosen = next(row for row in selection["tasks"]["detection"]["selected_seeds"] if row["seed"] == 42)
    run_path = reference_dir / chosen["run"]
    saved = load_run(run_path)
    if saved["metadata"].task != "detection" or saved["metadata"].cohort.cohort_id != cohort.cohort_id:
        raise ValueError("XGBoost reference differs in task or cohort")
    if saved["expected"] != dict(zip(data.scenario_ids, data.targets, strict=True)):
        raise ValueError("XGBoost reference labels differ from the Toto cohort")
    return run_path


def _record(path, name, cohort, data, scores, failures, predicted, *, train_cohort_id, scales,
            threshold, configuration, resources, bootstrap_replicates, coverage, membership, masks):
    detector = {"detector": name, "threshold": threshold, "channel_scales": scales.tolist(),
                "channel_order": list(NUMERIC_CHANNELS), "inference": INFERENCE}
    content = _json_bytes(detector)
    metadata = RunMetadata(
        model=f"{name}/detection/{cohort.split}", task="detection", cohort=cohort,
        train_cohort_id=train_cohort_id, seed=42, checkpoint=_digest(content),
        transformations={
            "channel_order": list(NUMERIC_CHANNELS), "channel_scales": scales.tolist(),
            "scale_policy": "population SD of observed values in normal training windows; 1 if constant",
            "class_mapping": LABEL_MAPPINGS["detection"],
            "evaluation_status": "provisional_exploratory" if cohort.mode == "exploratory" else "scored",
        },
        configuration={**configuration, "threshold": threshold, "decision": "score >= threshold"},
        versions={"python": platform.python_version(), "platform": platform.platform(),
                  "numpy": version("numpy"), "torch": _optional_version("torch"),
                  "toto-2": _optional_version("toto-2")},
        resources=resources,
    )
    expected = dict(zip(data.scenario_ids, data.targets, strict=True))
    groups = dict(zip(data.scenario_ids, data.group_ids, strict=True))
    rows = []
    for index, sid in enumerate(data.scenario_ids):
        if predicted[index] < 0:
            rows.append(CasePrediction(scenario_id=sid, task="detection", status="error",
                                       error=failures.get(index, "forecast failed")))
        else:
            rows.append(CasePrediction(scenario_id=sid, task="detection",
                                       predicted_label=int(predicted[index])))
    metrics = write_run(path, metadata, expected, groups, rows, bootstrap_replicates=bootstrap_replicates)
    (path / "detector.json").write_bytes(content)
    np.savez(path / "scores.npz", scores=scores, context_mask=masks[0], target_mask=masks[1])
    if not np.array_equal(apply_threshold(np.load(path / "scores.npz")["scores"], threshold), predicted):
        raise ValueError(f"{name} predictions changed after reload")
    covered = [sid for sid, value in coverage.items() if value]
    hits = {sid for sid, row in zip(data.scenario_ids, predicted, strict=True) if row == 1}
    _save_json(path / "errors.json", [
        {"scenario_id": sid, "expected": expected[sid], "predicted": int(predicted[i]),
         "score": None if not np.isfinite(scores[i]) else float(scores[i]),
         "origin": membership[sid]["origin"], "rca_label": membership[sid].get("rca_label"),
         "fault_in_forecast_segment": coverage.get(sid)}
        for i, sid in enumerate(data.scenario_ids) if int(predicted[i]) != expected[sid]])
    return {"run": str(path), "macro_f1": metrics["macro_f1"], "recall": metrics["recall"],
            "false_alarms": metrics["false_alarms"], "missed_faults": metrics["missed_faults"],
            "failed_predictions": len(failures), "detector_sha256": _digest(content),
            "forecast_segment_faults": {"with_interval": sum(v is not None for v in coverage.values()),
                                        "reaching_segment": len(covered),
                                        "detected": len(hits & set(covered))}}


def _score_all(context, mask, target, target_mask, scales, forecasts):
    return {
        "toto": window_scores(forecasts, target, target_mask, mask, scales),
        "last_value": window_scores(last_value_forecast(context, mask), target, target_mask, mask, scales),
    }


def run_toto(dataset, output_dir, *, xgboost_reference, cache_dir="artifacts/cache/toto",
             bootstrap_replicates=1000, allow_exploratory=False, device="cpu",
             batch_size=BATCH_SIZE, forecaster=None):
    """Fit channel scales on normal train windows and choose each detector's threshold on dev."""
    if batch_size <= 0 or bootstrap_replicates < 0:
        raise ValueError("batch size must be positive and bootstrap count nonnegative")
    require_ready(dataset, ("detection",), allow_exploratory=allow_exploratory)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("output exists; choose a new experiment directory")
    train_cohort, train = split_data(dataset, "detection", "train")
    dev_cohort, dev = split_data(dataset, "detection", "dev")
    reference = _reference(xgboost_reference, dataset, dev_cohort, dev)
    scales = fit_scales(window_arrays(train.inputs), train.targets)
    context, mask, target, target_mask = split_windows(window_arrays(dev.inputs))
    forecasts, cache_path, seconds, forecaster = _cached_forecast(
        cache_dir, dataset, dev_cohort, context, mask, forecaster, device=device, batch_size=batch_size)
    checks = None
    if forecaster is not None:
        if any(parameter.requires_grad for parameter in forecaster.parameters()):
            raise ValueError("Toto parameters must be frozen")
        before = state_digest(forecaster)
        checks = check_forecaster(forecaster, context, mask, device=device)
    coverage = _forecast_coverage(dataset, dev.scenario_ids, dev.targets)
    output_dir.mkdir(parents=True)
    summary = {"release_id": dataset.release_id, "selection_split": "dev",
               "evaluation_status": "provisional_exploratory" if dev_cohort.mode == "exploratory" else "scored",
               "train_windows": len(train.inputs), "dev_windows": len(dev.inputs),
               "train_ids_sha256": ids_hash(train.scenario_ids), "dev_cohort_id": dev_cohort.cohort_id,
               "xgboost_reference": str(reference), "forecast_cache": cache_path,
               "forecast_seconds": seconds, "model_checks": checks, "detectors": {}}
    for name, (scores, failures) in _score_all(context, mask, target, target_mask, scales, forecasts).items():
        selection = select_threshold(scores, dev.targets)
        _save_json(output_dir / f"{name}_thresholds.json", selection)
        configuration = {"selection": "dev macro-F1 over 101 score quantiles and an all-normal endpoint; "
                                      "ties: fewer false alarms, then higher threshold",
                         "dev_macro_f1_at_selection": selection["dev_macro_f1"], "seed": 42}
        if name == "toto":
            configuration |= {"checkpoint_repository": CHECKPOINT_REPOSITORY,
                              "checkpoint_revision": CHECKPOINT_REVISION,
                              "checkpoint_sha256": CHECKPOINT_SHA256, "package_version": PACKAGE_VERSION,
                              "parameters": CHECKPOINT_PARAMETERS, "weight_bytes": CHECKPOINT_PARAMETERS * 4,
                              "quantile_levels": list(QUANTILES), "device": device,
                              "batch_size": batch_size, "parameters_frozen": True}
        resources = {"forecast_seconds": seconds if name == "toto" else 0.0,
                     "process_peak_rss_bytes": _peak_rss()}
        summary["detectors"][name] = _record(
            output_dir / name, name, dev_cohort, dev, scores, failures,
            apply_threshold(scores, selection["threshold"]), train_cohort_id=train_cohort.cohort_id,
            scales=scales,
            threshold=selection["threshold"], configuration=configuration, resources=resources,
            bootstrap_replicates=bootstrap_replicates, coverage=coverage,
            membership=dataset.splits["membership"], masks=(mask, target_mask))
    if forecaster is not None and state_digest(forecaster) != before:
        raise ValueError("Toto weights changed during threshold selection")
    summary["comparison"] = compare_runs([reference, output_dir / "toto", output_dir / "last_value"])
    _save_json(output_dir / "selection.json", summary)
    dataset.check_source()
    return summary


def predict_saved_detector(dataset, detector_run_dir, output_dir, *, split="test",
                           cache_dir="artifacts/cache/toto", bootstrap_replicates=1000,
                           allow_exploratory=False, device="cpu", batch_size=BATCH_SIZE, forecaster=None):
    """Replay a saved threshold and channel scales on one split; nothing is refitted."""
    path, output_dir = Path(detector_run_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("output exists; choose a new experiment directory")
    if split not in ("dev", "test") or batch_size <= 0 or bootstrap_replicates < 0:
        raise ValueError("choose dev or test and use valid batch and bootstrap values")
    saved = load_run(path)
    metadata = saved["metadata"]
    require_ready(dataset, ("detection",), allow_exploratory=allow_exploratory)
    if metadata.cohort.release_id != dataset.release_id:
        raise ValueError("detector belongs to a different release")
    content = (path / "detector.json").read_bytes()
    if _digest(content) != metadata.checkpoint:
        raise ValueError("detector checksum mismatch")
    detector = json.loads(content)
    if detector["channel_order"] != list(NUMERIC_CHANNELS) or detector["inference"] != INFERENCE:
        raise ValueError("detector channel order or inference settings differ")
    scales = np.asarray(detector["channel_scales"], dtype=np.float64)
    cohort, data = split_data(dataset, "detection", split)
    context, mask, target, target_mask = split_windows(window_arrays(data.inputs))
    forecasts, seconds = None, 0.0
    if detector["detector"] == "toto":
        forecasts, _, seconds, forecaster = _cached_forecast(
            cache_dir, dataset, cohort, context, mask, forecaster, device=device, batch_size=batch_size)
    else:
        forecasts = last_value_forecast(context, mask)
    scores, failures = window_scores(forecasts, target, target_mask, mask, scales)
    configuration = {**metadata.configuration}
    configuration.pop("threshold")
    configuration.pop("decision")
    return _record(output_dir, detector["detector"], cohort, data, scores, failures,
                   apply_threshold(scores, detector["threshold"]), train_cohort_id=metadata.train_cohort_id,
                   scales=scales, threshold=detector["threshold"], configuration=configuration,
                   resources={"forecast_seconds": seconds, "process_peak_rss_bytes": _peak_rss()},
                   bootstrap_replicates=bootstrap_replicates,
                   coverage=_forecast_coverage(dataset, data.scenario_ids, data.targets),
                   membership=dataset.splits["membership"], masks=(mask, target_mask))
