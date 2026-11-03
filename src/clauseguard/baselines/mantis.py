"""Frozen Mantis features and development-selected logistic heads."""

import hashlib
import io
import json
import os
import platform
import resource
import sys
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np

from clauseguard.baselines.artifacts import compare_runs, load_run, write_run
from clauseguard.baselines.contracts import CasePrediction, RunMetadata
from clauseguard.baselines.tree_data import ids_hash, require_ready, split_data
from clauseguard.datasets.contracts import LABEL_MAPPINGS
from clauseguard.kpi.mantis import (
    CHECKPOINT_REPOSITORY,
    CHECKPOINT_REVISION,
    CHECKPOINT_SHA256,
    EMBEDDING_WIDTH,
    ENCODER_PARAMETERS,
    ENCODER_WEIGHT_BYTES,
    NUM_PATCHES,
    PACKAGE_VERSION,
    SEQ_LEN,
    UPSTREAM_COMMIT,
    WindowImputer,
    _variant_features,
    encode_windows,
    fit_head,
    load_encoder,
    raw_statistics,
    state_digest,
)

FEATURE_VARIANTS = ("embeddings", "embeddings+statistics")
BATCH_SIZE = 32


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _array_digest(values):
    view = memoryview(np.ascontiguousarray(values)).cast("B")
    return hashlib.sha256(view).hexdigest()


def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _optional_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return "not loaded"


def _cache_features(cache_dir, dataset, cohort, batch, imputer, encoder, *, split,
                    device, batch_size):
    windows = imputer.transform(batch.inputs)
    statistics = raw_statistics(windows)
    identity = {
        "version": 1,
        "checkpoint_repository": CHECKPOINT_REPOSITORY,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "release_id": dataset.release_id,
        "cohort_id": cohort.cohort_id,
        "task": cohort.task,
        "split": split,
        "scenario_ids_sha256": ids_hash(batch.scenario_ids),
        "input_sha256": _array_digest(windows),
        "preprocessing_sha256": _digest(_json_bytes(imputer.state())),
        "channel_order": list(imputer.state()["channel_order"]),
        "sequence_length": SEQ_LEN,
        "num_patches": NUM_PATCHES,
    }
    key = _digest(_json_bytes(identity))
    path = Path(cache_dir) / f"{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"].item()))
            embeddings = saved["embeddings"]
            saved_statistics = saved["statistics"]
        if metadata.get("identity") != identity:
            raise ValueError("Mantis feature cache identity differs")
        if metadata.get("embeddings_sha256") != _array_digest(embeddings):
            raise ValueError("Mantis embedding cache checksum mismatch")
        if metadata.get("statistics_sha256") != _array_digest(saved_statistics):
            raise ValueError("Mantis statistics cache checksum mismatch")
        if embeddings.shape != (len(batch.inputs), EMBEDDING_WIDTH):
            raise ValueError("Mantis feature cache has an invalid embedding shape")
        if saved_statistics.shape != statistics.shape or not np.array_equal(saved_statistics, statistics):
            raise ValueError("Mantis feature cache statistics differ from current inputs")
        return embeddings, saved_statistics, str(path), 0.0, encoder

    if encoder is None:
        encoder = load_encoder(device=device)
    before = state_digest(encoder)
    started = time.perf_counter()
    embeddings = encode_windows(encoder, windows, batch_size=batch_size, device=device)
    elapsed = time.perf_counter() - started
    if state_digest(encoder) != before:
        raise ValueError("Mantis encoder weights changed during feature extraction")
    metadata = {
        "identity": identity,
        "embeddings_sha256": _array_digest(embeddings),
        "statistics_sha256": _array_digest(statistics),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, embeddings=embeddings, statistics=statistics,
                 metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return embeddings, statistics, str(path), elapsed, encoder


def _reference(reference_dir, dataset, task, train_cohort, train, dev_cohort, dev, model_name):
    reference_dir = Path(reference_dir)
    selection = json.loads((reference_dir / "selection.json").read_text(encoding="utf-8"))
    if selection["release_id"] != dataset.release_id:
        raise ValueError(f"{model_name} reference belongs to a different release")
    selected = selection["tasks"][task]["selected_seeds"][0]
    if selected["seed"] != 42:
        raise ValueError(f"{model_name} reference must identify seed 42")
    run_path = reference_dir / selected["run"]
    saved = load_run(run_path)
    metadata = saved["metadata"]
    if (metadata.task != task or metadata.seed != 42
            or metadata.train_cohort_id != train_cohort.cohort_id
            or metadata.cohort.cohort_id != dev_cohort.cohort_id):
        raise ValueError(f"{model_name} reference differs in task or cohort")
    if json.loads((reference_dir / task / "training_ids.json").read_text(encoding="utf-8")) != list(
            train.scenario_ids):
        raise ValueError(f"{model_name} training IDs differ from Mantis")
    if saved["expected"] != dict(zip(dev.scenario_ids, dev.targets, strict=True)):
        raise ValueError(f"{model_name} development labels differ from Mantis")
    return run_path, metadata


def _head_bytes(scaler, classifier):
    output = io.BytesIO()
    joblib.dump({"scaler": scaler, "classifier": classifier}, output, compress=3)
    return output.getvalue()


def _run_variant(task, variant, scaler, classifier, train_cohort, dev_cohort, train, dev,
                 embeddings, statistics, output_dir, *, cache_paths, extraction_seconds,
                 imputer, membership, bootstrap_replicates, device, head_fit_seconds):
    features = _variant_features(variant, embeddings["train"], statistics["train"])
    dev_features = _variant_features(variant, embeddings["dev"], statistics["dev"])
    started = time.perf_counter()
    predictions = classifier.predict(scaler.transform(dev_features)).astype(int)
    predict_seconds = time.perf_counter() - started
    path = output_dir / task / variant
    content = _head_bytes(scaler, classifier)
    checksum = _digest(content)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    configuration = {
        "feature_variant": variant,
        "feature_width": int(features.shape[1]),
        "upstream_commit": UPSTREAM_COMMIT,
        "mantis_package_version": PACKAGE_VERSION,
        "checkpoint_repository": CHECKPOINT_REPOSITORY,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "num_patches": NUM_PATCHES,
        "sequence_length": SEQ_LEN,
        "features_per_channel": EMBEDDING_WIDTH // 16,
        "return_transf_layer": 2,
        "output_token": "combined",
        "classifier": "LogisticRegression",
        "C": 1,
        "solver": "lbfgs",
        "max_iter": 2000,
        "class_weight_policy": "exact_training_n_over_k_nclass",
        "class_weights": classifier.class_weight,
        "seed": 42,
        "device": device,
        "encoder_parameters_frozen": True,
        "encoder_parameter_count": ENCODER_PARAMETERS,
        "encoder_weight_bytes": ENCODER_WEIGHT_BYTES,
        "head_checkpoint_bytes": len(content),
        "cache_paths": cache_paths,
    }
    transformations = {
        **imputer.state(),
        "training_ids_sha256": ids_hash(train.scenario_ids),
        "training_class_counts": dict(sorted(Counter(map(str, train.targets)).items())),
        "class_mapping": LABEL_MAPPINGS[task],
        "evaluation_status": "provisional_exploratory" if train_cohort.mode == "exploratory" else "scored",
        "feature_order": ["mantis_embedding_channel_order", "optional_channel_means_then_stds"],
        "embedding_width": EMBEDDING_WIDTH,
        "statistics_width": 32,
    }
    metadata = RunMetadata(
        model=f"mantis/{task}/{variant}", task=task, cohort=dev_cohort,
        train_cohort_id=train_cohort.cohort_id, seed=42, checkpoint=checksum,
        transformations=transformations, configuration=configuration,
        versions={"python": platform.python_version(), "platform": platform.platform(),
                  "numpy": version("numpy"), "scikit-learn": version("scikit-learn"),
                  "joblib": version("joblib"), "torch": _optional_version("torch"),
                  "mantis-tsfm": PACKAGE_VERSION},
        resources={"encoder_extraction_seconds": extraction_seconds["train"]
                   + extraction_seconds["dev"], "head_fit_seconds": head_fit_seconds,
                   "prediction_seconds": predict_seconds,
                   "process_peak_rss_bytes": float(peak_rss * (1 if sys.platform == "darwin" else 1024))},
    )
    expected = dict(zip(dev.scenario_ids, dev.targets, strict=True))
    groups = dict(zip(dev.scenario_ids, dev.group_ids, strict=True))
    rows = [CasePrediction(scenario_id=sid, task=task, predicted_label=int(label))
            for sid, label in zip(dev.scenario_ids, predictions, strict=True)]
    metrics = write_run(path, metadata, expected, groups, rows,
                        bootstrap_replicates=bootstrap_replicates)
    head_path = path / "head.joblib"
    head_path.write_bytes(content)
    loaded = joblib.load(io.BytesIO(head_path.read_bytes()))
    replayed = loaded["classifier"].predict(loaded["scaler"].transform(dev_features)).astype(int)
    if not np.array_equal(predictions, replayed):
        raise ValueError("Mantis head predictions changed after reload")
    _save_json(path / "errors.json", [
        {"scenario_id": sid, "expected": expected[sid], "predicted": int(label),
         "origin": membership[sid]["origin"],
         "fault_label": membership[sid].get("fault_label"),
         "rca_label": membership[sid].get("rca_label")}
        for sid, label in zip(dev.scenario_ids, predictions, strict=True)
        if int(label) != expected[sid]
    ])
    return {"run": str(path.relative_to(output_dir)), "macro_f1": metrics["macro_f1"],
            "checkpoint_sha256": checksum, "head_fit_seconds": head_fit_seconds,
            "prediction_seconds": predict_seconds}


def run_mantis(dataset, output_dir, *, rf_reference, xgboost_reference,
               tasks=("detection", "category"), cache_dir="artifacts/cache/mantis",
               bootstrap_replicates=1000, allow_exploratory=False, device="cpu",
               batch_size=BATCH_SIZE, encoder=None):
    """Fit two fixed heads on train and choose the better variant on the full dev set."""
    tasks = tuple(tasks)
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("choose one or both distinct tasks")
    if batch_size <= 0 or bootstrap_replicates < 0:
        raise ValueError("batch size and bootstrap replicates must be nonnegative")
    require_ready(dataset, tasks, allow_exploratory=allow_exploratory)
    output_dir, cache_dir = Path(output_dir), Path(cache_dir)
    if output_dir.exists():
        raise FileExistsError("output exists; choose a new experiment directory")
    prepared = {}
    for task in tasks:
        train_cohort, train = split_data(dataset, task, "train")
        dev_cohort, dev = split_data(dataset, task, "dev")
        rf = _reference(rf_reference, dataset, task, train_cohort, train, dev_cohort, dev, "RF")
        xgb = _reference(xgboost_reference, dataset, task, train_cohort, train,
                         dev_cohort, dev, "XGBoost")
        prepared[task] = train_cohort, train, dev_cohort, dev, rf, xgb

    if encoder is not None and any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("Mantis encoder parameters must be frozen")
    if encoder is not None:
        encoder.eval()
    weights_before = state_digest(encoder) if encoder is not None else None
    output_dir.mkdir(parents=True)
    summary = {"release_id": dataset.release_id,
               "evaluation_status": "provisional_exploratory" if dataset.splits["mode"] == "exploratory" else "scored",
               "selection_split": "dev", "tasks": {}, "cache_dir": str(cache_dir)}

    for task, (train_cohort, train, dev_cohort, dev, rf, xgb) in prepared.items():
        imputer = WindowImputer().fit(train.inputs)
        embeddings, statistics, cache_paths, extraction_seconds = {}, {}, {}, {}
        for split, cohort, batch in (("train", train_cohort, train), ("dev", dev_cohort, dev)):
            (embeddings[split], statistics[split], cache_paths[split], extraction_seconds[split],
             encoder) = _cache_features(cache_dir, dataset, cohort, batch, imputer, encoder,
                                        split=split, device=device, batch_size=batch_size)
            if encoder is not None:
                if any(parameter.requires_grad for parameter in encoder.parameters()):
                    raise ValueError("Mantis encoder parameters must be frozen")
                encoder.eval()
                if weights_before is None:
                    weights_before = state_digest(encoder)
        if encoder is not None and state_digest(encoder) != weights_before:
            raise ValueError("Mantis encoder weights changed during feature extraction")

        trials = []
        train_seconds = {}
        for variant in FEATURE_VARIANTS:
            train_features = _variant_features(variant, embeddings["train"], statistics["train"])
            started = time.perf_counter()
            try:
                scaler, classifier, weights = fit_head(train_features, train.targets, task=task)
            except (ValueError, RuntimeError) as exc:
                trials.append({"feature_variant": variant, "status": "failed",
                               "error": f"{type(exc).__name__}: {exc}"})
                continue
            fit_seconds = time.perf_counter() - started
            train_seconds[variant] = fit_seconds
            result = _run_variant(
                task, variant, scaler, classifier, train_cohort, dev_cohort, train, dev,
                embeddings, statistics, output_dir,
                cache_paths=cache_paths, extraction_seconds=extraction_seconds,
                imputer=imputer, membership=dataset.splits["membership"],
                bootstrap_replicates=bootstrap_replicates,
                device=device, head_fit_seconds=fit_seconds)
            result.update(feature_variant=variant, status="ok", class_weights=weights,
                          head_fit_seconds=fit_seconds)
            trials.append(result)
        successful = [index for index, row in enumerate(trials) if row["status"] == "ok"]
        if not successful:
            summary["tasks"][task] = {"status": "failed", "feature_trials": trials}
            _save_json(output_dir / "selection.json", summary)
            continue
        selected = max(successful, key=lambda index: trials[index]["macro_f1"])
        selected_run = trials[selected]["run"]
        reference_runs = [rf[0], xgb[0], output_dir / selected_run]
        comparison = compare_runs(reference_runs)
        summary["tasks"][task] = {
            "status": "complete",
            "feature_trials": trials,
            "selected_feature_variant": FEATURE_VARIANTS[selected],
            "selected_run": selected_run,
            "dev_macro_f1": trials[selected]["macro_f1"],
            "rf_reference": str(rf[0]), "xgboost_reference": str(xgb[0]),
            "comparison": comparison,
            "train_windows": len(train.inputs), "dev_windows": len(dev.inputs),
            "train_ids_sha256": ids_hash(train.scenario_ids),
            "dev_cohort_id": dev_cohort.cohort_id,
            "cache_paths": cache_paths,
            "encoder_extraction_seconds": extraction_seconds,
            "head_fit_seconds": train_seconds,
        }
        _save_json(output_dir / "selection.json", summary)

    if encoder is not None and state_digest(encoder) != weights_before:
        raise ValueError("Mantis encoder weights changed during classifier fitting")
    dataset.check_source()
    return summary


def predict_saved_model(dataset, model_run_dir, output_dir, *, split="test",
                        cache_dir="artifacts/cache/mantis", bootstrap_replicates=1000,
                        allow_exploratory=False, device="cpu", batch_size=BATCH_SIZE,
                        encoder=None):
    """Reload a saved logistic head and score one complete release split."""
    path, output_dir = Path(model_run_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("output exists; choose a new experiment directory")
    if split not in ("dev", "test") or batch_size <= 0 or bootstrap_replicates < 0:
        raise ValueError("choose dev or test and use valid batch and bootstrap values")
    saved = load_run(path)
    metadata = saved["metadata"]
    task = metadata.task
    require_ready(dataset, (task,), allow_exploratory=allow_exploratory)
    if metadata.cohort.release_id != dataset.release_id:
        raise ValueError("Mantis checkpoint belongs to a different release")
    config = metadata.configuration
    if (config["checkpoint_sha256"] != CHECKPOINT_SHA256
            or config["checkpoint_revision"] != CHECKPOINT_REVISION
            or config["upstream_commit"] != UPSTREAM_COMMIT):
        raise ValueError("Mantis encoder reference differs from the pinned model")
    content = (path / "head.joblib").read_bytes()
    if _digest(content) != metadata.checkpoint:
        raise ValueError("Mantis head checksum mismatch")
    head = joblib.load(io.BytesIO(content))
    imputer = WindowImputer.from_state(metadata.transformations)
    cohort, batch = split_data(dataset, task, split)
    embeddings, statistics, cache_path, extract_seconds, encoder = _cache_features(
        cache_dir, dataset, cohort, batch, imputer, encoder, split=split,
        device=device, batch_size=batch_size)
    if encoder is not None and any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("Mantis encoder parameters must be frozen")
    features = _variant_features(config["feature_variant"], embeddings, statistics)
    predictions = head["classifier"].predict(head["scaler"].transform(features)).astype(int)
    run_metadata = metadata.model_copy(update={
        "model": f"mantis/{task}/{config['feature_variant']}/{split}", "cohort": cohort,
        "configuration": {**config, "device": device},
        "resources": {"encoder_extraction_seconds": extract_seconds},
    })
    expected = dict(zip(batch.scenario_ids, batch.targets, strict=True))
    groups = dict(zip(batch.scenario_ids, batch.group_ids, strict=True))
    rows = [CasePrediction(scenario_id=sid, task=task, predicted_label=int(label))
            for sid, label in zip(batch.scenario_ids, predictions, strict=True)]
    metrics = write_run(output_dir, run_metadata, expected, groups, rows,
                        bootstrap_replicates=bootstrap_replicates)
    return {"run_dir": str(output_dir), "split": split, "metrics": metrics,
            "cache_path": cache_path}
