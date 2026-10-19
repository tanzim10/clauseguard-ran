"""Development-only RF selection and separately invoked held-out prediction."""

import json
import platform
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from clauseguard.baselines.artifacts import load_run, write_run
from clauseguard.baselines.contracts import RunMetadata
from clauseguard.baselines.operations import predict_adapter
from clauseguard.baselines.tree_data import (
    ids_hash as _ids_hash,
)
from clauseguard.baselines.tree_data import (
    require_ready as _require_ready,
)
from clauseguard.baselines.tree_data import (
    select_training as _select_training,
)
from clauseguard.baselines.tree_data import (
    split_data as _split_data,
)
from clauseguard.baselines.tree_data import (
    training_ids as _training_ids,
)
from clauseguard.datasets.scenarios import ScenarioDataset
from clauseguard.kpi.features import ChannelMedianFeatures
from clauseguard.kpi.random_forest import (
    CONFIGURATIONS,
    RandomForestAdapter,
    RandomForestModel,
    fit_forest,
    load_model,
    serialize_model,
)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _versions():
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: version(name) for name in ("numpy", "scikit-learn", "joblib")}}


def _record_run(path, model, cohort, data, *, train_cohort_id, training_ids_hash, resources,
                bootstrap_replicates, allow_exploratory=False):
    started = time.perf_counter()
    predictions = predict_adapter(RandomForestAdapter(model.task), model, cohort, [data])
    prediction_seconds = time.perf_counter() - started
    content, checksum = serialize_model(model)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metadata = RunMetadata(
        model=f"random_forest/{model.task}/{path.name}", task=model.task, cohort=cohort,
        train_cohort_id=train_cohort_id, seed=model.seed, versions=_versions(),
        configuration={"n_estimators": 300, "class_weight": "balanced", "n_jobs": 1,
                       **model.configuration},
        transformations={"channel_medians": model.features.medians_.tolist(),
                         "feature_order": list(model.features.feature_names_),
                         "training_ids_sha256": training_ids_hash,
                         **({"evaluation_status": "provisional_exploratory",
                             "evaluation_note": "Recording independence is unresolved; interpret metrics as provisional."}
                            if allow_exploratory else {})},
        checkpoint=checksum,
        resources={**resources, "prediction_seconds": prediction_seconds,
                   "process_peak_rss_bytes": float(peak_rss * (1 if sys.platform == "darwin" else 1024))},
    )
    expected = dict(zip(data.scenario_ids, data.targets, strict=True))
    groups = dict(zip(data.scenario_ids, data.group_ids, strict=True))
    metrics = write_run(path, metadata, expected, groups, predictions,
                        bootstrap_replicates=bootstrap_replicates)
    (path / "model.joblib").write_bytes(content)
    errors = [{"scenario_id": row.scenario_id, "expected": expected[row.scenario_id],
               "predicted": row.predicted_label, "status": row.status, "error": row.error}
              for row in predictions
              if row.status != "ok" or expected[row.scenario_id] != row.predicted_label]
    _write_json(path / "errors.json", errors)
    return metrics


def run_random_forest(dataset: ScenarioDataset, output_dir: Path, *,
                      tasks=("detection", "category"), training_selection: Path | None = None,
                      bootstrap_replicates: int = 1000,
                      allow_exploratory: bool = False) -> dict:
    """Tune on dev only; keep all dev cases and save selected seeds42/43/44."""
    tasks = tuple(tasks)
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("choose one or both distinct tasks")
    _require_ready(dataset, tasks, allow_exploratory=allow_exploratory)
    provisional = dataset.splits["mode"] == "exploratory"
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("output exists; choose a new experiment directory")
    ids = _training_ids(dataset, training_selection)
    output_dir.mkdir(parents=True)
    summary = {"release_id": dataset.release_id, "selection_split": "dev",
               "evaluation_status": ("provisional_exploratory" if provisional else "scored"),
               "tasks": {}}
    for task in tasks:
        train_cohort, full_train = _split_data(dataset, task, "train")
        train = _select_training(full_train, ids)
        dev_cohort, dev = _split_data(dataset, task, "dev")
        started = time.perf_counter()
        transformer = ChannelMedianFeatures().fit(train.inputs)
        train_features = transformer.transform(train.inputs)
        feature_seconds = time.perf_counter() - started
        trials = []
        for index, configuration in enumerate(CONFIGURATIONS):
            path = output_dir / task / f"config-{index}-seed-42"
            started = time.perf_counter()
            forest = fit_forest(train_features, train.targets, task=task,
                                configuration=configuration, seed=42)
            fit_seconds = time.perf_counter() - started
            model = RandomForestModel(task, transformer, forest, dict(configuration), 42)
            metrics = _record_run(
                path, model, dev_cohort, dev, train_cohort_id=train_cohort.cohort_id,
                training_ids_hash=_ids_hash(train.scenario_ids),
                resources={"fit_seconds": fit_seconds, "feature_seconds": feature_seconds},
                bootstrap_replicates=bootstrap_replicates, allow_exploratory=provisional,
            )
            trials.append({"configuration": configuration, "dev_macro_f1": metrics["macro_f1"],
                           "run": str(path.relative_to(output_dir))})
        # Stable first-wins ties prefer the shallower tree and larger leaf from CONFIGURATIONS.
        best = max(trials, key=lambda trial: trial["dev_macro_f1"])
        selected = [{"seed": 42, "dev_macro_f1": best["dev_macro_f1"], "run": best["run"]}]
        for seed in (43, 44):
            path = output_dir / task / f"selected-seed-{seed}"
            started = time.perf_counter()
            forest = fit_forest(train_features, train.targets, task=task,
                                configuration=best["configuration"], seed=seed)
            fit_seconds = time.perf_counter() - started
            model = RandomForestModel(task, transformer, forest, dict(best["configuration"]), seed)
            metrics = _record_run(
                path, model, dev_cohort, dev, train_cohort_id=train_cohort.cohort_id,
                training_ids_hash=_ids_hash(train.scenario_ids),
                resources={"fit_seconds": fit_seconds, "feature_seconds": feature_seconds},
                bootstrap_replicates=bootstrap_replicates, allow_exploratory=provisional,
            )
            selected.append({"seed": seed, "dev_macro_f1": metrics["macro_f1"],
                             "run": str(path.relative_to(output_dir))})
        scores = [row["dev_macro_f1"] for row in selected]
        summary["tasks"][task] = {
            "trials": trials, "selected_configuration": best["configuration"],
            "selected_seeds": selected, "dev_macro_f1_mean": float(np.mean(scores)),
            "dev_macro_f1_std": float(np.std(scores)),
            "train_windows": len(train.inputs), "full_train_windows": len(full_train.inputs),
            "train_ids_sha256": _ids_hash(train.scenario_ids), "dev_cohort_id": dev_cohort.cohort_id,
        }
        _write_json(output_dir / task / "training_ids.json", list(train.scenario_ids))
        _write_json(output_dir / "selection.json", summary)
    dataset.check_source()
    return summary


def predict_saved_model(dataset: ScenarioDataset, model_run_dir: Path, output_dir: Path, *,
                         split: str = "test", bootstrap_replicates: int = 1000,
                         allow_exploratory: bool = False) -> dict:
    """Score a saved checkpoint on a requested split; never refit preprocessing or weights."""
    model_run_dir = Path(model_run_dir)
    saved = load_run(model_run_dir)
    metadata = saved["metadata"]
    if metadata.cohort.release_id != dataset.release_id:
        raise ValueError("checkpoint belongs to a different release")
    _require_ready(dataset, (metadata.task,), allow_exploratory=allow_exploratory)
    if metadata.cohort.mode == "exploratory" and not allow_exploratory:
        raise ValueError("exploratory checkpoint prediction requires explicit opt-in")
    model = load_model(model_run_dir / "model.joblib", expected_sha256=metadata.checkpoint)
    if model.task != metadata.task:
        raise ValueError("checkpoint task differs from saved run")
    if (list(model.features.feature_names_) != metadata.transformations["feature_order"]
            or model.features.medians_.tolist() != metadata.transformations["channel_medians"]):
        raise ValueError("checkpoint preprocessing differs from saved run")
    provisional = dataset.splits["mode"] == "exploratory"
    cohort, data = _split_data(dataset, model.task, split)
    return _record_run(
        Path(output_dir), model, cohort, data, train_cohort_id=metadata.train_cohort_id,
        training_ids_hash=metadata.transformations["training_ids_sha256"],
        resources={}, bootstrap_replicates=bootstrap_replicates,
        allow_exploratory=provisional,
    )
