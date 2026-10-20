"""Development-only comparison of saved baseline experiments, built from saved runs alone."""

import hashlib
import json
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from clauseguard.baselines.artifacts import load_run
from clauseguard.baselines.scoring import score_predictions

MODEL_FILES = ("model.joblib", "model.ubj", "model.pt")
SELECTION_TOLERANCE = 1e-9


def load_window_selection(path: Path) -> dict[str, Any]:
    """Read an audit selection of scenario IDs to rescore every model on."""
    selection = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = selection["scenario_ids"]
    if len(ids) != len(set(ids)):
        raise ValueError("window selection contains duplicate IDs")
    digest = hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()
    return {"policy": selection.get("policy"), "release_id": selection["release_id"],
            "ids": set(ids), "ids_sha256": digest}


def _restrict_run(run: dict[str, Any], selection: dict[str, Any], task: str) -> dict[str, Any]:
    """Rescore a saved run on the selected windows only; every fault window must stay."""
    if run["metadata"].cohort.release_id != selection["release_id"]:
        raise ValueError("window selection belongs to a different release")
    keep = selection["ids"]
    positives = [sid for sid, target in run["expected"].items() if task == "category" or target == 1]
    if any(sid not in keep for sid in positives):
        raise ValueError("window selection must retain every fault window")
    expected = {sid: target for sid, target in run["expected"].items() if sid in keep}
    groups = {sid: run["groups"][sid] for sid in expected}
    predictions = [p for p in run["predictions"] if p.scenario_id in expected]
    metrics = score_predictions(task, expected, groups, predictions, bootstrap_replicates=0)
    return {**run, "expected": expected, "groups": groups, "predictions": predictions,
            "metrics": metrics}


def _selected_runs(experiment: Path, task: str, window_selection: dict[str, Any] | None = None,
                   ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection = json.loads((experiment / "selection.json").read_text(encoding="utf-8"))
    entry = selection["tasks"].get(task)
    if entry is None:
        raise ValueError(f"{experiment.name} has no {task} selection")
    runs = []
    for chosen in entry["selected_seeds"]:
        run = load_run(experiment / chosen["run"])
        metadata = run["metadata"]
        if metadata.task != task or metadata.cohort.split != "dev":
            raise ValueError(f"{chosen['run']} is not a {task} development run")
        if metadata.seed != chosen["seed"] or metadata.cohort.cohort_id != entry["dev_cohort_id"]:
            raise ValueError(f"{chosen['run']} does not match its selection record")
        if abs(run["metrics"]["macro_f1"] - chosen["dev_macro_f1"]) > SELECTION_TOLERANCE:
            raise ValueError(f"{chosen['run']} saved score differs from the selection record")
        if window_selection is not None:
            run = _restrict_run(run, window_selection, task)
        runs.append({**run, "path": experiment / chosen["run"]})
    if not runs:
        raise ValueError(f"{experiment.name} selected no {task} runs")
    return {**selection, **entry}, runs


def _size_bytes(path: Path) -> int | None:
    for name in MODEL_FILES:
        if (path / name).exists():
            return (path / name).stat().st_size
    return None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_model(experiment: Path, task: str,
                    window_selection: dict[str, Any] | None = None) -> dict[str, Any]:
    """Summarize one experiment's selected development seeds for a single task."""
    entry, runs = _selected_runs(Path(experiment), task, window_selection)
    scores = [run["metrics"]["macro_f1"] for run in runs]
    seeds = []
    for run in runs:
        metrics = run["metrics"]
        row = {"seed": run["metadata"].seed, "macro_f1": metrics["macro_f1"],
               "accuracy": metrics["accuracy"],
               "prediction_status_counts": metrics["prediction_status_counts"]}
        if task == "detection":
            row.update(false_alarms=metrics["false_alarms"], missed_faults=metrics["missed_faults"])
        seeds.append(row)
    resources = [run["metadata"].resources for run in runs]
    sizes = [size for run in runs if (size := _size_bytes(run["path"])) is not None]
    labels = runs[0]["metrics"]["per_class"]
    summary = {
        "model": runs[0]["metadata"].model.split("/")[0],
        "experiment": str(experiment),
        "release_id": entry["release_id"],
        "evaluation_status": entry["evaluation_status"],
        "dev_cohort_id": entry["dev_cohort_id"],
        "train_ids_sha256": entry["train_ids_sha256"],
        "train_windows": entry["train_windows"],
        "selected_configuration": entry["selected_configuration"],
        "trials": len(entry["trials"]),
        "dev_macro_f1_mean": statistics.fmean(scores),
        "dev_macro_f1_std": statistics.pstdev(scores),
        "dev_macro_f1_median": statistics.median(scores),
        "dev_macro_f1_range": [min(scores), max(scores)],
        "seeds": seeds,
        "per_class_f1_mean": {label: statistics.fmean(run["metrics"]["per_class"][label]["f1"]
                                                      for run in runs) for label in labels},
        "class_support": runs[0]["metrics"]["class_support"],
        "cost": {
            "fit_seconds_mean": _mean([r["fit_seconds"] for r in resources if "fit_seconds" in r]),
            "prediction_seconds_mean": _mean([r["prediction_seconds"] for r in resources
                                              if "prediction_seconds" in r]),
            "process_peak_rss_bytes_max": max((r.get("process_peak_rss_bytes", 0.0)
                                               for r in resources), default=0.0) or None,
            "model_size_bytes": _mean(sizes),
            "platforms": sorted({run["metadata"].versions.get("platform", "unknown")
                                 for run in runs}),
        },
    }
    summary["_identity"] = (runs[0]["expected"], runs[0]["groups"])
    return summary


def compare_experiments(experiments: Mapping[str, Path], task: str,
                        window_selection: Path | None = None) -> dict[str, Any]:
    """Compare models only when release, cohort, scored cases, and training data are identical.

    With ``window_selection`` every model is rescored from its saved predictions on that subset.
    """
    if len(experiments) < 2:
        raise ValueError("comparison requires at least two experiments")
    selection = load_window_selection(window_selection) if window_selection else None
    summaries = {name: summarize_model(path, task, selection)
                 for name, path in experiments.items()}
    first_name, first = next(iter(summaries.items()))
    for name, summary in summaries.items():
        for key in ("release_id", "dev_cohort_id", "train_ids_sha256", "evaluation_status"):
            if summary[key] != first[key]:
                raise ValueError(f"{name} differs from {first_name} in {key}")
        if summary["_identity"] != first["_identity"]:
            raise ValueError(f"{name} differs from {first_name} in scored cases or groups")
    windows = len(first["_identity"][0])
    for summary in summaries.values():
        del summary["_identity"]
    report = {"task": task, "split": "dev", "release_id": first["release_id"],
              "evaluation_status": first["evaluation_status"],
              "dev_cohort_id": first["dev_cohort_id"], "models": summaries}
    if selection is not None:
        report["window_selection"] = {"policy": selection["policy"],
                                      "ids_sha256": selection["ids_sha256"], "windows": windows}
    return report


def _fmt(value: float | None, scale: float = 1.0, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value / scale:.{digits}f}"


def render_markdown(reports: list[dict[str, Any]]) -> str:
    """Render development tables, one section per task, with no test-set numbers."""
    lines = ["# KPI model comparison (development split only)", ""]
    for report in reports:
        task, models = report["task"], report["models"]
        identity = (f"Release `{report['release_id'][:8]}`, status `{report['evaluation_status']}`, "
                    f"dev cohort `{report['dev_cohort_id'][:8]}`.")
        lines += [f"## {task.capitalize()}", "", identity, ""]
        if "window_selection" in report:
            window = report["window_selection"]
            note = (f"Rescored from saved predictions on {window['windows']} selected windows "
                    f"(policy `{window['policy']}`, hash `{window['ids_sha256'][:8]}`).")
            lines += [note, ""]
        extra = " | False alarms | Missed faults" if task == "detection" else ""
        header = (f"| Model | Dev macro-F1 | Median | Seeds | Accuracy{extra} | Failed predictions | "
                  "Fit s | Predict s | Peak RSS GB | Size MB |")
        lines += [header,
                  "|---|---|---|---|---|" + ("---|---|" if extra else "") + "---|---|---|---|---|"]
        for name, model in models.items():
            seeds = model["seeds"]
            cells = [name, f"{model['dev_macro_f1_mean']:.3f} ± {model['dev_macro_f1_std']:.3f}",
                     f"{model['dev_macro_f1_median']:.3f}",
                     " / ".join(f"{s['macro_f1']:.3f}" for s in seeds),
                     " / ".join(f"{s['accuracy']:.3f}" for s in seeds)]
            if extra:
                cells += [" / ".join(str(s["false_alarms"]) for s in seeds),
                          " / ".join(str(s["missed_faults"]) for s in seeds)]
            failed = sum(sum(v for k, v in s["prediction_status_counts"].items() if k != "ok")
                         for s in seeds)
            cost = model["cost"]
            cells += [str(failed), _fmt(cost["fit_seconds_mean"]),
                      _fmt(cost["prediction_seconds_mean"], digits=2),
                      _fmt(cost["process_peak_rss_bytes_max"], 1e9, 2),
                      _fmt(cost["model_size_bytes"], 1e6, 2)]
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", "Selected settings:", ""]
        lines += [f"- {name}: `{json.dumps(model['selected_configuration'], sort_keys=True)}` "
                  f"({model['train_windows']} training windows, {model['trials']} trials)"
                  for name, model in models.items()]
        if task == "category":
            labels = list(next(iter(models.values()))["per_class_f1_mean"])
            support = next(iter(models.values()))["class_support"]
            lines += ["", "Per-class F1 (mean over seeds):", "",
                      "| Class | Dev support | " + " | ".join(models) + " |",
                      "|---|---|" + "---|" * len(models)]
            lines += [f"| {label} | {support[label]} | "
                      + " | ".join(f"{m['per_class_f1_mean'][label]:.3f}" for m in models.values())
                      + " |" for label in labels]
        lines.append("")
    return "\n".join(lines)
