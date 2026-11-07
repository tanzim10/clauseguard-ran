"""Frozen held-out evaluation and report for the Task 3 KPI baselines (Task 3.8)."""

import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from clauseguard.baselines.artifacts import load_run
from clauseguard.baselines.comparison import _selected_runs, compare_experiments
from clauseguard.baselines.contracts import CasePrediction
from clauseguard.baselines.data import experiment_cohort
from clauseguard.baselines.scoring import score_combined_diagnosis, score_predictions
from clauseguard.datasets.contracts import CATEGORY_LABELS, LABEL_MAPPINGS

MODELS = ("random_forest", "xgboost", "timesnet")
TASKS = ("detection", "category")
CLASS_NAMES = {
    "detection": ("normal", "abnormal"),
    "category": CATEGORY_LABELS,
    "diagnosis": ("normal", *CATEGORY_LABELS),
}
DIAGNOSIS_POLICY = (
    "The detection head decides first. Windows it calls abnormal take the category from the same "
    "model and seed. Scored windows are normal plus the ten synthetic categories; jamming is "
    "scored only in detection. Gold fault labels never replace detector outputs."
)


def _sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _predictor(model):
    """Import lazily so the protocol and report need no model libraries."""
    if model == "random_forest":
        from clauseguard.baselines.random_forest import predict_saved_model
    elif model == "xgboost":
        from clauseguard.baselines.xgboost import predict_saved_model
    elif model == "timesnet":
        from clauseguard.baselines.timesnet import predict_saved_model
    else:
        raise ValueError(f"unsupported model: {model}")
    return predict_saved_model


def find_test_exposure(paths, release_id) -> list[dict]:
    """Every saved test-split run for this release under the given directories."""
    found, seen = [], set()
    for root in paths:
        for run in sorted(Path(root).rglob("run.json")):
            if run.resolve() in seen:
                continue
            seen.add(run.resolve())
            metadata = json.loads(run.read_text(encoding="utf-8"))
            cohort = metadata["cohort"]
            if cohort["split"] == "test" and cohort["release_id"] == release_id:
                found.append({"run": str(run.parent), "model": metadata["model"],
                              "task": metadata["task"], "seed": metadata.get("seed")})
    return found


def _text_status(path):
    if path is None:
        return {"status": "not supplied", "reasons": ["no Task 3.7 text cohort was supplied"]}
    cohort = json.loads(Path(path).read_text(encoding="utf-8"))
    targets = cohort["targets"]
    runnable = all(target["runnable"] for target in targets.values())
    return {"status": "ready" if runnable else "blocked", "cohort": str(path),
            "cohort_sha256": cohort["cohort_sha256"], "selected": cohort["selected"],
            "reasons": sorted({reason for target in targets.values() for reason in target["reasons"]})}


def freeze_protocol(dataset, experiments, output_dir, *, frozen_on, exposure_dirs=(),
                    text_cohort=None) -> dict:
    """Record every choice made on development data before any test prediction is read."""
    if set(experiments) != set(MODELS):
        raise ValueError(f"protocol requires experiments for {', '.join(MODELS)}")
    output_dir = Path(output_dir)
    path = output_dir / "protocol.json"
    if path.exists():
        raise FileExistsError("protocol exists; a frozen protocol is never rewritten")
    experiments = {name: Path(experiments[name]) for name in MODELS}
    models = {name: {"experiment": str(path), "tasks": {}} for name, path in experiments.items()}
    comparators = {}
    for task in TASKS:
        report = compare_experiments(experiments, task)
        if report["release_id"] != dataset.release_id:
            raise ValueError("experiments belong to a different release")
        for name, summary in report["models"].items():
            _, runs = _selected_runs(experiments[name], task)
            models[name]["tasks"][task] = {
                "selected_configuration": summary["selected_configuration"],
                "train_ids_sha256": summary["train_ids_sha256"],
                "dev_cohort_id": summary["dev_cohort_id"],
                "dev_macro_f1_mean": summary["dev_macro_f1_mean"],
                "seeds": [{"seed": run["metadata"].seed, "run": str(run["path"]),
                           "checkpoint": run["metadata"].checkpoint,
                           "dev_macro_f1": run["metrics"]["macro_f1"]} for run in runs],
            }
        comparators[task] = max(report["models"], key=lambda n: report["models"][n]["dev_macro_f1_mean"])
    test_cohorts = {}
    for task in TASKS:
        cohort = experiment_cohort(dataset, split="test", task=task)
        coverage = cohort.coverage["test"]
        test_cohorts[task] = {"cohort_id": cohort.cohort_id, "windows": coverage["windows"],
                              "groups": coverage["groups"], "labels": coverage["labels"]}
    protocol = {
        "version": "1", "frozen_on": frozen_on, "release_id": dataset.release_id,
        "mode": dataset.splits["mode"],
        "evaluation_status": "provisional_exploratory" if dataset.splits["mode"] == "exploratory"
        else "scored",
        "settings_chosen_on": "dev", "seeds": [42, 43, 44], "label_mappings": LABEL_MAPPINGS,
        "score_policy": {
            "primary": "macro_f1",
            "failed_or_missing_predictions": "kept in the denominator and counted incorrect",
            "confidence_intervals": "recording-group bootstrap, only with at least 10 groups",
            "diagnosis": DIAGNOSIS_POLICY,
            "detection_reference": "normal-majority predictor (every window normal)",
        },
        "test_cohorts": test_cohorts, "models": models,
        "dev_comparators": comparators,
        "prior_test_exposure": find_test_exposure([*experiments.values(), *exposure_dirs],
                                             dataset.release_id),
        "text": _text_status(text_cohort),
    }
    protocol["protocol_sha256"] = _sha256(protocol)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(path, protocol)
    return protocol


def load_protocol(output_dir) -> dict:
    protocol = json.loads((Path(output_dir) / "protocol.json").read_text(encoding="utf-8"))
    digest = protocol.pop("protocol_sha256")
    if _sha256(protocol) != digest:
        raise ValueError("protocol changed after it was frozen")
    protocol["protocol_sha256"] = digest
    return protocol


def predict_test(dataset, output_dir, *, allow_exploratory=False, device="cpu",
                 bootstrap_replicates=1000) -> dict:
    """Replay every frozen checkpoint on test. Existing complete runs are reused."""
    protocol = load_protocol(output_dir)
    if protocol["release_id"] != dataset.release_id:
        raise ValueError("protocol belongs to a different release")
    for task in TASKS:
        if experiment_cohort(dataset, split="test", task=task).cohort_id != \
                protocol["test_cohorts"][task]["cohort_id"]:
            raise ValueError(f"{task} test cohort changed after the protocol was frozen")
    runs = {}
    for model in MODELS:
        for task in TASKS:
            for seed in protocol["models"][model]["tasks"][task]["seeds"]:
                source = load_run(seed["run"])["metadata"]
                if source.checkpoint != seed["checkpoint"] or source.seed != seed["seed"]:
                    raise ValueError(f"{seed['run']} differs from the frozen checkpoint")
                target = Path(output_dir) / "runs" / model / task / f"seed-{seed['seed']}"
                if not target.exists():
                    options = {"device": device} if model == "timesnet" else {}
                    _predictor(model)(dataset, seed["run"], target, split="test",
                                      bootstrap_replicates=bootstrap_replicates,
                                      allow_exploratory=allow_exploratory, **options)
                run = load_run(target)
                cohort = run["metadata"].cohort
                if cohort.split != "test" or cohort.cohort_id != protocol["test_cohorts"][task]["cohort_id"]:
                    raise ValueError(f"{target} is not a run on the frozen test cohort")
                runs[model, task, seed["seed"]] = run
    return runs


def matching_case_counts(expected, kpi_predictions, text_predictions) -> dict:
    """Descriptive KPI/text agreement on identical IDs; failures count as incorrect."""
    kpi = {p.scenario_id: p for p in kpi_predictions}
    text = {p.scenario_id: p for p in text_predictions}
    if set(kpi) != set(expected) or set(text) != set(expected):
        raise ValueError("matching-case counts need predictions for exactly the same IDs")

    def correct(prediction, target):
        return prediction.status == "ok" and prediction.predicted_label == target

    counts = Counter()
    for sid, target in expected.items():
        k, t = correct(kpi[sid], target), correct(text[sid], target)
        counts["both_correct" if k and t else "kpi_only" if k else "text_only" if t else "neither"] += 1
    return {key: counts[key] for key in ("both_correct", "kpi_only", "text_only", "neither")}


def _detection_row(metrics):
    return {key: metrics[key] for key in (
        "macro_f1", "accuracy", "precision", "recall", "f1", "balanced_accuracy",
        "false_alarms", "missed_faults", "normal_false_positive_rate")}


def _failed(metrics):
    return sum(v for k, v in metrics["prediction_status_counts"].items() if k != "ok")


def build_tables(dataset, protocol, runs, *, bootstrap_replicates=1000) -> dict:
    """Every table regenerates from the saved per-case predictions."""
    members = dataset.splits["membership"]
    tables = defaultdict(list)
    confusion = {}
    diagnosis_metrics = {}
    for (model, task, seed), run in sorted(runs.items()):
        metrics = run["metrics"]
        row = {"model": model, "seed": seed, "windows": metrics["count"],
               "failed_predictions": _failed(metrics)}
        if task == "detection":
            tables["detection"].append(row | _detection_row(metrics))
        else:
            tables["category"].append(row | {"macro_f1": metrics["macro_f1"],
                                             "accuracy": metrics["accuracy"]})
        confusion[f"{model}/{task}/seed-{seed}"] = metrics["confusion_matrix"]
        tables["per_class"] += [
            {"model": model, "task": task, "seed": seed, "class": int(label),
             "name": CLASS_NAMES[task][int(label)], **values}
            for label, values in metrics["per_class"].items()]
        if task == "detection":
            predicted = {p.scenario_id: p for p in run["predictions"]}
            buckets = defaultdict(Counter)
            for sid, target in run["expected"].items():
                prediction = predicted.get(sid)
                label = prediction.predicted_label if prediction and prediction.status == "ok" else None
                key = (members[sid]["origin"], members[sid]["rca_label"])
                buckets[key]["windows"] += 1
                buckets[key]["correct"] += label == target
                buckets[key]["missed_faults"] += target == 1 and label != 1
                buckets[key]["false_alarms"] += target == 0 and label == 1
                buckets[key]["failed"] += label is None
            tables["errors_by_origin"] += [
                {"model": model, "seed": seed, "origin": origin, "rca_label": rca,
                 **{k: counts[k] for k in ("windows", "correct", "missed_faults",
                                           "false_alarms", "failed")}}
                for (origin, rca), counts in sorted(buckets.items())]
    for model in MODELS:
        for seed in protocol["seeds"]:
            detection, category = runs[model, "detection", seed], runs[model, "category", seed]
            ids = {sid for sid, target in detection["expected"].items()
                   if target == 0 or sid in category["expected"]}
            metrics = score_combined_diagnosis(
                detection["expected"], category["expected"],
                {sid: detection["groups"][sid] for sid in ids},
                detection["predictions"], category["predictions"],
                bootstrap_replicates=bootstrap_replicates)
            diagnosis_metrics[model, seed] = metrics
            tables["diagnosis"].append({"model": model, "seed": seed, "windows": metrics["count"],
                                        "failed_predictions": _failed(metrics),
                                        "macro_f1": metrics["macro_f1"],
                                        "accuracy": metrics["accuracy"]})
            confusion[f"{model}/diagnosis/seed-{seed}"] = metrics["confusion_matrix"]
            tables["per_class"] += [
                {"model": model, "task": "diagnosis", "seed": seed, "class": int(label),
                 "name": CLASS_NAMES["diagnosis"][int(label)], **values}
                for label, values in metrics["per_class"].items()]
    reference_run = runs[MODELS[0], "detection", protocol["seeds"][0]]
    reference = score_predictions(
        "detection", reference_run["expected"], reference_run["groups"],
        [CasePrediction(scenario_id=sid, task="detection", predicted_label=0)
         for sid in reference_run["expected"]], bootstrap_replicates=0)
    tables["detection_reference"].append({"model": "normal_majority", "seed": "",
                                          "windows": reference["count"], "failed_predictions": 0,
                                          **_detection_row(reference)})
    for task, rows in (("detection", tables["detection"]), ("category", tables["category"]),
                       ("diagnosis", tables["diagnosis"])):
        for model in MODELS:
            scores = [row["macro_f1"] for row in rows if row["model"] == model]
            interval = (diagnosis_metrics[model, protocol["seeds"][0]] if task == "diagnosis"
                        else runs[model, task, protocol["seeds"][0]]["metrics"])["confidence_intervals"]
            dev = (protocol["models"][model]["tasks"][task]["dev_macro_f1_mean"]
                   if task != "diagnosis" else None)
            tables["seed_summary"].append({
                "model": model, "task": task, "dev_macro_f1_mean": dev,
                "test_macro_f1_mean": statistics.fmean(scores),
                "test_macro_f1_std": statistics.pstdev(scores),
                "test_macro_f1_min": min(scores), "test_macro_f1_max": max(scores),
                "group_interval": "available" if interval["available"] else
                f"unavailable ({interval['groups']} groups; {interval.get('reason', '')})",
            })
    for model in MODELS:
        for task in TASKS:
            costs = [runs[model, task, seed]["metadata"].resources for seed in protocol["seeds"]]
            sizes = [_model_size(Path(seed["run"])) for seed in
                     protocol["models"][model]["tasks"][task]["seeds"]]
            dev_resources = [load_run(seed["run"])["metadata"].resources for seed in
                             protocol["models"][model]["tasks"][task]["seeds"]]
            tables["cost"].append({
                "model": model, "task": task,
                "fit_seconds_mean": _mean_of(dev_resources, "fit_seconds"),
                "test_prediction_seconds_mean": _mean_of(costs, "prediction_seconds"),
                "process_peak_rss_gb_max": max((r.get("process_peak_rss_bytes", 0.0)
                                                for r in dev_resources), default=0.0) / 1e9,
                "model_size_mb_mean": statistics.fmean(sizes) / 1e6 if all(sizes) else None,
            })
    return {"tables": dict(tables), "confusion": confusion}


def _model_size(path):
    for name in ("model.joblib", "model.ubj", "model.pt"):
        if (path / name).exists():
            return (path / name).stat().st_size
    return None


def _mean_of(rows, key):
    values = [row[key] for row in rows if key in row]
    return statistics.fmean(values) if values else None


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_report(protocol, results) -> str:
    tables = results["tables"]
    summary = {(row["model"], row["task"]): row for row in tables["seed_summary"]}
    status = protocol["evaluation_status"]
    lines = [
        "# Task 3 held-out evaluation of the KPI baselines", "",
        (f"Release `{protocol['release_id'][:12]}` ({status}). Protocol `{protocol['protocol_sha256'][:12]}`, "
        f"frozen on {protocol['frozen_on']} from development results only. Seeds "
        f"{', '.join(map(str, protocol['seeds']))}."), "",
        ("**Provisional.** Recording independence is unresolved, so these are exploratory "
         "numbers, not a scored release." if status != "scored" else "Scored release."),
        "**Earlier test exposure:** "
        + (", ".join(dict.fromkeys(f"`{row['run']}`" for row in protocol["prior_test_exposure"]))
           or "none recorded")
        + ". These test results are therefore not a fresh, untouched evaluation.", "",
        "## Detection", "",
        (f"Test: {protocol['test_cohorts']['detection']['windows']} windows "
        f"({protocol['test_cohorts']['detection']['labels']}), "
        f"{protocol['test_cohorts']['detection']['groups']} recording groups."), "",
        "| Model | Dev macro-F1 | Test macro-F1 (mean ± sd) | Seeds | Fault recall | False alarms | Missed faults | Failed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model in MODELS:
        rows = [row for row in tables["detection"] if row["model"] == model]
        s = summary[model, "detection"]
        lines.append(
            f"| {model} | {_fmt(s['dev_macro_f1_mean'])} | {_fmt(s['test_macro_f1_mean'])} ± "
            f"{_fmt(s['test_macro_f1_std'])} | {' / '.join(_fmt(r['macro_f1']) for r in rows)} | "
            f"{' / '.join(_fmt(r['recall']) for r in rows)} | "
            f"{' / '.join(str(r['false_alarms']) for r in rows)} | "
            f"{' / '.join(str(r['missed_faults']) for r in rows)} | "
            f"{sum(r['failed_predictions'] for r in rows)} |")
    ref = tables["detection_reference"][0]
    lines += [
        (f"| normal-majority reference | n/a | {_fmt(ref['macro_f1'])} | n/a | {_fmt(ref['recall'])} | "
        f"{ref['false_alarms']} | {ref['missed_faults']} | 0 |"), "",
        (f"The reference shows why accuracy alone misleads: calling every window normal already "
        f"reaches {ref['accuracy']:.1%} accuracy."), "",
        "## Fault category (known synthetic faults)", "",
        (f"Test: {protocol['test_cohorts']['category']['windows']} windows, "
        f"{protocol['test_cohorts']['category']['groups']} recording groups."), "",
        "| Model | Dev macro-F1 | Test macro-F1 (mean ± sd) | Seeds | Accuracy | Failed |",
        "|---|---|---|---|---|---|",
    ]
    for model in MODELS:
        rows = [row for row in tables["category"] if row["model"] == model]
        s = summary[model, "category"]
        lines.append(
            f"| {model} | {_fmt(s['dev_macro_f1_mean'])} | {_fmt(s['test_macro_f1_mean'])} ± "
            f"{_fmt(s['test_macro_f1_std'])} | {' / '.join(_fmt(r['macro_f1']) for r in rows)} | "
            f"{' / '.join(_fmt(r['accuracy']) for r in rows)} | "
            f"{sum(r['failed_predictions'] for r in rows)} |")
    lines += ["", "## Combined diagnosis (detector, then category)", "", DIAGNOSIS_POLICY, "",
              "| Model | Test macro-F1 (mean ± sd) | Seeds | Accuracy | Windows |",
              "|---|---|---|---|---|"]
    for model in MODELS:
        rows = [row for row in tables["diagnosis"] if row["model"] == model]
        s = summary[model, "diagnosis"]
        lines.append(
            f"| {model} | {_fmt(s['test_macro_f1_mean'])} ± {_fmt(s['test_macro_f1_std'])} | "
            f"{' / '.join(_fmt(r['macro_f1']) for r in rows)} | "
            f"{' / '.join(_fmt(r['accuracy']) for r in rows)} | {rows[0]['windows']} |")
    lines += ["", "## Detection errors by origin (mean over seeds)", "",
              "| Origin | Label | Windows | " + " | ".join(f"{m} missed / false alarms" for m in MODELS) + " |",
              "|---|---|---|" + "---|" * len(MODELS)]
    keys = sorted({(r["origin"], r["rca_label"]) for r in tables["errors_by_origin"]})
    for origin, rca in keys:
        cells = []
        windows = None
        for model in MODELS:
            rows = [r for r in tables["errors_by_origin"]
                    if r["model"] == model and (r["origin"], r["rca_label"]) == (origin, rca)]
            windows = rows[0]["windows"]
            cells.append(f"{statistics.fmean(r['missed_faults'] for r in rows):.1f} / "
                         f"{statistics.fmean(r['false_alarms'] for r in rows):.1f}")
        lines.append(f"| {origin} | {rca} | {windows} | " + " | ".join(cells) + " |")
    lines += ["", "## Uncertainty", "",
              "Seed variation is shown above. Recording-group intervals need at least 10 groups:", ""]
    lines += [f"- {row['model']} {row['task']}: {row['group_interval']}" for row in tables["seed_summary"]
              if row["model"] == MODELS[0]]
    lines += ["", ("Overlapping windows from one recording are not independent samples, so the "
              "window counts above overstate the evidence."), "", "## Cost", "",
              "| Model | Task | Fit s (dev runs) | Test predict s | Peak RSS GB | Size MB |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {r['model']} | {r['task']} | {_fmt(r['fit_seconds_mean'], 1)} | "
              f"{_fmt(r['test_prediction_seconds_mean'], 2)} | {_fmt(r['process_peak_rss_gb_max'], 2)} | "
              f"{_fmt(r['model_size_mb_mean'], 2)} |" for r in tables["cost"]]
    lines += ["", ("Peak RSS is a process-wide high-water mark and fit times come from different "
              "devices, so cost columns are indicative only."), "", "## Text baseline", "",
              f"Status: **{protocol['text']['status']}**."]
    lines += [f"- {reason}" for reason in protocol["text"]["reasons"]]
    lines += ["", ("The KPI/text matching-case analysis (both correct, KPI only, text only, neither) "
              "needs approved text, so the full Task 3 KPI/text comparison is not complete. "
              "This is a KPI-only report."), "",
              "## Limitations and what Task 4 should use", "",
              (f"- Development comparators: detection `{protocol['dev_comparators']['detection']}`, "
              f"category `{protocol['dev_comparators']['category']}`."),
              ("- A predicted category is a pattern match to a synthetic label, not proof of the "
              "physical cause. Task 4 must add specification evidence before any causal claim."),
              ("- Task 4 should consume the per-case predictions in `runs/`, the missed-fault and "
              "false-alarm counts by origin, and the per-class F1, not a single headline score."),
              ("- The category test set has only two recording groups; per-class scores can "
              "change a lot with one recording."), ""]
    return "\n".join(lines)


def write_report(dataset, output_dir, runs, *, bootstrap_replicates=1000) -> dict:
    output_dir = Path(output_dir)
    protocol = load_protocol(output_dir)
    results = build_tables(dataset, protocol, runs, bootstrap_replicates=bootstrap_replicates)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    for name, rows in results["tables"].items():
        _write_csv(tables_dir / f"{name}.csv", rows)
    _save_json(output_dir / "confusion.json", results["confusion"])
    _save_json(output_dir / "results.json", {"protocol_sha256": protocol["protocol_sha256"],
                                             **results["tables"]})
    (output_dir / "report.md").write_text(render_report(protocol, results), encoding="utf-8")
    return results
