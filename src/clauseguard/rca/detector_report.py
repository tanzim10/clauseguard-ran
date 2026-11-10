"""Readiness report for Task 4.2 detector predictions.

Prediction records never carry labels. The evaluation section here is the authorized scoring
stage: it reads release targets separately to show how often the detector agrees with them.
"""

from collections import Counter
from collections.abc import Mapping, Sequence

from clauseguard.rca.detector import CaseRecord


def summarize_cases(records: Sequence[CaseRecord]) -> dict:
    status = {task: dict(Counter(getattr(r, task).prediction.status for r in records))
              for task in ("detection", "category", "diagnosis")}
    return {
        "cases": len(records), "status": status,
        "diagnosis_labels": dict(Counter(r.diagnosis.prediction.predicted_label for r in records
                                         if r.diagnosis.prediction.status == "ok")),
        "text": dict(Counter(f"{r.text.state}:{r.text.reason}" for r in records)),
        "absence_reasons": dict(Counter(r.category.prediction.absence_reason for r in records
                                        if r.category.prediction.status == "absent")),
        "errors": [{"scenario_id": r.scenario_id, "task": task, "error": getattr(r, task).error}
                   for r in records for task in ("detection", "category") if getattr(r, task).error],
    }


def case_table(dataset, selection: Mapping, records: Sequence[CaseRecord],
               case_ids: Mapping[str, str]) -> list[dict]:
    """Per-case predictions plus whether the case's family shaped the selected checkpoints."""
    membership = dataset.splits["membership"]
    models = selection["models"]
    rows = []
    for record in records:
        family = membership[record.scenario_id]["group_id"]
        rows.append({
            "case_id": case_ids[record.scenario_id], "scenario_id": record.scenario_id,
            "diagnosis": record.diagnosis.prediction.predicted_label,
            "detection_confidence": record.detection.prediction.confidence,
            "category_confidence": record.category.prediction.confidence,
            **{f"family_in_{task}_{part}": family in models[task]["training_history"][f"{part}_families"]
               for task in ("detection", "category") for part in ("training", "development")},
        })
    return rows


def evaluate_against_release(dataset, records: Sequence[CaseRecord]) -> dict:
    """Agreement with release targets; scoring only, never fed back into any record."""
    membership = dataset.splits["membership"]
    det = [(membership[r.scenario_id]["detection_target"], r.detection.prediction) for r in records]
    scored = [(target, p.predicted_label_id) for target, p in det if p.status == "ok"]
    faults = [pred for target, pred in scored if target == 1]
    normals = [pred for target, pred in scored if target == 0]
    category = [(membership[r.scenario_id]["category_target"], r.category.prediction) for r in records
                if membership[r.scenario_id]["category_target"] is not None]
    called = [(target, p.predicted_label_id) for target, p in category if p.status == "ok"]
    return {
        "detection": {"scored": len(scored), "faults": len(faults), "faults_detected": sum(faults),
                      "normals": len(normals), "false_alarms": sum(normals)},
        "category": {"category_eligible": len(category), "category_called": len(called),
                     "correct": sum(target == pred for target, pred in called)},
    }


def render_report(selection: Mapping, rca: Mapping, cross: Mapping | None,
                  reproduction: Mapping | None) -> str:
    models = selection["models"]
    lines = [
        "# Task 4.2 detector predictions — readiness report", "",
        (f"Selection `{selection['manifest_sha256'][:12]}` frozen on {selection['frozen_on']} for release "
         f"`{selection['release_id'][:12]}` ({selection['readiness']['evaluation_status']})."), "",
        "## Selected pipeline", "",
        "| Task | Model | Run | Checkpoint | Dev macro-F1 (seed 42) | Development use |",
        "|---|---|---|---|---:|---|",
    ]
    for task in ("detection", "category"):
        m = models[task]
        lines.append(f"| {task} | {m['model']} {m['configuration']} | `{m['run']}` | "
                     f"`{m['checkpoint_sha256'][:12]}` | {m['dev_macro_f1']:.3f} | "
                     f"{', '.join(m['training_history']['development_use'])} |")
    lines += ["", f"Rule: {selection['selection']['rule']}.", "",
              "Development candidates (mean macro-F1):", ""]
    for task, rows in selection["selection"]["candidates"].items():
        lines.append(f"- {task}: " + ", ".join(f"{r['model']} {r['dev_macro_f1']:.3f}" for r in rows))
    lines += ["", (f"Pipeline: detection is {selection['pipeline']['detection']}; category is "
                   f"{selection['pipeline']['category']}; diagnosis is {selection['pipeline']['diagnosis']}."),
              "", "## Readiness and unresolved conditions", ""]
    lines += [f"- {condition}" for condition in selection["readiness"]["conditions"]]
    lines.append(f"- scored eligibility: {selection['readiness']['scored_eligible']}")
    exposure = selection["prior_test_exposure"]
    lines.append(f"- prior test exposure: {len(exposure)} saved test-split runs"
                 + (f" ({', '.join(sorted({e['model'] for e in exposure}))})" if exposure else ""))
    lines += ["", "## RCA evaluation cases", "", _status_table(rca["summary"]), "",
              f"Saved-run agreement (selected dev runs): {rca['saved_run_agreement']}.",
              f"Text: {rca['summary']['text']}.", "",
              ("| Case | Diagnosis prediction | Detection confidence | Family in detection development "
               "| Family in category development |"), "|---|---|---:|---|---|"]
    for row in rca["case_table"]:
        lines.append(f"| {row['case_id']} | {row['diagnosis']} | {row['detection_confidence']:.3f} | "
                     f"{row['family_in_detection_development']} | {row['family_in_category_development']} |")
    lines += ["", "Confidence is an uncalibrated class probability, not a diagnosis probability.", ""]
    if cross is not None:
        lines += ["## SLM training candidates (out-of-fold)", "",
                  f"{cross['windows']} training windows, {cross['folds']} family folds. {cross['fold_rule']}.",
                  "", "| Fold | Held-out families | Detection model | Category model | Predicted windows |",
                  "|---:|---:|---|---|---:|"]
        for row in cross["fold_models"]:
            lines.append(f"| {row['fold']} | {len(row['held_out_families'])} | {_fold_cell(row['detection'])} | "
                         f"{_fold_cell(row['category'])} | {row['predicted_windows']} |")
        lines += ["", _status_table(cross["summary"]), "",
                  f"Separation: {cross['separation']}.", ""]
    lines += ["## Dataset compatibility", "", "| Dataset | Status | Detail |", "|---|---|---|"]
    for name, entry in selection["dataset_compatibility"].items():
        lines.append(f"| {name} | {entry['status']} | {entry.get('detail') or entry.get('reason')} |")
    if reproduction is not None:
        lines += ["", "## Reproduction", "", f"{reproduction}"]
    return "\n".join(lines) + "\n"


def _fold_cell(info: Mapping) -> str:
    if info["status"] != "fitted":
        return f"unavailable ({info['reason']})"
    return f"`{info['checkpoint_sha256'][:12]}`, {info['training_windows']} windows"


def _status_table(summary: Mapping) -> str:
    rows = ["| Head | " + " | ".join(sorted({s for c in summary["status"].values() for s in c})) + " |"]
    statuses = sorted({s for c in summary["status"].values() for s in c})
    rows.append("|---|" + "---:|" * len(statuses))
    for task, counts in summary["status"].items():
        rows.append(f"| {task} | " + " | ".join(str(counts.get(s, 0)) for s in statuses) + " |")
    return "\n".join(rows)
