"""Compare two finished evaluation runs row by row (Week 5).

Rows are matched by ID. Only rows present in both runs with the same input are scored against
each other; added, removed and input-changed rows are listed separately.

Two kinds of difference are kept apart:

- Non-equivalence: the runs used different golden data, corpus, scenario release, inputs or
  metric policy, so their scores do not measure the same thing.
- Experiment variables: mode, retrieval and model settings. These are what an experiment is
  meant to change, so they are shown but do not make the runs non-equivalent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clauseguard.eval.artifacts import METRICS_FILE, ArtifactError, RowRecord, read_run

# Identity keys in run.json and what changing each one means.
EQUIVALENCE_KEYS = {
    "golden_release": "golden release",
    "golden_files": "golden files",
    "corpus": "corpus identity",
    "scenario_release": "scenario release",
    "split": "evaluated split",
    "inputs": "row inputs",
    "policy": "metric policy",
}
_TOLERANCE = 1e-12


def compare_runs(current: Path | str, previous: Path | str) -> dict[str, Any]:
    """Compare `current` against `previous`; both must be finished runs."""
    current_run, current_rows, current_metrics = read_comparable(current)
    previous_run, previous_rows, previous_metrics = read_comparable(previous)
    current, previous = Path(current), Path(previous)

    now = {r.id: r for r in current_rows}
    before = {r.id: r for r in previous_rows}
    common = [row_id for row_id in now if row_id in before]
    input_changed = [i for i in common if now[i].input != before[i].input]
    compared = [i for i in common if i not in input_changed]

    ks = _common_ks(current_run, previous_run)
    per_row = []
    summary = {"improved": 0, "worsened": 0, "mixed": 0, "unchanged": 0}
    metric_changes: dict[str, dict[str, int]] = {}
    for row_id in compared:
        old, new = _row_values(before[row_id], ks), _row_values(now[row_id], ks)
        deltas = {}
        for name in sorted(set(old) & set(new)):
            delta = new[name] - old[name]
            counts = metric_changes.setdefault(name, {"improved": 0, "worsened": 0, "unchanged": 0})
            if delta > _TOLERANCE:
                counts["improved"] += 1
            elif delta < -_TOLERANCE:
                counts["worsened"] += 1
            else:
                counts["unchanged"] += 1
                continue
            deltas[name] = delta
        change = _classify(deltas)
        summary[change] += 1
        outputs = _changed_outputs(before[row_id], now[row_id])
        if deltas or outputs:
            per_row.append(
                {"id": row_id, "change": change, "deltas": deltas, "outputs_changed": outputs}
            )

    reasons = _non_equivalence(current_run, previous_run)
    return {
        "current": _describe(current_run, current),
        "previous": _describe(previous_run, previous),
        "equivalent": not reasons,
        "non_equivalence": reasons,
        "experiment_variables": _experiment_variables(current_run, previous_run),
        "rows": {
            "common": len(common),
            "compared": len(compared),
            "added": sorted(set(now) - set(before)),
            "removed": sorted(set(before) - set(now)),
            "input_changed": input_changed,
        },
        "summary": summary,
        "metric_changes": dict(sorted(metric_changes.items())),
        "headline_delta": _headline_delta(current_metrics, previous_metrics),
        "per_row": per_row,
    }


def read_comparable(
    path: Path | str,
) -> tuple[dict[str, Any], list[RowRecord], dict[str, Any]]:
    """Read a finished run plus its `metrics.json`, raising `ArtifactError` if either is unusable."""
    path = Path(path)
    run, rows = read_run(path)
    try:
        metrics = json.loads((path / METRICS_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactError(f"{path}: no {METRICS_FILE}; replay the run to score it first") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path / METRICS_FILE}: invalid JSON ({exc.msg})") from exc
    return run, rows, metrics


def _row_values(record: RowRecord, ks: list[int]) -> dict[str, float]:
    """Per-row metric values; failed rows score 0 like the headline averages."""
    if record.status == "excluded":
        return {}
    scores = record.scores or {}
    failed = record.status == "failed"
    values: dict[str, float] = {}
    if record.answerable:
        retrieval = scores.get("retrieval") or {}
        cutoffs = retrieval.get("cutoffs", {})
        values["mrr"] = 0.0 if failed else float(retrieval.get("mrr", 0.0))
        for k in ks:
            cut = {} if failed else cutoffs.get(str(k), {})
            values[f"clause_recall@{k}"] = float(cut.get("clause_recall", 0.0))
            values[f"group_coverage@{k}"] = float(cut.get("group_coverage", 0.0))
            values[f"precision@{k}"] = float(cut.get("precision", 0.0))
            values[f"complete_reference_match@{k}"] = float(
                bool(cut.get("complete_reference_match"))
            )
    answer = scores.get("answer")
    if not record.answerable and (answer or failed):
        values["correct_abstention"] = float(not failed and bool(answer["correct_abstention"]))
    return values


def _classify(deltas: dict[str, float]) -> str:
    up = any(d > 0 for d in deltas.values())
    down = any(d < 0 for d in deltas.values())
    if up and down:
        return "mixed"
    return "improved" if up else "worsened" if down else "unchanged"


def _changed_outputs(old: RowRecord, new: RowRecord) -> list[str]:
    changed = []
    if old.status != new.status:
        changed.append("status")
    if _ranking(old) != _ranking(new):
        changed.append("ranking")
    if (old.answer and old.answer.model_dump()) != (new.answer and new.answer.model_dump()):
        changed.append("answer")
    return changed


def _ranking(record: RowRecord) -> list[str] | None:
    return [p.chunk_id for p in record.retrieval.passages] if record.retrieval else None


def _non_equivalence(current: dict[str, Any], previous: dict[str, Any]) -> list[str]:
    now, before = current.get("identities", {}), previous.get("identities", {})
    reasons = []
    for key, label in EQUIVALENCE_KEYS.items():
        if now.get(key) != before.get(key):
            reasons.append(f"{label} changed")
    return reasons


def _experiment_variables(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    variables = {}
    if current.get("mode") != previous.get("mode"):
        variables["mode"] = {"previous": previous.get("mode"), "current": current.get("mode")}
    old = _flatten(previous.get("settings", {}))
    new = _flatten(current.get("settings", {}))
    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            variables[f"settings.{key}"] = {"previous": old.get(key), "current": new.get(key)}
    return variables


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat = {}
    for key, item in value.items():
        name = f"{prefix}{key}"
        if isinstance(item, dict):
            flat.update(_flatten(item, f"{name}."))
        else:
            flat[name] = item
    return flat


def _common_ks(current: dict[str, Any], previous: dict[str, Any]) -> list[int]:
    now = set(current.get("policy", {}).get("ks", []))
    before = set(previous.get("policy", {}).get("ks", []))
    return sorted(now & before)


def _headline_delta(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, float]:
    now = current.get("retrieval", {}).get("headline") or {}
    before = previous.get("retrieval", {}).get("headline") or {}
    return {name: now[name] - before[name] for name in sorted(set(now) & set(before))}


def _describe(run: dict[str, Any], path: Path) -> dict[str, Any]:
    return {"run_id": run.get("run_id"), "path": str(path), "mode": run.get("mode")}
