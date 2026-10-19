"""Write-once, replayable local baseline run artifacts."""

import hashlib
import json
from pathlib import Path
from typing import Any

from clauseguard.baselines.contracts import CasePrediction, RunMetadata
from clauseguard.baselines.data import experiment_targets
from clauseguard.baselines.scoring import score_predictions
from clauseguard.datasets.scenarios import ScenarioDataset

FILES = ("run.json", "targets.json", "predictions.jsonl", "scoring.json", "metrics.json")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _prediction_bytes(predictions) -> bytes:
    return b"".join(_json_bytes(item.model_dump(mode="json")) + b"\n" for item in predictions)


def write_run(path: Path, metadata: RunMetadata, expected: dict[str, int],
              groups: dict[str, str], predictions: list[CasePrediction], *, seed: int = 42,
              bootstrap_replicates: int = 1000) -> dict[str, Any]:
    """Save one immutable run directory and return its freshly computed metrics."""
    metadata.validate_cohort()
    if not set(expected).issubset(metadata.cohort.scenario_ids):
        raise ValueError("run targets include IDs outside the declared cohort")
    if metadata.task != "diagnosis" and set(expected) != set(metadata.cohort.scenario_ids):
        raise ValueError("run targets must cover the complete declared cohort")
    if set(groups) != set(expected):
        raise ValueError("group IDs must cover exactly the run targets")
    metrics = score_predictions(metadata.task, expected, groups, predictions, seed=seed,
                                bootstrap_replicates=bootstrap_replicates)
    values = {
        "run.json": _json_bytes(metadata.model_dump(mode="json")),
        "targets.json": _json_bytes({"expected": expected, "groups": groups}),
        "predictions.jsonl": _prediction_bytes(predictions),
        "scoring.json": _json_bytes({"seed": seed, "bootstrap_replicates": bootstrap_replicates}),
        "metrics.json": _json_bytes(metrics),
    }
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    for name, content in values.items():
        (path / name).write_bytes(content)
    integrity = {name: _digest(content) for name, content in values.items()}
    (path / "integrity.json").write_bytes(_json_bytes(integrity))
    return metrics


def write_experiment_run(path: Path, dataset: ScenarioDataset, metadata: RunMetadata,
                         predictions: list[CasePrediction], *, seed: int = 42,
                         bootstrap_replicates: int = 1000) -> dict[str, Any]:
    """Resolve labels and recording groups from the validated release before saving."""
    expected, groups = experiment_targets(dataset, metadata.cohort, task=metadata.task)
    return write_run(path, metadata, expected, groups, predictions, seed=seed,
                     bootstrap_replicates=bootstrap_replicates)


def load_run(path: Path) -> dict[str, Any]:
    """Verify and rescore a run using only its saved per-case records."""
    path = Path(path)
    integrity = json.loads((path / "integrity.json").read_text(encoding="utf-8"))
    contents = {name: (path / name).read_bytes() for name in FILES}
    if integrity != {name: _digest(content) for name, content in contents.items()}:
        raise ValueError("run artifact checksum mismatch")
    metadata = RunMetadata.model_validate_json(contents["run.json"])
    metadata.validate_cohort()
    targets = json.loads(contents["targets.json"])
    expected, groups = targets["expected"], targets["groups"]
    predictions = [CasePrediction.model_validate_json(line) for line in
                   contents["predictions.jsonl"].splitlines()]
    saved_metrics = json.loads(contents["metrics.json"])
    scoring = json.loads(contents["scoring.json"])
    recomputed = score_predictions(metadata.task, expected, groups, predictions, **scoring)
    if saved_metrics != recomputed:
        raise ValueError("saved metrics do not match predictions")
    return {"metadata": metadata, "expected": expected, "groups": groups,
            "predictions": predictions, "metrics": recomputed}


def compare_runs(paths: list[Path]) -> dict[str, Any]:
    """Summarize runs only when every model used identical targets and cohort identity."""
    if len(paths) < 2:
        raise ValueError("comparison requires at least two runs")
    runs = [load_run(path) for path in paths]
    model_names = [run["metadata"].model for run in runs]
    if len(set(model_names)) != len(model_names):
        raise ValueError("comparison requires unique model names")
    first = runs[0]
    for run in runs[1:]:
        a, b = first["metadata"], run["metadata"]
        if (a.task, a.cohort.cohort_id) != (b.task, b.cohort.cohort_id):
            raise ValueError("cannot compare runs with different task or cohort identity")
        if (first["expected"], first["groups"]) != (run["expected"], run["groups"]):
            raise ValueError("cannot compare runs with different scored cases or groups")
    return {
        "task": first["metadata"].task,
        "cohort_id": first["metadata"].cohort.cohort_id,
        "runs": {
            run["metadata"].model: {
                "metrics": run["metrics"],
                "prediction_status_counts": run["metrics"]["prediction_status_counts"],
            }
            for run in runs
        },
    }
