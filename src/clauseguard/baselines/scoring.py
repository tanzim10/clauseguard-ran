"""Fixed-cohort metrics for detection, category, and combined diagnosis runs."""

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from clauseguard.baselines.contracts import CasePrediction, Task

MISSING = "missing"


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def _class_metrics(labels: Sequence[int], actual: Sequence[int], predicted: Sequence[int | None]):
    confusion = {str(label): {str(col): 0 for col in (*labels, MISSING)} for label in labels}
    per_class = {}
    for label in labels:
        support = sum(value == label for value in actual)
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))
        per_class[str(label)] = {
            "support": support,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / support if support else 0.0,
            "f1": _f1(tp, fp, fn),
        }
    for a, p in zip(actual, predicted):
        confusion[str(a)][str(p) if p is not None else MISSING] += 1
    total = len(actual)
    accuracy = sum(a == p for a, p in zip(actual, predicted)) / total if total else 0.0
    return accuracy, per_class, confusion


def _metric_block(task: Task, actual: Sequence[int], predicted: Sequence[int | None]) -> dict[str, Any]:
    labels = (0, 1) if task == "detection" else tuple(range(11 if task == "diagnosis" else 10))
    accuracy, per_class, confusion = _class_metrics(labels, actual, predicted)
    result: dict[str, Any] = {
        "count": len(actual),
        "accuracy": accuracy,
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(labels),
        "class_support": {key: row["support"] for key, row in per_class.items()},
        "per_class": per_class,
        "confusion_matrix": confusion,
    }
    if task == "detection":
        positive = per_class["1"]
        normal_support = per_class["0"]["support"]
        false_alarms = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
        result.update(
            precision=positive["precision"], recall=positive["recall"], f1=positive["f1"],
            balanced_accuracy=sum(row["recall"] for row in per_class.values()) / 2,
            normal_false_positive_rate=false_alarms / normal_support if normal_support else 0.0,
            false_alarms=false_alarms,
            missed_faults=sum(a == 1 and p == 0 for a, p in zip(actual, predicted)),
        )
    return result


def _prepare(task: Task, expected: Mapping[str, int], predictions: Sequence[CasePrediction]):
    seen: dict[str, CasePrediction] = {}
    for prediction in predictions:
        if prediction.task != task:
            raise ValueError(f"prediction task mismatch for {prediction.scenario_id}")
        if prediction.scenario_id not in expected:
            raise ValueError(f"prediction outside cohort: {prediction.scenario_id}")
        if prediction.scenario_id in seen:
            raise ValueError(f"duplicate prediction: {prediction.scenario_id}")
        seen[prediction.scenario_id] = prediction
    labels = range(2) if task == "detection" else range(11 if task == "diagnosis" else 10)
    supported = set(labels)
    actual, all_predicted, successful = [], [], []
    status_counts = Counter()
    for scenario_id, target in expected.items():
        if target not in supported:
            raise ValueError(f"unsupported {task} target for {scenario_id}: {target}")
        prediction = seen.get(scenario_id)
        label = None if prediction is None else prediction.predicted_label
        status = "missing" if prediction is None else prediction.status
        if status == "ok" and (not isinstance(label, int) or isinstance(label, bool) or label not in supported):
            status = "invalid"
        if status != "ok":
            label = None
        actual.append(target)
        all_predicted.append(label)
        status_counts[status] += 1
        if status == "ok":
            successful.append((target, label))
    return actual, all_predicted, successful, status_counts


def _bootstrap(task: Task, actual: Sequence[int], predicted: Sequence[int | None],
               groups: Sequence[str], *, seed: int, replicates: int) -> dict[str, Any]:
    unique = sorted(set(groups))
    if len(unique) < 10:
        return {"available": False, "groups": len(unique), "reason": "requires at least 10 recording groups"}
    rows: dict[str, list[int]] = {group: [] for group in unique}
    for index, group in enumerate(groups):
        rows[group].append(index)
    rng = random.Random(seed)
    values = {key: [] for key in ("accuracy", "macro_f1")}
    for _ in range(replicates):
        indices = [index for group in (rng.choice(unique) for _ in unique) for index in rows[group]]
        metrics = _metric_block(task, [actual[i] for i in indices], [predicted[i] for i in indices])
        for key, samples in values.items():
            samples.append(metrics[key])
    return {
        "available": True, "groups": len(unique), "replicates": replicates, "seed": seed,
        "confidence_level": 0.95,
        "intervals": {
            key: [sorted(samples)[int(.025 * (replicates - 1))],
                  sorted(samples)[int(.975 * (replicates - 1))]]
            for key, samples in values.items()
        },
    }


def score_predictions(task: Task, expected: Mapping[str, int], groups: Mapping[str, str],
                      predictions: Sequence[CasePrediction], *, seed: int = 42,
                      bootstrap_replicates: int = 1000) -> dict[str, Any]:
    """Score exactly the expected IDs; omitted outputs count as missing failures."""
    if set(groups) != set(expected):
        raise ValueError("group IDs must cover exactly the scored cohort")
    actual, predicted, successful, statuses = _prepare(task, expected, predictions)
    metrics = _metric_block(task, actual, predicted)
    valid_actual = [row[0] for row in successful]
    valid_predicted = [row[1] for row in successful]
    metrics["successful_only"] = _metric_block(task, valid_actual, valid_predicted)
    metrics["prediction_status_counts"] = {
        status: statuses.get(status, 0) for status in ("ok", "error", "invalid", "missing")
    }
    metrics["confidence_intervals"] = _bootstrap(
        task, actual, predicted, [groups[key] for key in expected], seed=seed,
        replicates=bootstrap_replicates,
    ) if bootstrap_replicates else {
        "available": False, "groups": len(set(groups.values())), "reason": "disabled"}
    return metrics


def score_combined_diagnosis(detection_targets: Mapping[str, int], category_targets: Mapping[str, int],
                             groups: Mapping[str, str], detections: Sequence[CasePrediction],
                             categories: Sequence[CasePrediction], *, seed: int = 42,
                             bootstrap_replicates: int = 1000) -> dict[str, Any]:
    """Score normal=0 and supported fault categories=1..10 from separate model heads."""
    actual = {sid: 0 if target == 0 else category_targets[sid] + 1
              for sid, target in detection_targets.items()
              if target == 0 or sid in category_targets}
    ids = set(actual)
    if set(groups) != ids:
        raise ValueError("combined diagnosis group IDs must match eligible IDs")
    det = {prediction.scenario_id: prediction for prediction in detections}
    cat = {prediction.scenario_id: prediction for prediction in categories}
    if len(det) != len(detections) or len(cat) != len(categories):
        raise ValueError("duplicate head prediction")
    if set(det) - set(detection_targets) or set(cat) - set(category_targets):
        raise ValueError("head prediction outside its task cohort")
    combined = []
    for sid in actual:
        d = det.get(sid)
        if d is None:
            continue
        elif d.status != "ok":
            combined.append(CasePrediction(scenario_id=sid, task="diagnosis", status=d.status,
                                            error=d.error))
        elif d.predicted_label not in (0, 1):
            combined.append(CasePrediction(scenario_id=sid, task="diagnosis", status="invalid"))
        elif d.predicted_label == 0:
            combined.append(CasePrediction(scenario_id=sid, task="diagnosis", predicted_label=0))
        else:
            c = cat.get(sid)
            if c is None:
                continue
            elif c.status != "ok":
                combined.append(CasePrediction(scenario_id=sid, task="diagnosis", status=c.status,
                                                error=c.error))
            elif not isinstance(c.predicted_label, int) or c.predicted_label not in range(10):
                combined.append(CasePrediction(scenario_id=sid, task="diagnosis", status="invalid"))
            else:
                combined.append(CasePrediction(scenario_id=sid, task="diagnosis", predicted_label=c.predicted_label + 1))
    return score_predictions("diagnosis", actual, groups, combined, seed=seed,
                             bootstrap_replicates=bootstrap_replicates)
