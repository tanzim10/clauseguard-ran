"""Shared cohort loading and training selections for classical tree models."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from clauseguard.baselines.data import ExperimentBatch, experiment_cohort, iter_experiment_batches


def ids_hash(ids):
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def require_ready(dataset, tasks, *, allow_exploratory=False):
    dataset.check_source()
    if dataset.splits["mode"] != "scored" and not allow_exploratory:
        raise ValueError("real training/prediction requires a scored release; exploratory runs require explicit allow_exploratory")
    for task in tasks:
        if task not in ("detection", "category"):
            raise ValueError("invalid Random Forest task")
        readiness = dataset.quality["tasks"][task]["kpi"]
        valid = readiness["coverage_ready"] if dataset.splits["mode"] == "exploratory" else readiness["ready"]
        if not valid:
            raise ValueError(f"{task} KPI release lacks split coverage: {readiness['reasons']}")


def split_data(dataset, task, split):
    cohort = experiment_cohort(dataset, task=task, split=split)
    batches = list(iter_experiment_batches(dataset, cohort))
    return cohort, ExperimentBatch(
        cohort_id=cohort.cohort_id,
        **{field: tuple(value for batch in batches for value in getattr(batch, field))
           for field in ("inputs", "targets", "scenario_ids", "group_ids")},
    )


def training_ids(dataset, path):
    if path is None:
        return None
    selection = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = selection["scenario_ids"]
    if selection["release_id"] != dataset.release_id:
        raise ValueError("training selection belongs to a different release")
    if len(ids) != len(set(ids)) or set(ids) - set(dataset.splits["membership"]):
        raise ValueError("training selection contains duplicate or unknown IDs")
    kept = set(ids)
    if any(sid not in kept for sid, row in dataset.splits["membership"].items()
           if row["split"] == "train" and row["detection_target"] == 1):
        raise ValueError("normal-window reduction must retain every training fault")
    return kept


def select_training(batch, ids):
    if ids is None:
        return batch
    indices = [i for i, sid in enumerate(batch.scenario_ids) if sid in ids]
    if not indices:
        raise ValueError("training selection is empty")
    return replace(batch, **{
        field: tuple(getattr(batch, field)[i] for i in indices)
        for field in ("inputs", "targets", "scenario_ids", "group_ids")
    })
