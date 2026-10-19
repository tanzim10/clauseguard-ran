"""Experiment views over a validated release. Source membership and values stay unchanged."""

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from clauseguard.datasets.contracts import ModelInputs
from clauseguard.datasets.release import project_inputs
from clauseguard.datasets.scenarios import ScenarioDataset
from clauseguard.datasets.splits import select_task_cohort


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class ExperimentCohort(BaseModel):
    """Serializable input identity and membership, also usable by later run artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["1"] = "1"
    release_id: str
    source_hash: str
    split_hash: str
    input_hash: str
    task: Literal["detection", "category"]
    split: Literal["train", "dev", "test"]
    modality: Literal["kpi", "text", "paired"]
    mode: Literal["scored", "exploratory"]
    scenario_ids: tuple[str, ...]
    exclusions: dict[str, str]
    ready: bool
    coverage_ready: bool
    reasons: tuple[str, ...]
    coverage: dict

    @property
    def cohort_id(self) -> str:
        return hashlib.sha256(_json_bytes(self.model_dump(mode="json"))).hexdigest()


def experiment_cohort(dataset: ScenarioDataset, *, split: str, task: str,
                      modality: str = "kpi") -> ExperimentCohort:
    """Describe a view even if unready; only the loader can permit model iteration."""
    dataset.check_source()
    ids = select_task_cohort(dataset.splits, split=split, task=task, modality=modality)
    exclusions = {}
    for sid, member in sorted(dataset.splits["membership"].items()):
        if member["split"] not in (split, "excluded"):
            continue
        reason = member["exclusion_reason"] or member["kpi_reason"]
        if not reason and task == "category":
            reason = member["category_reason"]
        if not reason and modality != "kpi":
            reason = member["text_reasons"][task]
        if reason:
            exclusions[sid] = reason
    digest = hashlib.sha256()
    for sid in ids:
        inputs = project_inputs(dataset.load_scenario(sid), modality=modality, task=task)
        digest.update(_json_bytes([sid, inputs.model_dump(mode="json", exclude_none=True)]))
        digest.update(b"\n")
    status = dataset.quality["tasks"][task][modality]
    dataset.check_source()
    return ExperimentCohort(
        release_id=dataset.release_id, source_hash=dataset.manifest["scenarios_sha256"],
        split_hash=dataset.manifest["files"]["splits.json"], input_hash=digest.hexdigest(),
        task=task, split=split, modality=modality, mode=dataset.splits["mode"],
        scenario_ids=tuple(ids), exclusions=exclusions, ready=status["ready"],
        coverage_ready=status["coverage_ready"], reasons=tuple(status["reasons"]),
        coverage=deepcopy(status["splits"]),
    )


@dataclass(frozen=True)
class ExperimentBatch:
    """Pass only inputs to a predictor; targets and correlation metadata stay separate."""

    cohort_id: str
    inputs: tuple[ModelInputs, ...]
    targets: tuple[int, ...]
    scenario_ids: tuple[str, ...]
    group_ids: tuple[str, ...]


def _batch(examples, cohort_id):
    return ExperimentBatch(
        cohort_id=cohort_id, inputs=tuple(e.inputs for e in examples),
        targets=tuple(e.target for e in examples),
        scenario_ids=tuple(e.scenario_id for e in examples),
        group_ids=tuple(e.group_id for e in examples),
    )


def iter_experiment_batches(dataset: ScenarioDataset, cohort: ExperimentCohort, *, batch_size=32):
    """Yield unchanged observations, applying the release loader's existing readiness gates."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    current = experiment_cohort(dataset, split=cohort.split, task=cohort.task,
                                modality=cohort.modality)
    if current != cohort:
        raise ValueError("cohort differs from the validated release")
    cohort_id = cohort.cohort_id
    examples = []
    for example in dataset.iter_examples(split=cohort.split, task=cohort.task,
                                         modality=cohort.modality):
        examples.append(example)
        if len(examples) == batch_size:
            yield _batch(examples, cohort_id)
            examples = []
    if examples:
        yield _batch(examples, cohort_id)


def combined_diagnosis_ids(dataset: ScenarioDataset, *, split: str,
                           modality: str = "kpi") -> tuple[str, ...]:
    """Normal plus ten supported synthetic categories; retain source targets unchanged."""
    dataset.check_source()
    detection = select_task_cohort(dataset.splits, split=split, task="detection", modality=modality)
    category = set(select_task_cohort(dataset.splits, split=split, task="category", modality=modality))
    members = dataset.splits["membership"]
    return tuple(sid for sid in detection if members[sid]["detection_target"] == 0 or sid in category)
