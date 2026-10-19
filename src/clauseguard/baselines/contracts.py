"""Saved-run contracts shared by model adapters, scorers, and offline tools."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from clauseguard.baselines.data import ExperimentCohort

Task = Literal["detection", "category", "diagnosis"]
PredictionStatus = Literal["ok", "error", "invalid"]


class CasePrediction(BaseModel):
    """One attempted prediction; failed attempts remain part of the evaluation cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    task: Task
    predicted_label: int | str | None = None
    status: PredictionStatus = "ok"
    error: str | None = None


class RunMetadata(BaseModel):
    """Identity and settings needed to interpret or reproduce a baseline run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    task: Task
    cohort: ExperimentCohort
    train_cohort_id: str | None = None
    transformations: dict[str, Any] = Field(default_factory=dict)
    versions: dict[str, str] = Field(default_factory=dict)
    seed: int | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)
    checkpoint: str | None = None
    resources: dict[str, float] = Field(default_factory=dict)

    def validate_cohort(self) -> None:
        if self.task != "diagnosis" and self.task != self.cohort.task:
            raise ValueError("run task differs from experiment cohort")
        if self.task == "diagnosis" and self.cohort.task != "detection":
            raise ValueError("combined diagnosis requires a detection cohort")

