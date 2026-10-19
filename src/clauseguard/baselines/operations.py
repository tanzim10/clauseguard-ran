"""Small adapter seam for later baseline issues; no model is registered here."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

from clauseguard.baselines.contracts import CasePrediction
from clauseguard.baselines.data import ExperimentBatch, ExperimentCohort
from clauseguard.datasets.contracts import ModelInputs


class BaselineAdapter(Protocol):
    """Fit on labelled training batches and predict from observations alone."""

    def fit(self, batches: Iterable[ExperimentBatch], *, seed: int,
            configuration: Mapping[str, Any]) -> Any: ...

    def predict(self, model: Any, inputs: Sequence[ModelInputs]) -> Sequence[Any]: ...


def train_adapter(adapter: BaselineAdapter, cohort: ExperimentCohort,
                  batches: Iterable[ExperimentBatch], *, seed: int,
                  configuration: Mapping[str, Any]) -> Any:
    """Train only against the declared training cohort."""
    if cohort.split != "train":
        raise ValueError("adapter training requires a train cohort")

    def checked_batches():
        for batch in batches:
            if batch.cohort_id != cohort.cohort_id:
                raise ValueError("training batch differs from declared cohort")
            yield batch

    return adapter.fit(checked_batches(), seed=seed, configuration=configuration)


def predict_adapter(adapter: BaselineAdapter, model: Any, cohort: ExperimentCohort,
                    batches: Iterable[ExperimentBatch]) -> list[CasePrediction]:
    """Predict from observations only and retain batch failures as per-case errors."""
    if cohort.task not in ("detection", "category"):
        raise ValueError("model predictions require a detection or category cohort")
    predictions = []
    seen = set()
    for batch in batches:
        if batch.cohort_id != cohort.cohort_id:
            raise ValueError("prediction batch differs from declared cohort")
        if seen.intersection(batch.scenario_ids):
            raise ValueError("duplicate scenario in prediction batches")
        seen.update(batch.scenario_ids)
        try:
            outputs = list(adapter.predict(model, batch.inputs))
        except Exception as exc:  # noqa: BLE001 - retain any model failure in the score denominator.
            predictions.extend(CasePrediction(
                scenario_id=sid, task=cohort.task, status="error",
                error=f"{type(exc).__name__}: {exc}",
            ) for sid in batch.scenario_ids)
            continue
        if len(outputs) != len(batch.scenario_ids):
            predictions.extend(CasePrediction(
                scenario_id=sid, task=cohort.task, status="error",
                error=f"adapter returned {len(outputs)} outputs for {len(batch.scenario_ids)} cases",
            ) for sid in batch.scenario_ids)
            continue
        predictions.extend(CasePrediction(
            scenario_id=sid, task=cohort.task, predicted_label=label,
            status="ok" if isinstance(label, int) and not isinstance(label, bool) else "invalid",
        ) for sid, label in zip(batch.scenario_ids, outputs, strict=True))
    if seen != set(cohort.scenario_ids):
        raise ValueError("prediction batches must cover the declared cohort exactly")
    return predictions
