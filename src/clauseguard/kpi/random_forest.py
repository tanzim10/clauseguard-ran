"""CPU Random Forest using the shared train-fitted window features."""

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from clauseguard.datasets.contracts import ModelInputs
from clauseguard.kpi.features import ChannelMedianFeatures

CONFIGURATIONS = tuple({"max_depth": depth, "min_samples_leaf": leaf}
                       for depth in (12, None) for leaf in (5, 1))


def fit_forest(features, targets, *, task, configuration, seed):
    """Fit one of the four agreed configurations with training-derived class weights."""
    if task not in ("detection", "category"):
        raise ValueError("invalid Random Forest task")
    if configuration not in CONFIGURATIONS:
        raise ValueError("configuration is outside the four-config search")
    expected = {0, 1} if task == "detection" else set(range(10))
    if set(targets) != expected:
        raise ValueError("training targets must contain every task label")
    if len(features) != len(targets):
        raise ValueError("training inputs and targets differ in length")
    return RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=1,
        **configuration,
    ).fit(features, targets)


@dataclass
class RandomForestModel:
    task: str
    features: ChannelMedianFeatures
    forest: RandomForestClassifier
    configuration: dict
    seed: int

    def predict(self, inputs: list[ModelInputs]) -> list[int]:
        if not inputs:
            return []
        return self.forest.predict(self.features.transform(inputs)).astype(int).tolist()


def fit_model(inputs, targets, *, task, configuration, seed=42):
    """Fit preprocessing and a classifier on explicitly supplied training observations."""
    transformer = ChannelMedianFeatures().fit(inputs)
    forest = fit_forest(transformer.transform(inputs), targets, task=task,
                        configuration=configuration, seed=seed)
    return RandomForestModel(task, transformer, forest, dict(configuration), seed)


class RandomForestAdapter:
    """The shared adapter interface receives labels for fit and observations for predict."""

    def __init__(self, task):
        self.task = task

    def fit(self, batches, *, seed, configuration):
        inputs, targets = [], []
        for batch in batches:
            inputs.extend(batch.inputs)
            targets.extend(batch.targets)
        return fit_model(inputs, targets, task=self.task, configuration=configuration, seed=seed)

    def predict(self, model, inputs):
        if model.task != self.task:
            raise ValueError("adapter and checkpoint tasks differ")
        return model.predict(inputs)


def serialize_model(model: RandomForestModel) -> tuple[bytes, str]:
    stream = BytesIO()
    joblib.dump(model, stream, compress=3)
    content = stream.getvalue()
    return content, hashlib.sha256(content).hexdigest()


def load_model(path: Path, *, expected_sha256: str) -> RandomForestModel:
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("checkpoint checksum mismatch")
    model = joblib.load(BytesIO(content))
    if not isinstance(model, RandomForestModel):
        raise TypeError("unsupported checkpoint type")
    if model.features.feature_names_ != ChannelMedianFeatures.feature_names_:
        raise ValueError("checkpoint feature order differs from the current feature contract")
    if len(model.features.feature_names_) != model.forest.n_features_in_:
        raise ValueError("checkpoint feature order and classifier differ")
    if not np.isfinite(model.features.medians_).all():
        raise ValueError("checkpoint contains invalid imputation values")
    return model
