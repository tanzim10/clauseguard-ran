"""CPU histogram XGBoost on the shared, training-fitted KPI features."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

from clauseguard.datasets.contracts import NUMERIC_CHANNELS, ModelInputs
from clauseguard.kpi.features import ChannelMedianFeatures

CONFIGURATIONS = tuple({"max_depth": depth, "min_child_weight": weight}
                       for depth in (3, 6) for weight in (1, 5))


def _labels(task):
    if task not in ("detection", "category"):
        raise ValueError("invalid XGBoost task")
    return set(range(2 if task == "detection" else 10))


def _matrix(features, targets, *, labels, training):
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(ChannelMedianFeatures.feature_names_):
        raise ValueError("features must follow the fixed 224-column feature contract")
    if not len(matrix) or len(matrix) != len(targets):
        raise ValueError("inputs and targets must be nonempty and equal in length")
    if not np.isfinite(matrix).all():
        raise ValueError("features must be finite")
    observed = set(targets)
    if not observed <= labels or (training and observed != labels):
        raise ValueError("training targets must contain every task label; dev labels must be valid")
    return matrix


def fit_booster(train_features, train_targets, dev_features, dev_targets, *,
                task, configuration, seed):
    """Fit with balanced training weights and unweighted dev loss for early stopping."""
    labels = _labels(task)
    if configuration not in CONFIGURATIONS:
        raise ValueError("configuration is outside the four-config search")
    train = _matrix(train_features, train_targets, labels=labels, training=True)
    dev = _matrix(dev_features, dev_targets, labels=labels, training=False)
    metric = "logloss" if task == "detection" else "mlogloss"
    objective = "binary:logistic" if task == "detection" else "multi:softprob"
    task_parameters = {"scale_pos_weight": 1} if task == "detection" else {"num_class": 10}
    booster = XGBClassifier(
        objective=objective, eval_metric=metric, tree_method="hist", device="cpu",
        n_jobs=1, learning_rate=0.05, n_estimators=500, subsample=1,
        colsample_bytree=1, random_state=seed, **configuration, **task_parameters,
        callbacks=[EarlyStopping(rounds=30, data_name="validation_0", metric_name=metric,
                                 maximize=False, save_best=True)],
    )
    booster.fit(train, train_targets,
                sample_weight=compute_sample_weight("balanced", train_targets),
                eval_set=[(dev, dev_targets)], verbose=False)
    # Native UBJSON carries names so reload can check column order as well as count.
    booster.get_booster().feature_names = list(ChannelMedianFeatures.feature_names_)
    return booster


@dataclass
class XGBoostModel:
    task: str
    features: ChannelMedianFeatures
    booster: XGBClassifier
    configuration: dict
    seed: int

    def predict(self, inputs: list[ModelInputs]) -> list[int]:
        if not inputs:
            return []
        return self.booster.predict(self.features.transform(inputs)).astype(int).tolist()

    def predict_proba(self, inputs: list[ModelInputs]) -> np.ndarray:
        if not inputs:
            return np.empty((0, len(_labels(self.task))))
        return self.booster.predict_proba(self.features.transform(inputs))


def fit_model(inputs, targets, dev_inputs, dev_targets, *, task, configuration, seed=42):
    """Fit medians on training windows and use the separate dev set only for stopping."""
    transformer = ChannelMedianFeatures().fit(inputs)
    booster = fit_booster(transformer.transform(inputs), targets,
                          transformer.transform(dev_inputs), dev_targets,
                          task=task, configuration=configuration, seed=seed)
    return XGBoostModel(task, transformer, booster, dict(configuration), seed)


class XGBoostAdapter:
    def __init__(self, task):
        _labels(task)
        self.task = task

    def fit(self, batches, *, validation_batches, seed, configuration):
        inputs, targets, dev_inputs, dev_targets = [], [], [], []
        for batch in batches:
            inputs.extend(batch.inputs)
            targets.extend(batch.targets)
        for batch in validation_batches:
            dev_inputs.extend(batch.inputs)
            dev_targets.extend(batch.targets)
        return fit_model(inputs, targets, dev_inputs, dev_targets,
                         task=self.task, configuration=configuration, seed=seed)

    def predict(self, model, inputs):
        if model.task != self.task:
            raise ValueError("adapter and checkpoint tasks differ")
        return model.predict(inputs)


def serialize_model(model: XGBoostModel) -> tuple[bytes, str]:
    """Persist only native model data; preprocessing belongs in run metadata."""
    content = bytes(model.booster.get_booster().save_raw(raw_format="ubj"))
    return content, hashlib.sha256(content).hexdigest()


def load_model(path: Path, *, expected_sha256: str, task: str,
               features: ChannelMedianFeatures, configuration: dict, seed: int) -> XGBoostModel:
    labels = _labels(task)
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("checkpoint checksum mismatch")
    if tuple(features.feature_names_) != ChannelMedianFeatures.feature_names_:
        raise ValueError("checkpoint feature order differs from the current feature contract")
    medians = np.asarray(features.medians_, dtype=np.float64)
    if medians.shape != (len(NUMERIC_CHANNELS),) or not np.isfinite(medians).all():
        raise ValueError("checkpoint contains invalid imputation values")
    booster = XGBClassifier(n_jobs=1, device="cpu")
    booster.load_model(bytearray(content))
    native = booster.get_booster()
    if native.num_features() != len(features.feature_names_):
        raise ValueError("checkpoint feature count and preprocessing differ")
    if native.feature_names != list(features.feature_names_):
        raise ValueError("checkpoint feature order and classifier differ")
    model_config = json.loads(native.save_config())["learner"]
    objective = "binary:logistic" if task == "detection" else "multi:softprob"
    if model_config["objective"]["name"] != objective or set(booster.classes_) != labels:
        raise ValueError("checkpoint objective or class mapping differs from task")
    try:
        best_iteration = booster.best_iteration
    except AttributeError as error:
        raise ValueError("checkpoint has no early-stopping iteration") from error
    if best_iteration < 0 or native.num_boosted_rounds() != best_iteration + 1:
        raise ValueError("checkpoint must retain exactly the best early-stopping rounds")
    if configuration not in CONFIGURATIONS:
        raise ValueError("configuration is outside the four-config search")
    return XGBoostModel(task, features, booster, dict(configuration), seed)
