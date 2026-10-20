import hashlib

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("xgboost")

from clauseguard.datasets.contracts import NUMERIC_CHANNELS, ModelInputs, NumericInputs
from clauseguard.kpi.features import ChannelMedianFeatures
from clauseguard.kpi.xgboost import (
    CONFIGURATIONS,
    XGBoostAdapter,
    XGBoostModel,
    fit_booster,
    fit_model,
    load_model,
    serialize_model,
)


def window(value):
    return ModelInputs(kpi=NumericInputs(
        channel_names=list(NUMERIC_CHANNELS), sampling_rate_hz=10,
        values=[[value] * 16 for _ in range(128)]))


@pytest.mark.parametrize("task,classes", [("detection", 2), ("category", 10)])
def test_exact_train_weights_and_unweighted_dev_loss(task, classes, monkeypatch, tmp_path):
    import clauseguard.kpi.xgboost as module

    targets = np.repeat(np.arange(classes), np.arange(1, classes + 1) * 4)
    matrix = np.repeat(targets[:, None], 224, axis=1).astype(float)
    calls = []
    original = module.XGBClassifier.fit

    def observe(self, features, labels, **kwargs):
        calls.append(kwargs.copy())
        return original(self, features, labels, **kwargs)

    monkeypatch.setattr(module.XGBClassifier, "fit", observe)
    model = fit_booster(matrix, targets, matrix, targets,
                        task=task, configuration=CONFIGURATIONS[0], seed=42)
    weights = calls[0]["sample_weight"]
    expected = len(targets) / (classes * np.bincount(targets)[targets])
    np.testing.assert_array_equal(weights, expected)
    assert weights.mean() == pytest.approx(1)
    assert "sample_weight_eval_set" not in calls[0]
    params = model.get_params()
    assert params["eval_metric"] == ("logloss" if classes == 2 else "mlogloss")
    assert params["n_jobs"] == 1 and params["device"] == "cpu"
    assert params["subsample"] == params["colsample_bytree"] == 1
    assert model.predict_proba(matrix).shape == (len(targets), classes)
    wrapped = XGBoostModel(task, ChannelMedianFeatures().fit([window(0)]),
                           model, CONFIGURATIONS[0], 42)
    content, checksum = serialize_model(wrapped)
    checkpoint = tmp_path / "model.ubj"
    checkpoint.write_bytes(content)
    restored = load_model(checkpoint, expected_sha256=checksum, task=task,
                          features=wrapped.features, configuration=CONFIGURATIONS[0], seed=42)
    np.testing.assert_array_equal(restored.booster.predict_proba(matrix),
                                  model.predict_proba(matrix))


@pytest.fixture
def stopped_model():
    targets = [0] * 40 + [1] * 40
    # Opposite dev labels make additional fitting worse, exercising early stopping.
    return fit_model([window(value) for value in targets], targets,
                     [window(value) for value in targets], targets[::-1],
                     task="detection", configuration=CONFIGURATIONS[0])


def test_train_only_imputation_and_early_stopping(stopped_model):
    model = stopped_model
    np.testing.assert_array_equal(model.features.medians_, [0.5] * 16)
    assert model.booster.best_iteration < 469
    assert model.booster.get_booster().num_boosted_rounds() == model.booster.best_iteration + 1
    assert len(model.booster.evals_result()["validation_0"]["logloss"]) >= 31
    before = model.features.medians_.copy()
    model.predict([window(None), window(1000)])
    np.testing.assert_array_equal(model.features.medians_, before)
    assert model.predict([]) == []
    assert model.predict_proba([]).shape == (0, 2)


def test_native_reload_preserves_predictions_and_best_iteration(stopped_model, tmp_path):
    content, checksum = serialize_model(stopped_model)
    assert hashlib.sha256(content).hexdigest() == checksum
    checkpoint = tmp_path / "model.ubj"
    checkpoint.write_bytes(content)
    restored = load_model(checkpoint, expected_sha256=checksum, task="detection",
                          features=stopped_model.features, configuration=CONFIGURATIONS[0],
                          seed=42)
    inputs = [window(0), window(1), window(None)]
    np.testing.assert_array_equal(restored.predict_proba(inputs),
                                  stopped_model.predict_proba(inputs))
    assert restored.predict(inputs) == stopped_model.predict(inputs)
    assert restored.booster.best_iteration == stopped_model.booster.best_iteration
    assert restored.booster.n_jobs == 1
    assert restored.booster.device == "cpu"
    with pytest.raises(ValueError, match="checksum"):
        load_model(checkpoint, expected_sha256="wrong", task="detection",
                   features=stopped_model.features, configuration=CONFIGURATIONS[0], seed=42)
    with pytest.raises(ValueError, match="objective or class mapping"):
        load_model(checkpoint, expected_sha256=checksum, task="category",
                   features=stopped_model.features, configuration=CONFIGURATIONS[0], seed=42)


@pytest.mark.parametrize("corruption", ["order", "medians", "count", "native_order"])
def test_reload_rejects_preprocessing_or_native_feature_corruption(stopped_model, tmp_path,
                                                                  corruption):
    features = stopped_model.features
    if corruption == "order":
        features.feature_names_ = tuple(reversed(features.feature_names_))
    elif corruption == "medians":
        features.medians_ = np.array([np.nan] * 16)
    elif corruption == "count":
        features.medians_ = np.array([0.5] * 15)
    else:
        stopped_model.booster.get_booster().feature_names = list(reversed(features.feature_names_))
    content, checksum = serialize_model(stopped_model)
    checkpoint = tmp_path / "bad.ubj"
    checkpoint.write_bytes(content)
    with pytest.raises(ValueError, match="feature order|imputation"):
        load_model(checkpoint, expected_sha256=checksum, task="detection", features=features,
                   configuration=CONFIGURATIONS[0], seed=42)


def test_dev_extremes_do_not_change_fitted_medians():
    model = fit_model([window(0), window(1)] * 10, [0, 1] * 10,
                      [window(-100), window(100)], [0, 1], task="detection",
                      configuration=CONFIGURATIONS[0])
    np.testing.assert_array_equal(model.features.medians_, [0.5] * 16)


@pytest.mark.parametrize("change,match", [
    ("missing_train_label", "every task label"),
    ("unknown_dev_label", "dev labels"),
    ("wrong_shape", "224-column"),
    ("nonfinite", "finite"),
    ("empty_dev", "nonempty"),
])
def test_fit_validates_feature_contract_and_task_labels(change, match):
    train = np.zeros((4, 224))
    dev = np.zeros((2, 224))
    targets, dev_targets = [0, 1, 0, 1], [0, 1]
    if change == "missing_train_label":
        targets = [0] * 4
    elif change == "unknown_dev_label":
        dev_targets = [0, 2]
    elif change == "wrong_shape":
        train = train[:, :-1]
    elif change == "nonfinite":
        train[0, 0] = np.inf
    else:
        dev, dev_targets = np.empty((0, 224)), []
    with pytest.raises(ValueError, match=match):
        fit_booster(train, targets, dev, dev_targets,
                    task="detection", configuration=CONFIGURATIONS[0], seed=42)


def test_adapter_requires_separate_validation_and_matching_task(stopped_model):
    adapter = XGBoostAdapter("category")
    with pytest.raises(ValueError, match="tasks differ"):
        adapter.predict(stopped_model, [])
    with pytest.raises(TypeError, match="validation_batches"):
        adapter.fit([], seed=42, configuration=CONFIGURATIONS[0])
    assert isinstance(stopped_model, XGBoostModel)
