import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from clauseguard.datasets.contracts import NUMERIC_CHANNELS, ModelInputs, NumericInputs
from clauseguard.kpi.timesnet import (
    CONFIGURATIONS,
    TimesNetAdapter,
    TimesNetClassifier,
    TimesNetModel,
    WindowScaler,
    balanced_class_weights,
    fit_timesnet,
    load_model,
    predict_logits,
    serialize_model,
)


def inputs(rng, *, offset=0.0, scale=1.0, channels=None, missing=None):
    values = rng.standard_normal((128, 16)) * scale + offset
    names = list(NUMERIC_CHANNELS)
    if channels is not None:
        values, names = values[:, channels], [names[i] for i in channels]
    rows = [list(map(float, row)) for row in values]
    if missing:
        rows[missing[0]][missing[1]] = None
    return ModelInputs(kpi=NumericInputs(channel_names=names, sampling_rate_hz=10, values=rows))


@pytest.fixture
def windows():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 128, 16)).astype("float32")
    y = np.array([0] * 30 + [1] * 10)
    x[y == 1] += 1.0
    return x, y


@pytest.mark.parametrize("classes", [2, 10])
def test_logit_width_and_input_shape_contract(classes):
    net = TimesNetClassifier(classes, 0.1)
    assert net(torch.zeros(4, 128, 16)).shape == (4, classes)
    with pytest.raises(ValueError, match="expected"):
        net(torch.zeros(4, 100, 16))
    with pytest.raises(ValueError, match="expected"):
        net(torch.zeros(4, 128, 15))


def test_parameter_count_matches_reviewed_estimate():
    def count(classes):
        return sum(p.numel() for p in TimesNetClassifier(classes, 0.1).parameters())

    assert (count(2), count(10)) == (698_690, 731_466)


def test_scaler_uses_train_statistics_only_and_follows_channel_order():
    rng = np.random.default_rng(1)
    train = [inputs(rng, offset=5, scale=2) for _ in range(6)]
    scaler = WindowScaler().fit(train)
    state = scaler.state()
    shifted = [inputs(rng, offset=1000, scale=50) for _ in range(3)]
    scaler.transform(shifted)
    assert scaler.state() == state  # transforming dev/test never refits
    flipped = list(range(15, -1, -1))
    item = inputs(np.random.default_rng(2))
    permuted = inputs(np.random.default_rng(2), channels=flipped)
    # Same values under reversed channel names land in the NUMERIC_CHANNELS order.
    np.testing.assert_array_equal(scaler.transform([item]), scaler.transform([permuted]))
    assert abs(float(scaler.transform(train).mean())) < 0.5


def test_scaler_imputes_with_train_median_and_handles_constant_channels():
    rng = np.random.default_rng(3)
    train = [inputs(rng) for _ in range(4)]
    scaler = WindowScaler().fit(train)
    out = scaler.transform([inputs(rng, missing=(5, 3))])[0, 5, 3]
    assert out == pytest.approx((scaler.medians_[3] - scaler.means_[3]) / scaler.stds_[3], abs=1e-5)
    constant = ModelInputs(kpi=NumericInputs(channel_names=list(NUMERIC_CHANNELS), sampling_rate_hz=10,
                                             values=[[3.0] * 16 for _ in range(128)]))
    fitted = WindowScaler().fit([constant])
    assert (fitted.stds_ == 1).all() and np.isfinite(fitted.transform([constant])).all()
    assert WindowScaler.from_state(fitted.state()).state() == fitted.state()
    with pytest.raises(ValueError, match="fit the scaler"):
        WindowScaler().transform([constant])


def test_balanced_weights_match_tree_models():
    targets = np.array([0] * 30 + [1] * 10)
    np.testing.assert_allclose(balanced_class_weights(targets, 2), [40 / 60, 40 / 20])
    with pytest.raises(ValueError, match="every task label"):
        balanced_class_weights(targets, 3)


def test_training_weights_train_loss_but_not_dev_loss(windows, monkeypatch):
    x, y = windows
    calls = []
    original = torch.nn.functional.cross_entropy

    def observe(*args, **kwargs):
        calls.append(kwargs.get("weight"))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "cross_entropy", observe)
    fit_timesnet(x, y, x, y, task="detection", configuration=CONFIGURATIONS[0], seed=1,
                 max_epochs=1, batch_size=20)
    # The dev report makes two unweighted calls: raw and prior-corrected loss.
    train_weights, dev_weights = calls[:-2], calls[-2:]
    assert all(w is not None for w in train_weights) and dev_weights == [None, None]
    np.testing.assert_allclose(train_weights[0].numpy(), balanced_class_weights(y, 2))


def test_early_stopping_restores_best_dev_loss_epoch(windows):
    x, y = windows
    _, info = fit_timesnet(x, y, x[:20], y[:20][::-1].copy(), task="detection",
                             configuration=CONFIGURATIONS[0], seed=1, max_epochs=6, patience=1)
    history = info["history"]
    assert all(np.isfinite(r["train_loss"]) and np.isfinite(r["dev_loss"]) for r in history)
    assert info["best_dev_loss"] == min(r["dev_loss"] for r in history)
    assert info["best_epoch"] == int(np.argmin([r["dev_loss"] for r in history]))
    assert info["attempted_epochs"] == len(history) <= 6


def test_same_seed_is_reproducible_on_cpu(windows):
    x, y = windows
    runs = [fit_timesnet(x, y, x, y, task="detection", configuration=CONFIGURATIONS[1], seed=7,
                         max_epochs=2)[0] for _ in range(2)]
    np.testing.assert_array_equal(predict_logits(runs[0], x), predict_logits(runs[1], x))


def test_checkpoint_round_trip_and_rejections(windows):
    x, y = windows
    rng = np.random.default_rng(4)
    scaler = WindowScaler().fit([inputs(rng) for _ in range(4)])
    net, _ = fit_timesnet(x, y, x, y, task="detection", configuration=CONFIGURATIONS[2], seed=3,
                          max_epochs=1)
    model = TimesNetModel("detection", scaler, net, dict(CONFIGURATIONS[2]), 3)
    content, checksum = serialize_model(model)
    restored = load_model(content, expected_sha256=checksum, task="detection", scaler=scaler,
                          configuration=CONFIGURATIONS[2], seed=3)
    np.testing.assert_array_equal(predict_logits(restored.net, x), predict_logits(net, x))
    with pytest.raises(ValueError, match="checksum"):
        load_model(content + b"x", expected_sha256=checksum, task="detection", scaler=scaler,
                   configuration=CONFIGURATIONS[2], seed=3)
    with pytest.raises(RuntimeError):  # category head does not fit a detection checkpoint
        load_model(content, expected_sha256=checksum, task="category", scaler=scaler,
                   configuration=CONFIGURATIONS[2], seed=3)
    with pytest.raises(ValueError, match="outside"):
        load_model(content, expected_sha256=checksum, task="detection", scaler=scaler,
                   configuration={"learning_rate": 0.5, "dropout": 0.1}, seed=3)
    with pytest.raises(ValueError, match="differ"):
        TimesNetAdapter("category").predict(model, [])
    assert model.predict([]) == []
    assert set(model.predict([inputs(rng) for _ in range(3)])) <= {0, 1}


def test_dev_report_logs_per_group_and_prior_correction(windows):
    x, y = windows
    groups = ["a"] * 20 + ["b"] * 20
    _, info = fit_timesnet(x, y, x, y, task="detection", configuration=CONFIGURATIONS[0], seed=1,
                           max_epochs=2, dev_groups=groups, prior_corrected_stopping=True)
    row = info["history"][0]
    assert set(row["per_group"]) == {"a", "b"}
    assert all(s["windows"] == 20 and 0 <= s["accuracy"] <= 1 for s in row["per_group"].values())
    assert 0 <= row["dev_macro_f1"] <= 1
    assert info["selection_loss"] == "dev_prior_corrected_loss"
    assert row["dev_selection_loss"] == row["dev_prior_corrected_loss"]
    assert info["best_dev_loss"] == min(r["dev_prior_corrected_loss"] for r in info["history"])


def test_cosine_schedule_decays_learning_rate(windows):
    x, y = windows
    _, info = fit_timesnet(x, y, x, y, task="detection", configuration=CONFIGURATIONS[0], seed=1,
                           max_epochs=3, patience=5, lr_schedule="cosine")
    rates = [row["learning_rate"] for row in info["history"]]
    assert rates[0] == pytest.approx(1e-3) and rates[0] > rates[1] > rates[2]
    with pytest.raises(ValueError, match="lr_schedule"):
        fit_timesnet(x, y, x, y, task="detection", configuration=CONFIGURATIONS[0], seed=1,
                     max_epochs=1, lr_schedule="step")


def test_instance_norm_ignores_window_offsets_and_scale():
    torch.manual_seed(0)
    net = TimesNetClassifier(2, 0.1, instance_norm=True, num_kernels=3).eval()
    x = torch.randn(4, 128, 16)
    with torch.no_grad():
        torch.testing.assert_close(net(x), net(x * 3 + 7), atol=1e-4, rtol=1e-4)
    assert sum(p.numel() for p in net.parameters()) < 698_690


def test_architecture_and_instance_norm_round_trip(windows):
    x, y = windows
    scaler = WindowScaler().fit([inputs(np.random.default_rng(5)) for _ in range(2)])
    net, _ = fit_timesnet(x, y, x, y, task="detection", configuration=CONFIGURATIONS[0], seed=3,
                          max_epochs=1, architecture={"num_kernels": 3}, instance_norm=True)
    content, checksum = serialize_model(TimesNetModel("detection", scaler, net,
                                                      dict(CONFIGURATIONS[0]), 3))
    restored = load_model(content, expected_sha256=checksum, task="detection", scaler=scaler,
                          configuration=CONFIGURATIONS[0], seed=3,
                          architecture={"num_kernels": 3}, instance_norm=True)
    np.testing.assert_array_equal(predict_logits(restored.net, x), predict_logits(net, x))
    with pytest.raises(RuntimeError):  # default four kernels cannot load a three-kernel state
        load_model(content, expected_sha256=checksum, task="detection", scaler=scaler,
                   configuration=CONFIGURATIONS[0], seed=3)


def test_pooled_head_is_small_position_independent_and_round_trips(windows):
    from clauseguard.kpi.timesnet import HEADS

    assert HEADS == ("flatten", "pooled")
    flat, pooled = (TimesNetClassifier(10, 0.1, head=h) for h in HEADS)
    assert pooled.projection.in_features == 32 and flat.projection.in_features == 32 * 128
    assert (sum(p.numel() for p in flat.parameters())
            - sum(p.numel() for p in pooled.parameters())) == (4096 - 32) * 10
    with pytest.raises(ValueError, match="head"):
        TimesNetClassifier(2, 0.1, head="nope")
    x, y = windows
    scaler = WindowScaler().fit([inputs(np.random.default_rng(6)) for _ in range(2)])
    net, _ = fit_timesnet(x, y, x, y, task="detection", configuration=CONFIGURATIONS[0], seed=3,
                          max_epochs=1, head="pooled")
    assert net.head == "pooled"
    content, checksum = serialize_model(TimesNetModel("detection", scaler, net,
                                                      dict(CONFIGURATIONS[0]), 3))
    restored = load_model(content, expected_sha256=checksum, task="detection", scaler=scaler,
                          configuration=CONFIGURATIONS[0], seed=3, head="pooled")
    np.testing.assert_array_equal(predict_logits(restored.net, x), predict_logits(net, x))
    with pytest.raises(RuntimeError):  # a flatten head cannot load a pooled state
        load_model(content, expected_sha256=checksum, task="detection", scaler=scaler,
                   configuration=CONFIGURATIONS[0], seed=3)


def test_recording_balanced_weights_equalize_classes_and_recordings():
    from clauseguard.kpi.timesnet import recording_balanced_weights

    targets = np.array([0] * 100 + [0] * 10 + [1] * 6 + [1] * 2)
    groups = np.array(["long"] * 100 + ["short"] * 10 + ["f1"] * 6 + ["f2"] * 2)
    w = recording_balanced_weights(targets, groups, 2)
    assert w.mean() == pytest.approx(1.0)
    total = {g: float(w[groups == g].sum()) for g in set(groups)}
    assert total["long"] == pytest.approx(total["short"]) and total["f1"] == pytest.approx(total["f2"])
    assert w[targets == 0].sum() == pytest.approx(w[targets == 1].sum())
    with pytest.raises(ValueError, match="every task label"):
        recording_balanced_weights(np.zeros(4, dtype=int), np.arange(4), 2)
    with pytest.raises(ValueError, match="cover"):
        recording_balanced_weights(targets, groups[:-1], 2)


def test_recording_weights_change_training_and_need_groups(windows):
    x, y = windows
    groups = ["a"] * 25 + ["b"] * 5 + ["c"] * 6 + ["d"] * 4
    common = {"task": "detection", "configuration": CONFIGURATIONS[0], "seed": 1, "max_epochs": 1}
    with pytest.raises(ValueError, match="train_groups"):
        fit_timesnet(x, y, x, y, recording_weights=True, **common)
    plain, _ = fit_timesnet(x, y, x, y, **common)
    weighted, info = fit_timesnet(x, y, x, y, train_groups=groups, recording_weights=True, **common)
    assert np.isfinite(info["history"][0]["train_loss"])
    assert not np.array_equal(predict_logits(plain, x), predict_logits(weighted, x))
