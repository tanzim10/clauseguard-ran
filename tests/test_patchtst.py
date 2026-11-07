import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

import torch.nn.functional as F

from clauseguard.datasets.contracts import (
    NUMERIC_CHANNELS,
    ModelInputs,
    NumericInputs,
)
from clauseguard.kpi.patchtst import (
    CONFIGURATIONS,
    STRIDES,
    PatchTSTClassifier,
    PatchTSTModel,
    balanced_class_weights,
    create_patches,
    fit_patchtst,
    load_model,
    patch_count,
    serialize_model,
)
from clauseguard.kpi.timesnet import WindowScaler


def _inputs(values, names=NUMERIC_CHANNELS):
    return ModelInputs(kpi=NumericInputs(channel_names=list(names), sampling_rate_hz=10.0,
                                         values=[[None if np.isnan(v) else v for v in row]
                                                 for row in values.tolist()]))


@pytest.mark.parametrize("patch_length, patches", [(8, 31), (16, 15)])
def test_patches_cover_every_step_with_half_overlap(patch_length, patches):
    stride = STRIDES[patch_length]
    assert stride == patch_length // 2 and patch_count(patch_length, stride) == patches
    window = torch.arange(128, dtype=torch.float32)[None, :, None].repeat(1, 1, 16)
    window[..., 3] += 1000  # channel identity survives patching
    out = create_patches(window, patch_length, stride)
    assert tuple(out.shape) == (1, patches, 16, patch_length)
    covered = torch.unique(out[0, :, 0, :])
    assert covered.tolist() == list(range(128))
    assert out[0, 0, 0].tolist() == list(range(patch_length))
    assert out[0, -1, 0, -1] == 127 and out[0, 1, 0, 0] == stride
    assert out[0, 0, 3, 0] == 1000


def test_four_settings_in_fixed_grid_order():
    assert CONFIGURATIONS == (
        {"patch_length": 8, "learning_rate": 1e-3}, {"patch_length": 8, "learning_rate": 3e-4},
        {"patch_length": 16, "learning_rate": 1e-3}, {"patch_length": 16, "learning_rate": 3e-4})


@pytest.mark.parametrize("num_class, patch_length", [(2, 8), (10, 16)])
def test_output_shapes_and_channel_independent_encoder(num_class, patch_length):
    torch.manual_seed(0)
    net = PatchTSTClassifier(num_class, patch_length).eval()
    x = torch.randn(3, 128, 16)
    assert tuple(net(x).shape) == (3, num_class)
    features = net.encode(x)
    assert tuple(features.shape) == (3, 16, 32, patch_count(patch_length, STRIDES[patch_length]))
    assert net.head.in_features == 16 * 32 == 512
    changed = x.clone()
    changed[:, :, 5] += 10.0
    after = net.encode(changed)
    others = [c for c in range(16) if c != 5]
    assert torch.allclose(features[:, others], after[:, others])  # no attention across channels
    assert not torch.allclose(features[:, 5], after[:, 5])
    with pytest.raises(ValueError, match="expected"):
        net(torch.randn(2, 96, 16))


def test_scaler_uses_training_statistics_and_contract_channel_order():
    train = np.zeros((128, 16))
    train[:, 0] = np.r_[np.zeros(64), np.full(64, 2.0)]
    train[0, 1] = np.nan
    train[1:, 1] = 4.0
    scaler = WindowScaler().fit([_inputs(train)])
    assert scaler.means_[0] == 1.0 and scaler.stds_[0] == 1.0
    assert scaler.stds_[2] == 1.0  # constant channel
    assert scaler.medians_[1] == 4.0  # missing value imputed with the training median
    reversed_names = list(reversed(NUMERIC_CHANNELS))
    dev = np.zeros((128, 16))
    dev[:, reversed_names.index(NUMERIC_CHANNELS[0])] = 3.0
    scaled = scaler.transform([_inputs(dev, reversed_names)])
    assert (scaled[0, :, 0] == 2.0).all()  # (3 - 1) / 1 in contract order
    empty = np.zeros((128, 16))
    empty[:, 7] = np.nan
    with pytest.raises(ValueError, match="observed training measurement"):
        WindowScaler().fit([_inputs(empty)])


def test_weighted_loss_matches_hand_computation():
    targets = np.array([0, 0, 0, 1])
    weights = balanced_class_weights(targets, 2)
    assert weights.tolist() == pytest.approx([4 / (2 * 3), 4 / (2 * 1)])
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 3.0]])
    y = torch.as_tensor(targets)
    nll = -F.log_softmax(logits, dim=1)[torch.arange(4), y]
    w = torch.from_numpy(weights)[y]
    expected = float((w * nll).sum() / w.sum())
    assert float(F.cross_entropy(logits, y, weight=torch.from_numpy(weights))) == pytest.approx(expected)


def test_training_restores_best_epoch_and_reload_reproduces_logits():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(40, 128, 16)).astype(np.float32)
    targets = np.array([0, 1] * 20)
    train[targets == 1, 60:70, 2] += 4.0
    net, info = fit_patchtst(train, targets, train[:10], targets[:10], task="detection",
                             configuration=CONFIGURATIONS[2], seed=42, max_epochs=3, patience=1)
    assert info["attempted_epochs"] <= 3
    assert info["best_dev_loss"] == min(row["dev_loss"] for row in info["history"])
    repeat, _ = fit_patchtst(train, targets, train[:10], targets[:10], task="detection",
                             configuration=CONFIGURATIONS[2], seed=42, max_epochs=3, patience=1)
    for a, b in zip(net.state_dict().values(), repeat.state_dict().values(), strict=True):
        assert torch.equal(a, b)  # CPU training is deterministic for a fixed seed
    scaler = WindowScaler()
    scaler.medians_ = scaler.means_ = np.zeros(16)
    scaler.stds_ = np.ones(16)
    model = PatchTSTModel("detection", scaler, net, dict(CONFIGURATIONS[2]), 42)
    content, checksum = serialize_model(model)
    reloaded = load_model(content, expected_sha256=checksum, task="detection", scaler=scaler,
                          configuration=dict(CONFIGURATIONS[2]), seed=42)
    windows = [_inputs(train[i]) for i in range(5)]
    assert np.array_equal(model.logits(windows), reloaded.logits(windows))
    with pytest.raises(ValueError, match="checksum"):
        load_model(content + b"x", expected_sha256=checksum, task="detection", scaler=scaler,
                   configuration=dict(CONFIGURATIONS[2]), seed=42)
    with pytest.raises(RuntimeError):
        load_model(content, expected_sha256=checksum, task="category", scaler=scaler,
                   configuration=dict(CONFIGURATIONS[2]), seed=42)
