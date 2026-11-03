import hashlib

import numpy as np
import pytest

from clauseguard.datasets.contracts import NUMERIC_CHANNELS, ModelInputs, NumericInputs
from clauseguard.kpi.mantis import (
    EMBEDDING_WIDTH,
    NUM_PATCHES,
    SEQ_LEN,
    STATISTICS_WIDTH,
    WindowImputer,
    _variant_features,
    encode_windows,
    load_encoder,
    raw_statistics,
    state_digest,
)


def _window(offset=0, *, missing=None, names=NUMERIC_CHANNELS):
    values = []
    for step in range(SEQ_LEN):
        row = [float(offset + step + channel) for channel in range(len(names))]
        if missing is not None:
            row[names.index(missing)] = None
        values.append(row)
    return ModelInputs(kpi=NumericInputs(channel_names=list(names), sampling_rate_hz=10,
                                        values=values))


def test_imputer_uses_training_medians_and_fixed_channel_order():
    first = _window(0, missing=NUMERIC_CHANNELS[0])
    second = _window(10)
    imputer = WindowImputer().fit([first, second])

    transformed = imputer.transform([first])

    assert transformed.shape == (1, SEQ_LEN, len(NUMERIC_CHANNELS))
    assert transformed[0, 0, 0] == pytest.approx(73.5)
    assert imputer.state()["channel_order"] == list(NUMERIC_CHANNELS)
    restored = WindowImputer.from_state(imputer.state())
    np.testing.assert_array_equal(restored.medians_, imputer.medians_)


def test_imputer_rejects_channel_without_observed_training_values():
    inputs = [_window(missing=NUMERIC_CHANNELS[2])]

    with pytest.raises(ValueError, match="every channel"):
        WindowImputer().fit(inputs)


def test_imputer_rejects_wrong_channel_contract():
    names = NUMERIC_CHANNELS[:-1] + (NUMERIC_CHANNELS[0],)

    with pytest.raises(ValueError, match="exactly the 16 distinct"):
        WindowImputer().fit([_window(names=names)])


def test_raw_statistics_use_population_standard_deviation():
    values = np.zeros((1, SEQ_LEN, len(NUMERIC_CHANNELS)), dtype=np.float32)
    values[0, :, 0] = np.arange(SEQ_LEN)

    result = raw_statistics(values)

    assert result.shape == (1, STATISTICS_WIDTH)
    assert result[0, 0] == pytest.approx(63.5)
    assert result[0, len(NUMERIC_CHANNELS)] == pytest.approx(np.arange(SEQ_LEN).std())


def test_encoder_receives_one_series_per_channel_and_returns_fixed_width():
    torch = pytest.importorskip("torch")

    class FakeEncoder:
        def eval(self):
            return self

        def __call__(self, values):
            assert torch.is_inference_mode_enabled()
            assert tuple(values.shape[1:]) == (1, SEQ_LEN)
            return values.mean(dim=(1, 2))[:, None].repeat(1, 512)

    windows = np.zeros((2, SEQ_LEN, len(NUMERIC_CHANNELS)), dtype=np.float32)
    windows[:, :, 1] = 1

    result = encode_windows(FakeEncoder(), windows, batch_size=1)

    assert result.shape == (2, EMBEDDING_WIDTH)
    assert np.all(result[0, :512] == 0)
    assert np.all(result[0, 512:1024] == 1)


def test_feature_variants_enforce_exact_dimensions():
    embeddings = np.zeros((2, EMBEDDING_WIDTH), dtype=np.float32)
    statistics = np.zeros((2, STATISTICS_WIDTH), dtype=np.float32)

    assert _variant_features("embeddings", embeddings, statistics).shape == (2, 8192)
    assert _variant_features("embeddings+statistics", embeddings, statistics).shape == (2, 8224)
    with pytest.raises(ValueError, match="8,192"):
        _variant_features("embeddings", embeddings[:, :-1], statistics)


def test_sequence_length_must_match_patch_count():
    assert SEQ_LEN % NUM_PATCHES == 0
    assert SEQ_LEN // NUM_PATCHES == 4


def test_local_checkpoint_digest_loads_and_freezes_pretrained_tensors(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("mantis")
    from mantis.architecture.version2 import MantisV2
    from safetensors.torch import save_file

    import clauseguard.kpi.mantis as mantis_module

    source = MantisV2(num_patches=NUM_PATCHES, return_transf_layer=2,
                      output_token="combined", device="cpu")
    checkpoint = tmp_path / "model.safetensors"
    save_file(source.state_dict(), str(checkpoint))
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(mantis_module, "CHECKPOINT_SHA256", checksum)

    loaded = load_encoder(device="cpu", checkpoint_path=checkpoint)

    assert loaded.return_transf_layer == 2 and loaded.output_token == "combined"
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert state_digest(loaded) == state_digest(source)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_encoder(device="cpu", checkpoint_path=checkpoint)
