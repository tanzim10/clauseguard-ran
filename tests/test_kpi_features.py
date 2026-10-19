import numpy as np
import pytest

from clauseguard.datasets.contracts import NUMERIC_CHANNELS, ModelInputs, NumericInputs
from clauseguard.kpi.features import STATISTICS, ChannelMedianFeatures


def window(values, names=NUMERIC_CHANNELS, rate=10):
    rows = [[value] * len(names) for value in values]
    return ModelInputs(kpi=NumericInputs(channel_names=list(names), sampling_rate_hz=rate,
                                        values=rows))


def first_channel(features):
    return dict(zip(STATISTICS, features[0, :14]))


def test_constant_has_224_named_features_and_preserves_absolute_scale():
    fitted = ChannelMedianFeatures()
    result = fitted.fit_transform([window([7] * 128)])
    assert result.shape == (1, 224)
    assert fitted.feature_names_[0] == "DL_BLER__mean"
    assert fitted.feature_names_[-1] == "UL_SNR__half_std_change"
    np.testing.assert_array_equal(fitted.medians_, [7] * 16)
    np.testing.assert_array_equal(result[0, :14], [7, 0, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0])


def test_ramp_slope_uses_seconds_and_quartiles_are_linear():
    stats = first_channel(ChannelMedianFeatures().fit_transform([window(list(range(128)), rate=2)]))
    assert stats == pytest.approx({
        "mean": 63.5, "std": (1365.25) ** 0.5, "min": 0, "max": 127,
        "q25": 31.75, "median": 63.5, "q75": 95.25, "zero_fraction": 1/128,
        "missing_fraction": 0, "slope_per_second": 2, "mean_absolute_difference": 1,
        "max_jump": 1, "half_mean_change": 64, "half_std_change": 0,
    })


def test_late_spike_retains_jump_and_half_changes():
    stats = first_channel(ChannelMedianFeatures().fit_transform([window([0] * 127 + [128])]))
    assert stats["mean"] == 1
    assert stats["std"] == pytest.approx(127 ** 0.5)
    assert stats["max_jump"] == 128
    assert stats["mean_absolute_difference"] == pytest.approx(128 / 127)
    assert stats["half_mean_change"] == 2
    assert stats["half_std_change"] == pytest.approx(252 ** 0.5)
    assert stats["q75"] == 0
    assert stats["zero_fraction"] == 127 / 128


def test_imputation_fits_train_only_and_fractions_precede_imputation():
    transformer = ChannelMedianFeatures().fit([window([0, 2] + [None] * 126)])
    before = transformer.medians_.copy()
    stats = first_channel(transformer.transform([window([100] + [None] * 127)]))
    assert stats["mean"] == pytest.approx(227 / 128)
    assert stats["median"] == 1
    assert stats["missing_fraction"] == 127 / 128
    assert stats["zero_fraction"] == 0
    np.testing.assert_array_equal(transformer.medians_, before)
    all_missing = first_channel(transformer.transform([window([None] * 128)]))
    assert all_missing["mean"] == 1
    assert all_missing["missing_fraction"] == 1
    assert all_missing["zero_fraction"] == 0


def test_named_channels_are_reordered_before_fitting_and_transforming():
    names = tuple(reversed(NUMERIC_CHANNELS))
    reordered = window([0] * 128, names)
    reordered.kpi.values = [[NUMERIC_CHANNELS.index(name) for name in names] for _ in range(128)]
    transformer = ChannelMedianFeatures()
    output = transformer.fit_transform([reordered])
    np.testing.assert_array_equal(transformer.medians_, np.arange(16))
    np.testing.assert_array_equal(output[0, ::14], np.arange(16))


def test_fit_requires_observations_in_every_channel():
    with pytest.raises(ValueError, match="training windows"):
        ChannelMedianFeatures().fit([])
    with pytest.raises(ValueError, match="observed training"):
        ChannelMedianFeatures().fit([window([None] * 128)])
    sample = window([1] * 128)
    sample.kpi.values = [[None] + row[1:] for row in sample.kpi.values]
    with pytest.raises(ValueError, match="observed training"):
        ChannelMedianFeatures().fit([sample])


@pytest.mark.parametrize("field,value,message", [
    ("kpi", None, "KPI-only"),
    ("text", "context", "KPI-only"),
    ("values", [[1] * 16], "128 by 16"),
    ("values", [[1] * 15] * 128, "128 by 16"),
    ("channel_names", ["DL_BLER"] * 16, "distinct"),
    ("channel_names", list(NUMERIC_CHANNELS[:-1]), "distinct"),
    ("sampling_rate_hz", 0, "sampling rate"),
    ("sampling_rate_hz", float("nan"), "sampling rate"),
    ("values", [[float("nan")] * 16] * 128, "finite"),
    ("values", [[float("inf")] * 16] * 128, "finite"),
])
def test_invalid_windows_are_rejected(field, value, message):
    transformer = ChannelMedianFeatures().fit([window([1] * 128)])
    item = window([1] * 128)
    if field in ("kpi", "text"):
        item = item.model_copy(update={field: value})
    else:
        item = item.model_copy(update={"kpi": item.kpi.model_copy(update={field: value})})
    with pytest.raises(ValueError, match=message):
        transformer.transform([item])


def test_transform_requires_fit_and_accepts_an_empty_batch_after_fit():
    with pytest.raises(ValueError, match="fit channel medians"):
        ChannelMedianFeatures().transform([window([1] * 128)])
    assert ChannelMedianFeatures().fit([window([1] * 128)]).transform([]).shape == (0, 224)
