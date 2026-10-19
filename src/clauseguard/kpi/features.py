"""Train-fitted imputation and fixed-size statistics for 128-step KPI windows."""

import math
from collections.abc import Sequence

import numpy as np

from clauseguard.datasets.contracts import NUMERIC_CHANNELS, ModelInputs

STATISTICS = (
    "mean", "std", "min", "max", "q25", "median", "q75", "zero_fraction",
    "missing_fraction", "slope_per_second", "mean_absolute_difference", "max_jump",
    "half_mean_change", "half_std_change",
)


class ChannelMedianFeatures:
    """Impute with training channel medians, then emit 14 statistics per channel.

    Columns follow NUMERIC_CHANNELS, regardless of input column order. Standard
    deviations use population variance; quantiles use linear interpolation.
    Missing and zero fractions describe the original 128 measurements.
    """

    feature_names_ = tuple(f"{channel}__{stat}" for channel in NUMERIC_CHANNELS
                           for stat in STATISTICS)

    def __init__(self):
        self.medians_: np.ndarray | None = None
        self.feature_names_ = type(self).feature_names_

    @staticmethod
    def _window(inputs: ModelInputs) -> tuple[np.ndarray, float]:
        kpi = inputs.kpi
        if kpi is None or inputs.text is not None:
            raise ValueError("features require KPI-only inputs")
        names = kpi.channel_names
        if len(names) != len(NUMERIC_CHANNELS) or set(names) != set(NUMERIC_CHANNELS):
            raise ValueError("features require exactly the 16 distinct numeric channels")
        if not math.isfinite(kpi.sampling_rate_hz) or kpi.sampling_rate_hz <= 0:
            raise ValueError("sampling rate must be finite and positive")
        if len(kpi.values) != 128 or any(len(row) != len(names) for row in kpi.values):
            raise ValueError("features require a 128 by 16 window")
        if any(value is not None and not math.isfinite(value)
               for row in kpi.values for value in row):
            raise ValueError("observed KPI values must be finite; use null for missing data")
        array = np.asarray(kpi.values, dtype=np.float64)
        return array[:, [names.index(name) for name in NUMERIC_CHANNELS]], kpi.sampling_rate_hz

    def fit(self, inputs: Sequence[ModelInputs]):
        """Fit exact channel medians from observed training measurements only."""
        if not inputs:
            raise ValueError("cannot fit features without training windows")
        observed = np.concatenate([self._window(item)[0] for item in inputs], axis=0)
        if np.isnan(observed).all(axis=0).any():
            raise ValueError("every channel requires an observed training measurement")
        medians = np.nanmedian(observed, axis=0)
        if not np.isfinite(medians).all():
            raise ValueError("training channel medians exceed finite numeric range")
        self.medians_ = medians
        return self

    def transform(self, inputs: Sequence[ModelInputs]) -> np.ndarray:
        if self.medians_ is None:
            raise ValueError("fit channel medians before transforming windows")
        features = np.empty((len(inputs), len(self.feature_names_)), dtype=np.float64)
        for index, item in enumerate(inputs):
            original, rate = self._window(item)
            missing = np.isnan(original)
            values = np.where(missing, self.medians_, original)
            means = values.mean(axis=0)
            quantiles = np.quantile(values, [0.25, 0.5, 0.75], axis=0)
            differences = np.abs(np.diff(values, axis=0))
            times = (np.arange(128, dtype=np.float64) - 63.5) / rate
            stats = np.stack((
                means, values.std(axis=0), values.min(axis=0),
                values.max(axis=0), *quantiles, (original == 0).mean(axis=0),
                missing.mean(axis=0),
                (times[:, None] * (values - means)).sum(axis=0) / (times @ times),
                differences.mean(axis=0), differences.max(axis=0),
                values[64:].mean(axis=0) - values[:64].mean(axis=0),
                values[64:].std(axis=0) - values[:64].std(axis=0),
            ), axis=1)
            if not np.isfinite(stats).all():
                raise ValueError("KPI statistics exceed finite numeric range")
            features[index] = stats.ravel()
        return features

    def fit_transform(self, inputs: Sequence[ModelInputs]) -> np.ndarray:
        return self.fit(inputs).transform(inputs)
