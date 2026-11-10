"""Deterministic KPI statistics for RCA evidence (Task 4.4).

Values are used as observed: no classifier imputation, no unit conversion and no rate derived
from counters. A statistic reports a measurement, never an abnormality or a cause.
"""

import math
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CALCULATION_VERSION = "kpi-stats-v1"
STATISTICS = ("mean", "minimum", "maximum")
MIN_VALID_COUNT = {"mean": 1, "minimum": 1, "maximum": 1}
MISSING_RULE = ("a sample is missing when its value is absent (null or empty); zero is a valid value; "
                "non-finite numbers are excluded from statistics and counted separately")
EXTREMUM_RULE = "position of the first sample holding the extreme value, as an absolute sample index"
GAP_FACTOR = 1.5

Semantics = Literal["reported_value", "cumulative_counter", "categorical", "identifier"]


class ChannelRejection(BaseModel):
    """A channel (or some of its values) that produced no statistic, with the reason."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    channel: str
    scope: Literal["channel", "values"]
    reason: str
    sample_start: int | None = None
    sample_end: int | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class ChannelStatistic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    channel: str
    statistic: Literal["mean", "minimum", "maximum"]
    value: float = Field(allow_inf_nan=False)
    sample_start: int = Field(ge=0)
    sample_end: int = Field(ge=0)
    valid_count: int = Field(ge=1)
    sample_count: int = Field(ge=1)
    missing_fraction: float = Field(ge=0, le=1)
    non_finite_count: int = Field(ge=0)
    extremum_index: int | None = None


class TimeIntegrityError(ValueError):
    """Timestamps cannot attribute samples safely (duplicates or wrong order)."""


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def counter_resets(values: Sequence[Any], start: int = 0) -> list[int]:
    """Absolute indices where a cumulative counter decreases; a decrease is a reset, not a change."""
    resets, previous = [], None
    for index, value in enumerate(values):
        if _is_missing(value) or not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        if previous is not None and value < previous:
            resets.append(start + index)
        previous = value
    return resets


def summarize_channel(name: str, values: Sequence[Any], *, semantics: Semantics,
                      start: int = 0) -> tuple[list[ChannelStatistic], list[ChannelRejection]]:
    """Mean, minimum and maximum of observed values in samples ``start..start+len-1``."""
    end = start + len(values) - 1
    bounds = {"sample_start": start, "sample_end": end} if values else {}

    def reject(reason, **detail):
        return [], [ChannelRejection(channel=name, scope="channel", reason=reason, detail=detail, **bounds)]

    if not values:
        return reject("empty_interval")
    if semantics == "categorical":
        return reject("categorical_channel_no_numeric_statistics")
    if semantics == "identifier":
        return reject("identifier_field_not_a_measurement")
    missing = sum(_is_missing(v) for v in values)
    present = [(start + i, v) for i, v in enumerate(values) if not _is_missing(v)]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for _, v in present):
        return reject("non_numeric_value")
    if semantics == "cumulative_counter":
        return reject("counter_semantics_unverified", resets=counter_resets(values, start),
                      note="cumulative counters need verified semantics before deltas or rates")
    finite = [(i, float(v)) for i, v in present if math.isfinite(v)]
    non_finite = [i for i, v in present if not math.isfinite(v)]
    if len(finite) < min(MIN_VALID_COUNT.values()):
        return reject("no_valid_values", missing=missing, non_finite=len(non_finite))
    rejections = ([ChannelRejection(channel=name, scope="values", reason="non_finite_values_excluded",
                                    detail={"indices": non_finite}, **bounds)] if non_finite else [])
    numbers = [v for _, v in finite]
    common = {"channel": name, "sample_start": start, "sample_end": end, "valid_count": len(finite),
              "sample_count": len(values), "missing_fraction": missing / len(values),
              "non_finite_count": len(non_finite)}
    low = min(finite, key=lambda item: item[1])
    high = max(finite, key=lambda item: item[1])  # max/min keep the first occurrence
    return [
        ChannelStatistic(statistic="mean", value=math.fsum(numbers) / len(numbers), **common),
        ChannelStatistic(statistic="minimum", value=low[1], extremum_index=low[0], **common),
        ChannelStatistic(statistic="maximum", value=high[1], extremum_index=high[0], **common),
    ], rejections


def uniform_times(start_time: str, rate_hz: float, count: int) -> list[str]:
    """Sample timestamps of a fixed-cadence window (naive, millisecond or microsecond precision)."""
    origin = datetime.fromisoformat(start_time)
    digits = len(start_time.rpartition(".")[2]) if "." in start_time else 0
    timespec = "milliseconds" if digits == 3 else "microseconds"
    return [(origin + timedelta(seconds=i / rate_hz)).isoformat(sep=" ", timespec=timespec)
            for i in range(count)]


def check_timestamps(stamps: Sequence[datetime], *, nominal_period: float | None = None) -> list[dict]:
    """Reject duplicates and wrong order; return gaps longer than ``GAP_FACTOR`` periods."""
    gaps = []
    for index in range(1, len(stamps)):
        delta = (stamps[index] - stamps[index - 1]).total_seconds()
        if delta == 0:
            raise TimeIntegrityError(f"duplicate timestamp at sample {index}")
        if delta < 0:
            raise TimeIntegrityError(f"timestamps out of order at sample {index}")
        if nominal_period is not None and delta > GAP_FACTOR * nominal_period:
            gaps.append({"after_sample": index - 1, "seconds": delta})
    return gaps


def summarize_kpi_window(kpi_window) -> tuple[list[ChannelStatistic], list[ChannelRejection]]:
    """Statistics for every channel of a ``KpiWindow``; categorical channels are rejected."""
    statistics, rejections = [], []
    for name in sorted(kpi_window.channels):
        channel = kpi_window.channels[name]
        stats, rejected = summarize_channel(
            name, channel.values, semantics="categorical" if channel.kind == "categorical" else "reported_value")
        statistics.extend(stats)
        rejections.extend(rejected)
    return statistics, rejections
