"""Frozen Toto 2.0 forecasts and forecast-error detection for the TelecomTS window contract."""

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from clauseguard.datasets.contracts import NUMERIC_CHANNELS, ModelInputs
from clauseguard.kpi.features import ChannelMedianFeatures

PACKAGE = "toto-2"
PACKAGE_VERSION = "2.0.0"
CHECKPOINT_REPOSITORY = "Datadog/Toto-2.0-22m"
CHECKPOINT_REVISION = "685e4ae3e2be8d8998025e53dd98e7fdcb296a89"
CHECKPOINT_SHA256 = "9cd503d82df3aa71747862688f47a31c1d0a4b80f898df6e046189016eaa21dd"
CHECKPOINT_PARAMETERS = 21_915_584
PATCH_SIZE = 32
SEQ_LEN = 128
CONTEXT = 96
HORIZON = SEQ_LEN - CONTEXT
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MEDIAN_INDEX = QUANTILES.index(0.5)
THRESHOLD_QUANTILES = 101
CHANNELS = len(NUMERIC_CHANNELS)


def window_arrays(inputs: Sequence[ModelInputs]) -> np.ndarray:
    """(batch, 16, 128) in `NUMERIC_CHANNELS` order; missing values stay NaN."""
    windows = np.empty((len(inputs), CHANNELS, SEQ_LEN), dtype=np.float64)
    for index, item in enumerate(inputs):
        windows[index] = ChannelMedianFeatures._window(item)[0].T
    return windows


def fit_scales(windows: np.ndarray, targets: Sequence[int]) -> np.ndarray:
    """Population SD of observed values in normal training windows; 1 for a constant channel."""
    labels = np.asarray(targets)
    normal = np.asarray(windows, dtype=np.float64)[labels == 0]
    if not len(normal):
        raise ValueError("scales need at least one normal training window")
    values = normal.transpose(1, 0, 2).reshape(CHANNELS, -1)
    observed = ~np.isnan(values)
    if not observed.any(axis=1).all():
        raise ValueError("every channel needs an observed normal training measurement")
    scales = np.array([values[c, observed[c]].std(ddof=0) for c in range(CHANNELS)])
    return np.where(scales > 0, scales, 1.0)


def split_windows(windows: np.ndarray):
    """Context (first 96 steps) and forecast target (last 32 steps), each with observation masks."""
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (CHANNELS, SEQ_LEN):
        raise ValueError(f"expected windows shaped (batch, {CHANNELS}, {SEQ_LEN})")
    if CONTEXT % PATCH_SIZE or HORIZON % PATCH_SIZE:
        raise ValueError("context and horizon must be whole Toto patches")
    context, target = values[..., :CONTEXT], values[..., CONTEXT:]
    return context, ~np.isnan(context), target, ~np.isnan(target)


def fill_context(context: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Finite placeholders only where the mask is false: carry the last observation, else the next."""
    filled = np.array(context, dtype=np.float64)
    for index in np.ndindex(*filled.shape[:2]):
        row, observed = filled[index], mask[index]
        if observed.all():
            continue
        if not observed.any():
            filled[index] = 0.0
            continue
        positions = np.where(observed, np.arange(CONTEXT), -1)
        last = np.maximum.accumulate(positions)
        first = observed.argmax()
        filled[index] = row[np.where(last >= 0, last, first)]
    return filled


def last_value_forecast(context: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Repeat each channel's last observed context value; NaN when a channel has none."""
    steps = context.shape[2]
    positions = np.where(mask, np.arange(steps), -1).max(axis=2)
    values = np.take_along_axis(context, np.maximum(positions, 0)[..., None], axis=2)[..., 0]
    values = np.where(positions >= 0, values, np.nan)
    return np.repeat(values[..., None], HORIZON, axis=2)


def window_scores(forecast: np.ndarray, target: np.ndarray, target_mask: np.ndarray,
                  context_mask: np.ndarray, scales: np.ndarray):
    """Mean over channels of scaled mean absolute error on observed targets, plus failure reasons."""
    scores = np.full(len(target), np.nan)
    failures: dict[int, str] = {}
    for index in range(len(target)):
        errors = []
        for channel, name in enumerate(NUMERIC_CHANNELS):
            observed = target_mask[index, channel]
            if not context_mask[index, channel].any():
                failures[index] = f"{name} has no observed context"
                break
            if not observed.any():
                failures[index] = f"{name} has no observed forecast target"
                break
            predicted = forecast[index, channel, observed]
            if not np.isfinite(predicted).all():
                failures[index] = f"{name} forecast is not finite"
                break
            errors.append(np.abs(predicted - target[index, channel, observed]).mean() / scales[channel])
        else:
            scores[index] = float(np.mean(errors))
    return scores, failures


def _macro_f1(predicted: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    f1 = []
    for label in (0, 1):
        tp = int(((predicted == label) & (labels == label)).sum())
        fp = int(((predicted == label) & (labels != label)).sum())
        fn = int(((labels == label) & (predicted != label)).sum())
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return sum(f1) / 2, int(((predicted == 1) & (labels == 0)).sum())


def apply_threshold(scores: np.ndarray, threshold: float | None) -> np.ndarray:
    """1 = abnormal (`score >= threshold`), 0 = normal, -1 = failed. `None` calls every window normal."""
    values = np.asarray(scores, dtype=np.float64)
    abnormal = np.zeros(values.shape, dtype=int) if threshold is None else (values >= threshold).astype(int)
    return np.where(np.isfinite(values), abnormal, -1)


def select_threshold(scores: np.ndarray, labels: Sequence[int]) -> dict:
    """Dev macro-F1 over 101 score quantiles plus an all-normal endpoint; failures count as wrong."""
    values, targets = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("every development forecast failed; no threshold can be selected")
    candidates = [*sorted(set(np.quantile(finite, np.linspace(0, 1, THRESHOLD_QUANTILES)).tolist())),
                  None]
    trials = []
    for threshold in candidates:
        macro_f1, false_alarms = _macro_f1(apply_threshold(values, threshold), targets)
        trials.append({"threshold": threshold, "macro_f1": macro_f1, "false_alarms": false_alarms})
    best = max(trials, key=lambda row: (row["macro_f1"], -row["false_alarms"],
                                        np.inf if row["threshold"] is None else row["threshold"]))
    return {"threshold": best["threshold"], "dev_macro_f1": best["macro_f1"],
            "candidates": len(trials), "trials": trials}


def load_forecaster(*, device: str = "cpu", checkpoint_dir: Path | None = None):
    """Load the pinned checkpoint, check its bytes and size, and freeze every parameter."""
    try:
        from importlib.metadata import version

        import torch
        from huggingface_hub import snapshot_download
        from toto2 import Toto2Model
    except ImportError as exc:
        raise ImportError("install Toto on Python 3.12+ with `uv sync --extra toto`") from exc
    if version(PACKAGE) != PACKAGE_VERSION:
        raise RuntimeError(f"expected {PACKAGE} {PACKAGE_VERSION}")
    if device not in ("cpu", "cuda"):
        raise ValueError("Toto device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    directory = Path(checkpoint_dir) if checkpoint_dir is not None else Path(snapshot_download(
        repo_id=CHECKPOINT_REPOSITORY, revision=CHECKPOINT_REVISION,
        allow_patterns=["config.json", "model.safetensors"]))
    if hashlib.sha256((directory / "model.safetensors").read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Toto checkpoint checksum mismatch")
    model = Toto2Model.from_pretrained(str(directory), map_location=device)
    if sum(parameter.numel() for parameter in model.parameters()) != CHECKPOINT_PARAMETERS:
        raise RuntimeError("Toto checkpoint has an unexpected parameter count")
    if tuple(round(float(q), 6) for q in model.output_head.knots) != QUANTILES:
        raise RuntimeError("Toto quantile levels differ from the pinned contract")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device).eval()


def forecast_median(model, context: np.ndarray, mask: np.ndarray, *, batch_size: int = 64,
                    device: str = "cpu") -> np.ndarray:
    """Median forecast (batch, 16, 32). Future values never reach the model."""
    import torch

    if context.shape[1:] != (CHANNELS, CONTEXT) or mask.shape != context.shape:
        raise ValueError(f"expected context and mask shaped (batch, {CHANNELS}, {CONTEXT})")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    filled = fill_context(context, mask)
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(filled), batch_size):
            values = torch.as_tensor(filled[start:start + batch_size], dtype=torch.float32, device=device)
            observed = torch.as_tensor(mask[start:start + batch_size], dtype=torch.bool, device=device)
            series_ids = torch.zeros(values.shape[:2], dtype=torch.long, device=device)
            # has_missing_values stays True (the package default) for every batch, so a window's
            # forecast never depends on whether a batch neighbour has gaps.
            quantiles = model.forecast({"target": values, "target_mask": observed,
                                        "series_ids": series_ids}, horizon=HORIZON,
                                       decode_block_size=None, has_missing_values=True)
            expected = (len(QUANTILES), len(values), CHANNELS, HORIZON)
            if tuple(quantiles.shape) != expected:
                raise ValueError(f"Toto returned shape {tuple(quantiles.shape)}; expected {expected}")
            outputs.append(quantiles[MEDIAN_INDEX].float().cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, CHANNELS, HORIZON))
