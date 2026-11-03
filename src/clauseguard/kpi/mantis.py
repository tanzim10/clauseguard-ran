"""Frozen MantisV2 features for the fixed TelecomTS KPI window contract."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from clauseguard.datasets.contracts import NUMERIC_CHANNELS, ModelInputs
from clauseguard.kpi.features import ChannelMedianFeatures

UPSTREAM_COMMIT = "23ecfe493830db177546ef159947de7d79a573bd"
PACKAGE_VERSION = "1.1.0"
CHECKPOINT_REPOSITORY = "paris-noah/MantisV2"
CHECKPOINT_REVISION = "99fe0f548960e272fbfa4b82fd9b5b5956779dfd"
CHECKPOINT_SHA256 = "49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1"
SEQ_LEN = 128
NUM_PATCHES = 32
FEATURES_PER_CHANNEL = 512
EMBEDDING_WIDTH = len(NUMERIC_CHANNELS) * FEATURES_PER_CHANNEL
STATISTICS_WIDTH = len(NUMERIC_CHANNELS) * 2
ENCODER_PARAMETERS = 4_188_672
ENCODER_WEIGHT_BYTES = ENCODER_PARAMETERS * 4


def _task_labels(task: str) -> set[int]:
    if task == "detection":
        return {0, 1}
    if task == "category":
        return set(range(10))
    raise ValueError("invalid Mantis task")


class WindowImputer:
    """Fill missing values with medians learned from training windows only."""

    def __init__(self):
        self.medians_: np.ndarray | None = None

    def fit(self, inputs: Sequence[ModelInputs]):
        if not inputs:
            raise ValueError("cannot fit Mantis imputation without training windows")
        windows = [ChannelMedianFeatures._window(item)[0] for item in inputs]
        observed = np.concatenate(windows, axis=0)
        if np.isnan(observed).all(axis=0).any():
            raise ValueError("every channel needs an observed training measurement")
        self.medians_ = np.nanmedian(observed, axis=0)
        if not np.isfinite(self.medians_).all():
            raise ValueError("training medians must be finite")
        return self

    def transform(self, inputs: Sequence[ModelInputs]) -> np.ndarray:
        if self.medians_ is None:
            raise ValueError("fit Mantis imputation before transforming windows")
        output = np.empty((len(inputs), SEQ_LEN, len(NUMERIC_CHANNELS)), dtype=np.float32)
        for index, item in enumerate(inputs):
            values = ChannelMedianFeatures._window(item)[0]
            output[index] = np.where(np.isnan(values), self.medians_, values)
        if not np.isfinite(output).all():
            raise ValueError("imputed windows must contain finite values")
        return output

    def state(self) -> dict:
        if self.medians_ is None:
            raise ValueError("fit Mantis imputation before saving its state")
        return {"channel_order": list(NUMERIC_CHANNELS), "channel_medians": self.medians_.tolist()}

    @classmethod
    def from_state(cls, state: dict):
        if state["channel_order"] != list(NUMERIC_CHANNELS):
            raise ValueError("checkpoint channel order differs from the KPI contract")
        medians = np.asarray(state["channel_medians"], dtype=np.float64)
        if medians.shape != (len(NUMERIC_CHANNELS),) or not np.isfinite(medians).all():
            raise ValueError("checkpoint contains invalid channel medians")
        result = cls()
        result.medians_ = medians
        return result


def raw_statistics(windows: np.ndarray) -> np.ndarray:
    """Return channel means and population standard deviations after imputation."""
    values = np.asarray(windows, dtype=np.float64)
    expected = (SEQ_LEN, len(NUMERIC_CHANNELS))
    if values.ndim != 3 or tuple(values.shape[1:]) != expected:
        raise ValueError(f"expected windows shaped (batch, {expected[0]}, {expected[1]})")
    means = values.mean(axis=1)
    stds = values.std(axis=1, ddof=0)
    result = np.concatenate((means, stds), axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Mantis raw statistics must be finite")
    return result.astype(np.float32)


def load_encoder(*, device: str = "cpu", checkpoint_path: Path | None = None):
    """Load the pinned checkpoint and reject missing or changed weights."""
    try:
        from importlib.metadata import version

        import torch
        from huggingface_hub import hf_hub_download
        from mantis.architecture.version2 import MantisV2
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("install Mantis with `uv sync --extra baselines --extra mantis`") from exc
    if version("mantis-tsfm") != PACKAGE_VERSION:
        raise RuntimeError(f"expected mantis-tsfm {PACKAGE_VERSION}")
    if device not in ("cpu", "cuda"):
        raise ValueError("Mantis device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    path = (Path(checkpoint_path) if checkpoint_path is not None else Path(hf_hub_download(
        repo_id=CHECKPOINT_REPOSITORY, filename="model.safetensors",
        revision=CHECKPOINT_REVISION)))
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum != CHECKPOINT_SHA256:
        raise ValueError("Mantis checkpoint checksum mismatch")

    model = MantisV2(
        num_patches=NUM_PATCHES, return_transf_layer=2, output_token="combined", device=device,
    )
    state = load_file(str(path), device=device)
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Mantis encoder weights must be frozen")
    return model


def encode_windows(encoder, windows: np.ndarray, *, batch_size: int = 32,
                   device: str = "cpu") -> np.ndarray:
    """Encode all 16 channels with one frozen encoder and preserve channel order."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("install torch with `uv sync --extra timesnet`") from exc
    values = np.asarray(windows, dtype=np.float32)
    expected = (SEQ_LEN, len(NUMERIC_CHANNELS))
    if values.ndim != 3 or tuple(values.shape[1:]) != expected:
        raise ValueError(f"expected windows shaped (batch, {expected[0]}, {expected[1]})")
    if SEQ_LEN % NUM_PATCHES:
        raise ValueError("sequence length must be divisible by num_patches")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not np.isfinite(values).all():
        raise ValueError("Mantis inputs must be finite")
    if not len(values):
        return np.empty((0, EMBEDDING_WIDTH), dtype=np.float32)

    outputs = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = values[start:start + batch_size]
            series = batch.transpose(0, 2, 1).reshape(-1, 1, SEQ_LEN)
            result = encoder(torch.as_tensor(series, dtype=torch.float32, device=device))
            if not isinstance(result, torch.Tensor) or tuple(result.shape) != (
                    len(batch) * len(NUMERIC_CHANNELS), FEATURES_PER_CHANNEL):
                shape = getattr(result, "shape", None)
                raise ValueError(f"Mantis returned shape {shape}; expected 512 values per channel")
            outputs.append(result.reshape(len(batch), EMBEDDING_WIDTH).cpu().numpy())
    features = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(features).all():
        raise ValueError("Mantis embeddings must be finite")
    return features


def state_digest(model) -> str:
    """Hash all encoder tensors so callers can verify that fitting changed no weights."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _variant_features(variant: str, embeddings: np.ndarray, statistics: np.ndarray) -> np.ndarray:
    embedded = np.asarray(embeddings, dtype=np.float32)
    stats = np.asarray(statistics, dtype=np.float32)
    if embedded.ndim != 2 or embedded.shape[1] != EMBEDDING_WIDTH:
        raise ValueError("Mantis embeddings must contain 8,192 values per window")
    if stats.shape != (len(embedded), STATISTICS_WIDTH):
        raise ValueError("Mantis statistics must contain 32 values per window")
    if variant == "embeddings":
        return embedded
    if variant == "embeddings+statistics":
        return np.concatenate((embedded, stats), axis=1)
    raise ValueError("invalid Mantis feature variant")


def fit_head(features: np.ndarray, targets: Sequence[int], *, task: str):
    """Fit the fixed, balanced logistic head and feature scaler on training data."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    values = np.asarray(features, dtype=np.float32)
    labels = np.asarray(targets, dtype=np.int64)
    expected = _task_labels(task)
    if values.ndim != 2 or len(values) != len(labels) or not len(labels):
        raise ValueError("training features and targets must have matching rows")
    if not np.isfinite(values).all() or set(labels.tolist()) != expected:
        raise ValueError("training data must be finite and contain every task label")
    counts = np.bincount(labels, minlength=len(expected))
    weights = {label: len(labels) / (len(expected) * counts[label])
               for label in range(len(expected))}
    scaler = StandardScaler().fit(values)
    classifier = LogisticRegression(
        C=1, solver="lbfgs", max_iter=2000, class_weight=weights, random_state=42,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(scaler.transform(values), labels)
    if any(issubclass(item.category, ConvergenceWarning) for item in captured):
        raise RuntimeError("balanced logistic classifier did not converge")
    if set(classifier.classes_.tolist()) != expected:
        raise RuntimeError("logistic classifier returned an incomplete class set")
    return scaler, classifier, weights
