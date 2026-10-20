"""TimesNet classifier for 128-step KPI windows (PyTorch).

The network is adapted from thuml/Time-Series-Library at commit
4e938a1767106324dd753b2a44832bf870a0252e (MIT License, Copyright (c) 2021 THUML @
Tsinghua University): ``models/TimesNet.py`` (classification path only),
``Inception_Block_V1`` and the value/positional embedding. Upstream's ``exp/`` and
``data_provider/`` are not used: they select models on the test set and standardize each
split with its own statistics.
"""

import hashlib
import io
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from clauseguard.datasets.contracts import NUMERIC_CHANNELS, ModelInputs
from clauseguard.kpi.features import ChannelMedianFeatures

UPSTREAM_COMMIT = "4e938a1767106324dd753b2a44832bf870a0252e"
SEQ_LEN = 128
N_CHANNELS = len(NUMERIC_CHANNELS)
INFERENCE_BATCH_SIZE = 32
GRADIENT_CLIP = 4.0
ARCHITECTURE = {"e_layers": 2, "d_model": 32, "d_ff": 64, "top_k": 3, "num_kernels": 4}
CONFIGURATIONS = tuple({"learning_rate": lr, "dropout": dropout}
                       for lr in (1e-3, 3e-4) for dropout in (0.1, 0.3))
HEADS = ("flatten", "pooled")


def _num_classes(task):
    if task not in ("detection", "category"):
        raise ValueError("invalid TimesNet task")
    return 2 if task == "detection" else 10


class _InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels):
        super().__init__()
        self.kernels = nn.ModuleList(
            nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            for i in range(num_kernels))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return torch.stack([kernel(x) for kernel in self.kernels], dim=-1).mean(-1)


def _periods(x, k):
    """Top-k FFT periods. Amplitudes are averaged over the batch, as upstream does."""
    spectrum = torch.fft.rfft(x, dim=1).abs()
    frequency = spectrum.mean(0).mean(-1)
    frequency[0] = 0
    top = torch.topk(frequency, k).indices.detach().cpu().numpy()
    return x.shape[1] // top, spectrum.mean(-1)[:, top]


class _TimesBlock(nn.Module):
    def __init__(self, d_model, d_ff, top_k, num_kernels):
        super().__init__()
        self.k = top_k
        self.conv = nn.Sequential(
            _InceptionBlock(d_model, d_ff, num_kernels), nn.GELU(),
            _InceptionBlock(d_ff, d_model, num_kernels))

    def forward(self, x):
        batch, length, channels = x.shape
        periods, weights = _periods(x, self.k)
        outputs = []
        for period in periods:
            period = int(period)
            padded = math.ceil(length / period) * period
            out = x
            if padded != length:
                out = torch.cat([x, x.new_zeros(batch, padded - length, channels)], dim=1)
            out = out.reshape(batch, padded // period, period, channels).permute(0, 3, 1, 2)
            out = self.conv(out.contiguous())
            outputs.append(out.permute(0, 2, 3, 1).reshape(batch, -1, channels)[:, :length])
        stacked = torch.stack(outputs, dim=-1)
        weights = F.softmax(weights, dim=1)[:, None, None, :]
        return (stacked * weights).sum(-1) + x


class _Embedding(nn.Module):
    """Value (circular-padded conv) plus fixed sinusoidal positional embedding."""

    def __init__(self, c_in, d_model, dropout):
        super().__init__()
        self.token = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1,
                               padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.token.weight, mode="fan_in", nonlinearity="leaky_relu")
        position = torch.arange(SEQ_LEN, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * -(math.log(10000.0) / d_model))
        pe = torch.zeros(SEQ_LEN, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div), torch.cos(position * div)
        self.register_buffer("pe", pe[None], persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.token(x.permute(0, 2, 1)).transpose(1, 2) + self.pe)


class TimesNetClassifier(nn.Module):
    """Classification logits from ``(batch, 128, 16)`` windows."""

    def __init__(self, num_class, dropout, *, instance_norm=False, head="flatten", **architecture):
        super().__init__()
        arch = {**ARCHITECTURE, **architecture}
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}")
        self.instance_norm, self.head = instance_norm, head
        self.embedding = _Embedding(N_CHANNELS, arch["d_model"], dropout)
        self.blocks = nn.ModuleList(
            _TimesBlock(arch["d_model"], arch["d_ff"], arch["top_k"], arch["num_kernels"])
            for _ in range(arch["e_layers"]))
        self.norm = nn.LayerNorm(arch["d_model"])
        self.dropout = nn.Dropout(dropout)
        # "flatten" is upstream's head: every time step gets its own weights. "pooled" averages
        # over time first, so the head is position-independent and ~100x smaller.
        width = arch["d_model"] * (SEQ_LEN if head == "flatten" else 1)
        self.projection = nn.Linear(width, num_class)

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (SEQ_LEN, N_CHANNELS):
            raise ValueError(f"expected (batch, {SEQ_LEN}, {N_CHANNELS}) windows")
        if self.instance_norm:
            # Per-window, per-channel standardization (upstream's non-stationary normalization).
            # Removes recording-level offsets, and with them absolute KPI levels.
            x = x - x.mean(1, keepdim=True)
            x = x / torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        out = self.embedding(x)
        for block in self.blocks:
            out = self.norm(block(out))
        mask = x.new_ones(x.shape[0], x.shape[1])  # no padding: every window is 128 samples
        out = self.dropout(F.gelu(out)) * mask.unsqueeze(-1)
        if self.head == "pooled":
            return self.projection(out.sum(1) / mask.sum(1, keepdim=True))
        return self.projection(out.reshape(out.shape[0], -1))


class WindowScaler:
    """Train-fitted median imputation and per-channel z-scoring.

    Statistics come from the training cohort only; dev and test windows are transformed
    with them. Standard deviation is the population value; constant channels use 1.
    """

    def __init__(self):
        self.medians_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.stds_: np.ndarray | None = None

    def fit(self, inputs):
        windows = [ChannelMedianFeatures._window(item)[0] for item in inputs]
        if not windows:
            raise ValueError("cannot fit a scaler without training windows")
        observed = np.concatenate(windows, axis=0)
        if np.isnan(observed).all(axis=0).any():
            raise ValueError("every channel requires an observed training measurement")
        self.medians_ = np.nanmedian(observed, axis=0)
        filled = np.where(np.isnan(observed), self.medians_, observed)
        self.means_ = filled.mean(axis=0)
        stds = filled.std(axis=0)
        self.stds_ = np.where(stds > 0, stds, 1.0)
        if not all(np.isfinite(a).all() for a in (self.medians_, self.means_, self.stds_)):
            raise ValueError("scaler statistics exceed finite numeric range")
        return self

    def transform(self, inputs):
        if self.medians_ is None:
            raise ValueError("fit the scaler before transforming windows")
        out = np.empty((len(inputs), SEQ_LEN, N_CHANNELS), dtype=np.float32)
        for index, item in enumerate(inputs):
            window = ChannelMedianFeatures._window(item)[0]
            window = np.where(np.isnan(window), self.medians_, window)
            out[index] = (window - self.means_) / self.stds_
        if not np.isfinite(out).all():
            raise ValueError("scaled windows exceed finite float32 range")
        return out

    def state(self):
        return {"channel_order": list(NUMERIC_CHANNELS), "channel_medians": self.medians_.tolist(),
                "channel_means": self.means_.tolist(), "channel_stds": self.stds_.tolist()}

    @classmethod
    def from_state(cls, state):
        if state["channel_order"] != list(NUMERIC_CHANNELS):
            raise ValueError("checkpoint channel order differs from the current contract")
        scaler = cls()
        scaler.medians_ = np.asarray(state["channel_medians"], dtype=np.float64)
        scaler.means_ = np.asarray(state["channel_means"], dtype=np.float64)
        scaler.stds_ = np.asarray(state["channel_stds"], dtype=np.float64)
        if any(a.shape != (N_CHANNELS,) or not np.isfinite(a).all()
               for a in (scaler.medians_, scaler.means_, scaler.stds_)):
            raise ValueError("checkpoint contains invalid scaler statistics")
        return scaler


def balanced_class_weights(targets, num_class):
    """``N / (K * count)``, identical to the RF/XGBoost training weights."""
    counts = np.bincount(np.asarray(targets, dtype=np.int64), minlength=num_class)
    if (counts == 0).any():
        raise ValueError("training targets must contain every task label")
    return (len(targets) / (num_class * counts)).astype(np.float32)


def recording_balanced_weights(targets, groups, num_class):
    """Per-window weights: equal total weight per class, and equal per recording within a class.

    A 78-window recording then counts as much as a 775-window one, so long recordings cannot
    dominate. The mean weight is 1, matching the balanced class weights' scale.
    """
    targets = np.asarray(targets, dtype=np.int64)
    groups = np.asarray(groups)
    if len(groups) != len(targets):
        raise ValueError("groups must cover every training target")
    if set(np.unique(targets).tolist()) != set(range(num_class)):
        raise ValueError("training targets must contain every task label")
    weights = np.zeros(len(targets), dtype=np.float64)
    for label in range(num_class):
        in_class = targets == label
        recordings = np.unique(groups[in_class])
        for recording in recordings:
            index = in_class & (groups == recording)
            weights[index] = 1.0 / (num_class * len(recordings) * index.sum())
    return (weights * len(targets)).astype(np.float32)


@torch.no_grad()
def predict_logits(net, windows, *, batch_size=INFERENCE_BATCH_SIZE, device="cpu"):
    """Logits in cohort order. Periods are batch-dependent, so the batch size is part of the model."""
    net.eval()
    chunks = [net(torch.from_numpy(windows[i:i + batch_size]).to(device)).cpu()
              for i in range(0, len(windows), batch_size)]
    return torch.cat(chunks).numpy() if chunks else np.empty((0, net.projection.out_features))


def macro_f1(actual, predicted, num_class):
    scores = []
    for label in range(num_class):
        tp = int(((predicted == label) & (actual == label)).sum())
        fp = int(((predicted == label) & (actual != label)).sum())
        fn = int(((predicted != label) & (actual == label)).sum())
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(scores))


def _dev_report(logits, targets, groups, log_prior):
    """Raw and prior-corrected dev cross-entropy, macro-F1 and per-recording accuracy/loss."""
    logits = torch.from_numpy(logits)
    y = torch.as_tensor(targets, dtype=torch.long)
    losses = F.cross_entropy(logits, y, reduction="none").numpy()
    # Balanced weights shift the logits by -log(prior); adding it back recovers the real prior.
    corrected = float(F.cross_entropy(logits + log_prior, y))
    predicted = logits.argmax(1).numpy()
    actual = np.asarray(targets)
    per_group = {}
    if groups is not None:
        for group in sorted(set(groups)):
            index = np.asarray(groups) == group
            per_group[group] = {"windows": int(index.sum()), "loss": float(losses[index].mean()),
                                "accuracy": float((predicted[index] == actual[index]).mean())}
    return {"dev_loss": float(losses.mean()), "dev_prior_corrected_loss": corrected,
            "dev_macro_f1": macro_f1(actual, predicted, logits.shape[1]), "per_group": per_group}


def fit_timesnet(train_windows, train_targets, dev_windows, dev_targets, *, task, configuration,
                 seed, max_epochs=100, patience=10, batch_size=32, device="cpu", on_epoch=None,
                 dev_groups=None, prior_corrected_stopping=False, lr_schedule=None,
                 architecture=None, instance_norm=False, head="flatten",
                 train_groups=None, recording_weights=False):
    """Fit with balanced training weights; stop on dev cross-entropy.

    The default stops on unweighted dev cross-entropy. ``prior_corrected_stopping`` adds the
    log training prior to the logits first, undoing the balanced weights' shift toward the
    minority class. Returns the net restored to its best epoch and a per-epoch history. The
    weighted training loss uses PyTorch's ``mean`` reduction (divides by the batch's weight sum).
    """
    num_class = _num_classes(task)
    if configuration not in CONFIGURATIONS:
        raise ValueError("configuration is outside the four-config search")
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    if lr_schedule not in (None, "cosine"):
        raise ValueError("lr_schedule must be None or 'cosine'")
    if set(np.unique(dev_targets).tolist()) - set(range(num_class)):
        raise ValueError("dev targets contain invalid labels")
    if recording_weights and train_groups is None:
        raise ValueError("recording_weights needs train_groups")
    weights = torch.from_numpy(balanced_class_weights(train_targets, num_class)).to(device)
    sample_weights = (torch.from_numpy(recording_balanced_weights(
        train_targets, train_groups, num_class)).to(device) if recording_weights else None)
    counts = np.bincount(np.asarray(train_targets, dtype=np.int64), minlength=num_class)
    log_prior = torch.from_numpy(np.log(counts / counts.sum()).astype(np.float32))
    torch.manual_seed(seed)
    # MPS lacks deterministic kernels, so only CPU runs are bit-reproducible.
    torch.use_deterministic_algorithms(device == "cpu")
    generator = torch.Generator().manual_seed(seed)
    net = TimesNetClassifier(num_class, configuration["dropout"], instance_norm=instance_norm,
                             head=head, **(architecture or {})).to(device)
    optimizer = torch.optim.RAdam(net.parameters(), lr=configuration["learning_rate"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
                 if lr_schedule == "cosine" else None)
    train_x = torch.from_numpy(train_windows)
    train_y = torch.as_tensor(train_targets, dtype=torch.long)
    selection_key = "dev_prior_corrected_loss" if prior_corrected_stopping else "dev_loss"
    history, best_loss, best_epoch, best_state = [], math.inf, -1, None
    for epoch in range(max_epochs):
        started = time.perf_counter()
        net.train()
        order = torch.randperm(len(train_x), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            logits = net(train_x[index].to(device))
            if sample_weights is None:
                loss = F.cross_entropy(logits, train_y[index].to(device), weight=weights)
            else:  # weighted mean, the same reduction PyTorch uses for class weights
                batch_weights = sample_weights[index.to(device)]
                loss = (F.cross_entropy(logits, train_y[index].to(device), reduction="none")
                        * batch_weights).sum() / batch_weights.sum()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRADIENT_CLIP)
            optimizer.step()
            total += float(loss.detach()) * len(index)
        learning_rate = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()
        report = _dev_report(predict_logits(net, dev_windows, device=device), dev_targets,
                             dev_groups, log_prior)
        row = {"epoch": epoch, "train_loss": total / len(order), "learning_rate": learning_rate,
               **report, "dev_selection_loss": report[selection_key],
               "seconds": time.perf_counter() - started}
        history.append(row)
        if on_epoch:
            on_epoch(row)
        selection_loss = row["dev_selection_loss"]
        if not math.isfinite(selection_loss):
            raise ValueError("dev loss became non-finite; training diverged")
        if selection_loss < best_loss:
            best_loss, best_epoch = selection_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    net.load_state_dict(best_state)
    return net, {"history": history, "best_epoch": best_epoch, "best_dev_loss": best_loss,
                 "selection_loss": selection_key, "attempted_epochs": len(history),
                 "hit_epoch_cap": len(history) == max_epochs and best_epoch >= max_epochs - patience}


@dataclass
class TimesNetModel:
    task: str
    scaler: WindowScaler
    net: TimesNetClassifier
    configuration: dict
    seed: int
    device: str = "cpu"
    batch_size: int = INFERENCE_BATCH_SIZE

    def logits(self, inputs: list[ModelInputs]) -> np.ndarray:
        if not inputs:
            return np.empty((0, _num_classes(self.task)))
        return predict_logits(self.net, self.scaler.transform(inputs),
                              batch_size=self.batch_size, device=self.device)

    def predict(self, inputs: list[ModelInputs]) -> list[int]:
        # numpy argmax returns the first maximum, so ties go to the lowest class index.
        return np.argmax(self.logits(inputs), axis=1).astype(int).tolist()


class TimesNetAdapter:
    def __init__(self, task):
        _num_classes(task)
        self.task = task

    def predict(self, model, inputs):
        if model.task != self.task:
            raise ValueError("adapter and checkpoint tasks differ")
        return model.predict(inputs)


def serialize_model(model: TimesNetModel) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.net.state_dict().items()}, buffer)
    content = buffer.getvalue()
    return content, hashlib.sha256(content).hexdigest()


def load_model(content: bytes, *, expected_sha256: str, task: str, scaler: WindowScaler,
               configuration: dict[str, Any], seed: int, device="cpu",
               batch_size=INFERENCE_BATCH_SIZE, architecture=None,
               instance_norm=False, head="flatten") -> TimesNetModel:
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("checkpoint checksum mismatch")
    if configuration not in CONFIGURATIONS:
        raise ValueError("configuration is outside the four-config search")
    net = TimesNetClassifier(_num_classes(task), configuration["dropout"],
                             instance_norm=instance_norm, head=head, **(architecture or {}))
    state = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    net.load_state_dict(state)  # strict: rejects wrong task head or architecture
    return TimesNetModel(task, scaler, net.to(device).eval(), dict(configuration), seed,
                         device, batch_size)
