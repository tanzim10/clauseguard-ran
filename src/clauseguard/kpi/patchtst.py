"""Supervised PatchTST classifier for 128-step KPI windows (PyTorch).

The encoder and classification head are adapted from yuqinie98/PatchTST at commit
204c21efe0b39603ad6e2ca640ef5896646ab1a9 (Apache License 2.0; upstream ``LICENSE``):
``PatchTST_self_supervised/src/models/patchTST.py`` (``PatchTSTEncoder``, ``TSTEncoder``,
``TSTEncoderLayer``, ``ClassificationHead``), ``models/layers/attention.py``,
``models/layers/pos_encoding.py`` (``pe="zeros"``) and ``callback/patch_mask.py``
(``create_patch``). Upstream's training loop, RevIN and data loaders are not used.
Changes from upstream: classification path only, LayerNorm fixed, no stored attention or
masks, and shapes checked for the 16-channel window. License text: ``licenses/PatchTST-LICENSE``.
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
from clauseguard.kpi.timesnet import WindowScaler, balanced_class_weights, macro_f1

UPSTREAM_COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"
SEQ_LEN = 128
N_CHANNELS = len(NUMERIC_CHANNELS)
TRAIN_BATCH_SIZE = 32
INFERENCE_BATCH_SIZE = 32
GRADIENT_CLIP = 1.0
WEIGHT_DECAY = 1e-4
RELOAD_TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}
ARCHITECTURE = {"n_layers": 2, "d_model": 32, "n_heads": 4, "d_ff": 64, "norm": "LayerNorm",
                "dropout": 0.1, "head_dropout": 0.1, "attn_dropout": 0.0, "res_attention": True,
                "pre_norm": False, "pe": "zeros", "learn_pe": True, "shared_embedding": True,
                "activation": "gelu", "head_type": "classification"}
STRIDES = {8: 4, 16: 8}
CONFIGURATIONS = tuple({"patch_length": patch, "learning_rate": lr}
                       for patch in (8, 16) for lr in (1e-3, 3e-4))


def num_classes(task):
    if task not in ("detection", "category"):
        raise ValueError("invalid PatchTST task")
    return 2 if task == "detection" else 10


def patch_count(patch_length, stride):
    return 1 + (SEQ_LEN - patch_length) // stride


def create_patches(x, patch_length, stride):
    """``(batch, 128, 16)`` to ``(batch, patches, 16, patch_length)``; must cover every step."""
    patches = patch_count(patch_length, stride)
    if patch_length + stride * (patches - 1) != SEQ_LEN:
        raise ValueError("patches must cover the full window without cropping or padding")
    return x.unfold(dimension=1, size=patch_length, step=stride)


class _Attention(nn.Module):
    """Multi-head scaled dot-product attention with upstream's residual attention scores."""

    def __init__(self, d_model, n_heads, attn_dropout, proj_dropout):
        super().__init__()
        self.n_heads, self.d_k = n_heads, d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.scale = nn.Parameter(torch.tensor(self.d_k ** -0.5), requires_grad=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.to_out = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(proj_dropout))

    def forward(self, x, prev=None):
        bs = x.size(0)
        q = self.W_Q(x).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(x).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v = self.W_V(x).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k) * self.scale
        if prev is not None:
            scores = scores + prev
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_k)
        return self.to_out(out), scores


class _EncoderLayer(nn.Module):
    """Post-norm attention and feed-forward sublayers with residual connections."""

    def __init__(self, d_model, n_heads, d_ff, dropout, attn_dropout):
        super().__init__()
        self.self_attn = _Attention(d_model, n_heads, attn_dropout, dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, x, prev=None):
        attended, scores = self.self_attn(x, prev)
        x = self.norm_attn(x + self.dropout_attn(attended))
        return self.norm_ffn(x + self.dropout_ffn(self.ff(x))), scores


class PatchTSTClassifier(nn.Module):
    """Channel-independent Transformer over patches; the head reads each channel's last patch."""

    def __init__(self, num_class, patch_length):
        super().__init__()
        if patch_length not in STRIDES:
            raise ValueError(f"patch length must be one of {sorted(STRIDES)}")
        arch = ARCHITECTURE
        self.patch_length, self.stride = patch_length, STRIDES[patch_length]
        self.num_patch = patch_count(patch_length, self.stride)
        self.W_P = nn.Linear(patch_length, arch["d_model"])
        self.W_pos = nn.Parameter(torch.empty(self.num_patch, arch["d_model"]).uniform_(-0.02, 0.02))
        self.dropout = nn.Dropout(arch["dropout"])
        self.layers = nn.ModuleList(
            _EncoderLayer(arch["d_model"], arch["n_heads"], arch["d_ff"], arch["dropout"],
                          arch["attn_dropout"]) for _ in range(arch["n_layers"]))
        self.head_dropout = nn.Dropout(arch["head_dropout"])
        self.head = nn.Linear(N_CHANNELS * arch["d_model"], num_class)

    def encode(self, x):
        """``(batch, 128, 16)`` to ``(batch, 16, d_model, patches)``. No attention across channels."""
        if x.ndim != 3 or tuple(x.shape[1:]) != (SEQ_LEN, N_CHANNELS):
            raise ValueError(f"expected (batch, {SEQ_LEN}, {N_CHANNELS}) windows")
        patches = create_patches(x, self.patch_length, self.stride)
        bs, num_patch, n_vars, _ = patches.shape
        z = self.W_P(patches).transpose(1, 2)
        u = self.dropout(z.reshape(bs * n_vars, num_patch, -1) + self.W_pos)
        scores = None
        for layer in self.layers:
            u, scores = layer(u, scores)
        return u.reshape(bs, n_vars, num_patch, -1).permute(0, 1, 3, 2)

    def forward(self, x):
        features = self.encode(x)[:, :, :, -1].flatten(start_dim=1)
        return self.head(self.head_dropout(features))


@torch.no_grad()
def predict_logits(net, windows, *, batch_size=INFERENCE_BATCH_SIZE, device="cpu"):
    net.eval()
    chunks = [net(torch.from_numpy(windows[i:i + batch_size]).to(device)).cpu()
              for i in range(0, len(windows), batch_size)]
    return torch.cat(chunks).numpy() if chunks else np.empty((0, net.head.out_features))


def fit_patchtst(train_windows, train_targets, dev_windows, dev_targets, *, task, configuration,
                 seed, max_epochs=100, patience=10, device="cpu", on_epoch=None):
    """AdamW with balanced class weights; stop on unweighted dev cross-entropy, restore the best."""
    num_class = num_classes(task)
    if configuration not in CONFIGURATIONS:
        raise ValueError("configuration is outside the four-setting search")
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    if set(np.unique(dev_targets).tolist()) - set(range(num_class)):
        raise ValueError("dev targets contain invalid labels")
    weights = torch.from_numpy(balanced_class_weights(train_targets, num_class)).to(device)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(device == "cpu")
    generator = torch.Generator().manual_seed(seed)
    net = PatchTSTClassifier(num_class, configuration["patch_length"]).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=configuration["learning_rate"],
                                  weight_decay=WEIGHT_DECAY)
    train_x = torch.from_numpy(train_windows)
    train_y = torch.as_tensor(train_targets, dtype=torch.long)
    dev_y = torch.as_tensor(dev_targets, dtype=torch.long)
    history, best_loss, best_epoch, best_state = [], math.inf, -1, None
    for epoch in range(max_epochs):
        started = time.perf_counter()
        net.train()
        order = torch.randperm(len(train_x), generator=generator)
        total = 0.0
        for start in range(0, len(order), TRAIN_BATCH_SIZE):
            index = order[start:start + TRAIN_BATCH_SIZE]
            loss = F.cross_entropy(net(train_x[index].to(device)), train_y[index].to(device),
                                   weight=weights)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRADIENT_CLIP)
            optimizer.step()
            total += float(loss.detach()) * len(index)
        logits = torch.from_numpy(predict_logits(net, dev_windows, device=device))
        dev_loss = float(F.cross_entropy(logits, dev_y))
        row = {"epoch": epoch, "train_loss": total / len(order), "dev_loss": dev_loss,
               "dev_macro_f1": macro_f1(np.asarray(dev_targets), logits.argmax(1).numpy(), num_class),
               "seconds": time.perf_counter() - started}
        history.append(row)
        if on_epoch:
            on_epoch(row)
        if not math.isfinite(dev_loss):
            raise ValueError("dev loss became non-finite; training diverged")
        if dev_loss < best_loss:
            best_loss, best_epoch = dev_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    net.load_state_dict(best_state)
    return net, {"history": history, "best_epoch": best_epoch, "best_dev_loss": best_loss,
                 "attempted_epochs": len(history),
                 "hit_epoch_cap": len(history) == max_epochs and best_epoch >= max_epochs - patience}


@dataclass
class PatchTSTModel:
    task: str
    scaler: WindowScaler
    net: PatchTSTClassifier
    configuration: dict
    seed: int
    device: str = "cpu"
    batch_size: int = INFERENCE_BATCH_SIZE

    def logits(self, inputs: list[ModelInputs]) -> np.ndarray:
        if not inputs:
            return np.empty((0, num_classes(self.task)))
        return predict_logits(self.net, self.scaler.transform(inputs), batch_size=self.batch_size,
                              device=self.device)

    def predict(self, inputs: list[ModelInputs]) -> list[int]:
        # numpy argmax returns the first maximum, so ties go to the lowest class index.
        return np.argmax(self.logits(inputs), axis=1).astype(int).tolist()


class PatchTSTAdapter:
    def __init__(self, task):
        num_classes(task)
        self.task = task

    def predict(self, model, inputs):
        if model.task != self.task:
            raise ValueError("adapter and checkpoint tasks differ")
        return model.predict(inputs)


def serialize_model(model: PatchTSTModel) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.net.state_dict().items()}, buffer)
    content = buffer.getvalue()
    return content, hashlib.sha256(content).hexdigest()


def load_model(content: bytes, *, expected_sha256: str, task: str, scaler: WindowScaler,
               configuration: dict[str, Any], seed: int, device="cpu",
               batch_size=INFERENCE_BATCH_SIZE) -> PatchTSTModel:
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("checkpoint checksum mismatch")
    if configuration not in CONFIGURATIONS:
        raise ValueError("configuration is outside the four-setting search")
    net = PatchTSTClassifier(num_classes(task), configuration["patch_length"])
    net.load_state_dict(torch.load(io.BytesIO(content), map_location="cpu", weights_only=True))
    return PatchTSTModel(task, scaler, net.to(device).eval(), dict(configuration), seed, device,
                         batch_size)
