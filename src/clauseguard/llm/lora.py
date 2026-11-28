"""Task 4.10 QLoRA fine-tuning: versioned config, release verification, masks, selection, run record.

Heavy libraries (torch, transformers, peft, trl, bitsandbytes, liger-kernel) are imported inside the
functions that need them, so the base package and the tests never require them.

Why masks are built here and not by TRL's ``assistant_only_loss``: the Qwen3.5 chat template has no
``{% generation %}`` markers, so TRL could not locate the assistant span. We render with the real
template, tokenize with offsets and supervise exactly the assistant content plus its ``<|im_end|>``
(``locate_assistant``, the same span the 4.9 row check verifies). The trainer receives ``input_ids``
and ``labels`` (-100 elsewhere); no evidence is ever truncated: longer rows are rejected.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from clauseguard.rca.training_release import (
    DEFAULT_TOKENIZER,
    HFChatTokenizer,
    Message,
    check_row,
    locate_assistant,
)

QLORA_CONFIG_VERSION = "qlora-config-v1"
RUN_RECORD_VERSION = "qlora-run-record-v1"
SELECTION_RULE = (
    "metric: mean validation loss on assistant tokens (assistant content plus <|im_end|>), evaluated on the "
    "separate training validation set at the end of every epoch; the lowest value wins; values equal to 4 "
    "decimals tie and the earlier epoch wins"
)
APPROVED_STATUSES = ("sample_approved", "human_reviewed")
# Snapshot of Qwen/Qwen3.5-4B in the local Hugging Face cache (refs/main).
QWEN35_4B_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
# Language-model linear projections: full-attention q/k/v/o, linear-attention in_proj_qkv/in_proj_z/out_proj,
# MLP gate/up/down. The tiny in_proj_a/in_proj_b (32 outputs) and lm_head are left frozen; any visual tower
# is excluded by the lookahead. A regex string, so PEFT matches the full module name.
LORA_TARGET_REGEX = (
    r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)$"
)
LIBRARIES = ("torch", "transformers", "peft", "trl", "accelerate", "bitsandbytes", "liger-kernel",
             "flash-linear-attention", "pydantic", "clauseguard")


class ReleaseError(ValueError):
    """The training release is missing, altered or not approved."""


class OverLengthError(ValueError):
    """A row is longer than the context limit; it is rejected, never truncated."""


class QuantConfig(BaseModel):
    """4-bit base weights; shared by training and local inference so both run the same quantization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    load_in_4bit: bool = True
    quant_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"


class LoraSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    r: int = Field(default=16, ge=1)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: str = LORA_TARGET_REGEX
    bias: Literal["none"] = "none"


class QLoraConfig(BaseModel):
    """Everything fixed before training; its hash goes into the run record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["qlora-config-v1"] = QLORA_CONFIG_VERSION
    base_model: str = DEFAULT_TOKENIZER
    revision: str = QWEN35_4B_REVISION
    quantization: QuantConfig = QuantConfig()
    lora: LoraSettings = LoraSettings()
    optim: str = "paged_adamw_8bit"
    learning_rate: float = Field(default=2e-4, gt=0)
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = Field(default=0.03, ge=0, lt=1)
    epochs: int = Field(default=3, ge=1)
    per_device_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    max_grad_norm: float = Field(default=0.3, gt=0)
    seed: int = 42
    gradient_checkpointing: bool = True
    # Rows in the 4.9 release are 16,766-34,948 tokens; longer rows are rejected, never truncated.
    max_length: int = Field(default=40960, ge=1)
    # Off by default: liger-kernel 0.8.5 has no qwen3_5 patch, and `tail_loss` keeps logits small.
    # The train_lora CLI turns it on unless --no-liger is passed.
    use_liger_kernel: bool = False
    enable_thinking: Literal[False] = False
    selection_rule: str = SELECTION_RULE

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ hashing


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_release(release_dir: Path, *, allow_exploratory: bool = False) -> dict[str, Any]:
    """Check train/validation hashes against the manifest and refuse an unapproved release."""
    release_dir = Path(release_dir)
    manifest_path = release_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ReleaseError(f"no manifest.json in {release_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = manifest.get("release_status")
    if status not in APPROVED_STATUSES and not allow_exploratory:
        raise ReleaseError(f"release_status {status!r} is not one of {APPROVED_STATUSES}; "
                           "pass --allow-exploratory for dry runs only")
    outputs = manifest.get("outputs", {})
    hashes: dict[str, str] = {}
    for name in ("train.jsonl", "validation.jsonl"):
        path = release_dir / name
        if name not in outputs or not path.is_file():
            raise ReleaseError(f"{name} is missing from the release or its manifest")
        actual = sha256_file(path)
        if actual != outputs[name]:
            raise ReleaseError(f"{name} sha256 {actual} does not match manifest {outputs[name]}")
        hashes[name] = actual
    return {"release_status": status, "manifest_sha256": sha256_file(manifest_path), "files": hashes,
            "candidate_release_id": manifest.get("candidate_release_id"), "tokenizer": manifest.get("tokenizer")}


def load_rows(path: Path) -> list[list[Message]]:
    """Each line is ``{"messages": [system, user, assistant]}``."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append([Message.model_validate(m) for m in json.loads(line)["messages"]])
    return rows


# ------------------------------------------------------------- tokenization


class TrainTokenizer(Protocol):
    """ChatTokenizer plus ids; injectable so tests need no model files."""

    def render(self, messages: Sequence[Mapping[str, str]]) -> str: ...

    def token_spans(self, text: str) -> list[tuple[int, int]]: ...

    def encode(self, text: str) -> list[int]: ...


class HFTrainTokenizer(HFChatTokenizer):
    """Local Hugging Face tokenizer with ids; the chat template is the model's own."""

    def __init__(self, name: str = DEFAULT_TOKENIZER, revision: str | None = None) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is not installed; run `uv sync --extra slm`") from exc
        self.name = name
        self._tok = AutoTokenizer.from_pretrained(name, revision=revision)

    def encode(self, text: str) -> list[int]:
        return [int(i) for i in self._tok(text, add_special_tokens=False)["input_ids"]]

    @property
    def chat_template(self) -> str:
        return str(self._tok.chat_template or "")

    @property
    def raw(self) -> Any:
        return self._tok


def encode_example(messages: Sequence[Message], tokenizer: TrainTokenizer, max_length: int) -> dict[str, list[int]]:
    """``input_ids`` and ``labels`` (-100 outside the assistant span); rejects bad or over-long rows."""
    check = check_row("", "", messages, tokenizer, max_length)
    if any(p.startswith("over_token_limit") for p in check.problems):
        raise OverLengthError("; ".join(check.problems))
    if check.problems:
        raise ReleaseError("; ".join(check.problems))
    text = tokenizer.render([m.model_dump() for m in messages])
    span = locate_assistant(text)
    ids = tokenizer.encode(text)
    offsets = tokenizer.token_spans(text)
    if len(ids) != len(offsets):
        raise ReleaseError("token ids and offsets disagree")
    labels = [tok if a >= span.start and b <= span.mask_end else -100 for tok, (a, b) in zip(ids, offsets, strict=True)]
    return {"input_ids": ids, "labels": labels}


def build_dataset(rows: Sequence[Sequence[Message]], tokenizer: TrainTokenizer, max_length: int) -> list[dict]:
    return [encode_example(messages, tokenizer, max_length) for messages in rows]


# ---------------------------------------------------------------- selection


def select_checkpoint(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Lowest validation loss wins; equal to 4 decimals, the earlier epoch wins."""
    if not history:
        raise ValueError("no validation results to select from")
    return dict(min(history, key=lambda h: (round(float(h["eval_loss"]), 4), float(h["epoch"]))))


# -------------------------------------------------------------- environment


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def library_versions() -> dict[str, str | None]:
    return {name: _version(name) for name in LIBRARIES}


def _run(command: list[str]) -> str | None:
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def hardware_details() -> dict[str, Any]:
    """Host, CUDA device and nvidia-smi facts; never fails without a GPU."""
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(),
                            "machine": platform.machine(), "cuda_available": False}
    try:
        import torch

        info["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update(cuda_available=True, device_name=props.name, total_memory_bytes=props.total_memory,
                        compute_capability=list(torch.cuda.get_device_capability(0)),
                        bf16_supported=bool(torch.cuda.is_bf16_supported()),
                        device_count=torch.cuda.device_count())
    except ImportError:
        pass
    info["nvidia_smi"] = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                               "--format=csv,noheader"])
    return info


def git_commit() -> str | None:
    return _run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"])


def build_run_record(*, kind: str, config: QLoraConfig, release: Mapping[str, Any], tokenizer_info: Mapping[str, Any],
                     hardware: Mapping[str, Any], libraries: Mapping[str, Any], results: Mapping[str, Any],
                     timings: Mapping[str, float], command: Sequence[str] | None = None) -> dict[str, Any]:
    """Everything needed to reproduce and audit a run (issue #74 save list)."""
    return {
        "version": RUN_RECORD_VERSION, "kind": kind, "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(), "command": list(command or []),
        "config": config.model_dump(mode="json"), "config_sha256": config.sha256(),
        "model": {"base_model": config.base_model, "revision": config.revision},
        "tokenizer": dict(tokenizer_info),
        "dataset": {"release_status": release.get("release_status"), "manifest_sha256": release.get("manifest_sha256"),
                    "files": dict(release.get("files", {})),
                    "candidate_release_id": release.get("candidate_release_id")},
        "adapter_settings": config.lora.model_dump(mode="json"),
        "optimizer_settings": {"optim": config.optim, "learning_rate": config.learning_rate,
                               "lr_scheduler_type": config.lr_scheduler_type, "warmup_ratio": config.warmup_ratio,
                               "max_grad_norm": config.max_grad_norm, "epochs": config.epochs,
                               "per_device_batch_size": config.per_device_batch_size,
                               "gradient_accumulation_steps": config.gradient_accumulation_steps},
        "seeds": {"seed": config.seed, "data_seed": config.seed},
        "selection_rule": config.selection_rule,
        "libraries": dict(libraries), "hardware": dict(hardware), "results": dict(results),
        "timings_seconds": dict(timings),
    }


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _tokenizer_info(tok: HFTrainTokenizer, config: QLoraConfig) -> dict[str, Any]:
    template = tok.chat_template
    return {"name": tok.name, "revision": config.revision,
            "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "template_args": {"enable_thinking": False}}


def prepare_data(config: QLoraConfig, release_dir: Path, *, allow_exploratory: bool = False
                 ) -> tuple[dict, HFTrainTokenizer, list[dict], list[dict]]:
    release = verify_release(release_dir, allow_exploratory=allow_exploratory)
    tok = HFTrainTokenizer(config.base_model, config.revision)
    train = build_dataset(load_rows(Path(release_dir) / "train.jsonl"), tok, config.max_length)
    valid = build_dataset(load_rows(Path(release_dir) / "validation.jsonl"), tok, config.max_length)
    return release, tok, train, valid


# ------------------------------------------------------------------- model


def bnb_config(quant: QuantConfig) -> Any:
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(load_in_4bit=quant.load_in_4bit, bnb_4bit_quant_type=quant.quant_type,
                              bnb_4bit_use_double_quant=quant.double_quant,
                              bnb_4bit_compute_dtype=getattr(torch, quant.compute_dtype))


def load_quantized_model(config: QLoraConfig) -> Any:
    """Text-only 4-bit base model (AutoModelForCausalLM skips the vision tower)."""
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        config.base_model, revision=config.revision, quantization_config=bnb_config(config.quantization),
        dtype=getattr(torch, config.quantization.compute_dtype), device_map={"": 0})


def _apply_liger(model: Any) -> bool:
    try:
        from liger_kernel.transformers import _apply_liger_kernel_to_instance
    except ImportError:
        return False
    _apply_liger_kernel_to_instance(model=model)
    return True


def tail_loss(model: Any, input_ids: Any, labels: Any, attention_mask: Any = None) -> Any:
    """Assistant-token cross entropy computing logits only for the supervised tail of the sequence.

    The assistant span is at the end of every row, so ``logits_to_keep`` asks the model for the last k
    positions only (from the token before the first label to the end). That avoids a full sequence x
    248k-vocabulary logits tensor (about 17 GB in bf16 for a 35k-token row). Loss is computed in fp32
    and averaged over supervised tokens; the result equals the full-logits loss on the same labels.
    """
    from torch.nn import functional

    if input_ids.shape[0] != 1:
        raise ValueError("tail_loss expects one example per batch")
    supervised = (labels[0] != -100).nonzero()
    if supervised.numel() == 0:
        raise ValueError("a row without supervised tokens")
    first = int(supervised[0])
    length = int(input_ids.shape[1])
    keep = length - first + 1
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=keep).logits
    predicted = logits[:, :-1, :].float()
    target = labels[:, first:]
    return functional.cross_entropy(predicted.reshape(-1, predicted.shape[-1]), target.reshape(-1),
                                    ignore_index=-100)


def _tail_loss_trainer(base: type) -> type:
    """An SFTTrainer whose loss is `tail_loss`; evaluation reuses it (prediction_loss_only)."""

    class TailLossTrainer(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # tail_loss returns a per-example mean, so the trainer must divide it by the accumulation steps;
            # otherwise (Qwen3.5 accepts loss kwargs) 16 accumulated means would be summed, 16x the gradient.
            self.loss_is_scaled_for_ga = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = tail_loss(model, inputs["input_ids"], inputs["labels"], inputs.get("attention_mask"))
            return (loss, {"loss": loss}) if return_outputs else loss

    return TailLossTrainer


def _peft_config(config: QLoraConfig) -> Any:
    from peft import LoraConfig

    return LoraConfig(r=config.lora.r, lora_alpha=config.lora.alpha, lora_dropout=config.lora.dropout,
                      target_modules=config.lora.target_modules, bias=config.lora.bias, task_type="CAUSAL_LM")


def memory_trial(config: QLoraConfig, release_dir: Path, out: Path, *, allow_exploratory: bool = False) -> dict:
    """One forward and backward step on the longest train row; adapter gradients and peak GPU memory."""
    started = time.time()
    release, tok, train, _ = prepare_data(config, release_dir, allow_exploratory=allow_exploratory)
    import torch
    from peft import get_peft_model

    longest = max(train, key=lambda ex: len(ex["input_ids"]))
    model = load_quantized_model(config)
    liger = _apply_liger(model) if config.use_liger_kernel else False
    # Mirror what SFTTrainer does: no prepare_model_for_kbit_training, whose fp32 cast of the embeddings makes
    # every checkpointed activation fp32 and ran out of memory on the L4 at 35k tokens.
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, _peft_config(config))
    model.train()
    device = next(model.parameters()).device
    torch.cuda.reset_peak_memory_stats()
    ids = torch.tensor([longest["input_ids"]], device=device)
    labels = torch.tensor([longest["labels"]], device=device)
    step = time.time()
    dtype = getattr(torch, config.quantization.compute_dtype)
    with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):  # as the Trainer's bf16 mode
        loss = tail_loss(model, ids, labels)
    loss.backward()
    torch.cuda.synchronize()
    step_seconds = time.time() - step
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    missing = [n for n, p in trainable.items() if p.grad is None]
    zero_b = [n for n, p in trainable.items() if "lora_B" in n and p.grad is not None and not bool(p.grad.any())]
    # LoRA B starts at zero, so A's gradient is exactly zero on the first step; only B must be non-zero.
    ok = bool(trainable) and not missing and not zero_b and bool(torch.isfinite(loss).item())
    results = {"passed": ok, "loss": float(loss.item()), "tokens": int(ids.shape[1]),
               "assistant_tokens": sum(1 for x in longest["labels"] if x != -100),
               "trainable_tensors": len(trainable), "tensors_without_grad": missing[:10],
               "lora_B_with_zero_grad": zero_b[:10], "liger_applied": liger,
               "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
               "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
               "step_seconds": step_seconds}
    record = build_run_record(kind="memory_trial", config=config, release=release,
                              tokenizer_info=_tokenizer_info(tok, config), hardware=hardware_details(),
                              libraries=library_versions(), results=results,
                              timings={"total": time.time() - started})
    write_json(Path(out) / "memory_trial.json", record)
    return record


def train(config: QLoraConfig, release_dir: Path, out: Path, *, allow_exploratory: bool = False) -> dict:
    """SFT with TRL; per-epoch eval and adapter checkpoint; select by the pre-declared rule."""
    started = time.time()
    out = Path(out)
    release, tok, train_rows, valid_rows = prepare_data(config, release_dir, allow_exploratory=allow_exploratory)
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model = load_quantized_model(config)
    args = SFTConfig(
        output_dir=str(out / "checkpoints"), num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps, learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type, warmup_steps=config.warmup_ratio,  # a float < 1 is a ratio in transformers 5
        max_grad_norm=config.max_grad_norm, optim=config.optim, seed=config.seed, data_seed=config.seed,
        bf16=config.quantization.compute_dtype == "bfloat16",
        fp16=config.quantization.compute_dtype == "float16",
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False}, eval_strategy="epoch", save_strategy="epoch",
        logging_steps=1, max_length=config.max_length, use_liger_kernel=config.use_liger_kernel,
        report_to="none", assistant_only_loss=False, packing=False, load_best_model_at_end=False,
        prediction_loss_only=True)
    trainer = _tail_loss_trainer(SFTTrainer)(model=model, args=args, train_dataset=Dataset.from_list(train_rows),
                         eval_dataset=Dataset.from_list(valid_rows), processing_class=tok.raw,
                         peft_config=_peft_config(config))
    train_started = time.time()
    trainer.train()
    train_seconds = time.time() - train_started
    steps_per_epoch = trainer.state.max_steps / config.epochs
    history = []
    for entry in trainer.state.log_history:
        if "eval_loss" in entry:
            epoch = round(float(entry["epoch"]))
            step = int(entry["step"])
            history.append({"epoch": epoch, "step": step, "eval_loss": float(entry["eval_loss"]),
                            "checkpoint": f"checkpoint-{step}"})
    chosen = select_checkpoint(history)
    selected = out / "selected_adapter"
    source = out / "checkpoints" / chosen["checkpoint"]
    selected.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copy2(source / name, selected / name)
    tok.raw.save_pretrained(selected)
    results = {"validation_history": history, "selected": chosen, "steps_per_epoch": steps_per_epoch,
               "adapter_sha256": sha256_file(selected / "adapter_model.safetensors"),
               "selected_adapter": str(selected), "train_loss": trainer.state.log_history[-1].get("train_loss"),
               "peak_memory_allocated_bytes": peak_gpu_memory_bytes()}
    record = build_run_record(kind="train", config=config, release=release, tokenizer_info=_tokenizer_info(tok, config),
                              hardware=hardware_details(), libraries=library_versions(), results=results,
                              timings={"train": train_seconds, "total": time.time() - started})
    write_json(out / "run_record.json", record)
    return record


def peak_gpu_memory_bytes() -> int | None:
    try:
        import torch

        return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    except ImportError:
        return None


def train_lora() -> None:
    """Moved: use ``python -m clauseguard.cli.train_lora``."""
    raise NotImplementedError("use python -m clauseguard.cli.train_lora")
