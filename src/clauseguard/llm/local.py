"""Local Hugging Face generator for the RCA pipeline (Task 4.10): base model, optionally with a LoRA adapter.

It loads the base model with the same 4-bit settings as training, so the untuned and adapted runs differ
only by the adapter. Decoding is greedy at temperature 0; Qwen thinking mode is disabled.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from clauseguard.llm.clients import GenerationAttempt, GenerationResult, MalformedGenerationOutput
from clauseguard.llm.lora import QLoraConfig, peak_gpu_memory_bytes, sha256_file


def adapter_sha256(adapter_dir: Path) -> str:
    weights = Path(adapter_dir) / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"no adapter_model.safetensors in {adapter_dir}")
    return sha256_file(weights)


class LocalGenerator:
    """Same ``complete`` interface as ``LLMRouter``; ``model``/``tokenizer`` may be injected (tests)."""

    kind = "live"  # the run record only knows live or replay; a local model is a live generator

    def __init__(self, base_model: str, adapter: Path | None = None, *, config: QLoraConfig | None = None,
                 model: Any = None, tokenizer: Any = None) -> None:
        self.config = config or QLoraConfig(base_model=base_model)
        self.base_model = base_model
        self.adapter = Path(adapter) if adapter else None
        self.adapter_sha256 = adapter_sha256(self.adapter) if self.adapter else None
        self.model_id = base_model if not self.adapter_sha256 else f"{base_model}+adapter:{self.adapter_sha256}"
        self._model, self._tokenizer = model, tokenizer
        self._lock = threading.Lock()

    def _load(self) -> None:
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return
            from transformers import AutoTokenizer

            from clauseguard.llm.lora import load_quantized_model

            revision = self.config.revision if self.base_model == self.config.base_model else None
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model, revision=revision)
            model = load_quantized_model(self.config.model_copy(update={"base_model": self.base_model,
                                                                        "revision": revision}))
            if self.adapter:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(self.adapter))
            model.eval()
            self._model = model

    def complete(self, messages: list[dict[str, str]], *, task: str = "rca", models: tuple[str, ...] | None = None,
                 failover: bool = True, temperature: float | None = None, top_p: float | None = None,
                 max_tokens: int | None = None, seed: int | None = None, timeout: float | None = None,
                 **_: Any) -> GenerationResult:
        self._load()
        tok, model = self._tokenizer, self._model
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = tok(text, return_tensors="pt", add_special_tokens=False)
        if hasattr(model, "device"):
            encoded = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        sample = bool(temperature)
        kwargs: dict[str, Any] = {"max_new_tokens": max_tokens or 16384, "do_sample": sample}
        stops = _stop_token_ids(tok)
        if stops:  # the repo ships no generation_config, so eos would be <|endoftext|> only, not the turn end
            kwargs["eos_token_id"] = stops
            kwargs["pad_token_id"] = getattr(tok, "pad_token_id", None) or stops[-1]
        if sample:
            kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p
            if seed is not None:
                try:
                    import torch

                    torch.manual_seed(seed)
                except ImportError:
                    pass
        started = time.perf_counter()
        output = model.generate(**encoded, **kwargs)
        seconds = time.perf_counter() - started
        new_ids = output[0][prompt_tokens:]
        content = tok.decode(new_ids, skip_special_tokens=True)
        attempt = GenerationAttempt(model=self.model_id, outcome="ok" if content.strip() else "empty_output",
                                    seconds=seconds)
        if not content.strip():
            error = MalformedGenerationOutput()
            error.attempts = (attempt,)
            raise error
        completion_tokens = len(new_ids)
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                 "total_tokens": prompt_tokens + completion_tokens, "latency_seconds": seconds,
                 "peak_gpu_memory_bytes": peak_gpu_memory_bytes(), "adapter_sha256": self.adapter_sha256}
        return GenerationResult(content=content, requested_model=self.model_id, served_model=self.model_id,
                                attempts=[attempt], usage=usage)


def _stop_token_ids(tok: Any) -> list[int]:
    """Turn end (<|im_end|>, the token training supervises) first, then <|endoftext|>."""
    convert = getattr(tok, "convert_tokens_to_ids", None)
    if convert is None:
        return []
    unknown = getattr(tok, "unk_token_id", None)
    ids = [convert(t) for t in ("<|im_end|>", "<|endoftext|>")]
    return list(dict.fromkeys(i for i in ids if isinstance(i, int) and i >= 0 and i != unknown))
