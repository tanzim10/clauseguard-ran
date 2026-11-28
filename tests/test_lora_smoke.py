"""CPU smoke test for Task 4.10 training plumbing: a tiny random Qwen3.5 through the real `train()`.

Skipped unless transformers, peft, trl and datasets are installed and the Qwen3.5-4B tokenizer is in the
local Hugging Face cache (no network). It checks plumbing only: masks, trainer, LoRA targets, per-epoch
checkpoints, selection and the run record. It says nothing about quality or GPU memory.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

for module in ("transformers", "peft", "trl", "datasets"):
    pytest.importorskip(module)

from clauseguard.llm import lora

MODEL = "Qwen/Qwen3.5-4B"


def _tiny_model(config: lora.QLoraConfig):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    full = AutoConfig.from_pretrained(config.base_model, local_files_only=True)
    text = getattr(full, "text_config", full)
    text.update({"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
                 "layer_types": ["linear_attention"] * 3 + ["full_attention"], "num_attention_heads": 4,
                 "num_key_value_heads": 2, "head_dim": 16, "linear_num_key_heads": 2,
                 "linear_num_value_heads": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16})
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(text, dtype=torch.float32)


def _row(n: int) -> str:
    answer = {"status": "answered",
              "final_fault_categories": [{"category": "Jamming", "claim_ids": ["C1", "C2", "C3"]}],
              "claims": [
                  {"claim_id": "C1", "text": f"UL_SNR mean is {n}.", "claim_type": "observation",
                   "evidence_ids": ["K1"]},
                  {"claim_id": "C2", "text": "Uplink SNR falls under interference.", "claim_type": "mechanism",
                   "evidence_ids": ["S1"]},
                  {"claim_id": "C3", "text": f"A UL_SNR mean of {n} is consistent with interference.",
                   "claim_type": "diagnostic_inference", "evidence_ids": ["K1", "S1"]}],
              "uncertainty_reasons": ["units unknown"], "abstention_reason": None}
    return json.dumps({"messages": [
        {"role": "system", "content": "Return an RcaModelOutput JSON object."},
        {"role": "user", "content": f"Evidence K1: UL_SNR mean {n}. Explain the observed KPI changes."},
        {"role": "assistant", "content": json.dumps(answer)}]})


def _release(tmp: Path) -> Path:
    release = tmp / "release"
    release.mkdir()
    outputs = {}
    for name, count in (("train.jsonl", 4), ("validation.jsonl", 2)):
        text = "".join(_row(i) + "\n" for i in range(count))
        (release / name).write_text(text, encoding="utf-8")
        outputs[name] = hashlib.sha256(text.encode()).hexdigest()
    (release / "manifest.json").write_text(json.dumps({"release_status": "sample_approved", "outputs": outputs,
                                                       "candidate_release_id": "smoke"}), encoding="utf-8")
    return release


def test_tiny_qwen35_trains_saves_checkpoints_and_selects(tmp_path, monkeypatch):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except OSError:
        pytest.skip("Qwen3.5-4B tokenizer is not in the local Hugging Face cache")
    config = lora.QLoraConfig().model_copy(update={
        "epochs": 2, "gradient_accumulation_steps": 1, "optim": "adamw_torch", "use_liger_kernel": False,
        "gradient_checkpointing": False, "max_length": 2048,
        "quantization": lora.QuantConfig().model_copy(update={"compute_dtype": "float32"})})
    monkeypatch.setattr(lora, "load_quantized_model", _tiny_model)

    record = lora.train(config, _release(tmp_path), tmp_path / "out")

    results = record["results"]
    assert [h["epoch"] for h in results["validation_history"]] == [1, 2]
    assert results["selected"]["checkpoint"] in {h["checkpoint"] for h in results["validation_history"]}
    selected = Path(results["selected_adapter"])
    assert (selected / "adapter_model.safetensors").is_file() and (selected / "adapter_config.json").is_file()
    adapter = json.loads((selected / "adapter_config.json").read_text())
    targets = adapter["target_modules"]
    assert isinstance(targets, str) and "in_proj_qkv" in targets and "q_proj" in targets  # both layer types
    assert record["dataset"]["release_status"] == "sample_approved" and record["selection_rule"]
    assert (tmp_path / "out" / "run_record.json").is_file()


def test_local_generator_loads_the_trained_adapter_and_generates(tmp_path, monkeypatch):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except OSError:
        pytest.skip("Qwen3.5-4B tokenizer is not in the local Hugging Face cache")
    from clauseguard.llm.local import LocalGenerator

    config = lora.QLoraConfig().model_copy(update={
        "epochs": 1, "gradient_accumulation_steps": 1, "optim": "adamw_torch", "use_liger_kernel": False,
        "gradient_checkpointing": False, "max_length": 2048,
        "quantization": lora.QuantConfig().model_copy(update={"compute_dtype": "float32"})})
    monkeypatch.setattr(lora, "load_quantized_model", _tiny_model)
    record = lora.train(config, _release(tmp_path), tmp_path / "out")
    adapter = Path(record["results"]["selected_adapter"])

    generator = LocalGenerator(MODEL, adapter, config=config)
    messages = json.loads(_row(7))["messages"][:2]
    expected = f"{MODEL}+adapter:{record['results']['adapter_sha256']}"
    from clauseguard.llm.clients import MalformedGenerationOutput

    try:  # a random tiny model may emit only special tokens; generation itself must still run
        result = generator.complete(messages, max_tokens=8, temperature=0.0)
    except MalformedGenerationOutput as empty:
        assert empty.attempts[0].model == expected and empty.attempts[0].outcome == "empty_output"
    else:
        assert result.requested_model == expected and 0 < result.usage["completion_tokens"] <= 8


def test_tail_loss_equals_the_full_logits_loss_on_the_same_labels():
    import torch

    config = lora.QLoraConfig()
    model = _tiny_model(config)
    torch.manual_seed(1)
    ids = torch.randint(0, 1000, (1, 40))
    labels = torch.full_like(ids, -100)
    labels[0, 30:] = ids[0, 30:]  # the supervised span is the tail, as in every training row
    full = model(input_ids=ids, labels=labels).loss
    tail = lora.tail_loss(model, ids, labels)
    assert torch.allclose(full, tail, atol=1e-5)
    with pytest.raises(ValueError, match="one example"):
        lora.tail_loss(model, ids.repeat(2, 1), labels.repeat(2, 1))


def test_trainer_divides_the_per_example_loss_by_the_accumulation_steps():
    from trl import SFTTrainer

    trainer_class = lora._tail_loss_trainer(SFTTrainer)
    assert "loss_is_scaled_for_ga" in trainer_class.__init__.__code__.co_names
