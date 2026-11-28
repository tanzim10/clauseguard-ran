"""Task 4.10: LocalGenerator with an injected fake model and tokenizer, and run_rca wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

from clauseguard.cli import run_rca
from clauseguard.llm.clients import GenerationResult, GenerationUnavailable
from clauseguard.llm.local import LocalGenerator

FIXTURES = Path(__file__).parent / "fixtures" / "rca_pipeline"


class Ids(list):
    @property
    def shape(self):
        return (1, len(self))


class FakeTokenizer:
    def __init__(self):
        self.template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return "|".join(m["content"] for m in messages)

    def __call__(self, text, **kwargs):
        return {"input_ids": Ids([1, 2, 3])}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(96 + i) for i in ids)

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 248046, "<|endoftext|>": 248044}[token]


class FakeModel:
    def __init__(self, reply=(1, 2)):
        self.reply, self.kwargs = reply, None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return [[1, 2, 3, *self.reply]]


def _adapter(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    return tmp_path


def test_untuned_generator_reports_base_model_usage_and_disables_thinking():
    tok, model = FakeTokenizer(), FakeModel()
    result = LocalGenerator("Qwen/Qwen3.5-4B", model=model, tokenizer=tok).complete(
        [{"role": "user", "content": "hi"}], task="rca", models=("Qwen/Qwen3.5-4B",), failover=False,
        temperature=0.0, top_p=1.0, max_tokens=50, seed=1, timeout=5.0)
    assert isinstance(result, GenerationResult) and result.content == "ab"
    assert result.requested_model == result.served_model == "Qwen/Qwen3.5-4B"
    assert tok.template_kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    assert model.kwargs["do_sample"] is False and model.kwargs["max_new_tokens"] == 50
    assert model.kwargs["eos_token_id"] == [248046, 248044]  # stop at the turn end training supervises
    assert result.usage["prompt_tokens"] == 3 and result.usage["completion_tokens"] == 2
    assert result.usage["latency_seconds"] >= 0 and "peak_gpu_memory_bytes" in result.usage
    assert result.attempts[0].outcome == "ok" and LocalGenerator.kind == "live"


def test_adapter_hash_is_part_of_the_model_id(tmp_path):
    import hashlib

    gen = LocalGenerator("Qwen/Qwen3.5-4B", _adapter(tmp_path), model=FakeModel(), tokenizer=FakeTokenizer())
    digest = hashlib.sha256(b"weights").hexdigest()
    assert gen.model_id == f"Qwen/Qwen3.5-4B+adapter:{digest}"
    out = gen.complete([{"role": "user", "content": "x"}])
    assert out.requested_model == out.served_model == gen.model_id and out.usage["adapter_sha256"] == digest


def test_missing_adapter_weights_fail_early(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalGenerator("Qwen/Qwen3.5-4B", tmp_path)


def test_empty_generation_is_a_malformed_output():
    from clauseguard.llm.clients import MalformedGenerationOutput

    gen = LocalGenerator("m", model=FakeModel(reply=()), tokenizer=FakeTokenizer())
    with pytest.raises(MalformedGenerationOutput):
        gen.complete([{"role": "user", "content": "x"}])


class StubLocal:
    """Stands in for LocalGenerator; never loads a model."""

    kind = "live"
    instances: ClassVar[list] = []

    def __init__(self, base_model, adapter=None):
        self.base_model, self.adapter = base_model, adapter
        self.model_id = base_model + (f"+adapter:{'a' * 64}" if adapter else "")
        StubLocal.instances.append(self)

    def complete(self, messages, **kwargs):
        raise GenerationUnavailable("stub")


def _args(output, *extra):
    return ["--manifest", str(FIXTURES / "retrieval-manifest.json"), "--packets", str(FIXTURES / "packets.jsonl"),
            "--audits", str(FIXTURES / "audits.jsonl"), "--run-dir", str(FIXTURES / "snapshots"),
            "--output", str(output), "--case", "rca_fx_001", *extra]


@pytest.fixture(autouse=True)
def _patched(monkeypatch, tmp_path):
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    monkeypatch.chdir(tmp_path)
    StubLocal.instances = []
    monkeypatch.setattr("clauseguard.llm.local.LocalGenerator", StubLocal)


def test_run_rca_uses_local_generator_for_the_untuned_model(tmp_path):
    out = tmp_path / "base"
    assert run_rca.main(_args(out, "--local-model", "Qwen/Qwen3.5-4B", "--run-kind", "scored")) == 0
    index = json.loads((out / "index.json").read_text())
    assert StubLocal.instances[0].adapter is None
    assert index["config"]["configuration"] == "base_llm_full" and index["config"]["model_id"] == "Qwen/Qwen3.5-4B"
    assert index["runs"][0]["generator_kind"] == "live" and index["runs"][0]["generator_called"]


def test_run_rca_adapter_selects_adapted_configuration_and_model_id(tmp_path):
    out = tmp_path / "adapted"
    assert run_rca.main(_args(out, "--local-model", "Qwen/Qwen3.5-4B", "--adapter", str(tmp_path),
                              "--run-kind", "scored")) == 0
    index = json.loads((out / "index.json").read_text())
    assert index["config"]["configuration"] == "adapted_slm_full"
    assert index["config"]["model_id"] == "Qwen/Qwen3.5-4B+adapter:" + "a" * 64


def test_run_rca_rejects_inconsistent_local_options(tmp_path):
    with pytest.raises(SystemExit):
        run_rca.main(_args(tmp_path / "x", "--adapter", str(tmp_path)))
    with pytest.raises(SystemExit):
        run_rca.main(_args(tmp_path / "y", "--local-model", "m", "--model", "other"))
    with pytest.raises(SystemExit):
        run_rca.main(_args(tmp_path / "z", "--local-model", "m", "--configuration", "adapted_slm_full"))


def test_real_qwen_tokenizer_stops_on_the_turn_end():
    import os

    pytest = __import__("pytest")
    pytest.importorskip("transformers")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    from clauseguard.llm.local import _stop_token_ids

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3.5-4B tokenizer is not in the local Hugging Face cache")
    assert _stop_token_ids(tok) == [248046, 248044]
