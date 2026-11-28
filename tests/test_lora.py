"""Task 4.10: QLoRA config, release verification, training masks, selection and run record (no GPU, no network)."""

from __future__ import annotations

import hashlib
import json
import re
import sys

import pytest

from clauseguard.api.schemas import RcaModelOutput
from clauseguard.cli import train_lora
from clauseguard.llm import lora
from clauseguard.rca.training_release import Message

ABSTAIN = RcaModelOutput(status="abstained", final_fault_categories=[], claims=[],
                         abstention_reason="insufficient_mechanism").model_dump_json()


class FakeTokenizer:
    def render(self, messages):
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages[:-1]]
        return "".join(parts) + f"<|im_start|>assistant\n<think>\n\n</think>\n\n{messages[-1]['content']}<|im_end|>\n"

    def token_spans(self, text):
        return [m.span() for m in re.finditer(r"<\|[^|]+\|>|\S+|\s+", text)]

    def encode(self, text):
        return list(range(1, len(self.token_spans(text)) + 1))


def _messages(user="u"):
    return [Message(role="system", content="s"), Message(role="user", content=user),
            Message(role="assistant", content=ABSTAIN)]


def _release(tmp_path, status="sample_approved", tamper=False):
    files = {"train.jsonl": json.dumps({"messages": [m.model_dump() for m in _messages()]}) + "\n",
             "validation.jsonl": json.dumps({"messages": [m.model_dump() for m in _messages("v")]}) + "\n"}
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    manifest = {"release_status": status, "tokenizer": "Qwen/Qwen3.5-4B",
                "outputs": {n: hashlib.sha256(t.encode()).hexdigest() for n, t in files.items()}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    if tamper:
        (tmp_path / "train.jsonl").write_text(files["train.jsonl"] + "\n\n")
    return tmp_path


def test_config_defaults_are_the_decided_ones_and_hash_is_stable():
    c = lora.QLoraConfig()
    assert (c.version, c.base_model) == ("qlora-config-v1", "Qwen/Qwen3.5-4B")
    assert c.revision == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert (c.quantization.quant_type, c.quantization.double_quant, c.quantization.compute_dtype) == ("nf4", True, "bfloat16")
    assert (c.lora.r, c.lora.alpha, c.lora.dropout) == (16, 32, 0.05)
    assert (c.per_device_batch_size, c.gradient_accumulation_steps, c.epochs, c.seed) == (1, 4, 3, 42)
    assert (c.optim, c.learning_rate, c.max_grad_norm, c.max_length) == ("paged_adamw_8bit", 2e-4, 0.3, 40960)
    assert not c.use_liger_kernel and c.gradient_checkpointing and c.enable_thinking is False
    assert c.sha256() == lora.QLoraConfig().sha256() != lora.QLoraConfig(seed=1).sha256()
    assert "tie" in c.selection_rule and "every epoch" in c.selection_rule


def test_config_rejects_unknown_fields_and_enabled_thinking():
    with pytest.raises(ValueError):
        lora.QLoraConfig(bogus=1)
    with pytest.raises(ValueError):
        lora.QLoraConfig(enable_thinking=True)


def test_target_regex_matches_language_projections_only():
    rx = re.compile(lora.LORA_TARGET_REGEX)
    for name in ("model.layers.3.self_attn.q_proj", "model.layers.0.linear_attn.in_proj_qkv",
                 "model.layers.0.linear_attn.out_proj", "model.layers.5.mlp.down_proj"):
        assert rx.match(name)
    for name in ("lm_head", "model.visual.blocks.0.mlp.up_proj", "model.layers.0.linear_attn.in_proj_a"):
        assert not rx.match(name)


def test_verify_release_accepts_good_and_refuses_hash_mismatch_and_wrong_status(tmp_path):
    good = lora.verify_release(_release(tmp_path / "a" if (tmp_path / "a").mkdir() is None else tmp_path))
    assert good["release_status"] == "sample_approved" and set(good["files"]) == {"train.jsonl", "validation.jsonl"}
    (tmp_path / "b").mkdir()
    with pytest.raises(lora.ReleaseError, match="does not match"):
        lora.verify_release(_release(tmp_path / "b", tamper=True))
    (tmp_path / "c").mkdir()
    with pytest.raises(lora.ReleaseError, match="release_status"):
        lora.verify_release(_release(tmp_path / "c", status="exploratory"))
    assert lora.verify_release(tmp_path / "c", allow_exploratory=True)["release_status"] == "exploratory"
    (tmp_path / "d").mkdir()
    _release(tmp_path / "d", status="human_reviewed")
    assert lora.verify_release(tmp_path / "d")["release_status"] == "human_reviewed"


def test_labels_cover_only_assistant_content_and_end_marker():
    tok = FakeTokenizer()
    example = lora.encode_example(_messages(), tok, 1000)
    text = tok.render([m.model_dump() for m in _messages()])
    spans = tok.token_spans(text)
    kept = "".join(text[a:b] for (a, b), label in zip(spans, example["labels"], strict=True) if label != -100)
    assert kept == ABSTAIN + "<|im_end|>"
    assert all(label in (-100, tid) for label, tid in zip(example["labels"], example["input_ids"], strict=True))


def test_over_long_rows_are_rejected_not_truncated():
    with pytest.raises(lora.OverLengthError):
        lora.encode_example(_messages(), FakeTokenizer(), 10)


def test_selection_lowest_loss_and_ties_go_to_the_earlier_epoch():
    history = [{"epoch": 1, "eval_loss": 0.5}, {"epoch": 2, "eval_loss": 0.30001}, {"epoch": 3, "eval_loss": 0.30004}]
    assert lora.select_checkpoint(history)["epoch"] == 2  # 0.3000 == 0.3000 at 4 decimals
    assert lora.select_checkpoint([{"epoch": 1, "eval_loss": 0.2}, {"epoch": 2, "eval_loss": 0.3}])["epoch"] == 1
    assert lora.select_checkpoint([{"epoch": 3, "eval_loss": 0.1}, {"epoch": 1, "eval_loss": 0.4}])["epoch"] == 3
    with pytest.raises(ValueError):
        lora.select_checkpoint([])


def test_run_record_has_everything_issue_65_lists():
    config = lora.QLoraConfig()
    record = lora.build_run_record(
        kind="train", config=config, release={"release_status": "sample_approved", "manifest_sha256": "m",
                                              "files": {"train.jsonl": "t"}},
        tokenizer_info={"chat_template_sha256": "x"}, hardware={"device_name": "L4"},
        libraries=lora.library_versions(), results={"selected": {"epoch": 2}}, timings={"total": 1.0})
    for key in ("model", "tokenizer", "dataset", "adapter_settings", "optimizer_settings", "seeds", "libraries",
                "hardware", "selection_rule", "config_sha256", "timings_seconds"):
        assert key in record
    assert record["model"]["revision"] == config.revision and record["seeds"]["seed"] == 42
    assert record["selection_rule"] == lora.SELECTION_RULE and "transformers" in record["libraries"]


def test_importing_the_modules_loads_no_heavy_library():
    import subprocess

    code = ("import sys, clauseguard.llm.lora, clauseguard.llm.local, clauseguard.cli.train_lora;"
            "print(sorted({'torch','transformers','peft','trl','bitsandbytes'} & set(sys.modules)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_check_command_reports_without_a_tokenizer(tmp_path, monkeypatch, capsys):
    release = tmp_path / "r"
    release.mkdir()
    _release(release)

    def no_tokenizer(*a, **k):
        raise RuntimeError("transformers is not installed")

    monkeypatch.setattr(lora, "HFTrainTokenizer", no_tokenizer)
    assert train_lora.main(["check", "--release-dir", str(release), "--out", str(tmp_path / "o")]) == 0
    report = json.loads((tmp_path / "o" / "check_report.json").read_text())
    assert report["tokenizer"]["status"] == "skipped" and report["release"]["release_status"] == "sample_approved"


def test_check_command_with_fake_tokenizer_checks_rows(tmp_path, monkeypatch):
    release = tmp_path / "r"
    release.mkdir()
    _release(release)
    monkeypatch.setattr(lora, "HFTrainTokenizer", lambda *a, **k: FakeTokenizer())
    assert train_lora.main(["check", "--release-dir", str(release), "--out", str(tmp_path / "o")]) == 0
    report = json.loads((tmp_path / "o" / "check_report.json").read_text())
    assert report["passed"] and report["rows"]["train.jsonl"]["mask_aligned"] == 1


def test_train_refuses_unapproved_release(tmp_path, capsys):
    release = tmp_path / "r"
    release.mkdir()
    _release(release, status="exploratory")
    assert train_lora.main(["train", "--release-dir", str(release), "--out", str(tmp_path / "o")]) == 1
    assert "release_status" in capsys.readouterr().err
