"""Task 4.9: cited SLM training release — canonical records, exports, exclusions, Qwen check."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from clauseguard.api.schemas import RcaModelOutput
from clauseguard.cli import build_training_release as cli
from clauseguard.rca.candidates import CandidateReference
from clauseguard.rca.training_inputs import SelectedEvent, TrainingSelection
from clauseguard.rca.training_release import (
    HumanReview,
    MachineAcceptance,
    assemble_release,
    check_row,
    latest_complete,
)

ABSTAIN = RcaModelOutput(status="abstained", final_fault_categories=[], claims=[],
                         abstention_reason="insufficient_mechanism")


class FakeTokenizer:
    """Qwen-style template; one token per special tag, word or whitespace run."""

    def __init__(self, think: bool = True, closing: bool = True) -> None:
        self.think, self.closing = think, closing

    def render(self, messages):
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages[:-1]]
        last = messages[-1]
        block = "<think>\n\n</think>\n\n" if self.think else ""
        end = "<|im_end|>\n" if self.closing else ""
        return "".join(parts) + f"<|im_start|>assistant\n{block}{last['content']}{end}"

    def token_spans(self, text):
        return [m.span() for m in re.finditer(r"<\|[^|]+\|>|\S+|\s+", text)]


def _event(n: int, *, split="train", group=None, stratum="fault", source="TelecomTS") -> SelectedEvent:
    return SelectedEvent(candidate_id=f"c{n}", event_id=f"e{n}", source_dataset=source,
                         group_id=group or f"g{n}", split=split, stratum=stratum,
                         causes=["Jamming"] if stratum == "fault" else [], pilot=True, packet_sha256="a" * 64)


def _reference(n: int, stratum="fault") -> CandidateReference:
    return CandidateReference(candidate_id=f"c{n}", taxonomy_id="tax", causes=["Jamming"] if stratum == "fault" else [],
                              label_status="labelled_fault" if stratum == "fault" else "labelled_normal")


def _record(n: int, *, status="complete", parsed=ABSTAIN, user=None, created="2026-01-01", run=None):
    return SimpleNamespace(
        case_id=f"c{n}", run_id=run or f"r{n}", created_at=created, parsed_output=parsed,
        response=SimpleNamespace(execution_status=status),
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": user or f"user {n}"}],
        requested_model="m", served_model="m", prompt_version="p", prompt_sha256="b" * 64, config_sha256="c" * 64,
        source_packet_sha256="d" * 64, projected_packet_sha256="e" * 64, retrieval_manifest_sha256="f" * 64)


def _accept(*ids, rejected=()):
    cases = {f"c{n}": {"run_id": f"r{n}", "machine_status": "accepted", "reasons": [], "judge_model": "j"}
             for n in ids}
    cases |= {f"c{n}": {"run_id": f"r{n}", "machine_status": "rejected", "reasons": ["bad claim"]}
              for n in rejected}
    return MachineAcceptance.model_validate({"version": "v", "cases": cases})


def _release(events, records, acceptance, human=None, tokenizer=None, max_tokens=1000):
    selection = TrainingSelection(candidate_release_id="rel", seed=1, normal_share=0.1, pilot_size=1, events=events)
    refs = {e.candidate_id: _reference(int(e.candidate_id[1:]), e.stratum) for e in events}
    return assemble_release(selection=selection, references=refs, records=records, acceptance=acceptance,
                            human_review=human, tokenizer=tokenizer, tokenizer_name="fake" if tokenizer else None,
                            max_tokens=max_tokens, created_on="2026-11-24", input_hashes={"runs": {}})


def _lines(files, name):
    return [json.loads(line) for line in files[name].splitlines()]


def test_export_is_chat_only_and_references_stay_out():
    files = _release([_event(1), _event(2, split="validation", stratum="normal")], [_record(1), _record(2)],
                     _accept(1, 2))
    train, validation = _lines(files, "train.jsonl"), _lines(files, "validation.jsonl")
    assert len(train) == len(validation) == 1
    assert set(train[0]) == {"messages"}
    assert [m["role"] for m in train[0]["messages"]] == ["system", "user", "assistant"]
    assert RcaModelOutput.model_validate_json(train[0]["messages"][2]["content"]) == ABSTAIN
    assert "Jamming" not in files["train.jsonl"] and "labelled" not in files["train.jsonl"]
    canonical = _lines(files, "canonical.jsonl")[0]
    assert set(canonical) >= {"identity", "input", "target", "provenance", "references", "review"}
    assert canonical["references"]["causes"] == ["Jamming"]
    assert canonical["provenance"]["retrieval_manifest_sha256"] == "f" * 64
    index = _lines(files, "index.jsonl")
    assert [(i["file"], i["line"], i["candidate_id"]) for i in index] == [
        ("train.jsonl", 1, "c1"), ("validation.jsonl", 1, "c2")]
    manifest = json.loads(files["manifest.json"])
    assert manifest["release_status"] == "pending_human_review"
    assert manifest["outputs"]["train.jsonl"] and manifest["counts"]["exported"] == 2


def test_exclusions_carry_reasons():
    events = [_event(n) for n in range(1, 7)]
    records = [_record(1), _record(2), _record(3, parsed=None), _record(4, status="failed"), _record(6)]
    acceptance = _accept(1, 3, 6, rejected=[2])
    acceptance.cases["c6"].run_id = "stale"
    human = HumanReview.model_validate({"cases": {"c1": {"status": "rejected", "reviewer": "x",
                                                         "reviewed_on": "2026-11-24"}}})
    files = _release(events, records, acceptance, human)
    manifest = json.loads(files["manifest.json"])
    reasons = {e["candidate_id"]: " ".join(e["reasons"]) for e in manifest["exclusions"]}
    assert "human_rejected" in reasons["c1"]
    assert "machine_rejected: bad claim" in reasons["c2"]
    assert "parse_failure" in reasons["c3"]
    assert "no_complete_run" in reasons["c4"] and "no_complete_run" in reasons["c5"]
    assert "acceptance_run_mismatch" in reasons["c6"]
    assert files["train.jsonl"] == "" and manifest["release_status"] == "exploratory"


def test_unlisted_human_status_is_pending_and_still_exported():
    files = _release([_event(1)], [_record(1)], _accept(1),
                     HumanReview.model_validate({"cases": {}}))
    assert _lines(files, "canonical.jsonl")[0]["review"]["human_status"] == "pending"
    assert len(_lines(files, "train.jsonl")) == 1


def test_duplicate_user_message_is_excluded():
    files = _release([_event(1), _event(2)], [_record(1, user="same"), _record(2, user="same")], _accept(1, 2))
    manifest = json.loads(files["manifest.json"])
    assert len(_lines(files, "train.jsonl")) == 1
    assert "duplicate_user_message" in manifest["exclusions"][0]["reasons"][0]
    assert manifest["exclusions"][0]["candidate_id"] == "c2"


def test_group_in_both_splits_is_refused():
    events = [_event(1, group="g"), _event(2, split="validation", group="g")]
    with pytest.raises(ValueError, match="both splits"):
        _release(events, [_record(1), _record(2)], _accept(1, 2))


def test_latest_complete_prefers_newest_complete_record():
    old, new = _record(1, created="2026-01-01", run="a"), _record(1, created="2026-02-01", run="b")
    failed = _record(1, status="failed", created="2026-03-01", run="c")
    assert latest_complete([old, failed, new])["c1"].run_id == "b"


def test_deterministic_bytes():
    args = ([_event(2), _event(1)], [_record(2), _record(1)], _accept(1, 2))
    assert _release(*args, tokenizer=FakeTokenizer()) == _release(*args, tokenizer=FakeTokenizer())


def test_qwen_check_reports_tokens_and_rejects_over_limit():
    events = [_event(1), _event(2)]
    records = [_record(1), _record(2, user="word " * 400)]
    files = _release(events, records, _accept(1, 2), tokenizer=FakeTokenizer(), max_tokens=500)
    qwen = json.loads(files["qwen_check.json"])
    assert qwen["summary"]["rows"] == 2 and qwen["summary"]["rejected"] == 1
    assert qwen["summary"]["think_block_present"] == 2 and qwen["summary"]["mask_aligned"] == 2
    manifest = json.loads(files["manifest.json"])
    assert "over_token_limit" in manifest["exclusions"][0]["reasons"][0]
    assert len(_lines(files, "train.jsonl")) == 1


def test_check_row_flags_missing_end_and_handles_no_think_block():
    content = ABSTAIN.model_dump_json()
    messages = [SimpleNamespace(role="system", content="s", model_dump=lambda: {"role": "system", "content": "s"}),
                SimpleNamespace(role="user", content="u", model_dump=lambda: {"role": "user", "content": "u"}),
                SimpleNamespace(role="assistant", content=content,
                                model_dump=lambda: {"role": "assistant", "content": content})]
    good = check_row("r", "c", messages, FakeTokenizer(think=False), 1000)
    assert not good.problems and not good.think_block_present and good.assistant_tokens > 0
    bad = check_row("r", "c", messages, FakeTokenizer(closing=False), 1000)
    assert any("<|im_end|>" in p for p in bad.problems)


def test_without_tokenizer_check_is_marked_skipped():
    files = _release([_event(1)], [_record(1)], _accept(1))
    assert json.loads(files["qwen_check.json"])["status"] == "skipped"


def test_cli_refuses_existing_output_dir(tmp_path, capsys):
    argv = ["--runs", str(tmp_path), "--selection", "s", "--references", "r", "--acceptance", "a",
            "--no-tokenizer", "--created-on", "2026-11-24", "--output-dir", str(tmp_path)]
    assert cli.main(argv) == 1
    assert "output exists" in capsys.readouterr().err


def test_an_approved_sample_with_no_rejection_approves_the_release():
    decision = {"status": "approved", "reviewer": "lead", "reviewed_on": "2026-11-25", "notes": "ok"}
    events, records = [_event(1), _event(2)], [_record(1), _record(2)]
    sampled = _release(events, records, _accept(1, 2), HumanReview.model_validate({"cases": {"c1": decision}}))
    assert json.loads(sampled["manifest.json"])["release_status"] == "sample_approved"
    events, records = [*events, _event(3)], [*records, _record(3)]
    rejected = HumanReview.model_validate({"cases": {"c1": decision, "c2": decision | {"status": "rejected"}}})
    manifest = json.loads(_release(events, records, _accept(1, 2, 3), rejected)["manifest.json"])
    assert manifest["release_status"] == "exploratory"  # a rejected sample row blocks release approval
