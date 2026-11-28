"""Task 4.9 cited SLM training release: canonical records, train/validation chat files, Qwen check.

References (causes, label status) stay in the canonical record and never enter the chat messages.
A row is exported only when the machine judge accepted it and no human rejected it; every other case
is listed in the manifest with its reasons.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol

from pydantic import Field

from clauseguard.api.schemas import RcaModelOutput
from clauseguard.rca.candidates import CandidateReference, StrictModel
from clauseguard.rca.pipeline import RcaRunRecord
from clauseguard.rca.training_inputs import SelectedEvent, TrainingSelection

TRAINING_RELEASE_VERSION = "training-release-v1"
CANONICAL_VERSION = "training-canonical-v1"
QWEN_CHECK_VERSION = "qwen-check-v1"
DEFAULT_MAX_TOKENS = 32768
DEFAULT_TOKENIZER = "Qwen/Qwen3.5-4B"
ASSISTANT_MARKER = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"
EMPTY_THINK = "<think>\n\n</think>\n\n"

MachineStatus = Literal["accepted", "rejected", "missing"]
HumanStatus = Literal["approved", "rejected", "pending"]


class CaseAcceptance(StrictModel):
    run_id: str
    machine_status: Literal["accepted", "rejected"]
    reasons: list[str] = Field(default_factory=list)
    judge_model: str | None = None


class MachineAcceptance(StrictModel):
    version: str
    cases: dict[str, CaseAcceptance]
    rule: str | None = None
    judge: dict[str, Any] | None = None


class HumanDecision(StrictModel):
    status: Literal["approved", "rejected"]
    reviewer: str
    reviewed_on: str
    notes: str = ""


class HumanReview(StrictModel):
    cases: dict[str, HumanDecision] = Field(default_factory=dict)


class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Identity(StrictModel):
    record_id: str
    candidate_id: str
    event_id: str
    group_id: str
    source_dataset: str
    split: Literal["train", "validation"]


class Provenance(StrictModel):
    requested_model: str | None
    served_model: str | None
    run_id: str
    prompt_version: str
    prompt_sha256: str | None
    config_sha256: str
    source_packet_sha256: str
    projected_packet_sha256: str | None
    retrieval_manifest_sha256: str | None


class References(StrictModel):
    causes: list[str]
    label_status: str
    taxonomy_id: str
    stratum: Literal["fault", "normal"]


class Review(StrictModel):
    machine_status: MachineStatus
    machine_reasons: list[str] = Field(default_factory=list)
    judge_model: str | None = None
    human_status: HumanStatus
    reviewer: str | None = None
    reviewed_on: str | None = None


class CanonicalRecord(StrictModel):
    version: Literal["training-canonical-v1"] = CANONICAL_VERSION
    identity: Identity
    input: list[Message]
    target: Message | None
    provenance: Provenance
    references: References
    review: Review
    exported: bool = False
    exclusion_reasons: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- Qwen check

class ChatTokenizer(Protocol):
    def render(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Chat-template text for the whole conversation."""

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """Character span of every token of ``text``."""


class HFChatTokenizer:
    """Adapter over a local Hugging Face fast tokenizer (no network)."""

    def __init__(self, name: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is not installed; install it or pass --no-tokenizer") from exc
        self._tok = AutoTokenizer.from_pretrained(name, local_files_only=True)

    def render(self, messages: Sequence[Mapping[str, str]]) -> str:
        return self._tok.apply_chat_template(list(messages), tokenize=False, enable_thinking=False)

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        encoded = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [(int(a), int(b)) for a, b in encoded["offset_mapping"]]


class RowCheck(StrictModel):
    record_id: str
    candidate_id: str
    total_tokens: int
    assistant_tokens: int
    think_block_present: bool
    ends_with_im_end: bool
    assistant_nonempty: bool
    parses_as_rca_output: bool
    content_matches_target: bool
    mask_aligned: bool
    problems: list[str] = Field(default_factory=list)


class AssistantSpan(NamedTuple):
    """Character offsets of the final assistant turn in rendered chat text."""

    found: bool
    start: int  # first character of the assistant content (after marker and empty think block)
    end: int  # position of the closing <|im_end|>
    mask_end: int  # end of the supervised text: content plus its <|im_end|>
    think: bool
    ends: bool


def locate_assistant(text: str) -> AssistantSpan:
    """Where the supervised assistant text sits in ``text``; shared by the row check and training masks."""
    marker = text.rfind(ASSISTANT_MARKER)
    ends = text.rstrip("\n").endswith(IM_END)
    after = marker + len(ASSISTANT_MARKER) if marker >= 0 else 0
    think = marker >= 0 and text.startswith(EMPTY_THINK, after)
    start = after + len(EMPTY_THINK) if think else after
    end = text.rfind(IM_END)
    mask_end = end + len(IM_END) if ends and end >= start else start
    return AssistantSpan(marker >= 0, start, end, mask_end, think, ends)


def check_row(record_id: str, candidate_id: str, messages: Sequence[Message], tokenizer: ChatTokenizer,
              max_tokens: int) -> RowCheck:
    """Render one row and check template, assistant span, loss mask and length."""
    text = tokenizer.render([m.model_dump() for m in messages])
    spans = tokenizer.token_spans(text)
    problems: list[str] = []
    found, start, end, mask_end, think, ends = locate_assistant(text)
    if not found:
        problems.append("no assistant turn in rendered text")
    content = text[start:end] if found and end >= start else ""
    parses = True
    try:
        RcaModelOutput.model_validate_json(content)
    except ValueError:
        parses = False
    matches = content == messages[-1].content
    # Supervised characters are the assistant content plus its closing <|im_end|>; no token may straddle that edge.
    aligned = all(not (a < start < b or a < mask_end < b) for a, b in spans)
    assistant_tokens = sum(1 for a, b in spans if a >= start and b <= mask_end)
    if not content.strip():
        problems.append("assistant content is empty")
    if not ends:
        problems.append("rendered text does not end the assistant turn with <|im_end|>")
    if not parses:
        problems.append("assistant content does not parse as RcaModelOutput")
    if not matches:
        problems.append("rendered assistant content differs from the target")
    if not aligned:
        problems.append("a token straddles the assistant loss-mask boundary")
    if assistant_tokens == 0:
        problems.append("no assistant tokens under the loss mask")
    if len(spans) > max_tokens:
        problems.append(f"over_token_limit: {len(spans)} > {max_tokens}")
    return RowCheck(record_id=record_id, candidate_id=candidate_id, total_tokens=len(spans),
                    assistant_tokens=assistant_tokens, think_block_present=think, ends_with_im_end=ends,
                    assistant_nonempty=bool(content.strip()), parses_as_rca_output=parses,
                    content_matches_target=matches, mask_aligned=aligned, problems=problems)


def qwen_summary(checks: Sequence[RowCheck], tokenizer_name: str, max_tokens: int) -> dict:
    totals = [c.total_tokens for c in checks]
    return {
        "version": QWEN_CHECK_VERSION, "status": "checked", "tokenizer": tokenizer_name, "max_tokens": max_tokens,
        "template_args": {"enable_thinking": False},
        "summary": {
            "rows": len(checks), "rejected": sum(1 for c in checks if c.problems),
            "tokens_min": min(totals, default=0), "tokens_median": statistics.median(totals) if totals else 0,
            "tokens_max": max(totals, default=0),
            "think_block_present": sum(1 for c in checks if c.think_block_present),
            "mask_aligned": sum(1 for c in checks if c.mask_aligned),
        },
        "rows": [c.model_dump() for c in checks],
    }


# ------------------------------------------------------------------- loading

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def load_run_records(directories: Iterable[Path]) -> tuple[list[RcaRunRecord], dict[str, dict]]:
    """Every ``runs/*.json`` record plus a per-directory file count and combined digest."""
    records: list[RcaRunRecord] = []
    inputs: dict[str, dict] = {}
    for directory in directories:
        digest = hashlib.sha256()
        files = sorted((directory / "runs").glob("*.json"))
        for path in files:
            data = path.read_bytes()
            digest.update(path.name.encode() + b"\0" + sha256_bytes(data).encode() + b"\n")
            records.append(RcaRunRecord.model_validate_json(data))
        inputs[str(directory)] = {"files": len(files), "sha256": digest.hexdigest()}
    return records, inputs


def latest_complete(records: Iterable[RcaRunRecord]) -> dict[str, RcaRunRecord]:
    """Latest complete run per case; ties on time break on run ID."""
    best: dict[str, RcaRunRecord] = {}
    for record in records:
        if record.response.execution_status != "complete":
            continue
        current = best.get(record.case_id)
        if current is None or (record.created_at, record.run_id) > (current.created_at, current.run_id):
            best[record.case_id] = record
    return best


# ------------------------------------------------------------------ assembly

def _canonical(item: SelectedEvent, record: RcaRunRecord, reference: CandidateReference,
               acceptance: CaseAcceptance | None, human: HumanDecision | None) -> CanonicalRecord:
    messages = [Message.model_validate(m) for m in record.messages or []]
    target = (Message(role="assistant", content=record.parsed_output.model_dump_json())
              if record.parsed_output is not None else None)
    return CanonicalRecord(
        identity=Identity(record_id=sha256_text(f"{item.candidate_id}|{record.run_id}"),
                          candidate_id=item.candidate_id, event_id=item.event_id, group_id=item.group_id,
                          source_dataset=item.source_dataset, split=item.split),
        input=messages, target=target,
        provenance=Provenance(
            requested_model=record.requested_model, served_model=record.served_model, run_id=record.run_id,
            prompt_version=record.prompt_version, prompt_sha256=record.prompt_sha256,
            config_sha256=record.config_sha256, source_packet_sha256=record.source_packet_sha256,
            projected_packet_sha256=record.projected_packet_sha256,
            retrieval_manifest_sha256=record.retrieval_manifest_sha256),
        references=References(causes=list(reference.causes), label_status=reference.label_status,
                              taxonomy_id=reference.taxonomy_id, stratum=item.stratum),
        review=Review(
            machine_status=acceptance.machine_status if acceptance else "missing",
            machine_reasons=list(acceptance.reasons) if acceptance else [],
            judge_model=acceptance.judge_model if acceptance else None,
            human_status=human.status if human else "pending",
            reviewer=human.reviewer if human else None, reviewed_on=human.reviewed_on if human else None))


def _exclusion_reasons(record: RcaRunRecord, canonical: CanonicalRecord,
                       acceptance: CaseAcceptance | None) -> list[str]:
    reasons: list[str] = []
    if canonical.target is None:
        reasons.append("parse_failure: complete run has no parsed output")
    if not canonical.input:
        reasons.append("no_messages: run record holds no prompt messages")
    if acceptance is None:
        reasons.append("machine_not_judged: no acceptance entry")
    elif acceptance.run_id != record.run_id:
        reasons.append(f"acceptance_run_mismatch: judged {acceptance.run_id}, latest complete is {record.run_id}")
    elif acceptance.machine_status == "rejected":
        reasons.extend(f"machine_rejected: {r}" for r in acceptance.reasons or ["no reason given"])
    if canonical.review.human_status == "rejected":
        reasons.append("human_rejected")
    return reasons


def assert_no_group_leakage(events: Sequence[SelectedEvent]) -> None:
    splits: dict[str, set[str]] = {}
    for event in events:
        splits.setdefault(event.group_id, set()).add(event.split)
    leaked = sorted(group for group, seen in splits.items() if len(seen) > 1)
    if leaked:
        raise ValueError(f"group_id appears in both splits: {leaked[:5]}")


def _chat_line(record: CanonicalRecord) -> str:
    assert record.target is not None
    messages = [{"role": m.role, "content": m.content} for m in [*record.input, record.target]]
    return json.dumps({"messages": messages}, ensure_ascii=False)


def _jsonl(rows: Iterable[str]) -> str:
    return "".join(f"{row}\n" for row in rows)


def _tally(records: Iterable[CanonicalRecord]) -> dict:
    rows = list(records)
    causes = Counter(c for r in rows for c in (r.references.causes or ["normal"]))
    return {
        "by_split": dict(sorted(Counter(r.identity.split for r in rows).items())),
        "by_source": dict(sorted(Counter(r.identity.source_dataset for r in rows).items())),
        "by_stratum": dict(sorted(Counter(r.references.stratum for r in rows).items())),
        "by_cause": dict(sorted(causes.items())),
        "by_machine_status": dict(sorted(Counter(r.review.machine_status for r in rows).items())),
        "by_human_status": dict(sorted(Counter(r.review.human_status for r in rows).items())),
    }


def _release_status(exported: Sequence[CanonicalRecord], human_review: HumanReview | None) -> str:
    """Name the review state of the exported rows. The acceptance rule is unchanged."""
    if human_review is None:
        return "pending_human_review"
    if exported and all(row.review.human_status == "approved" for row in exported):
        return "human_reviewed"
    sampled = (
        exported
        and any(row.review.human_status == "approved" for row in exported)
        and not any(decision.status == "rejected" for decision in human_review.cases.values())
    )
    # Step 9 sampled review: every reviewed sample approved and none rejected approves the release.
    if sampled:
        return "sample_approved"
    return "exploratory"


def assemble_release(*, selection: TrainingSelection, references: Mapping[str, CandidateReference],
                     records: Sequence[RcaRunRecord], acceptance: MachineAcceptance,
                     human_review: HumanReview | None, tokenizer: ChatTokenizer | None, tokenizer_name: str | None,
                     max_tokens: int, created_on: str, input_hashes: Mapping[str, object]) -> dict[str, str]:
    """Output file name to text. Deterministic for the same inputs."""
    events = sorted(selection.events, key=lambda e: e.candidate_id)
    assert_no_group_leakage(events)
    latest = latest_complete(records)
    decisions = human_review.cases if human_review else {}
    canonical: list[CanonicalRecord] = []
    exclusions: list[dict] = []
    for item in events:
        record = latest.get(item.candidate_id)
        if record is None:
            exclusions.append({"candidate_id": item.candidate_id, "run_id": None,
                               "reasons": ["no_complete_run: no complete run record for this case"]})
            continue
        accepted = acceptance.cases.get(item.candidate_id)
        row = _canonical(item, record, references[item.candidate_id], accepted, decisions.get(item.candidate_id))
        row.exclusion_reasons = _exclusion_reasons(record, row, accepted)
        canonical.append(row)

    seen_users: dict[str, str] = {}
    for row in canonical:
        if row.exclusion_reasons:
            continue
        digest = sha256_text(next(m.content for m in row.input if m.role == "user"))
        first = seen_users.setdefault(digest, row.identity.candidate_id)
        if first != row.identity.candidate_id:
            row.exclusion_reasons.append(f"duplicate_user_message: same user message as {first}")

    checks: list[RowCheck] = []
    if tokenizer is not None:
        for row in canonical:
            if row.exclusion_reasons or row.target is None:
                continue
            check = check_row(row.identity.record_id, row.identity.candidate_id, [*row.input, row.target],
                              tokenizer, max_tokens)
            checks.append(check)
            row.exclusion_reasons.extend(f"qwen_check: {p}" for p in check.problems)

    for row in canonical:
        row.exported = not row.exclusion_reasons
        if row.exclusion_reasons:
            exclusions.append({"candidate_id": row.identity.candidate_id, "run_id": row.provenance.run_id,
                               "reasons": row.exclusion_reasons})
    exported = [r for r in canonical if r.exported]
    files: dict[str, str] = {"canonical.jsonl": _jsonl(r.model_dump_json() for r in canonical)}
    index: list[str] = []
    for split in ("train", "validation"):
        rows = [r for r in exported if r.identity.split == split]
        files[f"{split}.jsonl"] = _jsonl(_chat_line(r) for r in rows)
        index.extend(json.dumps({
            "file": f"{split}.jsonl", "line": number, "record_id": r.identity.record_id,
            "candidate_id": r.identity.candidate_id, "run_id": r.provenance.run_id, "split": split,
            "source": r.identity.source_dataset, "group_id": r.identity.group_id, "event_id": r.identity.event_id,
        }) for number, r in enumerate(rows, start=1))
    files["index.jsonl"] = _jsonl(index)
    if tokenizer is None:
        qwen: dict = {"version": QWEN_CHECK_VERSION, "status": "skipped", "reason": "no tokenizer was provided"}
    else:
        qwen = qwen_summary(checks, tokenizer_name or "", max_tokens)
    files["qwen_check.json"] = json.dumps(qwen, indent=2) + "\n"

    status = _release_status(exported, human_review)
    manifest = {
        "training_release_version": TRAINING_RELEASE_VERSION, "canonical_version": CANONICAL_VERSION,
        "selection_version": selection.selection_version, "candidate_release_id": selection.candidate_release_id,
        "acceptance_version": acceptance.version, "acceptance_rule": acceptance.rule,
        "judge": acceptance.judge, "created_on": created_on, "release_status": status,
        "max_tokens": max_tokens, "tokenizer": tokenizer_name,
        "inputs": dict(input_hashes),
        "outputs": {name: sha256_text(text) for name, text in sorted(files.items())},
        "counts": {"selected": len(events), "canonical": len(canonical), "exported": len(exported),
                   "excluded": len(exclusions), "exported_detail": _tally(exported),
                   "canonical_detail": _tally(canonical)},
        "exclusions": exclusions,
    }
    files["manifest.json"] = json.dumps(manifest, indent=2) + "\n"
    return files
