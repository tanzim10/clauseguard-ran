"""Independent model review of golden RCA rows (Task 4.3).

A reviewer model on an OpenAI-compatible endpoint (NVIDIA NIM, Gemini or Groq) judges each RCA row against its measured KPI evidence and the text
of every cited PDF page. Use a model from a different family than the one that drafted the rows.
The verdict is an audit record, not an approval: only a person sets an annotation to `approved`.

One model per run, no failover, so every saved verdict names the model that produced it. Provider
errors and malformed replies are saved as failed reviews instead of being dropped.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pypdf import PdfReader

from clauseguard.datasets.schema import RcaRow
from clauseguard.eval.annotations import Annotation, SourceLocator
from clauseguard.llm.clients import _retry_after_seconds

DEFAULT_MODEL = "openai/gpt-oss-120b"
PASSAGE_CHARS = 4000

TELECOMTS_CONTEXT = (
    "TelecomTS testbed: OpenAirInterface 5G NR SA, one gNB, one cell, band n78, phones at fixed "
    "zones, 10 Hz sampling, 128-sample windows. Synthetic faults are injected into real recordings. "
    "Units per the dataset paper: RSRP dBm, UL_SNR dB, BLER fraction 0-1, PRB_Utilization percent "
    "of the cell's PRBs, UL_NPRB PRBs per TTI, RX_Bytes uplink and TX_Bytes downlink bytes per "
    "100 ms, packet counts per 100 ms, Estimated_UL_Buffer presumably bytes. Baseline = pre-fault "
    "samples of the same window; comparison = fault samples."
)

SYSTEM_PROMPT = (
    "You are an independent, strict reviewer of a golden evaluation dataset for root-cause "
    "analysis (RCA) of 5G RAN faults. Each case was drafted by a different AI model. Judge ONLY "
    "from the measured KPI evidence and the specification page text given; do not fill gaps with "
    "outside knowledge, but point out claims that are wrong or unsupported. The dataset label is a "
    "claim from a synthetic dataset, not ground truth. Return only JSON."
)

RUBRIC = """Return JSON with exactly these keys:
{
 "question_leaks_answer": true|false,
 "question_leak_note": "...",
 "evidence_supports_conclusion": "yes"|"partial"|"no",
 "evidence_note": "...",
 "unsupported_claims_in_conclusion": ["..."],
 "citations": [{"spec_id": "...", "section": "...", "role_claimed": "definition"|"mechanism",
                "passage_supports_explanation": "yes"|"partial"|"no", "role_correct": true|false,
                "note": "..."}],
 "has_adequate_mechanism_citation": true|false,
 "acceptable_conclusions_reasonable": true|false,
 "uncertainty_honest_and_complete": true|false,
 "uncertainty_missing": ["..."],
 "verdict": "valid"|"valid_with_changes"|"invalid",
 "required_changes": ["..."],
 "confidence": 0.0-1.0
}
A definition says what a KPI or measurement is; a mechanism explains how a condition causes an
effect. "question_leaks_answer" is true when the question names or strongly implies the fault
category or the expected conclusion."""


class CitationReview(BaseModel):
    model_config = ConfigDict(extra="ignore")

    spec_id: str
    section: str
    role_claimed: str | None = None
    passage_supports_explanation: Literal["yes", "partial", "no"]
    role_correct: bool
    note: str = ""


class ReviewVerdict(BaseModel):
    """The reviewer's judgement of one RCA row."""

    model_config = ConfigDict(extra="ignore")

    question_leaks_answer: bool
    question_leak_note: str = ""
    evidence_supports_conclusion: Literal["yes", "partial", "no"]
    evidence_note: str = ""
    unsupported_claims_in_conclusion: list[str] = Field(default_factory=list)
    citations: list[CitationReview]
    has_adequate_mechanism_citation: bool
    acceptable_conclusions_reasonable: bool
    uncertainty_honest_and_complete: bool
    uncertainty_missing: list[str] = Field(default_factory=list)
    verdict: Literal["valid", "valid_with_changes", "invalid"]
    required_changes: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


PassageReader = Callable[[SourceLocator], str]


def pdf_passage_reader(manifest: Sequence[dict[str, Any]], raw_dir: Path | str) -> PassageReader:
    """Read the cited page's text from the corpus PDF named in the manifest."""
    files = {row["spec_id"]: Path(raw_dir) / row["filename"] for row in manifest if row.get("spec_id")}
    readers: dict[str, PdfReader] = {}

    def read(locator: SourceLocator) -> str:
        if locator.pdf_page is None:
            return f"(DOCX source; heading: {locator.docx_heading}; text not extracted)"
        if locator.spec_id not in readers:
            readers[locator.spec_id] = PdfReader(files[locator.spec_id])
        text = readers[locator.spec_id].pages[locator.pdf_page - 1].extract_text() or ""
        return re.sub(r"\s+", " ", text).strip()[:PASSAGE_CHARS]

    return read


def build_prompt(
    row: RcaRow, annotation: Annotation, read_passage: PassageReader, context: str
) -> str:
    """Render one row, its evidence and its cited page text as the reviewer's user message."""
    locators = {
        (loc.spec_id, loc.clause): loc for group in annotation.evidence for loc in group.alternatives
    }
    citations = []
    for citation in row.expected_citations:
        locator = locators[(citation.spec_id, citation.section.split()[0])]
        citations.append(
            {
                **citation.model_dump(),
                "version": locator.version,
                "pdf_page": locator.pdf_page,
                "page_text": read_passage(locator),
            }
        )
    case = {
        "id": row.id,
        "split": annotation.membership,
        "inputs": row.inputs,
        "question": row.question,
        "kpi_evidence": [evidence.model_dump() for evidence in row.kpi_evidence],
        "dataset_label_claim": row.source_labels,
        "expected_conclusion": row.expected_conclusion,
        "acceptable_conclusions": row.acceptable_conclusions,
        "uncertainty": row.uncertainty,
        "citations": citations,
        "context": context,
    }
    return "CASE:\n" + json.dumps(case, ensure_ascii=False, indent=1) + "\n\n" + RUBRIC


@dataclass(frozen=True)
class ReviewSettings:
    model: str = DEFAULT_MODEL
    provider: str = "nvidia"
    max_attempts: int = 4
    max_wait_seconds: float = 120.0
    delay_seconds: float = 0.0
    max_tokens: int = 4000


def _request_review(client: Any, prompt: str, settings: ReviewSettings) -> tuple[Any, str]:
    """Call the reviewer, retrying rate limits and transient failures within the wait budget."""
    last_error = "no attempt made"
    for attempt in range(settings.max_attempts):
        try:
            completion = client.chat.completions.create(
                model=settings.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=settings.max_tokens,
                response_format={"type": "json_object"},
            )
            return completion, ""
        except (APITimeoutError, APIConnectionError) as exc:
            last_error = type(exc).__name__
            wait = 5.0 * (attempt + 1)
        except APIStatusError as exc:
            last_error = f"HTTP {exc.status_code}"
            if exc.status_code not in {429, 500, 502, 503, 504}:
                break
            wait = _retry_after_seconds(exc) or 10.0 * (attempt + 1)
        if wait > settings.max_wait_seconds or attempt == settings.max_attempts - 1:
            break
        time.sleep(wait)
    return None, last_error


def review_row(
    client: Any,
    row: RcaRow,
    annotation: Annotation,
    read_passage: PassageReader,
    settings: ReviewSettings,
    context: str = TELECOMTS_CONTEXT,
) -> dict[str, Any]:
    """Review one row and return its audit record (`status` is reviewed or failed)."""
    prompt = build_prompt(row, annotation, read_passage, context)
    record: dict[str, Any] = {
        "id": row.id,
        "provider": settings.provider,
        "model": settings.model,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "row_sha256": hashlib.sha256(row.model_dump_json().encode()).hexdigest(),
    }
    completion, error = _request_review(client, prompt, settings)
    if completion is None:
        return {**record, "status": "failed", "error": error}
    try:
        content = completion.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError):
        content = ""
    record.update(
        served_model=getattr(completion, "model", None),
        usage=getattr(getattr(completion, "usage", None), "model_dump", lambda: None)(),
        raw_content=content,
    )
    try:
        verdict = ReviewVerdict.model_validate_json(content)
    except ValidationError as exc:
        return {**record, "status": "failed", "error": f"malformed reply: {exc.error_count()} errors"}
    return {**record, "status": "reviewed", "review": verdict.model_dump()}


def review_rows(
    client: Any,
    rows: Sequence[RcaRow],
    annotations: dict[str, Annotation],
    read_passage: PassageReader,
    out_dir: Path,
    settings: ReviewSettings,
    *,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Review every row, resuming finished ones with the same model, row and prompt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in enumerate(rows):
        path = out_dir / f"{row.id}.json"
        prompt_sha = hashlib.sha256(
            build_prompt(row, annotations[row.id], read_passage, TELECOMTS_CONTEXT).encode()
        ).hexdigest()
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if (
                previous.get("status") == "reviewed"
                and previous.get("model") == settings.model
                and previous.get("provider") == settings.provider
                and previous.get("prompt_sha256") == prompt_sha
            ):
                records.append(previous)
                log(f"{row.id}: kept ({previous['review']['verdict']})")
                continue
        if index and settings.delay_seconds:
            time.sleep(settings.delay_seconds)
        record = review_row(client, row, annotations[row.id], read_passage, settings)
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        records.append(record)
        outcome = record["review"]["verdict"] if record["status"] == "reviewed" else record["error"]
        log(f"{row.id}: {record['status']} ({outcome})")
    summary = {
        "provider": settings.provider,
        "model": settings.model,
        "rows": len(records),
        "failed": [r["id"] for r in records if r["status"] != "reviewed"],
        "verdicts": dict(
            Counter(r["review"]["verdict"] for r in records if r["status"] == "reviewed")
        ),
        "question_leaks": [
            r["id"] for r in records if r["status"] == "reviewed" and r["review"]["question_leaks_answer"]
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary
