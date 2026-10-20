"""Evaluation run artifacts: one new directory per run under `artifacts/eval/` (Week 5).

A run directory holds:

- `rows.jsonl`: one record per golden row, in golden order: the input actually sent, the raw
  ranked passages (scores and provenance included), the answer and citations when captured,
  any error, timings and the row's scores.
- `metrics.json`: aggregated scores with denominators, cohorts and the metric policy.
- `run.json`: mode, status, identities (golden release, corpus, scenario release, inputs,
  policy), code and retrieval/model settings, and row counts with exclusions.
- `comparison.json` (optional): changes against a previous run.

`run.json` is written first with status `running` and rewritten when the run ends, so an
interrupted run keeps its partial rows and is recognisably unfinished. Directories are never
reused, and outputs are never replaced with golden answers.
"""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from clauseguard.datasets.schema import NonEmptyStr

ROWS_FILE = "rows.jsonl"
METRICS_FILE = "metrics.json"
RUN_FILE = "run.json"
COMPARISON_FILE = "comparison.json"

Mode = Literal["retrieval", "query", "replay"]
RunStatus = Literal["running", "complete", "row_failures"]
FINISHED_STATUSES = ("complete", "row_failures")


class ArtifactError(ValueError):
    """A run directory is missing, unfinished or malformed."""


class Passage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    id: str
    score: float
    spec_id: str
    section: str
    source_file: str
    chunk_id: str
    text: str


class RetrievalOutput(BaseModel):
    """The ranking exactly as retrieval returned it, with the depth that was requested."""

    model_config = ConfigDict(extra="forbid")

    requested_top_k: int = Field(ge=1)
    passages: list[Passage]

    @model_validator(mode="after")
    def _check_ranks(self) -> RetrievalOutput:
        if [p.rank for p in self.passages] != list(range(1, len(self.passages) + 1)):
            raise ValueError("passage ranks must be 1..n in order")
        if len(self.passages) > self.requested_top_k:
            raise ValueError("more passages than requested_top_k")
        return self


class AnswerCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_id: str
    section: str
    source_file: str
    chunk_id: str


class AnswerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[AnswerCitation]


class RowError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: NonEmptyStr
    message: str


class RowInput(BaseModel):
    """What was sent for the row; golden answers and annotations never appear here."""

    model_config = ConfigDict(extra="forbid")

    question: NonEmptyStr
    top_k: int = Field(ge=1)


class RowRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: NonEmptyStr
    type: Literal["spec_qa", "rca"]
    split: str | None = None
    families: list[str] = Field(default_factory=list)
    answerable: bool
    input: RowInput
    status: Literal["ok", "failed", "excluded"]
    exclusion_reason: str | None = None
    error: RowError | None = None
    retrieval: RetrievalOutput | None = None
    answer: AnswerOutput | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    scores: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check_status(self) -> RowRecord:
        if self.status == "ok" and self.retrieval is None:
            raise ValueError("ok rows need retrieval output")
        if self.status == "failed" and self.error is None:
            raise ValueError("failed rows need an error")
        if self.status == "excluded" and not self.exclusion_reason:
            raise ValueError("excluded rows need an exclusion_reason")
        if self.status != "excluded" and self.exclusion_reason:
            raise ValueError("only excluded rows may have an exclusion_reason")
        return self


class RunDirectory:
    """Writes one run's artifacts into a directory that did not exist before."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def create(cls, root: Path | str, mode: Mode) -> RunDirectory:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = root / f"{stamp}-{mode}-{secrets.token_hex(3)}"
        path.mkdir()  # raises if it already exists; runs are never overwritten
        (path / ROWS_FILE).touch()
        return cls(path)

    def write_run(self, run: dict[str, Any]) -> None:
        _write_json(self.path / RUN_FILE, run)

    def append_row(self, record: RowRecord) -> None:
        with open(self.path / ROWS_FILE, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        _write_json(self.path / METRICS_FILE, metrics)

    def write_comparison(self, comparison: dict[str, Any]) -> None:
        _write_json(self.path / COMPARISON_FILE, comparison)


def read_run(path: Path | str) -> tuple[dict[str, Any], list[RowRecord]]:
    """Read a finished run's `run.json` and rows, rejecting unfinished or inconsistent runs."""
    path = Path(path)
    try:
        run = json.loads((path / RUN_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactError(f"{path}: no {RUN_FILE}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path / RUN_FILE}: invalid JSON ({exc.msg})") from exc
    if not isinstance(run, dict):
        raise ArtifactError(f"{path / RUN_FILE}: expected a JSON object")
    if run.get("status") not in FINISHED_STATUSES:
        raise ArtifactError(f"{path}: run status is {run.get('status')!r}; it did not finish")

    records: list[RowRecord] = []
    seen: set[str] = set()
    try:
        lines = (path / ROWS_FILE).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ArtifactError(f"{path}: no {ROWS_FILE}") from exc
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = RowRecord.model_validate_json(line)
        except ValidationError as exc:
            reason = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            raise ArtifactError(f"{path / ROWS_FILE}:{line_no}: {reason}") from exc
        if record.id in seen:
            raise ArtifactError(f"{path / ROWS_FILE}:{line_no}: duplicate row id {record.id!r}")
        seen.add(record.id)
        records.append(record)

    expected = run.get("rows", {}).get("total")
    if expected != len(records):
        raise ArtifactError(
            f"{path}: {RUN_FILE} lists {expected} rows but {ROWS_FILE} has {len(records)}"
        )
    return run, records


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
