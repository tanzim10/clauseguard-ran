"""Golden annotation records: review status and source evidence per golden row (Week 5).

Annotations live beside the golden rows in `data/golden/annotations.jsonl`, keyed by row `id`.
Each answerable row lists required evidence groups: every group is one claim the answer needs,
and any one of its alternatives supports that claim. No-answer rows instead record which
sources were checked and what is missing.

These checks cover structure and provenance only. They cannot show that a cited clause says
what the claim says; that still needs a person reading the source.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from clauseguard.datasets.schema import GoldenEvalRow, NonEmptyStr
from clauseguard.eval.golden import GoldenLoadError, _describe

Family = Literal["direct", "multi_clause", "difficult", "no_answer"]

# A clause number only, such as "5.1.1.3.1" or "A.1.2"; titles go in `title`.
ClauseId = Annotated[str, StringConstraints(pattern=r"^\S+$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-fA-F]{64}$")]


class SourceLocator(BaseModel):
    """Where a supporting clause sits in one exact source file.

    PDFs are located by page and DOCX files by heading (plus an optional paragraph number
    under it). `pdf_page` is the 1-based physical page in the file, which can differ from the
    number printed on the page. Chunk IDs are deliberately absent: they change whenever the
    corpus is re-chunked.
    """

    model_config = ConfigDict(extra="forbid")

    spec_id: NonEmptyStr
    version: NonEmptyStr
    sha256: Sha256
    clause: ClauseId
    title: NonEmptyStr | None = None
    pdf_page: int | None = Field(default=None, ge=1)
    docx_heading: NonEmptyStr | None = None
    docx_paragraph: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_position(self) -> SourceLocator:
        if (self.pdf_page is None) == (self.docx_heading is None):
            raise ValueError("locate the clause with exactly one of pdf_page or docx_heading")
        if self.docx_paragraph is not None and self.docx_heading is None:
            raise ValueError("docx_paragraph needs docx_heading")
        return self


class EvidenceGroup(BaseModel):
    """One claim the answer depends on; any single alternative supports it."""

    model_config = ConfigDict(extra="forbid")

    claim: NonEmptyStr
    alternatives: list[SourceLocator] = Field(min_length=1)


class NoAnswerRecord(BaseModel):
    """Why a row is unanswerable: the sources searched and the evidence they lack."""

    model_config = ConfigDict(extra="forbid")

    sources_checked: list[NonEmptyStr] = Field(min_length=1)
    missing_evidence: NonEmptyStr


class Annotation(BaseModel):
    """Review record for one golden row.

    Families may overlap (a row can be both `multi_clause` and `difficult`), except
    `no_answer`, which stands alone. Rows start as `draft`; only a person sets `approved`.
    """

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyStr
    status: Literal["draft", "approved"] = "draft"
    reviewer: NonEmptyStr | None = None
    reviewed_on: date | None = None
    families: list[Family] = Field(min_length=1)
    tags: list[NonEmptyStr] = Field(default_factory=list)
    evidence: list[EvidenceGroup] = Field(default_factory=list)
    no_answer: NoAnswerRecord | None = None
    corrections: list[NonEmptyStr] = Field(default_factory=list)

    @field_validator("families")
    @classmethod
    def _reject_repeated_families(cls, families: list[str]) -> list[str]:
        repeated = sorted({family for family in families if families.count(family) > 1})
        if repeated:
            raise ValueError(f"repeated family {', '.join(map(repr, repeated))}")
        return families

    @model_validator(mode="after")
    def _check_consistency(self) -> Annotation:
        if self.status == "approved" and (self.reviewer is None or self.reviewed_on is None):
            raise ValueError("approved annotations need a reviewer and reviewed_on")

        if "no_answer" in self.families:
            if len(self.families) > 1:
                raise ValueError("no_answer cannot be combined with other families")
            if self.no_answer is None:
                raise ValueError("no_answer annotations need a no_answer record")
            if self.evidence:
                raise ValueError("no_answer annotations must not have evidence groups")
        else:
            if not self.evidence:
                raise ValueError("answerable annotations need at least one evidence group")
            if self.no_answer is not None:
                raise ValueError("only no_answer annotations may have a no_answer record")
            if "multi_clause" in self.families and len(self.evidence) < 2:
                raise ValueError(
                    "multi_clause annotations need at least 2 required evidence groups"
                )
        return self


class AnnotationLoadError(GoldenLoadError):
    """An annotation file row failed to load; the message names the file, line and reason."""


def load_annotations(
    path: Path | str, manifest: Iterable[dict[str, Any]] | None = None
) -> list[Annotation]:
    """Load and validate an annotations JSONL file, raising `AnnotationLoadError` on a bad row.

    Blank lines are skipped. Pass `manifest` (rows from `load_manifest`) to check every
    locator's spec, version, hash, file format and page range, and every no-answer source.
    """
    path = Path(path)
    specs = {row["spec_id"]: row for row in manifest if row.get("spec_id")} if manifest else None
    annotations: list[Annotation] = []
    first_seen: dict[str, int] = {}

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnnotationLoadError(path, line_no, f"invalid JSON ({exc.msg})") from exc
            if not isinstance(raw, dict):
                raise AnnotationLoadError(path, line_no, "each line must be a JSON object")

            try:
                annotation = Annotation.model_validate(raw)
            except ValidationError as exc:
                raise AnnotationLoadError(path, line_no, _describe(exc)) from exc

            if annotation.id in first_seen:
                raise AnnotationLoadError(
                    path,
                    line_no,
                    f"duplicate id {annotation.id!r} (first on line {first_seen[annotation.id]})",
                )
            first_seen[annotation.id] = line_no

            if specs is not None:
                problem = _manifest_problem(annotation, specs)
                if problem:
                    raise AnnotationLoadError(path, line_no, problem)
            annotations.append(annotation)

    return annotations


def _manifest_problem(annotation: Annotation, specs: dict[str, dict[str, Any]]) -> str | None:
    for group in annotation.evidence:
        for locator in group.alternatives:
            row = specs.get(locator.spec_id)
            if row is None:
                return f"unknown spec_id {locator.spec_id!r} (not in manifest)"
            if locator.version != row.get("version"):
                return (
                    f"version {locator.version!r} for {locator.spec_id} does not match; "
                    f"manifest has {row.get('version')!r}"
                )
            if locator.sha256.lower() != (row.get("sha256") or "").strip().lower():
                return f"sha256 does not match manifest for {locator.spec_id}"

            file_format = (row.get("file_format") or "").strip().lower()
            if file_format == "pdf" and locator.pdf_page is None:
                return f"{locator.spec_id} is a pdf file; locate the clause with pdf_page"
            if file_format == "docx" and locator.docx_heading is None:
                return f"{locator.spec_id} is a docx file; locate the clause with docx_heading"

            page_count = (row.get("page_count") or "").strip()
            if (
                locator.pdf_page is not None
                and page_count.isdigit()
                and locator.pdf_page > int(page_count)
            ):
                return (
                    f"pdf_page {locator.pdf_page} is past the end of {locator.spec_id} "
                    f"({page_count} pages)"
                )

    if annotation.no_answer is not None:
        for spec_id in annotation.no_answer.sources_checked:
            if spec_id not in specs:
                return f"no_answer.sources_checked has unknown spec_id {spec_id!r}"
    return None


def section_clause(section: str) -> str:
    """Return the clause number at the start of a golden `section`, e.g. "8.2.1.2"."""
    return section.split()[0]


def check_against_golden(
    annotations: Iterable[Annotation], rows: Iterable[GoldenEvalRow]
) -> list[str]:
    """List every disagreement between annotations and golden rows (empty when consistent).

    Golden rows without an annotation are allowed here; requiring full coverage is a release
    check, not part of the record contract.
    """
    rows_by_id = {row.id: row for row in rows}
    problems: list[str] = []

    for annotation in annotations:
        row = rows_by_id.get(annotation.id)
        if row is None:
            problems.append(f"{annotation.id}: no golden row with this id")
            continue

        if ("no_answer" in annotation.families) == row.is_answerable:
            problems.append(
                f"{annotation.id}: no_answer family disagrees with "
                f"is_answerable={row.is_answerable}"
            )
            continue

        supported = {
            (locator.spec_id, locator.clause)
            for group in annotation.evidence
            for locator in group.alternatives
        }
        for citation in row.expected_citations:
            clause = section_clause(citation.section)
            if (citation.spec_id, clause) not in supported:
                problems.append(
                    f"{annotation.id}: expected citation {citation.spec_id} {clause} "
                    "is not in any evidence group"
                )

    return problems
