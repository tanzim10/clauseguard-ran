"""Golden dataset JSONL loading and validation (Week 5).

Checks structure and references only: unique IDs, required fields, answerability rules and,
when `known_spec_ids` is given, that every cited `spec_id` exists in the corpus manifest.
It does not verify that an expected answer is true.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from clauseguard.corpus.manifest import load_manifest
from clauseguard.datasets.schema import GoldenEvalRow


class GoldenLoadError(ValueError):
    """A golden file row failed to load; the message names the file, line and reason."""

    def __init__(self, path: Path, line: int, reason: str) -> None:
        super().__init__(f"{path}:{line}: {reason}")
        self.path = path
        self.line = line
        self.reason = reason


def load_known_spec_ids(manifest_path: Path | str) -> set[str]:
    """Return every `spec_id` listed in the corpus manifest, including documented blockers."""
    return {row["spec_id"] for row in load_manifest(manifest_path) if row.get("spec_id")}


def load_golden(
    path: Path | str, known_spec_ids: Iterable[str] | None = None
) -> list[GoldenEvalRow]:
    """Load and validate a golden JSONL file, raising `GoldenLoadError` on the first bad row.

    Blank lines are skipped. Pass `known_spec_ids` (see `load_known_spec_ids`) to reject
    citations to specifications that are not in the manifest.
    """
    path = Path(path)
    known = set(known_spec_ids) if known_spec_ids is not None else None
    rows: list[GoldenEvalRow] = []
    first_seen: dict[str, int] = {}

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GoldenLoadError(path, line_no, f"invalid JSON ({exc.msg})") from exc
            if not isinstance(raw, dict):
                raise GoldenLoadError(path, line_no, "each line must be a JSON object")

            try:
                row = GoldenEvalRow.model_validate(raw)
            except ValidationError as exc:
                raise GoldenLoadError(path, line_no, _describe(exc)) from exc

            if row.id in first_seen:
                raise GoldenLoadError(
                    path, line_no, f"duplicate id {row.id!r} (first on line {first_seen[row.id]})"
                )
            first_seen[row.id] = line_no

            if known is not None:
                for citation in row.expected_citations:
                    if citation.spec_id not in known:
                        raise GoldenLoadError(
                            path, line_no, f"unknown spec_id {citation.spec_id!r} (not in manifest)"
                        )
            rows.append(row)

    return rows


def _describe(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        message = error["msg"].removeprefix("Value error, ")
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts)
