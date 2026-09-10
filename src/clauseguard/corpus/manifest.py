"""Corpus manifest load/validate (Week 1)."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("filename", "source_url", "spec_id", "doc_type", "working_group", "version")
VALID_DOC_TYPES = {"oran", "3gpp"}
VALID_TEXT_LAYER_VALUES = {"", "true", "false"}
_PLACEHOLDER_SHA256 = {"", "pending"}


def load_manifest(path: Path | str) -> list[dict[str, Any]]:
    """Parse the manifest CSV into a list of row dicts.

    Reads whatever columns are present in the CSV header, so both the
    minimal schema (`manifest.example.csv`) and the richer real-corpus
    schema (extra provenance columns) parse the same way.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def validate_manifest_row(row: dict[str, Any], corpus_dir: Path | str | None = None) -> bool:
    """Validate one manifest row.

    A row must have every field in `REQUIRED_FIELDS` and a `doc_type`
    of `oran` or `3gpp`. From there it is either:

    - a documented blocker: `sha256` is empty or `"pending"`, and
      `license_note` explains why there's no file yet, or
    - an acquired row: `sha256` is present, and — when `corpus_dir` is
      given — matches the hash of the actual file on disk rather than
      being trusted as written.
    """
    for field in REQUIRED_FIELDS:
        if not row.get(field):
            return False

    if row.get("doc_type") not in VALID_DOC_TYPES:
        return False

    if (row.get("has_text_layer") or "").strip().lower() not in VALID_TEXT_LAYER_VALUES:
        return False

    sha256 = (row.get("sha256") or "").strip()
    is_blocker = sha256 in _PLACEHOLDER_SHA256

    if is_blocker:
        return not row.get("acquired_at") and bool((row.get("license_note") or "").strip())

    if not row.get("acquired_at"):
        return False

    if corpus_dir is not None:
        file_path = Path(corpus_dir) / row["filename"]
        if not file_path.is_file():
            return False
        actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if actual != sha256.lower():
            return False

    return True
