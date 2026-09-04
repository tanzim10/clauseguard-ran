"""Corpus manifest load/validate (Week 1). Placeholder only."""

from pathlib import Path
from typing import Any


def load_manifest(path: Path | str) -> list[dict[str, Any]]:
    """Load manifest CSV rows. Not implemented in scaffold."""
    raise NotImplementedError("Week 1 — load and validate corpus manifest CSV")


def validate_manifest_row(row: dict[str, Any]) -> bool:
    """Validate one manifest row. Not implemented in scaffold."""
    raise NotImplementedError("Week 1 — validate manifest fields")
