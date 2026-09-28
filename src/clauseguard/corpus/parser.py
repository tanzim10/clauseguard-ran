"""Parse acquired corpus documents into manifest-aligned plain text files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from pypdf import PdfReader

from clauseguard.config import Settings, get_settings
from clauseguard.corpus.manifest import load_manifest, validate_manifest_row


@dataclass(frozen=True)
class ParseSummary:
    """Outputs produced by one manifest-validated corpus parsing run."""

    documents: int
    outputs: tuple[Path, ...]


def _output_path(source: Path, output_dir: Path | str) -> Path:
    return Path(output_dir) / f"{source.stem}.txt"


def _write_text(source: Path, output_dir: Path | str, text: str) -> Path:
    normalized = text.strip()
    if not normalized:
        raise ValueError(f"no extractable text found in {source}")
    output = _output_path(source, output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{normalized}\n", encoding="utf-8")
    return output


def parse_pdf(source: Path | str, output_dir: Path | str) -> Path:
    """Extract text from a text-layer PDF into a manifest-aligned `.txt` file."""
    source_path = Path(source)
    pages = [page.extract_text() or "" for page in PdfReader(source_path).pages]
    return _write_text(source_path, output_dir, "\n\n".join(pages))


def parse_docx(source: Path | str, output_dir: Path | str) -> Path:
    """Extract paragraph text from a DOCX file without requiring host applications."""
    source_path = Path(source)
    with ZipFile(source_path) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs = [
        "".join(node.text or "" for node in paragraph.iter(f"{namespace}t"))
        for paragraph in document.iter(f"{namespace}p")
    ]
    return _write_text(source_path, output_dir, "\n\n".join(paragraphs))


def parse_document(source: Path | str, output_dir: Path | str) -> Path:
    """Parse one supported corpus document into text."""
    source_path = Path(source)
    if source_path.suffix.casefold() == ".pdf":
        return parse_pdf(source_path, output_dir)
    if source_path.suffix.casefold() == ".docx":
        return parse_docx(source_path, output_dir)
    raise ValueError(f"unsupported corpus format: {source_path.suffix}")


def parse_corpus(
    *,
    manifest_path: Path | str | None = None,
    raw_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    settings: Settings | None = None,
) -> ParseSummary:
    """Validate and parse every acquired manifest source into plain text."""
    resolved_settings = settings or get_settings()
    manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else Path(resolved_settings.data_dir) / "corpus" / "manifest.csv"
    )
    raw_root = Path(raw_dir or resolved_settings.raw_corpus_dir)
    parsed_root = Path(output_dir or resolved_settings.parsed_corpus_dir)
    rows = [row for row in load_manifest(manifest) if row.get("acquired_at")]

    invalid_sources = [
        row["filename"]
        for row in rows
        if not validate_manifest_row(row, corpus_dir=raw_root)
    ]
    if invalid_sources:
        raise ValueError(
            "manifest validation failed for acquired sources: "
            + ", ".join(invalid_sources)
        )

    outputs = tuple(
        parse_document(raw_root / row["filename"], parsed_root)
        for row in rows
    )
    return ParseSummary(documents=len(rows), outputs=outputs)
