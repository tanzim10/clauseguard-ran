"""Parsed-text → chunk → embed → Qdrant indexing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clauseguard.config import Settings, get_settings
from clauseguard.corpus.chunker import chunk_document
from clauseguard.corpus.manifest import load_manifest
from clauseguard.retrieval.embeddings import embed_texts
from clauseguard.retrieval.qdrant_store import ensure_collection, upsert_chunks


@dataclass(frozen=True)
class IndexSummary:
    """Counts produced by one indexing run."""

    documents: int
    chunks: int


def _default_manifest(settings: Settings) -> Path:
    return Path(settings.data_dir) / "corpus" / "manifest.csv"


def _parsed_path(parsed_dir: Path, filename: str) -> Path:
    return parsed_dir / f"{Path(filename).stem}.txt"


def index_corpus(
    *,
    manifest_path: Path | str | None = None,
    parsed_dir: Path | str | None = None,
    collection: str | None = None,
    settings: Settings | None = None,
    embedding_client: Any | None = None,
    qdrant_client: Any | None = None,
) -> IndexSummary:
    """Rebuild the configured Qdrant index from existing parsed text files.

    Parsed files must be named after their manifest source file with a ``.txt``
    suffix. This function deliberately does not invoke the unfinished PDF
    parser; missing parsed inputs fail before embedding or writing.
    """
    resolved_settings = settings or get_settings()
    manifest = Path(manifest_path) if manifest_path else _default_manifest(resolved_settings)
    parsed_root = Path(parsed_dir) if parsed_dir else Path(resolved_settings.parsed_corpus_dir)
    rows = [row for row in load_manifest(manifest) if row.get("acquired_at")]

    missing = [
        _parsed_path(parsed_root, row["filename"])
        for row in rows
        if not _parsed_path(parsed_root, row["filename"]).is_file()
    ]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"parsed corpus files are missing: {paths}")

    chunks: list[dict[str, Any]] = []
    for row in rows:
        source_path = _parsed_path(parsed_root, row["filename"])
        text = source_path.read_text(encoding="utf-8")
        chunks.extend(
            chunk_document(
                text,
                spec_id=row["spec_id"],
                source_file=row["filename"],
            )
        )

    embeddings = embed_texts(
        [chunk["text"] for chunk in chunks],
        client=embedding_client,
        settings=resolved_settings,
        input_type="passage",
    )
    target_collection = collection or resolved_settings.qdrant_collection
    ensure_collection(target_collection, client=qdrant_client, settings=resolved_settings)
    upsert_chunks(
        target_collection,
        chunks,
        embeddings=embeddings,
        client=qdrant_client,
        settings=resolved_settings,
    )
    return IndexSummary(documents=len(rows), chunks=len(chunks))
