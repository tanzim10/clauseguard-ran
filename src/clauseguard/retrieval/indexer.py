"""Parsed-text → chunk → embed → Qdrant indexing pipeline."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, UnprocessableEntityError

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
    rejected_chunk_ids: tuple[str, ...] = field(default_factory=tuple)

_TRANSIENT_ATTEMPTS = 5
_TRANSIENT_WAIT_SECONDS = 20.0


def _embed_with_retry(embed: Callable[[list[str]], list[list[float]]], texts: list[str],
                      sleep: Callable[[float], None]) -> list[list[float]]:
    """Retry connection errors and 5xx responses; other errors propagate."""
    for attempt in range(_TRANSIENT_ATTEMPTS):
        try:
            return embed(texts)
        except (APIConnectionError, APIStatusError) as error:
            transient = isinstance(error, APIConnectionError) or (error.status_code or 0) >= 500
            if not transient or attempt == _TRANSIENT_ATTEMPTS - 1:
                raise
            sleep(_TRANSIENT_WAIT_SECONDS * (attempt + 1))
    raise AssertionError("unreachable")


def embed_screened(chunks: Sequence[dict[str, Any]], embed: Callable[[list[str]], list[list[float]]],
                   *, batch_size: int, sleep: Callable[[float], None] = time.sleep,
                   ) -> tuple[list[dict[str, Any]], list[list[float]], list[str]]:
    """Embed chunks, dropping only those the provider rejects (HTTP 422 content screening).

    A rejected batch is halved until the single rejected chunks are found, so one flagged
    passage does not fail the whole index. Rejected chunk IDs are returned for the record.
    """
    kept: list[dict[str, Any]] = []
    vectors: list[list[float]] = []
    rejected: list[str] = []

    def run(batch: list[dict[str, Any]]) -> None:
        try:
            out = _embed_with_retry(embed, [c["text"] for c in batch], sleep)
        except UnprocessableEntityError:
            if len(batch) == 1:
                rejected.append(batch[0]["id"])
                return
            middle = len(batch) // 2
            run(batch[:middle])
            run(batch[middle:])
            return
        kept.extend(batch)
        vectors.extend(out)

    for start in range(0, len(chunks), batch_size):
        run(list(chunks[start:start + batch_size]))
    return kept, vectors, rejected


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

    chunks, embeddings, rejected = embed_screened(
        chunks,
        lambda texts: embed_texts(texts, client=embedding_client, settings=resolved_settings,
                                  input_type="passage"),
        batch_size=resolved_settings.embedding_batch_size,
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
    return IndexSummary(documents=len(rows), chunks=len(chunks), rejected_chunk_ids=tuple(rejected))
