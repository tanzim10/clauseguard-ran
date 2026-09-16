"""Deterministic section- and paragraph-aware chunking for specification text."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from clauseguard.config import Settings

_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+)*(?:\.)?[ \t]+\S.*\s*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_document(text: str, *, spec_id: str, source_file: str) -> list[dict[str, Any]]:
    """Split specification text into deterministic, metadata-bearing chunks.

    Numbered lines are treated as section headings. If no such line exists, blank-line
    separated paragraphs provide the chunk boundaries. Sections and paragraphs are kept
    intact where possible; oversized content is progressively split at sentence and then
    word boundaries to honor the configured token budget.
    """
    budget = max(1, Settings().chunk_size_tokens)
    lines = text.splitlines()
    heading_indexes = [index for index, line in enumerate(lines) if _is_heading(line)]

    section_chunks: list[tuple[str, str]] = []
    if heading_indexes:
        first_heading = heading_indexes[0]
        preamble = _paragraphs(lines[:first_heading])
        section_chunks.extend(_chunk_content("", preamble, budget))

        for position, heading_index in enumerate(heading_indexes):
            next_heading = (
                heading_indexes[position + 1]
                if position + 1 < len(heading_indexes)
                else len(lines)
            )
            section = lines[heading_index].strip()
            body = _paragraphs(lines[heading_index + 1 : next_heading])
            section_chunks.extend(_chunk_content(section, body, budget, heading=section))
    else:
        section_chunks.extend(_chunk_paragraphs(_paragraphs(lines), budget))

    chunks: list[dict[str, Any]] = []
    for index, (section, chunk_text) in enumerate(section_chunks):
        chunk_id = f"{spec_id}::{section}::{index}"
        chunks.append(
            {
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "spec_id": spec_id,
                    "source_file": source_file,
                    "section": section,
                    "chunk_id": chunk_id,
                },
            }
        )
    return chunks


def _is_heading(line: str) -> bool:
    """Return whether a non-blank line has a numbered section/clause prefix."""
    return bool(line.strip()) and bool(_HEADING_RE.fullmatch(line))


def _paragraphs(lines: Iterable[str]) -> list[str]:
    """Normalize non-empty lines into blank-line separated paragraphs."""
    paragraphs: list[str] = []
    current: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped:
            current.append(stripped)
        elif current:
            paragraphs.append(" ".join(current))
            current = []

    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def _chunk_content(
    section: str,
    paragraphs: list[str],
    budget: int,
    *,
    heading: str | None = None,
) -> list[tuple[str, str]]:
    """Pack paragraphs into chunks, splitting only when a unit cannot fit."""
    if heading is None and not paragraphs:
        return []

    chunks: list[tuple[str, str]] = []
    heading_parts = _split_words(heading, budget) if heading else []
    for heading_part in heading_parts[:-1]:
        chunks.append((section, heading_part))
    current_parts: list[str] = heading_parts[-1:]
    current_tokens = _token_count(current_parts[0]) if current_parts else 0

    for paragraph in paragraphs:
        paragraph_tokens = _token_count(paragraph)
        if paragraph_tokens <= budget and current_tokens + paragraph_tokens <= budget:
            current_parts.append(paragraph)
            current_tokens += paragraph_tokens
            continue

        if current_parts:
            chunks.append((section, _join_parts(current_parts)))
            current_parts = []
            current_tokens = 0

        if paragraph_tokens <= budget:
            current_parts = [paragraph]
            current_tokens = paragraph_tokens
            continue

        for piece in _split_oversized_paragraph(paragraph, budget):
            if current_parts and current_tokens + _token_count(piece) > budget:
                chunks.append((section, _join_parts(current_parts)))
                current_parts = []
                current_tokens = 0
            current_parts.append(piece)
            current_tokens += _token_count(piece)

    if current_parts:
        chunks.append((section, _join_parts(current_parts)))
    return chunks


def _chunk_paragraphs(paragraphs: list[str], budget: int) -> list[tuple[str, str]]:
    """Keep each fallback paragraph independent while enforcing the token budget."""
    chunks: list[tuple[str, str]] = []
    for paragraph in paragraphs:
        chunks.extend(_chunk_content("", [paragraph], budget))
    return chunks


def _split_oversized_paragraph(paragraph: str, budget: int) -> list[str]:
    """Split a paragraph on sentences, using words only for oversized sentences."""
    sentences = [part.strip() for part in _SENTENCE_RE.split(paragraph) if part.strip()]
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _token_count(sentence)
        if sentence_tokens > budget:
            if current:
                pieces.append(" ".join(current))
                current = []
                current_tokens = 0
            pieces.extend(_split_words(sentence, budget))
        elif current_tokens + sentence_tokens <= budget:
            current.append(sentence)
            current_tokens += sentence_tokens
        else:
            pieces.append(" ".join(current))
            current = [sentence]
            current_tokens = sentence_tokens

    if current:
        pieces.append(" ".join(current))
    return pieces


def _split_words(text: str, budget: int) -> list[str]:
    words = text.split()
    return [" ".join(words[start : start + budget]) for start in range(0, len(words), budget)]


def _token_count(text: str | None) -> int:
    return len(text.split()) if text else 0


def _join_parts(parts: list[str]) -> str:
    return "\n\n".join(part for part in parts if part)
