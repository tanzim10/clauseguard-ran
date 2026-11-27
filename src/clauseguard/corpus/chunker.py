"""Deterministic section- and paragraph-aware chunking for specification text."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from clauseguard.config import Settings

CHUNKER_VERSIONS = ("v1", "v2")
_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+)*(?:\.)?[ \t]+\S.*\s*$")
# v2 also accepts annex clauses such as "A.9 Monitoring of RF performance".
_HEADING_V2_RE = re.compile(r"^\s*(?:\d+|[A-Z](?=\.\d))(?:\.\d+)*(?:\.)?[ \t]+\S.*\s*$")
# Table-of-contents entries end in a dot leader and an optional page number.
_TOC_RE = re.compile(r"(?:\.\s?){5,}\s*\d*\s*$")
# The change-history annex closes every 3GPP/O-RAN document; its table rows look like headings.
_CHANGE_HISTORY_RE = re.compile(r"^\s*(?:Annex\s*[A-Z]?\s*\(informative\):\s*)?Change\s+history\b.*$", re.IGNORECASE)
# Front matter matches almost any query and never explains a fault.
_FRONT_MATTER_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:(?:Normative|Informative)\s+)?(?:References|Foreword|"
    r"Intellectual Property Rights|Legal Notice|Modal verbs terminology)\s*$", re.IGNORECASE)
# A preamble shorter than this is a cover page; longer preambles hold unnumbered content.
_TITLE_PAGE_MAX_WORDS = 200
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_PREAMBLE_SECTION = "preamble"
_UNSECTIONED_SECTION = "unsectioned"


def chunk_document(
    text: str, *, spec_id: str, source_file: str, version: str | None = None
) -> list[dict[str, Any]]:
    """Split specification text into deterministic, metadata-bearing chunks.

    Numbered lines are treated as section headings. If no such line exists, blank-line
    separated paragraphs provide the chunk boundaries. Sections and paragraphs are kept
    intact where possible; oversized content is progressively split at sentence and then
    word boundaries to honor the configured token budget.

    ``v2`` (the default) drops table-of-contents lines, the closing change-history annex and
    front-matter sections (References, Foreword, IPR and legal notices, a cover-page preamble),
    and accepts lettered
    annex clauses as headings. ``v1`` is kept so frozen retrieval manifests stay reproducible.
    """
    settings = Settings()
    version = version or settings.chunker_version
    if version not in CHUNKER_VERSIONS:
        raise ValueError(f"unknown chunker version {version!r}")
    budget = max(1, settings.chunk_size_tokens)
    lines = text.splitlines()
    if version == "v2":
        lines = _strip_change_history([line for line in lines if not _TOC_RE.search(line)])
    pattern = _HEADING_V2_RE if version == "v2" else _HEADING_RE
    heading_indexes = [index for index, line in enumerate(lines) if _is_heading(line, pattern)]

    section_chunks: list[tuple[str, str]] = []
    if heading_indexes:
        first_heading = heading_indexes[0]
        preamble = _paragraphs(lines[:first_heading])
        if version == "v1" or _token_count(" ".join(preamble)) >= _TITLE_PAGE_MAX_WORDS:
            section_chunks.extend(_chunk_content(_PREAMBLE_SECTION, preamble, budget))

        for position, heading_index in enumerate(heading_indexes):
            next_heading = (
                heading_indexes[position + 1]
                if position + 1 < len(heading_indexes)
                else len(lines)
            )
            section = lines[heading_index].strip()
            if version == "v2" and _FRONT_MATTER_RE.match(section):
                continue
            body = _paragraphs(lines[heading_index + 1 : next_heading])
            section_chunks.extend(_chunk_content(section, body, budget, heading=section))
    else:
        section_chunks.extend(
            _chunk_paragraphs(_paragraphs(lines), budget, section=_UNSECTIONED_SECTION)
        )

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


def _strip_change_history(lines: list[str]) -> list[str]:
    """Drop the closing change-history annex: the first standalone heading in the second half."""
    for index in range(len(lines) // 2, len(lines)):
        if _CHANGE_HISTORY_RE.match(lines[index]):
            return lines[:index]
    return lines


def _is_heading(line: str, pattern: re.Pattern[str] = _HEADING_RE) -> bool:
    """Return whether a non-blank line has a numbered section/clause prefix."""
    return bool(line.strip()) and bool(pattern.fullmatch(line))


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
            if (
                heading
                and len(current_parts) == 1
                and current_tokens < budget
                and current_tokens + paragraph_tokens > budget
            ):
                sentences = _sentences(paragraph)
                remaining_capacity = budget - current_tokens
                leading_sentences: list[str] = []
                while sentences and _token_count(sentences[0]) <= remaining_capacity:
                    leading_sentence = sentences.pop(0)
                    leading_sentences.append(leading_sentence)
                    remaining_capacity -= _token_count(leading_sentence)

                if leading_sentences:
                    current_parts.extend(leading_sentences)
                    current_tokens += sum(_token_count(part) for part in leading_sentences)
                    if not sentences:
                        continue
                    paragraph = " ".join(sentences)
                    paragraph_tokens = _token_count(paragraph)

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


def _chunk_paragraphs(
    paragraphs: list[str], budget: int, *, section: str
) -> list[tuple[str, str]]:
    """Keep each fallback paragraph independent while enforcing the token budget."""
    chunks: list[tuple[str, str]] = []
    for paragraph in paragraphs:
        chunks.extend(_chunk_content(section, [paragraph], budget))
    return chunks


def _split_oversized_paragraph(paragraph: str, budget: int) -> list[str]:
    """Split a paragraph on sentences, using words only for oversized sentences."""
    sentences = _sentences(paragraph)
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


def _sentences(paragraph: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.split(paragraph) if part.strip()]


def _split_words(text: str, budget: int) -> list[str]:
    words = text.split()
    return [" ".join(words[start : start + budget]) for start in range(0, len(words), budget)]


def _token_count(text: str | None) -> int:
    return len(text.split()) if text else 0


def _join_parts(parts: list[str]) -> str:
    return "\n\n".join(part for part in parts if part)
