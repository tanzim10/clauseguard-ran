from __future__ import annotations

from clauseguard.config import Settings
from clauseguard.corpus.chunker import chunk_document


def test_chunk_document_keeps_a_single_section_together():
    text = """1. Introduction
The near-real-time RAN intelligent controller coordinates policy and optimization.
This section describes the boundary between the controller and its applications.
"""

    chunks = chunk_document(
        text,
        spec_id="O-RAN.WG2.A1AP",
        source_file="oran-wg2-a1ap.txt",
    )

    assert len(chunks) == 1
    assert chunks[0]["text"].startswith("1. Introduction")
    assert "near-real-time RAN intelligent controller" in chunks[0]["text"]


def test_chunk_document_starts_a_new_chunk_at_each_section_heading():
    text = """1. Introduction
The controller manages policy guidance.

2. Interface behavior
The interface carries policy updates and measurement reports.
"""

    chunks = chunk_document(
        text,
        spec_id="O-RAN.WG2.A1AP",
        source_file="oran-wg2-a1ap.txt",
    )

    assert len(chunks) == 2
    assert [chunk["metadata"]["section"] for chunk in chunks] == [
        "1. Introduction",
        "2. Interface behavior",
    ]
    assert "policy guidance" in chunks[0]["text"]
    assert "measurement reports" in chunks[1]["text"]


def test_chunk_document_uses_paragraphs_when_no_headings_are_detectable():
    text = """The first paragraph explains the control-plane context and its purpose.
It contains enough context to stand on its own.

The second paragraph describes the reported behavior and the relevant observation.
It is intentionally separated from the first paragraph.
"""

    chunks = chunk_document(
        text,
        spec_id="3GPP TS 28.552",
        source_file="ts-28-552.txt",
    )

    assert len(chunks) == 2
    assert "control-plane context" in chunks[0]["text"]
    assert "reported behavior" in chunks[1]["text"]
    assert all(chunk["metadata"]["section"] == "" for chunk in chunks)


def test_chunk_document_populates_metadata_and_deterministic_ids():
    text = """3.1 Measurements
The measurement report contains the counters required for analysis.
"""
    kwargs = {
        "spec_id": "3GPP TS 28.552",
        "source_file": "ts-28-552.txt",
    }

    first = chunk_document(text, **kwargs)
    second = chunk_document(text, **kwargs)

    expected_id = "3GPP TS 28.552::3.1 Measurements::0"
    assert first[0]["id"] == expected_id
    assert first[0]["id"] == second[0]["id"]
    assert first[0]["metadata"] == {
        "spec_id": "3GPP TS 28.552",
        "source_file": "ts-28-552.txt",
        "section": "3.1 Measurements",
        "chunk_id": expected_id,
    }


def test_chunk_document_keeps_numeric_content_on_paragraph_fallback():
    text = """12345

The paragraph contains the actual specification content and should not inherit
the numeric line as a section heading.
"""

    chunks = chunk_document(
        text,
        spec_id="3GPP TS 28.552",
        source_file="ts-28-552.txt",
    )

    assert len(chunks) == 2
    assert all(chunk["metadata"]["section"] == "" for chunk in chunks)


def test_chunk_document_respects_the_configured_token_budget():
    text = "1. Large section\n" + " ".join(f"token-{index}" for index in range(600))

    chunks = chunk_document(
        text,
        spec_id="O-RAN.WG2.A1AP",
        source_file="oran-wg2-a1ap.txt",
    )

    token_budget = Settings().chunk_size_tokens
    assert len(chunks) > 1
    assert all(len(chunk["text"].split()) <= token_budget for chunk in chunks)


def test_chunk_document_packs_sentence_with_heading_before_overflowing():
    first_sentence = " ".join(f"signal-{index}" for index in range(500)) + "."
    second_sentence = " ".join(f"metric-{index}" for index in range(11)) + "."
    text = f"1. Measurements\n{first_sentence} {second_sentence}\n"

    chunks = chunk_document(
        text,
        spec_id="O-RAN.WG2.A1AP",
        source_file="oran-wg2-a1ap.txt",
    )

    assert len(chunks) == 2
    assert chunks[0]["text"] == f"1. Measurements\n\n{first_sentence}"
    assert chunks[1]["text"] == second_sentence
