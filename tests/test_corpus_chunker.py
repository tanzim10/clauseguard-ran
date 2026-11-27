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
    assert all(chunk["metadata"]["section"] == "unsectioned" for chunk in chunks)


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
    assert all(chunk["metadata"]["section"] == "unsectioned" for chunk in chunks)


def test_chunk_document_labels_text_before_the_first_heading_as_preamble():
    # v2 drops a preamble under 200 words as a cover page, so this one is long enough to keep.
    text = " ".join(["Scope and terminology for the specification."] * 40) + """

1. Introduction
The controller coordinates policy guidance.
"""

    chunks = chunk_document(
        text,
        spec_id="O-RAN.WG2.A1AP",
        source_file="oran-wg2-a1ap.txt",
    )

    assert chunks[0]["metadata"]["section"] == "preamble"
    assert chunks[0]["id"] == "O-RAN.WG2.A1AP::preamble::0"



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


def test_chunk_document_preserves_sentence_that_fits_without_heading():
    sentence = " ".join(f"signal-{index}" for index in range(509)) + " terminal."
    text = f"1. Measurements overview\n{sentence}\n"

    chunks = chunk_document(
        text,
        spec_id="O-RAN.WG2.A1AP",
        source_file="oran-wg2-a1ap.txt",
    )

    assert [chunk["text"] for chunk in chunks] == [
        "1. Measurements overview",
        sentence,
    ]


_TOC_AND_ANNEX = """Contents
6.6 SINR related measurements ........................................ 94
A.9 Monitoring of RF performance ..................................... 103
.............. 104

6.6 SINR related measurements
The SINR measurement reports the uplink signal quality.

Annex A (informative):
A.9 Monitoring of RF performance
Uplink interference disturbs the receiver and causes receive errors.
"""


def test_v2_drops_table_of_contents_lines_and_splits_annex_clauses():
    chunks = chunk_document(_TOC_AND_ANNEX, spec_id="3GPP TS 32.425", source_file="ts.txt", version="v2")

    sections = [chunk["metadata"]["section"] for chunk in chunks]
    assert "A.9 Monitoring of RF performance" in sections
    assert not any("....." in chunk["text"] for chunk in chunks)
    annex = next(chunk for chunk in chunks if chunk["metadata"]["section"].startswith("A.9"))
    assert "receive errors" in annex["text"]
    sinr = next(chunk for chunk in chunks if chunk["metadata"]["section"].startswith("6.6"))
    assert "receive errors" not in sinr["text"]


def test_v1_keeps_its_original_behaviour_for_frozen_manifests():
    chunks = chunk_document(_TOC_AND_ANNEX, spec_id="3GPP TS 32.425", source_file="ts.txt", version="v1")

    sections = [chunk["metadata"]["section"] for chunk in chunks]
    assert not any(section.startswith("A.9") for section in sections)
    assert any("....." in chunk["text"] for chunk in chunks)


def test_default_version_comes_from_settings(monkeypatch):
    monkeypatch.setenv("CHUNKER_VERSION", "v1")
    v1 = chunk_document(_TOC_AND_ANNEX, spec_id="s", source_file="f")
    monkeypatch.setenv("CHUNKER_VERSION", "v2")
    v2 = chunk_document(_TOC_AND_ANNEX, spec_id="s", source_file="f")

    assert v1 == chunk_document(_TOC_AND_ANNEX, spec_id="s", source_file="f", version="v1")
    assert v2 == chunk_document(_TOC_AND_ANNEX, spec_id="s", source_file="f", version="v2")
    assert v1 != v2


def test_unknown_chunker_version_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="unknown chunker version"):
        chunk_document("1 Scope\ntext", spec_id="s", source_file="f", version="v9")


def test_v2_drops_a_cover_page_preamble_only():
    cover = "ETSI TS 128 552 V18.7.0 5G; Management and orchestration\n\n1 Scope\nThe scope text."
    assert [c["metadata"]["section"] for c in chunk_document(cover, spec_id="s", source_file="f",
                                                              version="v2")] == ["1 Scope"]
    assert "preamble" in [c["metadata"]["section"] for c in chunk_document(cover, spec_id="s", source_file="f",
                                                                            version="v1")]


def test_v2_drops_references_and_the_closing_change_history_but_keeps_the_preamble():
    body = "\n".join(f"{n} Clause {n}\nThe clause {n} explains a measurement." for n in range(4, 12))
    intro = " ".join(["Intro text before any numbered heading stays."] * 40)
    text = f"""{intro}

2 References
[1] 3GPP TS 23.502: "Procedures for the 5G System".

{body}

Change history
1 B Add new use case for extended 5QI 16.6.0
2019-06 SA#84 SP-190371 0086 2 F Correction
"""
    chunks = chunk_document(text, spec_id="s", source_file="f", version="v2")

    sections = [chunk["metadata"]["section"] for chunk in chunks]
    joined = " ".join(chunk["text"] for chunk in chunks)
    assert "preamble" in sections and "Intro text" in joined
    assert "2 References" not in sections and "Procedures for the 5G System" not in joined
    assert "SA#84" not in joined and not any(section.startswith("1 B") for section in sections)
    assert "11 Clause 11" in sections
