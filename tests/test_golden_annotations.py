from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from clauseguard.corpus.manifest import load_manifest
from clauseguard.datasets.schema import GoldenEvalRow
from clauseguard.eval.annotations import (
    Annotation,
    AnnotationLoadError,
    SourceLocator,
    check_against_golden,
    load_annotations,
    section_clause,
)
from clauseguard.eval.golden import GoldenLoadError, load_golden

PDF_SHA = "a" * 64
DOCX_SHA = "b" * 64

MANIFEST = [
    {
        "spec_id": "3GPP TS 28.552",
        "version": "18.7.0",
        "sha256": PDF_SHA,
        "file_format": "pdf",
        "page_count": "376",
    },
    {
        "spec_id": "O-RAN.WG3.TS.E2SM-KPM",
        "version": "08.00",
        "sha256": DOCX_SHA,
        "file_format": "docx",
        "page_count": "",
    },
]


def _pdf_locator(**overrides):
    locator = {
        "spec_id": "3GPP TS 28.552",
        "version": "18.7.0",
        "sha256": PDF_SHA,
        "clause": "5.1.1.2.5",
        "title": "Mean DL PRB used for data traffic",
        "pdf_page": 30,
    }
    locator.update(overrides)
    return locator


def _docx_locator(**overrides):
    locator = {
        "spec_id": "O-RAN.WG3.TS.E2SM-KPM",
        "version": "08.00",
        "sha256": DOCX_SHA,
        "clause": "7.4.1",
        "docx_heading": "7.4.1 REPORT Service Style 1",
        "docx_paragraph": 2,
    }
    locator.update(overrides)
    return locator


def _group(claim="RRU.PrbUsedDl is the mean DL PRB count.", *alternatives):
    return {"claim": claim, "alternatives": list(alternatives) or [_pdf_locator()]}


def _answerable(**overrides):
    annotation = {
        "id": "spec_qa_001",
        "status": "approved",
        "reviewer": "tanzim10",
        "reviewed_on": "2026-10-14",
        "families": ["direct"],
        "tags": ["kpi"],
        "evidence": [_group()],
    }
    annotation.update(overrides)
    return annotation


def _unanswerable(**overrides):
    annotation = {
        "id": "spec_qa_002",
        "families": ["no_answer"],
        "no_answer": {
            "sources_checked": ["3GPP TS 28.552"],
            "missing_evidence": "No clause defines an NTN beam steering measurement.",
        },
    }
    annotation.update(overrides)
    return annotation


def _write(tmp_path, rows):
    path = tmp_path / "annotations.jsonl"
    lines = [row if isinstance(row, str) else json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- locator ----------------------------------------------------------------


def test_pdf_locator_keeps_clause_separate_from_title():
    locator = SourceLocator.model_validate(_pdf_locator())

    assert locator.clause == "5.1.1.2.5"
    assert locator.title == "Mean DL PRB used for data traffic"


def test_docx_locator_is_accepted():
    locator = SourceLocator.model_validate(_docx_locator())

    assert locator.docx_heading == "7.4.1 REPORT Service Style 1"
    assert locator.pdf_page is None


def test_locator_requires_a_position():
    raw = _pdf_locator()
    del raw["pdf_page"]

    with pytest.raises(ValidationError, match="pdf_page or docx_heading"):
        SourceLocator.model_validate(raw)


def test_locator_rejects_both_pdf_and_docx_positions():
    with pytest.raises(ValidationError, match="pdf_page or docx_heading"):
        SourceLocator.model_validate(_pdf_locator(docx_heading="5.1"))


def test_locator_rejects_paragraph_without_heading():
    with pytest.raises(ValidationError, match="docx_paragraph"):
        SourceLocator.model_validate(_pdf_locator(docx_paragraph=1))


def test_locator_rejects_clause_with_title_text():
    with pytest.raises(ValidationError, match="clause"):
        SourceLocator.model_validate(_pdf_locator(clause="5.1.1.2.5 Mean DL PRB"))


@pytest.mark.parametrize("sha", ["abc", "g" * 64, ""])
def test_locator_rejects_malformed_sha256(sha):
    with pytest.raises(ValidationError, match="sha256"):
        SourceLocator.model_validate(_pdf_locator(sha256=sha))


def test_locator_rejects_page_zero():
    with pytest.raises(ValidationError, match="pdf_page"):
        SourceLocator.model_validate(_pdf_locator(pdf_page=0))


def test_locator_rejects_chunk_id():
    with pytest.raises(ValidationError, match="chunk_id"):
        SourceLocator.model_validate(_pdf_locator(chunk_id="28552-0042"))


# --- annotation -------------------------------------------------------------


def test_answerable_annotation_round_trips():
    annotation = Annotation.model_validate(_answerable())

    assert annotation.reviewed_on == date(2026, 10, 14)
    assert annotation.evidence[0].alternatives[0].pdf_page == 30


def test_annotation_defaults_to_draft():
    annotation = Annotation.model_validate(_unanswerable())

    assert annotation.status == "draft"


@pytest.mark.parametrize("missing", ["reviewer", "reviewed_on"])
def test_approved_annotation_requires_reviewer_and_date(missing):
    raw = _answerable()
    del raw[missing]

    with pytest.raises(ValidationError, match="approved.*reviewer"):
        Annotation.model_validate(raw)


def test_families_may_overlap():
    raw = _answerable(
        families=["multi_clause", "difficult"],
        evidence=[_group(), _group("Second claim.", _docx_locator())],
    )

    assert Annotation.model_validate(raw).families == ["multi_clause", "difficult"]


def test_annotation_requires_a_family():
    with pytest.raises(ValidationError, match="families"):
        Annotation.model_validate(_answerable(families=[]))


def test_annotation_rejects_unknown_family():
    with pytest.raises(ValidationError, match="families"):
        Annotation.model_validate(_answerable(families=["clarification"]))


def test_annotation_rejects_repeated_family():
    with pytest.raises(ValidationError, match="repeated"):
        Annotation.model_validate(_answerable(families=["direct", "direct"]))


def test_multi_clause_needs_two_required_groups():
    with pytest.raises(ValidationError, match="multi_clause.*2"):
        Annotation.model_validate(_answerable(families=["multi_clause"]))


def test_alternatives_in_one_group_do_not_count_as_multi_clause():
    group = _group("One claim.", _pdf_locator(), _pdf_locator(clause="5.1.1.2.6"))

    with pytest.raises(ValidationError, match="multi_clause"):
        Annotation.model_validate(_answerable(families=["multi_clause"], evidence=[group]))


def test_evidence_group_needs_an_alternative():
    with pytest.raises(ValidationError, match="alternatives"):
        Annotation.model_validate(_answerable(evidence=[{"claim": "x", "alternatives": []}]))


def test_answerable_annotation_requires_evidence():
    with pytest.raises(ValidationError, match="evidence"):
        Annotation.model_validate(_answerable(evidence=[]))


def test_answerable_annotation_rejects_no_answer_record():
    raw = _answerable(no_answer=_unanswerable()["no_answer"])

    with pytest.raises(ValidationError, match="no_answer"):
        Annotation.model_validate(raw)


@pytest.mark.parametrize("other", ["direct", "multi_clause", "difficult"])
def test_no_answer_family_stands_alone(other):
    with pytest.raises(ValidationError, match="no_answer"):
        Annotation.model_validate(_unanswerable(families=["no_answer", other]))


def test_no_answer_annotation_requires_record():
    raw = _unanswerable()
    del raw["no_answer"]

    with pytest.raises(ValidationError, match="no_answer"):
        Annotation.model_validate(raw)


def test_no_answer_annotation_rejects_evidence():
    with pytest.raises(ValidationError, match="evidence"):
        Annotation.model_validate(_unanswerable(evidence=[_group()]))


@pytest.mark.parametrize(
    "record",
    [
        {"sources_checked": [], "missing_evidence": "x"},
        {"sources_checked": ["3GPP TS 28.552"], "missing_evidence": " "},
    ],
)
def test_no_answer_record_needs_sources_and_missing_evidence(record):
    with pytest.raises(ValidationError, match="sources_checked|missing_evidence"):
        Annotation.model_validate(_unanswerable(no_answer=record))


def test_annotation_records_corrections():
    raw = _answerable(corrections=["Answer said 'percentage'; clause defines an integer."])

    assert Annotation.model_validate(raw).corrections[0].startswith("Answer said")


def test_annotation_rejects_unexpected_field():
    with pytest.raises(ValidationError, match="surprise"):
        Annotation.model_validate(_answerable(surprise=1))


# --- loader -----------------------------------------------------------------


def test_load_annotations_returns_rows_in_file_order(tmp_path):
    path = _write(tmp_path, [_answerable(), "", _unanswerable()])

    rows = load_annotations(path, manifest=MANIFEST)

    assert [row.id for row in rows] == ["spec_qa_001", "spec_qa_002"]


def test_load_annotations_error_is_a_golden_load_error(tmp_path):
    path = _write(tmp_path, ["{not json"])

    with pytest.raises(GoldenLoadError, match=rf"{path.name}:1: invalid JSON"):
        load_annotations(path)


def test_load_annotations_reports_non_object_line(tmp_path):
    path = _write(tmp_path, ["[1]"])

    with pytest.raises(AnnotationLoadError, match=r":1: .*JSON object"):
        load_annotations(path)


def test_load_annotations_reports_validation_error_with_line(tmp_path):
    path = _write(tmp_path, [_answerable(), _answerable(id="spec_qa_003", evidence=[])])

    with pytest.raises(AnnotationLoadError, match=r":2: .*evidence"):
        load_annotations(path)


def test_load_annotations_reports_duplicate_id(tmp_path):
    path = _write(tmp_path, [_answerable(), _unanswerable(id="spec_qa_001")])

    with pytest.raises(AnnotationLoadError, match=r":2: duplicate id 'spec_qa_001'.*line 1"):
        load_annotations(path)


def test_load_annotations_skips_manifest_checks_without_manifest(tmp_path):
    path = _write(tmp_path, [_answerable(evidence=[_group("x", _pdf_locator(spec_id="Nope"))])])

    assert len(load_annotations(path)) == 1


@pytest.mark.parametrize(
    ("locator", "reason"),
    [
        (_pdf_locator(spec_id="O-RAN.WG9.Nope"), "unknown spec_id 'O-RAN.WG9.Nope'"),
        (_pdf_locator(version="18.6.0"), "version '18.6.0'.*manifest has '18.7.0'"),
        (_pdf_locator(sha256="c" * 64), "sha256 does not match"),
        (_pdf_locator(pdf_page=377), "pdf_page 377.*376 pages"),
        (_docx_locator(docx_heading=None, pdf_page=3), "docx.*docx_heading"),
        (_pdf_locator(pdf_page=None, docx_heading="5.1"), "pdf.*pdf_page"),
    ],
)
def test_load_annotations_checks_locators_against_manifest(tmp_path, locator, reason):
    path = _write(tmp_path, [_answerable(evidence=[_group("x", locator)])])

    with pytest.raises(AnnotationLoadError, match=rf":1: .*{reason}"):
        load_annotations(path, manifest=MANIFEST)


def test_load_annotations_matches_sha256_case_insensitively(tmp_path):
    locator = _pdf_locator(sha256=PDF_SHA.upper())
    path = _write(tmp_path, [_answerable(evidence=[_group("x", locator)])])

    assert len(load_annotations(path, manifest=MANIFEST)) == 1


def test_load_annotations_checks_no_answer_sources_against_manifest(tmp_path):
    record = {"sources_checked": ["3GPP TS 99.999"], "missing_evidence": "x"}
    path = _write(tmp_path, [_unanswerable(no_answer=record)])

    with pytest.raises(AnnotationLoadError, match=r":1: .*sources_checked.*'3GPP TS 99.999'"):
        load_annotations(path, manifest=MANIFEST)


# --- cross-check against golden rows ----------------------------------------


@pytest.mark.parametrize(
    ("section", "clause"),
    [
        ("5.1.1.3.1 Average DL UE throughput in gNB", "5.1.1.3.1"),
        ("8.2.1.2 Successful Operation", "8.2.1.2"),
        ("A.1.2 Annex example", "A.1.2"),
        ("4.2", "4.2"),
    ],
)
def test_section_clause_takes_leading_clause_number(section, clause):
    assert section_clause(section) == clause


def _golden(**overrides):
    row = {
        "id": "spec_qa_001",
        "question": "What does RRU.PrbUsedDl measure?",
        "expected_answer": "Average DL PRBs used for data traffic.",
        "expected_citations": [
            {"spec_id": "3GPP TS 28.552", "section": "5.1.1.2.5 Mean DL PRB used for data traffic"}
        ],
    }
    row.update(overrides)
    return GoldenEvalRow.model_validate(row)


def _golden_unanswerable(**overrides):
    row = {
        "id": "spec_qa_002",
        "question": "How does E2 report 6G NTN beam steering?",
        "expected_answer": "not found",
        "expected_citations": [],
        "is_answerable": False,
    }
    row.update(overrides)
    return GoldenEvalRow.model_validate(row)


def test_check_against_golden_accepts_consistent_rows():
    annotations = [
        Annotation.model_validate(_answerable()),
        Annotation.model_validate(_unanswerable()),
    ]

    assert check_against_golden(annotations, [_golden(), _golden_unanswerable()]) == []


def test_check_against_golden_allows_unannotated_rows():
    annotations = [Annotation.model_validate(_answerable())]

    assert check_against_golden(annotations, [_golden(), _golden_unanswerable()]) == []


def test_check_against_golden_reports_unknown_id():
    annotations = [Annotation.model_validate(_answerable(id="spec_qa_999"))]

    problems = check_against_golden(annotations, [_golden()])

    assert problems == ["spec_qa_999: no golden row with this id"]


def test_check_against_golden_reports_answerability_mismatch():
    annotations = [Annotation.model_validate(_unanswerable(id="spec_qa_001"))]

    problems = check_against_golden(annotations, [_golden()])

    assert problems == ["spec_qa_001: no_answer family disagrees with is_answerable=True"]


def test_check_against_golden_reports_citation_missing_from_evidence():
    group = _group("x", _pdf_locator(clause="5.1.1.2.6"))
    annotations = [Annotation.model_validate(_answerable(evidence=[group]))]

    problems = check_against_golden(annotations, [_golden()])

    assert problems == [
        "spec_qa_001: expected citation 3GPP TS 28.552 5.1.1.2.5 is not in any evidence group"
    ]


def test_check_against_golden_matches_citation_to_any_alternative():
    group = _group("x", _pdf_locator(clause="5.1.1.2.6"), _pdf_locator())
    annotations = [Annotation.model_validate(_answerable(evidence=[group]))]

    assert check_against_golden(annotations, [_golden()]) == []


# --- real data --------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "data" / "golden"


def test_annotations_file_matches_manifest_and_golden_rows():
    annotations = load_annotations(
        GOLDEN_DIR / "annotations.jsonl",
        manifest=load_manifest(REPO_ROOT / "data" / "corpus" / "manifest.csv"),
    )

    assert check_against_golden(annotations, load_golden(GOLDEN_DIR / "spec_qa.jsonl")) == []


def test_every_golden_row_has_an_approved_annotation():
    annotations = load_annotations(GOLDEN_DIR / "annotations.jsonl")
    golden_ids = [row.id for row in load_golden(GOLDEN_DIR / "spec_qa.jsonl")]

    approved = {a.id for a in annotations if a.status == "approved"}

    assert len(golden_ids) == 24
    assert approved == set(golden_ids)
