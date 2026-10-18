from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from clauseguard.corpus.manifest import load_manifest
from clauseguard.datasets.schema import GoldenCitation, GoldenEvalRow
from clauseguard.eval.golden import GoldenLoadError, load_golden, load_known_spec_ids

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = REPO_ROOT / "data" / "golden" / "spec_qa.jsonl"
EXAMPLE_PATH = REPO_ROOT / "data" / "golden" / "golden_eval.example.jsonl"
MANIFEST_PATH = REPO_ROOT / "data" / "corpus" / "manifest.csv"

KNOWN = {"3GPP TS 28.552", "O-RAN.WG2.A1AP"}


def _answerable(**overrides):
    row = {
        "id": "spec_qa_001",
        "question": "What does RRU.PrbUsedDl measure?",
        "expected_answer": "Average DL PRBs used for data traffic.",
        "expected_citations": [{"spec_id": "3GPP TS 28.552", "section": "5.1.1.2.5"}],
        "is_answerable": True,
    }
    row.update(overrides)
    return row


def _unanswerable(**overrides):
    row = {
        "id": "spec_qa_002",
        "question": "How does E2 report 6G NTN beam steering?",
        "expected_answer": "not found",
        "expected_citations": [],
        "is_answerable": False,
    }
    row.update(overrides)
    return row


def _write(tmp_path, rows):
    path = tmp_path / "golden.jsonl"
    lines = [row if isinstance(row, str) else json.dumps(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- models -----------------------------------------------------------------


def test_row_round_trips_without_losing_answer_or_citations():
    raw = _answerable()

    row = GoldenEvalRow.model_validate(raw)

    assert row.type == "spec_qa"
    assert row.expected_citations == [GoldenCitation(spec_id="3GPP TS 28.552", section="5.1.1.2.5")]
    assert row.model_dump(exclude_unset=True) == raw


@pytest.mark.parametrize("field", ["id", "question", "expected_answer", "expected_citations"])
def test_row_requires_field(field):
    raw = _answerable()
    del raw[field]

    with pytest.raises(ValidationError, match=field):
        GoldenEvalRow.model_validate(raw)


def test_row_rejects_unexpected_field():
    with pytest.raises(ValidationError, match="surprise"):
        GoldenEvalRow.model_validate(_answerable(surprise=1))


def test_row_rejects_unsupported_type():
    with pytest.raises(ValidationError, match="type"):
        GoldenEvalRow.model_validate(_answerable(type="rca"))


@pytest.mark.parametrize("field", ["spec_id", "section"])
def test_citation_requires_nonempty_field(field):
    citation = {"spec_id": "3GPP TS 28.552", "section": "5.1"}
    citation[field] = "  "

    with pytest.raises(ValidationError, match=field):
        GoldenEvalRow.model_validate(_answerable(expected_citations=[citation]))


def test_row_rejects_blank_question():
    with pytest.raises(ValidationError, match="question"):
        GoldenEvalRow.model_validate(_answerable(question="   "))


@pytest.mark.parametrize(
    "legacy", ["expected_spec_id", "expected_section", "spec_refs_golden", "scenario_id"]
)
def test_row_rejects_legacy_field_alongside_citations(legacy):
    with pytest.raises(ValidationError, match=f"legacy field.*{legacy}.*conflict"):
        GoldenEvalRow.model_validate(_answerable(**{legacy: "x"}))


def test_row_rejects_legacy_only_representation():
    raw = _answerable(expected_spec_id="3GPP TS 28.552", expected_section="5.1")
    del raw["expected_citations"]

    with pytest.raises(ValidationError, match="legacy field.*expected_citations"):
        GoldenEvalRow.model_validate(raw)


def test_answerable_row_requires_citations():
    with pytest.raises(ValidationError, match="answerable.*citation"):
        GoldenEvalRow.model_validate(_answerable(expected_citations=[]))


def test_unanswerable_row_is_accepted():
    row = GoldenEvalRow.model_validate(_unanswerable())

    assert row.expected_answer == "not found"
    assert row.expected_citations == []


@pytest.mark.parametrize("answer", ["Not found", "not found.", "No answer"])
def test_unanswerable_row_requires_exact_not_found(answer):
    with pytest.raises(ValidationError, match="not found"):
        GoldenEvalRow.model_validate(_unanswerable(expected_answer=answer))


def test_unanswerable_row_rejects_citations():
    citations = [{"spec_id": "3GPP TS 28.552", "section": "5.1"}]

    with pytest.raises(ValidationError, match="citation"):
        GoldenEvalRow.model_validate(_unanswerable(expected_citations=citations))


# --- loader -----------------------------------------------------------------


def test_load_golden_returns_rows_in_file_order(tmp_path):
    path = _write(tmp_path, [_answerable(), "", _unanswerable()])

    rows = load_golden(path, known_spec_ids=KNOWN)

    assert [row.id for row in rows] == ["spec_qa_001", "spec_qa_002"]


def test_load_golden_reports_malformed_json_with_line(tmp_path):
    path = _write(tmp_path, [_answerable(), "{not json"])

    with pytest.raises(GoldenLoadError, match=rf"{path.name}:2: invalid JSON"):
        load_golden(path, known_spec_ids=KNOWN)


def test_load_golden_reports_non_object_line(tmp_path):
    path = _write(tmp_path, ["[1, 2]"])

    with pytest.raises(GoldenLoadError, match=rf"{path.name}:1: .*JSON object"):
        load_golden(path, known_spec_ids=KNOWN)


def test_load_golden_reports_validation_error_with_line(tmp_path):
    path = _write(tmp_path, [_answerable(), _unanswerable(expected_answer="Not found")])

    with pytest.raises(GoldenLoadError, match=rf"{path.name}:2: .*not found"):
        load_golden(path, known_spec_ids=KNOWN)


def test_load_golden_reports_duplicate_id(tmp_path):
    path = _write(tmp_path, [_answerable(), _unanswerable(id="spec_qa_001")])

    with pytest.raises(GoldenLoadError, match=r":2: duplicate id 'spec_qa_001'.*line 1"):
        load_golden(path, known_spec_ids=KNOWN)


def test_load_golden_reports_unknown_spec_id(tmp_path):
    citations = [{"spec_id": "O-RAN.WG9.Nope", "section": "1"}]
    path = _write(tmp_path, [_answerable(expected_citations=citations)])

    with pytest.raises(GoldenLoadError, match=r":1: unknown spec_id 'O-RAN.WG9.Nope'"):
        load_golden(path, known_spec_ids=KNOWN)


def test_load_golden_skips_reference_check_without_known_ids(tmp_path):
    citations = [{"spec_id": "O-RAN.WG9.Nope", "section": "1"}]
    path = _write(tmp_path, [_answerable(expected_citations=citations)])

    assert len(load_golden(path)) == 1


def test_load_known_spec_ids_reads_manifest(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "filename,source_url,spec_id,doc_type,working_group,version\n"
        "a.pdf,https://example.com/a,O-RAN.WG2.A1AP,oran,WG2,1.0\n"
        "b.pdf,https://example.com/b,3GPP TS 28.552,3gpp,SA5,18.7\n",
        encoding="utf-8",
    )

    assert load_known_spec_ids(manifest) == KNOWN


# --- real data --------------------------------------------------------------


def test_seed_loads_all_rows_unchanged():
    before = hashlib.sha256(SEED_PATH.read_bytes()).hexdigest()
    raw_rows = [json.loads(line) for line in SEED_PATH.read_text().splitlines() if line.strip()]

    rows = load_golden(SEED_PATH, known_spec_ids=load_known_spec_ids(MANIFEST_PATH))

    assert len(rows) == 24
    assert [row.model_dump(exclude_unset=True) for row in rows] == raw_rows
    assert sum(not row.is_answerable for row in rows) == 4
    assert hashlib.sha256(SEED_PATH.read_bytes()).hexdigest() == before


def test_example_file_loads_against_manifest():
    known = {row["spec_id"] for row in load_manifest(MANIFEST_PATH)}

    rows = load_golden(EXAMPLE_PATH, known_spec_ids=known)

    assert {row.is_answerable for row in rows} == {True, False}
