from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from clauseguard.datasets.schema import RcaRow
from clauseguard.eval.annotations import Annotation, check_against_golden
from clauseguard.eval.golden import GoldenLoadError, load_golden, load_rca


def _rca_row(**overrides):
    row = {
        "id": "rca_001",
        "scenario_id": "telecomts_0001",
        "origin": "real",
        "inputs": "kpi_only",
        "question": "Why did downlink throughput in cell 12 fall during the evening window?",
        "kpi_evidence": [
            {
                "channel": "RRU.PrbUsedDl",
                "unit": "%",
                "baseline": {"start": "2024-03-01T17:00", "end": "2024-03-01T18:00"},
                "comparison": {"start": "2024-03-01T19:00", "end": "2024-03-01T20:00"},
                "baseline_value": 41.0,
                "comparison_value": 95.5,
                "direction": "increase",
            }
        ],
        "source_labels": {"fault_label": "congestion"},
        "expected_conclusion": "PRB saturation limited per-UE downlink throughput.",
        "acceptable_conclusions": ["Downlink capacity exhaustion in the cell."],
        "uncertainty": "Single cell; no alarm data to rule out backhaul limits.",
        "expected_citations": [
            {
                "spec_id": "3GPP TS 28.552",
                "section": "5.1.1.2.5 Mean DL PRB used for data traffic",
                "role": "definition",
                "explanation": "Defines what the PRB usage channel measures.",
            },
            {
                "spec_id": "3GPP TS 38.300",
                "section": "10.1 Scheduling",
                "role": "mechanism",
                "explanation": "The scheduler shares PRBs between UEs, so saturation caps each UE.",
            },
        ],
    }
    row.update(overrides)
    return row


def _write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_valid_rca_row_loads():
    row = RcaRow.model_validate(_rca_row())

    assert row.type == "rca"
    assert row.is_answerable
    assert row.kpi_evidence[0].direction == "increase"


def test_rca_row_rejects_direction_that_contradicts_measured_values():
    evidence = _rca_row()["kpi_evidence"][0] | {"direction": "decrease"}

    with pytest.raises(ValidationError, match="disagrees with measured values"):
        RcaRow.model_validate(_rca_row(kpi_evidence=[evidence]))


def test_rca_row_rejects_question_that_reveals_a_source_label():
    question = "Why did Congestion reduce downlink throughput in cell 12?"

    with pytest.raises(ValidationError, match="reveals source label.*congestion"):
        RcaRow.model_validate(_rca_row(question=question))


def test_rca_row_rejects_definition_only_citations():
    citations = [_rca_row()["expected_citations"][0]]

    with pytest.raises(ValidationError, match="mechanism citation"):
        RcaRow.model_validate(_rca_row(expected_citations=citations))


@pytest.mark.parametrize(
    "field", ["kpi_evidence", "source_labels", "expected_citations", "uncertainty", "inputs"]
)
def test_rca_row_requires_evidence_fields(field):
    row = _rca_row()
    del row[field]

    with pytest.raises(ValidationError):
        RcaRow.model_validate(row)


def test_rca_row_rejects_unknown_fields():
    with pytest.raises(ValidationError, match="extra"):
        RcaRow.model_validate(_rca_row(fault_label="congestion"))


def test_load_rca_reads_rows_and_checks_spec_ids(tmp_path):
    path = _write_jsonl(tmp_path / "rca.jsonl", [_rca_row(), _rca_row(id="rca_002")])

    rows = load_rca(path, known_spec_ids={"3GPP TS 28.552", "3GPP TS 38.300"})

    assert [row.id for row in rows] == ["rca_001", "rca_002"]
    with pytest.raises(GoldenLoadError, match="unknown spec_id '3GPP TS 38.300'"):
        load_rca(path, known_spec_ids={"3GPP TS 28.552"})


def test_load_rca_rejects_duplicate_ids(tmp_path):
    path = _write_jsonl(tmp_path / "rca.jsonl", [_rca_row(), _rca_row()])

    with pytest.raises(GoldenLoadError, match=r":2: duplicate id 'rca_001' \(first on line 1\)"):
        load_rca(path)


def test_empty_rca_file_loads_no_rows(tmp_path):
    path = tmp_path / "rca.jsonl"
    path.write_text("", encoding="utf-8")

    assert load_rca(path) == []


def test_spec_loader_still_rejects_rca_rows(tmp_path):
    path = _write_jsonl(tmp_path / "spec_qa.jsonl", [_rca_row()])

    with pytest.raises(GoldenLoadError, match="scenario_id"):
        load_golden(path)


def test_check_against_golden_covers_rca_citations():
    row = RcaRow.model_validate(_rca_row())
    locator = {
        "spec_id": "3GPP TS 28.552",
        "version": "18.7.0",
        "sha256": "a" * 64,
        "clause": "5.1.1.2.5",
        "pdf_page": 30,
    }
    annotation = Annotation.model_validate(
        {
            "id": "rca_001",
            "families": ["direct"],
            "evidence": [{"claim": "PRB usage definition.", "alternatives": [locator]}],
        }
    )

    assert check_against_golden([annotation], [row]) == [
        "rca_001: expected citation 3GPP TS 38.300 10.1 is not in any evidence group"
    ]
