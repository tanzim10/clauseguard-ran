"""Observation validation uses small local windows, not live dataset calls."""

import pytest
from pydantic import ValidationError

from clauseguard.datasets.schema import KpiWindow, Provenance, ScenarioRecord


def window():
    return {
        "channels": {
            "packets": {"kind": "numeric", "values": [1, 2.5, None]},
            "protocol": {"kind": "categorical", "values": ["TCP", "None", None]},
        },
        "sampling_rate_hz": 10,
        "start_time": "2025-01-01 00:00:00.000",
        "end_time": "2025-01-01 00:00:00.200",
    }


def test_preserves_native_values_and_unknown_units():
    parsed = KpiWindow.model_validate(window())
    assert parsed.sample_count == 3
    assert parsed.channels["packets"].values == [1, 2.5, None]
    assert type(parsed.channels["packets"].values[0]) is int
    assert parsed.channels["protocol"].values == ["TCP", "None", None]
    assert parsed.channels["packets"].unit is None
    assert parsed.start_time == "2025-01-01 00:00:00.000"
    assert KpiWindow.model_validate_json(parsed.model_dump_json()) == parsed


@pytest.mark.parametrize("values", [[True, 2, 3], [float("nan"), 2, 3], ["1", 2, 3]])
def test_invalid_numeric_values(values):
    data = window()
    data["channels"]["packets"]["values"] = values
    with pytest.raises(ValidationError):
        KpiWindow.model_validate(data)


def test_lengths_and_timestamp_bounds():
    data = window()
    data["channels"]["packets"]["values"] = [1]
    with pytest.raises(ValidationError, match="lengths"):
        KpiWindow.model_validate(data)


def scenario():
    return {
        "scenario_id": "fixture-1", "kpi_window": window(),
        "fault_label": "normal", "rca_label": "normal", "origin": "normal",
        "provenance": {"revision": "a" * 40, "source_file": "normal/chunked.jsonl",
                       "source_row": 1, "source_sha256": "b" * 64},
        "anomaly": {"exists": False},
        "reference": {"description": "Do not send a diagnosis to the model."},
    }


def test_reference_text_is_separate_from_approved_symptoms():
    data = scenario()
    assert ScenarioRecord.model_validate(data).nl_context == ""
    data["nl_context"] = "Throughput fell."
    with pytest.raises(ValidationError, match="unreviewed"):
        ScenarioRecord.model_validate(data)
    data["text_review"] = {"status": "approved", "source": "human symptoms",
                           "reviewer": "reviewer", "reviewed_at": "2026-10-14"}
    parsed = ScenarioRecord.model_validate(data)
    assert parsed.nl_context == "Throughput fell."
    assert "diagnosis" in parsed.reference["description"]


def test_verified_lineage_requires_evidence():
    data = scenario()["provenance"]
    data["relationship_status"] = "verified"
    with pytest.raises(ValidationError, match="evidence"):
        Provenance.model_validate(data)


def test_anomaly_inclusive_bounds_and_labels():
    data = scenario()
    data.update(fault_label="abnormal", rca_label="unknown", origin="unknown")
    data["anomaly"] = {"exists": True, "start_index": 0, "end_index": 2}
    ScenarioRecord.model_validate(data)
    data["anomaly"]["end_index"] = 3
    with pytest.raises(ValidationError, match="bounds"):
        ScenarioRecord.model_validate(data)
    data = window()
    data["end_time"] = "2025-01-01 00:00:00.300"
    with pytest.raises(ValidationError, match="cadence"):
        KpiWindow.model_validate(data)
