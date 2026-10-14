import copy
import json

import pytest

from clauseguard.datasets.telecomts import (
    build_scenarios,
    convert_record,
    file_sha256,
    scenario_id,
)


@pytest.fixture
def source_record():
    return {
        "KPIs": {"RX_Bytes": [2, 1.5, None], "DL_Protocol": ["TCP", "None", None]},
        "sampling_rate": 10, "start_time": "2025-01-01 00:00:00.000",
        "end_time": "2025-01-01 00:00:00.200",
        "description": "Source description with the diagnosis.",
        "anomalies": {"exists": False, "type": "None", "affected_kpis": [""],
                      "anomaly_duration": {"start": "", "end": ""}},
        "labels": {"congestion": "No"}, "statistics": {"mean": 1},
        "QnA": {"a": "Do not expose this answer."}, "extra_field": "preserve me",
    }


def convert(record, **kwargs):
    return convert_record(record, revision="a" * 40, source_file="normal/example.jsonl",
                          source_row=1, source_sha256="b" * 64, **kwargs)


def test_preserves_reference_and_normalizes_only_placeholders(source_record):
    original = copy.deepcopy(source_record)
    result = convert(source_record)
    assert source_record == original
    assert result.reference == {k: v for k, v in original.items() if k != "KPIs"}
    assert result.anomaly.affected_channels == []
    assert result.anomaly.start_index is None
    assert result.rca_label == "normal"
    assert result.nl_context == ""
    assert result.text_review.status == "pending"
    assert result.provenance.relationship_status == "unresolved"
    assert convert(source_record).scenario_id == result.scenario_id
    assert scenario_id("a" * 40, "file.jsonl", 2) != result.scenario_id


def test_injected_fault_is_not_experiment_congestion(source_record):
    source_record["anomalies"] = {"exists": True,
                                 "type": "High Network Congestion (Sudden Spike)",
                                 "affected_kpis": ["RX_Bytes"],
                                 "anomaly_duration": {"start": 1, "end": 2}}
    result = convert(source_record)
    assert result.fault_label == "abnormal"
    assert result.rca_label == "High Network Congestion (Sudden Spike)"
    assert result.reference["labels"]["congestion"] == "No"


def test_unknown_cause_and_explicit_human_approval(source_record):
    source_record["anomalies"] = {"exists": True, "type": "Unverified cause"}
    result = convert(source_record, review={"text": "Measured bytes fell.",
                     "status": "approved", "reviewer": "human",
                     "reviewed_at": "2026-10-14", "source": "human observations"})
    assert result.rca_label == "unknown"
    assert result.nl_context == "Measured bytes fell."


def test_sampling_alias_conflict(source_record):
    source_record["sampling_rate_hz"] = 20
    with pytest.raises(ValueError, match="conflicting"):
        convert(source_record)


def test_stream_accounts_for_invalid_rows_and_duplicates(tmp_path, source_record):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(source_record) + "\nnot json\n" + json.dumps(source_record) + "\n")
    kwargs = {"revision": "a" * 40, "checksums": {"normal/file.jsonl": file_sha256(source)}}
    report = build_scenarios({"normal/file.jsonl": source}, output_dir=tmp_path / "first", **kwargs)
    assert report["counts"] == {"selected_rows": 3, "accepted": 2, "rejected": 1}
    assert len(report["duplicates"]) == 1
    assert report["splits_ready"] is False
    second = tmp_path / "second"
    build_scenarios({"normal/file.jsonl": source}, output_dir=second, **kwargs)
    assert (second / "scenarios.jsonl").read_bytes() == (tmp_path / "first/scenarios.jsonl").read_bytes()
    rejected = json.loads((second / "rejected.jsonl").read_text())
    assert rejected["source_row"] == 2
    assert rejected["raw_line"] == "not json"


def test_checksum_failure_preserves_diagnostics(tmp_path, source_record):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(source_record) + "\n")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="checksum"):
        build_scenarios({"source.jsonl": source}, revision="a" * 40,
                        checksums={"source.jsonl": "0" * 64}, output_dir=output)
    assert "checksum" in json.loads((output / "conversion.json").read_text())["error"]


def test_refuses_overwriting_outputs(tmp_path, source_record):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(source_record) + "\n")
    args = {"revision": "a" * 40, "checksums": {"source.jsonl": file_sha256(source)},
            "output_dir": tmp_path / "output"}
    build_scenarios({"source.jsonl": source}, **args)
    with pytest.raises(FileExistsError):
        build_scenarios({"source.jsonl": source}, **args)
