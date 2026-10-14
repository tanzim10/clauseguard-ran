import csv
import json
from pathlib import Path

import pytest

from clauseguard.datasets.csv_export import export_analysis_csv

EXAMPLES = Path(__file__).resolve().parents[1] / "data/scenarios/scenarios.example.jsonl"


def test_csv_preserves_values_masks_and_joins(tmp_path):
    report = export_analysis_csv(EXAMPLES, tmp_path)
    with (tmp_path / "scenarios_overview.csv").open(newline="") as stream:
        overview = list(csv.DictReader(stream))
    with (tmp_path / "kpi_samples.csv").open(newline="") as stream:
        samples = list(csv.DictReader(stream))
    assert report["scenarios"] == len(overview) == 2
    assert report["samples"] == len(samples) == 6
    assert {r["scenario_id"] for r in samples} == {r["scenario_id"] for r in overview}
    assert samples[1]["RX_Bytes"] == "80.5"
    assert samples[1]["DL_Protocol"] == "None"
    assert samples[1]["DL_Protocol__missing"] == "False"
    assert samples[2]["DL_Protocol"] == ""
    assert samples[2]["DL_Protocol__missing"] == "True"
    assert samples[1]["relative_time_seconds"] == "0.1"
    assert all(row["relationship_status"] == "unresolved" for row in overview)
    assert report["channels"]["RX_Bytes"]["unit"] is None


def test_csv_is_reproducible_and_refuses_overwrites(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    export_analysis_csv(EXAMPLES, first)
    export_analysis_csv(EXAMPLES, second)
    for name in ("scenarios_overview.csv", "kpi_samples.csv", "csv_schema.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    with pytest.raises(FileExistsError):
        export_analysis_csv(EXAMPLES, first)


def test_reserved_columns_fail_before_export(tmp_path):
    record = json.loads(EXAMPLES.open().readline())
    record["kpi_window"]["channels"]["scenario_id"] = {"kind": "numeric", "values": [1, 2, 3]}
    path = tmp_path / "scenarios.jsonl"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="collide"):
        export_analysis_csv(path)
    assert not (tmp_path / "kpi_samples.csv").exists()
