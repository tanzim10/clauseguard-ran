"""Streaming analysis tables. These contain IDs/targets and are not inference exports."""

import csv
import json
from collections.abc import Iterator
from pathlib import Path

from clauseguard.datasets.schema import ScenarioRecord
from clauseguard.datasets.telecomts import file_sha256


def iter_analysis_scenarios(path: Path) -> Iterator[ScenarioRecord]:
    """Read validated acquisition output, without assigning membership or release eligibility."""
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                yield ScenarioRecord.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"{path.name}:{line_number}: {exc}") from exc


def export_analysis_csv(scenarios_path: Path, output_dir: Path | None = None) -> dict:
    """Two streaming passes retain values without keeping all time-series samples in memory."""
    output_dir = output_dir or scenarios_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    overview_path, samples_path, schema_path = [output_dir / name for name in
        ("scenarios_overview.csv", "kpi_samples.csv", "csv_schema.json")]
    if any(path.exists() for path in (overview_path, samples_path, schema_path)):
        raise FileExistsError("CSV analysis exports already exist")
    channels: dict[str, dict] = {}
    scenarios, samples = 0, 0
    for record in iter_analysis_scenarios(scenarios_path):
        scenarios += 1
        samples += record.kpi_window.sample_count
        for name, channel in record.kpi_window.channels.items():
            metadata = {"kind": channel.kind, "unit": channel.unit}
            if name in channels and channels[name] != metadata:
                raise ValueError(f"inconsistent channel metadata: {name}")
            channels[name] = metadata
    names = sorted(channels)
    sample_header = ["scenario_id", "sample_index", "relative_time_seconds", *names,
                     *(name + "__missing" for name in names)]
    if len(sample_header) != len(set(sample_header)):
        raise ValueError("KPI channel names collide with CSV metadata or missing-value columns")
    overview_header = ["scenario_id", "origin", "fault_label", "rca_label", "sample_count",
                       "sampling_rate_hz", "start_time", "end_time", "source_file", "source_row",
                       "recording_id", "parent_id", "relationship_status", "text_review_status",
                       "has_approved_text", "channel_names"]
    with overview_path.open("w", encoding="utf-8", newline="") as overview, \
            samples_path.open("w", encoding="utf-8", newline="") as samples_stream:
        overview_writer = csv.DictWriter(overview, fieldnames=overview_header)
        sample_writer = csv.DictWriter(samples_stream, fieldnames=sample_header)
        overview_writer.writeheader()
        sample_writer.writeheader()
        for record in iter_analysis_scenarios(scenarios_path):
            window, source = record.kpi_window, record.provenance
            overview_writer.writerow({
                "scenario_id": record.scenario_id, "origin": record.origin,
                "fault_label": record.fault_label, "rca_label": record.rca_label,
                "sample_count": window.sample_count, "sampling_rate_hz": window.sampling_rate_hz,
                "start_time": window.start_time, "end_time": window.end_time,
                "source_file": source.source_file, "source_row": source.source_row,
                "recording_id": source.recording_id, "parent_id": source.parent_id,
                "relationship_status": source.relationship_status,
                "text_review_status": record.text_review.status,
                "has_approved_text": bool(record.nl_context),
                "channel_names": json.dumps(sorted(window.channels)),
            })
            for index in range(window.sample_count):
                values = {name: window.channels[name].values[index] if name in window.channels
                          else None for name in names}
                sample_writer.writerow({
                    "scenario_id": record.scenario_id, "sample_index": index,
                    "relative_time_seconds": index / window.sampling_rate_hz, **values,
                    **{name + "__missing": value is None for name, value in values.items()},
                })
    schema = {"scenarios": scenarios, "samples": samples, "channels": channels,
              "source_scenarios_sha256": file_sha256(scenarios_path),
              "files": {path.name: file_sha256(path) for path in (overview_path, samples_path)},
              "purpose": "analysis only; labels and provenance are not model inputs",
              "missing_values": "empty value with channel__missing=True; literal strings are retained",
              "absent_channels": "consult overview channel_names to distinguish absent from null",
              "time": "relative sample time; source timezone is not inferred"}
    schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return schema
