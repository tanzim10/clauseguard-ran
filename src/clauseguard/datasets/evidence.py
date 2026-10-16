"""Raw-window evidence and human recording review. This module never assigns splits."""

import csv
import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

from clauseguard.datasets.acquisition import source_path, write_manifest
from clauseguard.datasets.contracts import CATEGORY_LABELS, NUMERIC_CHANNELS, RecordingReview
from clauseguard.datasets.csv_export import iter_analysis_scenarios
from clauseguard.datasets.schema import ScenarioRecord
from clauseguard.datasets.telecomts import file_sha256


def text_eligibility(record: ScenarioRecord, task: str) -> str:
    review = record.text_review
    if review.status != "approved" or not record.nl_context.strip():
        return "no_approved_text"
    if task not in review.approved_tasks:
        return "text_not_approved_for_task"
    if review.text_origin not in ("independent_observation", "kpi_derived"):
        return "text_origin_not_permitted"
    # A secondary check, not a substitute for human review of answer-informed wording.
    text = record.nl_context.casefold()
    if "there is a jammer" in text or any(label.casefold() in text for label in CATEGORY_LABELS):
        return "text_discloses_diagnosis"
    return ""


def summarize_record(record: ScenarioRecord) -> dict:
    window = record.kpi_window
    numeric_ok = window.sample_count == 128 and all(
        name in window.channels and window.channels[name].kind == "numeric"
        for name in NUMERIC_CHANNELS
    )
    category_reason = ("not_faulty" if record.fault_label == "normal" else
                       "unsupported_category")
    if (record.fault_label == "abnormal" and record.origin == "synthetic"
            and record.rca_label in CATEGORY_LABELS):
        category_reason = ""
    return {
        "scenario_id": record.scenario_id, "source_file": record.provenance.source_file,
        "source_row": record.provenance.source_row, "fault_label": record.fault_label,
        "rca_label": record.rca_label, "origin": record.origin,
        "start_time": window.start_time, "end_time": window.end_time,
        "kpi_reason": "" if numeric_ok else "requires_128_steps_and_16_numeric_channels",
        "category_reason": category_reason,
        "text_reasons": {task: text_eligibility(record, task) for task in ("detection", "category")},
        "unknown_units": sorted(name for name, c in window.channels.items() if c.unit is None),
        "excluded_channels": sorted(set(window.channels) - set(NUMERIC_CHANNELS)),
    }


def _read_raw(path: Path) -> tuple[list[dict], dict[datetime, int]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("missing or duplicate raw CSV columns")
        rows = list(reader)
    indices = {}
    for index, row in enumerate(rows):
        stamp = datetime.fromisoformat(row["timestamp"])
        if stamp in indices:
            raise ValueError("duplicate raw CSV timestamp")
        indices[stamp] = index
    return rows, indices


def _match_window(record, rows, indices, tolerance):
    window = record.kpi_window
    start_time = datetime.fromisoformat(window.start_time)
    start = indices.get(start_time)
    if start is None or start + window.sample_count > len(rows):
        raise ValueError("window not found in raw timeline")
    error = 0.0
    for offset in range(window.sample_count):
        row = rows[start + offset]
        expected = start_time + timedelta(seconds=offset / window.sampling_rate_hz)
        if datetime.fromisoformat(row["timestamp"]) != expected:
            raise ValueError("raw timestamps disagree with window cadence")
        for name, channel in window.channels.items():
            observed, value = channel.values[offset], row[name]
            if observed is None:
                if value != "":
                    raise ValueError(f"raw missing value mismatch: {name}")
            elif channel.kind == "categorical":
                if value != observed:
                    raise ValueError(f"raw categorical mismatch: {name}")
            else:
                numeric = float(value)
                if not math.isfinite(numeric) or abs(numeric - observed) > tolerance:
                    raise ValueError(f"raw numeric mismatch: {name}")
                error = max(error, abs(numeric - observed))
    return start + 1, start + window.sample_count, error


def _verified_files(manifest_path, expected_revision=None):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError(f"incomplete acquisition: {manifest_path}")
    if expected_revision and manifest["revision"] != expected_revision:
        raise ValueError("supplemental acquisition revision differs from processed sources")
    files = {entry["source_file"]: entry for entry in manifest["files"]}
    if len(files) != len(manifest["files"]):
        raise ValueError("duplicate acquisition source path")
    for name, entry in files.items():
        path = source_path(manifest_path.parent / "raw", name, supplemental=True)
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"acquired file checksum mismatch: {name}")
    return manifest["revision"], files


def build_recording_evidence(
    scenarios_path: Path, acquisition_dir: Path, output_dir: Path, *,
    reviews: dict[str, dict] | None = None, tolerance: float = 1e-9,
) -> dict:
    """Verify raw measurements; retain unresolved independence instead of inventing groups."""
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 1e-6:
        raise ValueError("numeric match tolerance must be between zero and 1e-6")
    names = ("recording_evidence.csv", "window_sources.csv", "evidence.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError("recording evidence exists; choose a new output directory")
    revision, processed = _verified_files(acquisition_dir / "acquisition.json")
    _, supplements = _verified_files(acquisition_dir / "supplements.json", revision)
    reviews = {name: RecordingReview.model_validate(value) for name, value in (reviews or {}).items()}
    if set(reviews) - set(processed):
        raise ValueError("recording review references an unknown source")
    traces, records, duplicate_links = {}, [], set()
    signatures, seen = {}, set()
    current_source, raw_rows, raw_index = None, [], {}
    mapping_columns = ["scenario_id", "source_file", "source_row", "start_time", "end_time",
                       "raw_start_row", "raw_end_row", "channels_compared", "max_absolute_error",
                       "raw_match", "reason"]
    with (output_dir / "window_sources.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=mapping_columns)
        writer.writeheader()
        for record in iter_analysis_scenarios(scenarios_path):
            source, sid = record.provenance.source_file, record.scenario_id
            if sid in seen:
                raise ValueError(f"duplicate scenario ID: {sid}")
            seen.add(sid)
            if (source not in processed or record.provenance.revision != revision
                    or record.provenance.source_sha256 != processed[source]["sha256"]):
                raise ValueError(f"scenario source identity differs from acquisition: {sid}")
            family = source.removesuffix("/processed/chunked.jsonl")
            if family == source:
                raise ValueError(f"raw mapping requires a processed trace: {source}")
            raw_name, description_name = (f"{family}/raw/{name}" for name in
                                          ("metrics.csv", "description.txt"))
            if raw_name not in supplements or description_name not in supplements:
                raise ValueError(f"missing supplemental acquisition: {source}")
            if source not in traces:
                review = reviews.get(source, RecordingReview())
                if (review.status == "verified" and record.origin == "synthetic"
                        and not review.parent_recording_ids):
                    raise ValueError(f"verified synthetic trace needs parent evidence: {source}")
                traces[source] = {
                    "source_file": source, "revision": revision,
                    "processed_sha256": processed[source]["sha256"],
                    "raw_file": raw_name, "raw_sha256": supplements[raw_name]["sha256"],
                    "description_sha256": supplements[description_name]["sha256"],
                    "window_count": 0, "matched_windows": 0, "review": review.model_dump(),
                    "provenance_recording_ids": [], "provenance_parent_ids": [],
                }
            for key, value in (("provenance_recording_ids", record.provenance.recording_id),
                               ("provenance_parent_ids", record.provenance.parent_id)):
                if value and value not in traces[source][key]:
                    traces[source][key].append(value)
            reason, first, last, max_error = "", None, None, None
            try:
                if current_source != source:
                    raw_rows, raw_index = _read_raw(source_path(
                        acquisition_dir / "raw", raw_name, supplemental=True))
                    current_source = source
                first, last, max_error = _match_window(record, raw_rows, raw_index, tolerance)
            except (ValueError, KeyError, TypeError) as exc:
                reason = str(exc)
            summary = summarize_record(record)
            summary.update(raw_match=not reason, raw_reason=reason)
            records.append(summary)
            traces[source]["window_count"] += 1
            traces[source]["matched_windows"] += int(not reason)
            writer.writerow({key: summary[key] for key in mapping_columns if key in summary} | {
                "raw_start_row": first, "raw_end_row": last, "max_absolute_error": max_error,
                "channels_compared": json.dumps(sorted(record.kpi_window.channels)), "reason": reason,
            })
            # Exact repeated complete observations are another conservative cross-source link.
            signature = hashlib.sha256(json.dumps(
                {name: c.values for name, c in record.kpi_window.channels.items()}, sort_keys=True,
                separators=(",", ":"),
            ).encode()).hexdigest()
            if signature in signatures and signatures[signature] != source:
                duplicate_links.add(tuple(sorted((source, signatures[signature]))))
            else:
                signatures[signature] = source
    for trace in traces.values():
        unknown = set(trace["review"]["candidate_parent_sources"]) - set(traces)
        if unknown:
            raise ValueError(f"candidate parent source absent from release: {sorted(unknown)}")
    trace_rows = []
    for source, trace in sorted(traces.items()):
        review = trace["review"]
        trace_rows.append({key: value for key, value in trace.items() if key != "review"} | {
            "raw_match_status": ("matched" if trace["matched_windows"] == trace["window_count"]
                                 else "mismatch"),
            "recording_status": review["status"], "recording_id": review["recording_id"],
            "parent_recording_ids": json.dumps(review["parent_recording_ids"]),
            "candidate_parent_sources": json.dumps(review["candidate_parent_sources"]),
            "session_evidence": review["evidence"], "reviewer": review["reviewer"],
            "reviewed_at": review["reviewed_at"], "window_mapping": "window_sources.csv",
        })
    with (output_dir / "recording_evidence.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trace_rows[0]) if trace_rows else ["source_file"])
        writer.writeheader()
        writer.writerows(trace_rows)
    evidence = {
        "version": "1", "revision": revision, "scenarios_sha256": file_sha256(scenarios_path),
        "numeric_tolerance": tolerance, "records": records, "traces": traces,
        "duplicate_source_links": sorted(duplicate_links),
        "acquisition_sha256": file_sha256(acquisition_dir / "acquisition.json"),
        "supplements_sha256": file_sha256(acquisition_dir / "supplements.json"),
        "files": {name: file_sha256(output_dir / name) for name in names[:2]},
    }
    write_manifest(output_dir / "evidence.json", evidence)
    return evidence
