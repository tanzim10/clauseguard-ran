"""Pinned TelecomTS source conversion; this module never assigns dataset splits."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from clauseguard.datasets.schema import (
    AnomalyAnnotation,
    KpiChannel,
    KpiWindow,
    Provenance,
    ScenarioRecord,
    TextReview,
)

REPOSITORY = "AliMaatouk/TelecomTS"
ADAPTER_VERSION = "1"
SCHEMA_VERSION = "1"
CATEGORICAL_CHANNELS = {"UL_Protocol", "DL_Protocol"}
KNOWN_CAUSES = {
    "Antenna Failure", "Co-Channel Interference (Mild)", "Co-Channel Interference (Severe)",
    "Faulty RF Filters (Temporal)", "Doppler Shift (Severe)",
    "Faulty Handover Algorithm (Too Frequent)", "Buffer Overflow (Gradual Buildup)",
    "Resource Allocation Bugs", "High Network Congestion (Gradual Buildup)",
    "High Network Congestion (Sudden Spike)", "Jamming",
}


def scenario_id(revision: str, source_file: str, source_row: int) -> str:
    """IDs derive from pinned source identity, not diagnoses or changing descriptions."""
    identity = json.dumps([REPOSITORY, revision, source_file, source_row], separators=(",", ":"))
    return "telecomts-" + hashlib.sha256(identity.encode()).hexdigest()


def convert_record(
    record: dict[str, Any], *, revision: str, source_file: str, source_row: int,
    source_sha256: str, review: dict[str, Any] | None = None,
    lineage: dict[str, Any] | None = None, units: dict[str, str] | None = None,
) -> ScenarioRecord:
    """Convert one row without promoting target-informed source text into observations."""
    rate = record.get("sampling_rate", record.get("sampling_rate_hz"))
    if ("sampling_rate" in record and "sampling_rate_hz" in record
            and record["sampling_rate"] != record["sampling_rate_hz"]):
        raise ValueError("conflicting sampling_rate and sampling_rate_hz")
    if isinstance(rate, bool):
        raise TypeError("sampling rate must be numeric")
    channels = {
        name: KpiChannel(
            kind="categorical" if name in CATEGORICAL_CHANNELS else "numeric",
            values=values, unit=(units or {}).get(name),
        ) for name, values in record["KPIs"].items()
    }
    window = KpiWindow(channels=channels, sampling_rate_hz=rate,
                       start_time=record["start_time"], end_time=record["end_time"])
    native = record["anomalies"]
    exists = native["exists"]
    if not isinstance(exists, bool):
        raise TypeError("anomalies.exists must be a boolean")
    cause = native.get("type") or None
    duration = native.get("anomaly_duration") or {}
    start, end = duration.get("start"), duration.get("end")
    start = None if start in (None, "") else start
    end = None if end in (None, "") else end
    affected = native.get("affected_kpis") or []
    # Only normal-case empty-string placeholders are removed; native values stay in reference.
    if not exists:
        affected = [name for name in affected if name != ""]
        if cause in (None, "None", "normal", "Normal"):
            cause = None
    annotation = AnomalyAnnotation(exists=exists, source_type=cause,
                                   affected_channels=affected, start_index=start, end_index=end)
    rca = cause if exists and cause in KNOWN_CAUSES else ("unknown" if exists else "normal")
    origin = "normal" if not exists else "unknown"
    if exists and source_file.startswith("anomalous/jammer/"):
        origin = "real"
    elif exists and source_file.startswith("anomalous/synthetic/"):
        origin = "synthetic"
    reference = {key: value for key, value in record.items() if key != "KPIs"}
    review_data = dict(review or {})
    text = review_data.pop("text", "")
    if review:
        reference["text_review_candidate"] = review
    if not review_data:
        review_data = {"status": "pending" if record.get("description") else "absent"}
    text_review = TextReview.model_validate(review_data)
    return ScenarioRecord(
        scenario_id=scenario_id(revision, source_file, source_row), kpi_window=window,
        fault_label="abnormal" if exists else "normal", rca_label=rca, origin=origin,
        provenance=Provenance(revision=revision, source_file=source_file, source_row=source_row,
                              source_sha256=source_sha256, **(lineage or {})),
        anomaly=annotation, text_review=text_review,
        nl_context=text if text_review.status == "approved" else "", reference=reference,
    )


def download_telecomts(**kwargs: Any) -> dict[str, Any]:
    """Acquire pinned remote or local sources, with retained integrity diagnostics."""
    from clauseguard.datasets.acquisition import acquire_telecomts

    return acquire_telecomts(**kwargs)


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def _parse_and_convert_row(
    line: str, *, revision: str, source_file: str, source_row: int, source_sha256: str,
    review: dict[str, Any] | None, lineage: dict[str, Any] | None,
    units: dict[str, str] | None,
) -> tuple[ScenarioRecord, str]:
    """Parse a source line and return its scenario and KPI-window signature."""
    record = json.loads(line, parse_constant=_reject_constant)
    if not isinstance(record, dict):
        raise TypeError("source row must be a JSON object")

    scenario = convert_record(
        record, revision=revision, source_file=source_file, source_row=source_row,
        source_sha256=source_sha256, review=review, lineage=lineage, units=units,
    )
    signature = hashlib.sha256(json.dumps(
        [record["KPIs"], scenario.kpi_window.start_time, scenario.kpi_window.end_time,
         scenario.kpi_window.sampling_rate_hz], sort_keys=True,
    ).encode()).hexdigest()
    return scenario, signature


def build_scenarios(
    sources: dict[str, Path], *, revision: str, checksums: dict[str, str], output_dir: Path,
    reviews: dict[str, dict[str, Any]] | None = None,
    lineage: dict[str, dict[str, Any]] | None = None, units: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Stream every selected physical line to a scenario or a rejection with diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / "scenarios.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    report_path = output_dir / "conversion.json"
    if any(path.exists() for path in (accepted_path, rejected_path, report_path)):
        raise FileExistsError("conversion outputs already exist; choose a new output directory")
    counts: Counter[str] = Counter()
    origins: Counter[str] = Counter()
    causes: Counter[str] = Counter()
    observed: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    report: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "adapter_version": ADAPTER_VERSION,
                            "revision": revision, "splits_ready": False}
    try:
        if not sources:
            raise ValueError("at least one source file is required")
        for name, path in sources.items():
            if file_sha256(path) != checksums[name]:
                raise ValueError(f"source checksum mismatch: {name}")
        with accepted_path.open("w", encoding="utf-8") as accepted, \
                rejected_path.open("w", encoding="utf-8") as rejected:
            for name, path in sorted(sources.items()):
                with path.open(encoding="utf-8") as stream:
                    for row_index, line in enumerate(stream, 1):
                        counts["selected_rows"] += 1
                        sid = scenario_id(revision, name, row_index)
                        seen_ids.add(sid)
                        try:
                            row, signature = _parse_and_convert_row(
                                line, revision=revision, source_file=name, source_row=row_index,
                                source_sha256=checksums[name], review=(reviews or {}).get(sid),
                                lineage=(lineage or {}).get(sid), units=units,
                            )
                            if signature in observed:
                                duplicates.append({"scenario_id": sid,
                                                   "matches": observed[signature]})
                            else:
                                observed[signature] = sid
                            accepted.write(row.model_dump_json() + "\n")
                            counts["accepted"] += 1
                            origins[row.origin] += 1
                            causes[row.rca_label] += 1
                        except (ValueError, TypeError, KeyError, AttributeError) as exc:
                            rejected.write(json.dumps({"source_file": name, "source_row": row_index,
                                "scenario_id": sid, "reason": str(exc), "raw_line": line.rstrip("\n")},
                                ensure_ascii=False) + "\n")
                            counts["rejected"] += 1
        if (set(reviews or {}) | set(lineage or {})) - seen_ids:
            raise ValueError("review or lineage sidecar contains IDs outside the selected source rows")
    except (OSError, ValueError, KeyError) as exc:
        report["error"] = str(exc)
        raise
    finally:
        report.update(counts={key: counts[key] for key in ("selected_rows", "accepted", "rejected")},
                      origins=dict(origins), causes=dict(causes), duplicates=duplicates)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
