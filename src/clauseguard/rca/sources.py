"""Source adapters that turn one source window into traceable evidence (Task 4.4).

Adapters read observations and provenance only. Labels, anomaly annotations, descriptions kept
as references, golden answers and review decisions are never read here.
"""

import csv
import hashlib
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from clauseguard.datasets.schema import ScenarioRecord
from clauseguard.rca.contracts import ModalityState, SourceIdentity
from clauseguard.rca.detector import rca_text_eligibility
from clauseguard.rca.evidence import EvidenceRegistry
from clauseguard.rca.summarizer import (
    ChannelRejection,
    Semantics,
    TimeIntegrityError,
    check_timestamps,
    summarize_channel,
    uniform_times,
)

TELECOMTS_TIME_RULE = ("start_time + index / sampling_rate_hz; the release matched every sample "
                       "to the raw metrics.csv timeline at this cadence")
UNAVAILABLE_SOURCES = {
    "TeleLogs": "source files are not available locally; fields, taxonomy and terms are unverified",
    "5G core faults": "source files are not available locally; troubleshooting responses stay "
                      "outside inference observations when added",
}
BUBBLERAN_IDENTIFIERS = {"", "timestamp", "timestamp_iso", "ran_ue_id", "e2node_nb_id", "mac_frame",
                         "mac_slot"}
BUBBLERAN_COUNTER = re.compile(r"(_aggr_|_harq\d|_errors$)")
# Labels stored beside the measurements; their values are never read.
REFERENCE_COLUMNS = {"BubbleRAN": {"persistent_anomaly"}, "SpotLight": {"anomaly"}}


class SourceEvidence(BaseModel):
    """Evidence emitted for one source window, plus every rejected channel and its reason."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    source: SourceIdentity
    kpi_state: ModalityState
    text_state: ModalityState
    text_reason: str | None = None
    rejections: list[ChannelRejection] = Field(default_factory=list)
    time: dict[str, Any] = Field(default_factory=dict)
    registry: EvidenceRegistry


def _add_channels(registry: EvidenceRegistry, source: SourceIdentity, channels: Mapping[str, Sequence],
                  semantics: Mapping[str, Semantics], *, location: dict, times: list[str] | None,
                  time_rule: str | None, timezone: str) -> list[ChannelRejection]:
    rejections = []
    for name in sorted(channels):
        statistics, rejected = summarize_channel(name, channels[name], semantics=semantics[name])
        rejections.extend(rejected)
        for statistic in statistics:
            registry.add_kpi(statistic, source=source, location={**location, "channel": name},
                             times=times, time_rule=time_rule, timezone=timezone)
    return rejections


def telecomts_evidence(record: ScenarioRecord, *, family_id: str | None,
                       family_status: str = "unresolved") -> SourceEvidence:
    """KPI statistics for one TelecomTS window; units stay null because the source has none."""
    window, provenance = record.kpi_window, record.provenance
    source = SourceIdentity(dataset="TelecomTS", revision=provenance.revision, case_id=record.scenario_id,
                            artifact_sha256=provenance.source_sha256, recording_family_id=family_id,
                            recording_family_status=family_status)
    times = uniform_times(window.start_time, window.sampling_rate_hz, window.sample_count)
    if datetime.fromisoformat(times[-1]) != datetime.fromisoformat(window.end_time):
        raise TimeIntegrityError("window end_time disagrees with its start time and cadence")
    times[0], times[-1] = window.start_time, window.end_time  # keep the source's own strings
    registry = EvidenceRegistry()
    channels = {name: channel.values for name, channel in window.channels.items()}
    semantics = {name: "categorical" if channel.kind == "categorical" else "reported_value"
                 for name, channel in window.channels.items()}
    units = {name: channel.unit for name, channel in window.channels.items()}
    if any(units.values()):
        raise ValueError("TelecomTS units are unverified; a unit needs review before use")
    rejections = _add_channels(registry, source, channels, semantics,
                               location={"source_file": provenance.source_file, "source_row": provenance.source_row},
                               times=times, time_rule=TELECOMTS_TIME_RULE, timezone="unknown (naive source timestamps)")
    text = rca_text_eligibility(record)
    if text.state == "eligible":
        registry.add_text(record_id=f"{record.scenario_id}:nl_context", text=record.nl_context, origin=text.origin,
                          source="TelecomTS nl_context", permission=record.text_review.model_dump_json())
    return SourceEvidence(
        source=source, kpi_state="included" if registry.records("kpi") else "missing",
        text_state="included" if text.state == "eligible" else "missing",
        text_reason=None if text.state == "eligible" else f"{text.state}: {text.reason}",
        rejections=rejections, registry=registry,
        time={"start": window.start_time, "end": window.end_time, "rate_hz": window.sampling_rate_hz,
              "rule": TELECOMTS_TIME_RULE, "timezone": "unknown"})


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value: str) -> float | None:
    return None if value is None or not value.strip() else float(value)


def table_evidence(path: Path, *, dataset: str, rows: Sequence[Mapping[str, str]], first_row: int,
                   time_column: str, parse_time, format_time, nominal_period: float | None,
                   semantics: Mapping[str, str], timezone: str) -> SourceEvidence:
    """Evidence from consecutive rows of one CSV stream; time integrity is checked first.

    ``semantics`` maps each column to a ``Semantics`` value or ``reference`` (a label column that
    is skipped without reading its values).
    """
    if not rows:
        raise ValueError("source window has no rows")
    stamps = [parse_time(row[time_column]) for row in rows]
    gaps = check_timestamps(stamps, nominal_period=nominal_period)
    last_row = first_row + len(rows) - 1
    source = SourceIdentity(dataset=dataset, revision=f"local-copy-sha256:{_sha256(path)[:16]}",
                            case_id=f"{Path(path).name}:rows-{first_row}-{last_row}",
                            artifact_sha256=_sha256(path), recording_family_status="unresolved")
    channels, kinds, rejections = {}, {}, []
    for name, kind in semantics.items():
        if kind == "reference":
            rejections.append(ChannelRejection(channel=name, scope="channel", reason="reference_label_excluded",
                                               detail={"note": "label column; values are never read"}))
            continue
        if kind in ("identifier", "categorical"):
            rejections.extend(summarize_channel(name, [row[name] for row in rows], semantics=kind)[1])
            continue
        try:
            channels[name], kinds[name] = [_number(row[name]) for row in rows], kind
        except ValueError:
            rejections.append(ChannelRejection(channel=name, scope="channel", reason="non_numeric_value",
                                               sample_start=0, sample_end=len(rows) - 1))
    times = [format_time(row[time_column]) for row in rows]
    registry = EvidenceRegistry()
    rejections.extend(_add_channels(registry, source, channels, kinds,
                                    location={"file": Path(path).name, "first_row": first_row},
                                    times=times, time_rule=f"{time_column} column", timezone=timezone))
    return SourceEvidence(source=source, kpi_state="included" if registry.records("kpi") else "missing",
                          text_state="missing", text_reason="source has no reviewed text",
                          rejections=rejections, registry=registry,
                          time={"column": time_column, "gaps": gaps, "nominal_period_s": nominal_period,
                                "timezone": timezone})


def _read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("missing or duplicate CSV columns")
        return list(reader.fieldnames), list(reader)


def bubbleran_evidence(path: Path, *, start_row: int, end_row: int) -> SourceEvidence:
    """One UE on one gNB over consecutive 1 s rows; cumulative counters are rejected, not differenced."""
    columns, rows = _read_rows(path)
    window = rows[start_row:end_row + 1]
    if len(window) != end_row - start_row + 1:
        raise ValueError("row range lies outside the file")
    if len({(row["ran_ue_id"], row["e2node_nb_id"]) for row in window}) != 1:
        raise ValueError("a BubbleRAN window must cover one UE on one gNB")
    semantics = {name: ("reference" if name in REFERENCE_COLUMNS["BubbleRAN"]
                        else "identifier" if name in BUBBLERAN_IDENTIFIERS
                        else "cumulative_counter" if BUBBLERAN_COUNTER.search(name) else "reported_value")
                 for name in columns}

    def parse(value):
        return datetime.fromtimestamp(int(value), tz=UTC)

    return table_evidence(path, dataset="BubbleRAN", rows=window, first_row=start_row,
                          time_column="timestamp", nominal_period=1.0, semantics=semantics,
                          parse_time=parse, format_time=lambda v: parse(v).isoformat(sep=" "),
                          timezone="UTC (Unix seconds)")


def spotlight_evidence(path: Path, *, start_time: str, end_time: str) -> SourceEvidence:
    """One SpotLight stream (radio or platform); streams are never combined without verified alignment.

    A repeated header row starts a new capture; a window may not cross it. Timestamps keep their
    source strings (nanoseconds); ordering checks use microsecond precision.
    """
    columns, rows = _read_rows(path)
    capture, data = 0, []
    for index, row in enumerate(rows):
        if row["timestamp"] == "timestamp":
            capture += 1
            continue
        data.append((index, capture, datetime.fromisoformat(row["timestamp"])))
    begin, finish = datetime.fromisoformat(start_time), datetime.fromisoformat(end_time)
    selected = [(index, part) for index, part, stamp in data if begin <= stamp <= finish]
    if not selected:
        raise ValueError("no rows inside the requested interval")
    if len({part for _, part in selected}) != 1:
        raise TimeIntegrityError("interval crosses a repeated header (capture boundary)")
    indices = [index for index, _ in selected]
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise TimeIntegrityError("rows inside the interval are not contiguous in file order")
    semantics = {name: ("reference" if name in REFERENCE_COLUMNS["SpotLight"]
                        else "identifier" if name in ("", "timestamp") else "reported_value")
                 for name in columns}
    return table_evidence(path, dataset="SpotLight", rows=[rows[i] for i in indices], first_row=indices[0],
                          time_column="timestamp", nominal_period=None, semantics=semantics,
                          parse_time=datetime.fromisoformat, format_time=lambda v: v,
                          timezone="unknown (naive source timestamps)")


def unavailable_source(dataset: str) -> dict:
    if dataset not in UNAVAILABLE_SOURCES:
        raise ValueError(f"{dataset} has an adapter")
    return {"dataset": dataset, "status": "unavailable", "reason": UNAVAILABLE_SOURCES[dataset]}
