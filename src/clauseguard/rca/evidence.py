"""Traceable evidence records and the registry that assigns prompt citation IDs (Task 4.4).

Each emitted record gets a short prompt ID (``K1``, ``T1``, ``S1``) mapped to one persistent
source ID, plus an audit record with the source revision, artifact hash, location and
calculation settings needed to reproduce it. Duplicate IDs and conflicting mappings are errors.
"""

import hashlib
import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from clauseguard.rca.contracts import (
    DetectorPrediction,
    EvidencePacket,
    KpiStatisticEvidence,
    ModalityState,
    SourceIdentity,
    SpecificationEvidence,
    TextEvidence,
)
from clauseguard.rca.summarizer import (
    CALCULATION_VERSION,
    EXTREMUM_RULE,
    MIN_VALID_COUNT,
    MISSING_RULE,
    ChannelStatistic,
)

PREFIXES = {"kpi": "K", "text": "T", "specification": "S"}
BASELINE_RULES = {
    "window_first_half": "samples 0..n//2-1 of the same source window, compared with samples n//2..n-1",
}


class EvidenceAudit(BaseModel):
    """Everything needed to trace and reproduce one emitted record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    persistent_id: str
    kind: Literal["kpi", "text", "specification"]
    source: dict[str, Any]
    calculation: dict[str, Any] | None = None
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BaselineVerification(BaseModel):
    """Why a baseline is comparable; ``kind`` keeps an earlier interval apart from verified normal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["earlier_interval", "verified_normal"]
    interval_rule: str
    status: Literal["verified", "unverified"]
    verified_by: str | None = None
    evidence: str | None = None


class EvidenceError(ValueError):
    """An evidence ID, mapping or baseline cannot be attributed safely."""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def relative_change(baseline: float, value: float) -> float | None:
    """Fractional change; undefined (``None``) when the baseline is zero."""
    if not (math.isfinite(baseline) and math.isfinite(value)):
        raise EvidenceError("relative change needs finite values")
    return None if baseline == 0 else (value - baseline) / abs(baseline)


class EvidenceRegistry:
    def __init__(self):
        self._records: dict[str, BaseModel] = {}
        self._audits: dict[str, EvidenceAudit] = {}
        self._by_persistent: dict[str, str] = {}
        self._counts = dict.fromkeys(PREFIXES, 0)

    def _register(self, kind: str, persistent_id: str, build, source: dict, calculation: dict | None,
                  evidence_id: str | None = None):
        if persistent_id in self._by_persistent:
            existing = self._by_persistent[persistent_id]
            candidate = build(existing)
            if candidate != self._records[existing]:
                raise EvidenceError(f"conflicting records for {persistent_id}")
            return self._records[existing]
        if evidence_id is None:
            self._counts[kind] += 1
            evidence_id = f"{PREFIXES[kind]}{self._counts[kind]}"
        if evidence_id in self._records:
            raise EvidenceError(f"duplicate evidence ID {evidence_id}")
        record = build(evidence_id)
        self._records[evidence_id] = record
        self._by_persistent[persistent_id] = evidence_id
        self._audits[evidence_id] = EvidenceAudit(
            evidence_id=evidence_id, persistent_id=persistent_id, kind=kind, source=source,
            calculation=calculation, content_sha256=_digest(record.model_dump(mode="json")))
        return record

    def add_kpi(self, statistic: ChannelStatistic, *, source: SourceIdentity, location: dict,
                unit: str | None = None, times: list[str] | None = None, time_rule: str | None = None,
                timezone: str = "unknown") -> KpiStatisticEvidence:
        start, end = statistic.sample_start, statistic.sample_end
        persistent_id = (f"{source.dataset}:{source.case_id}:{statistic.channel}:{statistic.statistic}:"
                         f"{start}-{end}:{CALCULATION_VERSION}")

        def build(evidence_id):
            return KpiStatisticEvidence(
                evidence_id=evidence_id, source_window_id=f"{source.dataset}:{source.case_id}",
                channel=statistic.channel, statistic=statistic.statistic, value=statistic.value, unit=unit,
                sample_start=start, sample_end=end,
                time_start=times[start] if times else None, time_end=times[end] if times else None,
                valid_count=statistic.valid_count, sample_count=statistic.sample_count,
                missing_fraction=statistic.missing_fraction, calculation_version=CALCULATION_VERSION)

        extremum = statistic.extremum_index
        calculation = {
            "version": CALCULATION_VERSION, "statistic": statistic.statistic,
            "sample_range": [start, end], "missing_rule": MISSING_RULE,
            "min_valid_count": MIN_VALID_COUNT[statistic.statistic],
            "non_finite_count": statistic.non_finite_count, "imputation": "none",
            "time_rule": time_rule, "timezone": timezone,
            **({"extremum_index": extremum, "extremum_rule": EXTREMUM_RULE,
                "extremum_time": times[extremum] if times else None} if extremum is not None else {}),
        }
        return self._register("kpi", persistent_id, build,
                              {**source.model_dump(mode="json"), "location": location}, calculation)

    def add_scalar(self, *, channel: str, value: float, source: SourceIdentity, location: dict,
                   unit: str | None = None) -> KpiStatisticEvidence:
        """A single reported measurement: no sample range or time is invented for it."""
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise EvidenceError(f"scalar {channel} must be a finite number")
        persistent_id = f"{source.dataset}:{source.case_id}:{channel}:scalar:{CALCULATION_VERSION}"

        def build(evidence_id):
            return KpiStatisticEvidence(
                evidence_id=evidence_id, source_window_id=f"{source.dataset}:{source.case_id}", channel=channel,
                statistic="scalar", value=float(value), unit=unit, valid_count=1, sample_count=1,
                missing_fraction=0.0, calculation_version=CALCULATION_VERSION)

        return self._register("kpi", persistent_id, build, {**source.model_dump(mode="json"), "location": location},
                              {"version": CALCULATION_VERSION, "statistic": "scalar",
                               "note": "reported value; no sample range, time or missing fraction is derived"})

    def add_text(self, *, record_id: str, text: str, origin: str, source: str, permission: str,
                 timestamp: str | None = None) -> TextEvidence:
        def build(evidence_id):
            return TextEvidence(evidence_id=evidence_id, record_id=record_id, text=text, origin=origin,
                                approval_status="approved_for_rca", source=source, timestamp=timestamp)

        return self._register("text", f"text:{record_id}:{_digest(text)}", build,
                              {"source": source, "record_id": record_id, "rca_permission": permission}, None)

    def add_specification(self, *, chunk_id: str, document_id: str, version: str, clause: str,
                          chunk_text: str, span: tuple[int, int] | None = None,
                          evidence_id: str | None = None) -> SpecificationEvidence:
        """Supply a whole chunk or an exact span of it; a span keeps its link to the original chunk."""
        if span is None:
            text = chunk_text
        else:
            begin, finish = span
            if not 0 <= begin < finish <= len(chunk_text):
                raise EvidenceError("excerpt span lies outside the retrieved chunk")
            text = chunk_text[begin:finish]
        content = hashlib.sha256(text.encode()).hexdigest()

        def build(evidence_id):
            return SpecificationEvidence(evidence_id=evidence_id, chunk_id=chunk_id, document_id=document_id,
                                         version=version, clause=clause, text=text, content_sha256=content)

        source = {"chunk_id": chunk_id, "document_id": document_id, "version": version,
                  "chunk_sha256": hashlib.sha256(chunk_text.encode()).hexdigest(),
                  "shortened": span is not None, "span": list(span) if span else None}
        return self._register("specification", f"spec:{chunk_id}:{content}", build, source, None,
                              evidence_id=evidence_id)

    def attach_baseline(self, incident_id: str, baseline_id: str,
                        verification: BaselineVerification) -> KpiStatisticEvidence:
        """Link a comparable, verified baseline; anything else keeps ``baseline_evidence_id`` null."""
        incident, baseline = self._records.get(incident_id), self._records.get(baseline_id)
        if not isinstance(incident, KpiStatisticEvidence) or not isinstance(baseline, KpiStatisticEvidence):
            raise EvidenceError("baselines link two KPI records in the same registry")
        if verification.status != "verified":
            raise EvidenceError("baseline is unverified; baseline stays null")
        if (baseline.channel, baseline.statistic, baseline.unit, baseline.calculation_version) != (
                incident.channel, incident.statistic, incident.unit, incident.calculation_version):
            raise EvidenceError("baseline channel, statistic, unit or calculation differs")
        if verification.kind == "earlier_interval":
            if verification.interval_rule not in BASELINE_RULES:
                raise EvidenceError("earlier-interval baselines need a fixed rule available at inference")
            if (baseline.source_window_id != incident.source_window_id
                    or baseline.sample_end is None or incident.sample_start is None
                    or baseline.sample_end >= incident.sample_start):
                raise EvidenceError("an earlier interval must end before the incident in the same window")
        elif not (verification.verified_by and verification.evidence):
            raise EvidenceError("verified normal operation needs a reviewer and evidence")
        linked = incident.model_copy(update={"baseline_evidence_id": baseline_id})
        self._records[incident_id] = linked
        audit = self._audits[incident_id]
        self._audits[incident_id] = audit.model_copy(update={
            "calculation": {**audit.calculation, "baseline": verification.model_dump(mode="json")},
            "content_sha256": _digest(linked.model_dump(mode="json"))})
        return linked

    def records(self, kind: str) -> list:
        return [self._records[eid] for eid, audit in self._audits.items() if audit.kind == kind]

    def audits(self) -> list[EvidenceAudit]:
        return list(self._audits.values())

    def packet(self, *, source: SourceIdentity, question: str, kpi_state: ModalityState,
               text_state: ModalityState, retrieval_status: str,
               detector_prediction: DetectorPrediction | None = None) -> EvidencePacket:
        packet = EvidencePacket(
            source=source, question=question, detector_prediction=detector_prediction,
            kpi_state=kpi_state, text_state=text_state, kpi_evidence=self.records("kpi"),
            text_evidence=self.records("text"), specification_evidence=self.records("specification"),
            retrieval_status=retrieval_status)
        check_packet_ids(packet)
        return packet


def check_packet_ids(packet: EvidencePacket) -> None:
    """Every citation ID is unique and every baseline points at a KPI record in the packet."""
    ids = [item.evidence_id for item in
           (*packet.kpi_evidence, *packet.text_evidence, *packet.specification_evidence)]
    duplicates = sorted({eid for eid in ids if ids.count(eid) > 1})
    if duplicates:
        raise EvidenceError(f"duplicate evidence IDs: {duplicates}")
    kpi_ids = {item.evidence_id for item in packet.kpi_evidence}
    for item in packet.kpi_evidence:
        if item.baseline_evidence_id is not None and item.baseline_evidence_id not in kpi_ids:
            raise EvidenceError(f"{item.evidence_id} cites a baseline outside the packet")


def check_audits(packet: EvidencePacket, audits: list[EvidenceAudit]) -> None:
    """Audits map each packet ID to one persistent source and match the record content."""
    by_id = {audit.evidence_id: audit for audit in audits}
    persistent = [audit.persistent_id for audit in audits]
    if len(by_id) != len(audits) or len(set(persistent)) != len(persistent):
        raise EvidenceError("audit records contain duplicate IDs or conflicting mappings")
    for item in (*packet.kpi_evidence, *packet.text_evidence, *packet.specification_evidence):
        audit = by_id.get(item.evidence_id)
        if audit is None or audit.content_sha256 != _digest(item.model_dump(mode="json")):
            raise EvidenceError(f"{item.evidence_id} has no matching audit record")
