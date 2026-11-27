"""Task 4.9 baselines: add comparison records to training packets with fixed, label-blind rules.

Without a baseline a target cannot claim a change (#51, rca-contract.md). Rules here never read labels,
anomaly bounds or references; they use only the source location of the window, which is known at
inference:

- BubbleRAN: ``preceding_window`` - the equal-length window that ends just before, in the same file.
- TelecomTS: ``same_zone_application_normal`` - one train-split window from the normal recording of the
  same zone and application (no-congestion first, then congestion), first by sha256(seed:candidate_id).
- Anything else (TelecomTS mobile or jammer captures, which name no zone) keeps a null baseline.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from clauseguard.rca.candidates import CandidateRecord, StrictModel
from clauseguard.rca.contracts import EvidencePacket
from clauseguard.rca.evidence import BaselineVerification, EvidenceError
from clauseguard.rca.sources import SourceEvidence
from clauseguard.rca.summarizer import ChannelStatistic

BASELINE_SELECTION_VERSION = "training-baselines-v1"
NORMAL_PROFILE_RULE = ("same_zone_application_normal: one train-split eligible window of the normal recording "
                       "normal/stationary/<zone>/<no_congestion|congestion>/<application> matching the incident's "
                       "zone and application, no_congestion preferred, first by sha256(seed:candidate_id)")
REVIEW_NOTE = "automatic compatibility check (same channel, statistic, unit, calculation); review pending"
_ZONE_APP = re.compile(r"/(Zone_[A-Z])/(?:(?:no_)?congestion/)?([^/]+)/processed/")

Rebuild = Callable[[CandidateRecord], tuple[EvidencePacket, SourceEvidence]]


class BaselinePlan(StrictModel):
    candidate_id: str
    rule: Literal["preceding_window", "same_zone_application_normal", "none"]
    baseline_candidate_id: str | None = None
    baseline_location: dict | None = None
    reason: str


def zone_application(file: str) -> tuple[str, str] | None:
    match = _ZONE_APP.search("/" + file)
    return (match.group(1), match.group(2)) if match else None


def plan_baselines(selected_ids: Sequence[str], candidates: Mapping[str, CandidateRecord], *,
                   seed: int) -> list[BaselinePlan]:
    """Choose a baseline source for each selected window from its location only."""
    normals: dict[tuple[str, str, bool], list[str]] = {}
    for cid, c in candidates.items():
        file = c.identity.source_location.get("file", "")
        if (c.identity.source_dataset == "TelecomTS" and file.startswith("normal/stationary/")
                and c.eligibility.status == "eligible_pilot" and c.eligibility.split == "train"):
            key = zone_application(file)
            if key:
                normals.setdefault((*key, "/no_congestion/" in file), []).append(cid)
    plans = []
    for cid in selected_ids:
        c = candidates[cid]
        loc = c.identity.source_location
        if c.identity.source_dataset == "BubbleRAN":
            length = loc["last_row"] - loc["first_row"] + 1
            if loc["first_row"] >= length:
                plans.append(BaselinePlan(candidate_id=cid, rule="preceding_window", reason="earlier rows exist",
                                          baseline_location={"file": loc["file"], "first_row": loc["first_row"] - length,
                                                             "last_row": loc["first_row"] - 1}))
            else:
                plans.append(BaselinePlan(candidate_id=cid, rule="none", reason="no earlier rows in the file"))
            continue
        key = zone_application(loc.get("file", "")) if c.identity.source_dataset == "TelecomTS" else None
        pool = [x for x in (normals.get((*key, True), []) or normals.get((*key, False), [])) if x != cid] if key \
            else []
        if not pool:
            plans.append(BaselinePlan(candidate_id=cid, rule="none",
                                      reason="no zone named in the location" if not key
                                      else "no train-split normal recording for this zone and application"))
            continue
        pick = min(pool, key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())
        plans.append(BaselinePlan(candidate_id=cid, rule="same_zone_application_normal", baseline_candidate_id=pick,
                                  baseline_location=candidates[pick].identity.source_location,
                                  reason=f"normal recording {candidates[pick].identity.source_location['file']}"))
    return plans


def _statistic(record, calculation: dict) -> ChannelStatistic:
    return ChannelStatistic(
        channel=record.channel, statistic=record.statistic, value=record.value, sample_start=record.sample_start,
        sample_end=record.sample_end, valid_count=record.valid_count, sample_count=record.sample_count,
        missing_fraction=record.missing_fraction, non_finite_count=calculation.get("non_finite_count", 0),
        extremum_index=calculation.get("extremum_index"))


def apply_baseline(packet: EvidencePacket, evidence: SourceEvidence, baseline: SourceEvidence,
                   plan: BaselinePlan) -> tuple[EvidencePacket, int]:
    """Add the baseline window's KPI records and link every matching incident record; returns links made."""
    registry = evidence.registry
    if plan.rule == "preceding_window":
        verification = BaselineVerification(kind="earlier_interval", interval_rule="preceding_window",
                                            status="verified", verified_by=REVIEW_NOTE)
    else:
        verification = BaselineVerification(kind="verified_normal", interval_rule=NORMAL_PROFILE_RULE,
                                            status="verified", verified_by=REVIEW_NOTE, evidence=plan.reason)
    incidents = {(r.channel, r.statistic, r.unit): r.evidence_id for r in registry.records("kpi")}
    audits = {a.evidence_id: a for a in baseline.registry.audits()}
    links = 0
    for record in baseline.registry.records("kpi"):
        incident_id = incidents.get((record.channel, record.statistic, record.unit))
        if incident_id is None or record.statistic == "scalar":
            continue
        calc = audits[record.evidence_id].calculation or {}
        added = registry.add_kpi(_statistic(record, calc), source=baseline.source,
                                 location={"baseline_for": packet.source.case_id, "rule": plan.rule,
                                           "channel": record.channel},
                                 unit=record.unit, times=_times(record), time_rule=calc.get("time_rule"),
                                 timezone=calc.get("timezone", "unknown"))
        try:
            registry.attach_baseline(incident_id, added.evidence_id, verification)
        except EvidenceError:
            continue
        links += 1
    rebuilt = registry.packet(source=packet.source, question=packet.question, kpi_state=packet.kpi_state,
                              text_state=packet.text_state, retrieval_status=packet.retrieval_status,
                              detector_prediction=packet.detector_prediction)
    return rebuilt, links


def _times(record) -> list[str] | None:
    """Rebuild a time list that reproduces the record's own start and end times at its sample bounds."""
    if record.time_start is None or record.sample_start is None or record.sample_end is None:
        return None
    times = [record.time_start] * (record.sample_end + 1)
    times[record.sample_end] = record.time_end
    return times
