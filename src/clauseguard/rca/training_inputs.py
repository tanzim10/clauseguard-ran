"""Task 4.9 inputs: pick one window per eligible event and rebuild its packet and audits.

Selection reads candidate references (label status and causes) only to stratify; references never
enter a packet. Every rebuilt packet must match the hash its candidate recorded in Task 4.6.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from pydantic import Field

from clauseguard.rca.candidate_pool import packet_digest
from clauseguard.rca.candidates import CandidateRecord, CandidateReference, StrictModel
from clauseguard.rca.contracts import EvidencePacket
from clauseguard.rca.sources import SourceEvidence

SELECTION_VERSION = "training-selection-v1"
SELECTION_RULE = (
    "eligible_pilot candidates only; one window per event, the first by sha256(seed:candidate_id); every "
    "labelled-fault event is kept; normal events are added at normal_share of the total, drawn round-robin "
    "across sources in sha256(seed:event_id) order; the pilot takes normal events at the same share and "
    "fault events round-robin across causes in sha256(seed:event_id) order")


class SelectedEvent(StrictModel):
    candidate_id: str
    event_id: str
    source_dataset: str
    group_id: str
    split: Literal["train", "validation"]
    stratum: Literal["fault", "normal"]
    causes: list[str] = Field(default_factory=list)
    pilot: bool
    packet_sha256: str


class TrainingSelection(StrictModel):
    selection_version: Literal["training-selection-v1"] = SELECTION_VERSION
    rule: str = SELECTION_RULE
    candidate_release_id: str
    seed: int
    normal_share: float
    pilot_size: int
    events: list[SelectedEvent]


def _order(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _round_robin(groups: Mapping[str, list[str]], limit: int) -> list[str]:
    """Take one item per group in turn (groups in key order) until ``limit`` items are taken."""
    queues = {key: list(items) for key, items in sorted(groups.items())}
    picked: list[str] = []
    while len(picked) < limit and any(queues.values()):
        for key in list(queues):
            if queues[key] and len(picked) < limit:
                picked.append(queues[key].pop(0))
    return picked


def normal_count(faults: int, share: float) -> int:
    """Normal events so that they are ``share`` of the total, rounded to the nearest whole event."""
    return 0 if share <= 0 else math.floor(faults * share / (1 - share) + 0.5)


def select_events(candidates: Sequence[CandidateRecord], references: Mapping[str, CandidateReference], *,
                  seed: int, normal_share: float, pilot_size: int) -> list[SelectedEvent]:
    """One window per eligible event: all faults, normals at ``normal_share``, and a marked pilot subset."""
    if not 0 <= normal_share < 1:
        raise ValueError("normal_share must be in [0, 1)")
    windows: dict[str, list[CandidateRecord]] = defaultdict(list)
    for candidate in candidates:
        status = references[candidate.identity.candidate_id].label_status
        if candidate.eligibility.status == "eligible_pilot" and status in ("labelled_fault", "labelled_normal"):
            windows[candidate.grouping.event_id].append(candidate)
    chosen = {event: min(items, key=lambda c: _order(seed, c.identity.candidate_id))
              for event, items in windows.items()}
    stratum = {event: "fault" if references[c.identity.candidate_id].label_status == "labelled_fault" else "normal"
               for event, c in chosen.items()}
    ordered = sorted(chosen, key=lambda event: _order(seed, event))
    faults = [e for e in ordered if stratum[e] == "fault"]
    normals_by_source: dict[str, list[str]] = defaultdict(list)
    for event in ordered:
        if stratum[event] == "normal":
            normals_by_source[chosen[event].identity.source_dataset].append(event)
    normals = _round_robin(normals_by_source, normal_count(len(faults), normal_share))

    pilot_normals = _round_robin(normals_by_source, min(len(normals), round(pilot_size * normal_share)))
    by_cause: dict[str, list[str]] = defaultdict(list)
    for event in faults:
        causes = references[chosen[event].identity.candidate_id].causes
        by_cause["+".join(sorted(causes))].append(event)
    pilot = {*pilot_normals, *_round_robin(by_cause, max(0, pilot_size - len(pilot_normals)))}

    selected = []
    for event in [*faults, *normals]:
        candidate = chosen[event]
        if candidate.eligibility.split is None or candidate.observations.packet_sha256 is None:
            raise ValueError(f"eligible candidate without split or packet hash: {candidate.identity.candidate_id}")
        selected.append(SelectedEvent(
            candidate_id=candidate.identity.candidate_id, event_id=event,
            source_dataset=candidate.identity.source_dataset, group_id=candidate.grouping.group_id,
            split=candidate.eligibility.split, stratum=stratum[event],
            causes=list(references[candidate.identity.candidate_id].causes), pilot=event in pilot,
            packet_sha256=candidate.observations.packet_sha256))
    return selected


def rebuild_inputs(selected: Sequence[SelectedEvent], candidates: Mapping[str, CandidateRecord],
                   rebuild: Callable[[CandidateRecord], tuple[EvidencePacket, SourceEvidence]],
                   augment: Callable[[SelectedEvent, EvidencePacket, SourceEvidence], EvidencePacket] | None = None,
                   ) -> tuple[list[dict], list[dict]]:
    """Bare packets and audit rows (keyed by candidate ID) for ``run_rca``.

    The rebuilt packet must match its Task 4.6 hash before ``augment`` (baselines) may change it.
    """
    from clauseguard.rca.evidence import check_audits

    packets, audits = [], []
    for item in selected:
        packet, evidence = rebuild(candidates[item.candidate_id])
        if packet_digest(packet) != item.packet_sha256:
            raise ValueError(f"rebuilt packet differs from its candidate hash: {item.candidate_id}")
        if augment is not None:
            packet = augment(item, packet, evidence)
        case_audits = evidence.registry.audits()
        check_audits(packet, case_audits)
        packets.append(packet.model_dump(mode="json"))
        audits.extend({"case_id": item.candidate_id, **a.model_dump(mode="json")} for a in case_audits)
    return packets, audits
