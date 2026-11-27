"""Grouping, duplicate checks, eligibility, grouped splits, batch and manifest (Task 4.6)."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from clauseguard.datasets.window_overlap import WindowSpan
from clauseguard.rca.candidate_pool import (
    ADAPTER_VERSION,
    GROUPING_VERSION,
    BuiltCandidate,
    CandidateConfig,
    Reservations,
    canonical_sha256,
    jaccard,
    load_reservations,
    packet_digest,
    shingles,
    write_json,
)
from clauseguard.rca.candidates import (
    CANDIDATE_VERSION,
    REFERENCE_VERSION,
    STATUS_ORDER,
    BatchRecord,
    CandidateManifest,
    CandidateRecord,
    CandidateReference,
    Check,
    Eligibility,
    EligibilityStatus,
    FileEntry,
    SourceInventory,
)
from clauseguard.rca.contracts import EvidencePacket
from clauseguard.rca.summarizer import CALCULATION_VERSION

DEDUP_VERSION = "candidate-dedup-v1"
EVENT_RULE = ("windows of one source file with the same labels whose raw-row intervals overlap form one "
              "event; events never cross source files or labels")


def _event_id(source: str, label: tuple, start: int, end: int) -> str:
    return "event-" + hashlib.sha256(f"{source}:{label}:{start}-{end}".encode()).hexdigest()[:16]


def _overlap_runs(spans: Sequence[WindowSpan]) -> list[list[WindowSpan]]:
    runs, current, end = [], [], -1
    for span in sorted(spans, key=lambda s: (s.start, s.end, s.scenario_id)):
        if current and span.start > end:
            runs.append(current)
            current = []
        current.append(span)
        end = max(end, span.end)
    return [*runs, current] if current else runs


def telecomts_events(spans: Sequence[WindowSpan]) -> tuple[dict[str, str], list[list[str]]]:
    """Event per window, plus groups of events whose windows share raw rows across labels.

    Overlapping events with different labels stay separate events but share raw rows; they are in
    one recording family, so a family-level split keeps them together.
    """
    by_source = defaultdict(list)
    for span in spans:
        by_source[span.source].append(span)
    events, shared = {}, []
    for source, items in sorted(by_source.items()):
        by_label = defaultdict(list)
        for span in items:
            by_label[(span.detection_target, span.category_target)].append(span)
        for label, group in by_label.items():
            for run in _overlap_runs(group):
                event = _event_id(source, label, run[0].start, max(s.end for s in run))
                events.update((s.scenario_id, event) for s in run)
        for run in _overlap_runs(items):
            linked = sorted({events[s.scenario_id] for s in run})
            if len(linked) > 1:
                shared.append(linked)
    return events, sorted(shared)


def assign_events(built: Iterable[BuiltCandidate], events: Mapping[str, str]) -> list[BuiltCandidate]:
    out = []
    for item in built:
        event = events.get(item.candidate.identity.source_record_id)
        if event is None:
            raise ValueError(f"no event for {item.candidate.identity.candidate_id}")
        grouping = item.candidate.grouping.model_copy(update={"event_id": event})
        out.append(item._replace(candidate=item.candidate.model_copy(update={"grouping": grouping})))
    return out


def duplicate_checks(built: Sequence[BuiltCandidate], reservations: Reservations,
                     config: CandidateConfig, *, shared_row_events: Iterable[list[str]] = ()) -> dict:
    """Exact observation duplicates, near-duplicate text, and matches with reserved evaluation text.

    Duplicates inside one event or one family stay together through grouping. Any other match is
    unresolved and its candidates are quarantined. Reserved-text matches are excluded later.
    """
    quarantine: dict[str, list[str]] = defaultdict(list)
    exact = defaultdict(list)
    for item in built:
        digest = item.candidate.observations.observation_sha256
        if digest is not None:
            exact[digest].append(item.candidate)
    exact_groups = []
    for digest, members in sorted(exact.items()):
        if len(members) < 2:
            continue
        families = {(c.identity.source_dataset, c.grouping.group_id) for c in members}
        resolution = ("same_event" if len({c.grouping.event_id for c in members}) == 1
                      else "same_family" if len(families) == 1 else "unresolved_cross_group")
        ids = sorted(c.identity.candidate_id for c in members)
        exact_groups.append({"observation_sha256": digest, "candidates": ids, "resolution": resolution})
        if resolution == "unresolved_cross_group":
            for cid in ids:
                quarantine[cid].append(f"exact observation duplicate across groups ({digest[:12]})")
    threshold, size = config.near_duplicate_threshold, config.shingle_size
    reserved_text = {}
    texts = [(item.candidate, text) for item in built for text in item.texts]
    for candidate, text in texts:
        matches = reservations.text_matches(text, threshold=threshold, size=size)
        if matches:
            reserved_text.setdefault(candidate.identity.candidate_id, []).extend(matches)
    near_text = []
    grams = [shingles(text, size) for _, text in texts]
    for i, (left, _) in enumerate(texts):
        for j in range(i + 1, len(texts)):
            right = texts[j][0]
            if left.identity.candidate_id == right.identity.candidate_id or jaccard(grams[i], grams[j]) < threshold:
                continue
            same = (left.identity.source_dataset, left.grouping.group_id) == (
                right.identity.source_dataset, right.grouping.group_id)
            pair = sorted([left.identity.candidate_id, right.identity.candidate_id])
            near_text.append({"candidates": pair, "resolution": "same_family" if same else "unresolved_cross_group"})
            if not same:
                for cid in pair:
                    quarantine[cid].append("near-duplicate text across groups")
    templates = {key: reservations.text_matches(text, threshold=threshold, size=size)
                 for key, text in config.question_templates.items()}
    return {
        "version": DEDUP_VERSION, "event_rule": EVENT_RULE,
        "near_text_rule": f"character {size}-gram Jaccard >= {threshold} after normalization (threshold not tuned)",
        "candidates_checked": len(built), "texts_checked": len(texts),
        "exact_duplicate_groups": exact_groups, "near_duplicate_text": near_text,
        "reserved_text_matches": {k: sorted(set(v)) for k, v in sorted(reserved_text.items())},
        "question_template_matches": templates,
        "events_sharing_raw_rows": [list(group) for group in shared_row_events],
        "quarantined": {k: v for k, v in sorted(quarantine.items())},
    }


# --- Eligibility and grouped train/validation split --------------------------------------------

CHECK_VERSION = "1"
SPLIT_RULE = ("families are split, never rows: within each (source, origin) stratum, families are "
              "ordered by sha256(seed:family) and the first round(n * fraction) go to validation; a "
              "stratum with one family stays in train")


def _status(statuses: Iterable[EligibilityStatus]) -> EligibilityStatus:
    return min(statuses, key=STATUS_ORDER.index, default="eligible_pilot")


def _eligibility(candidate: CandidateRecord, terms: SourceInventory, reservations: Reservations,
                 dedup: Mapping, config: CandidateConfig) -> Eligibility:
    cid = candidate.identity.candidate_id
    checks, statuses, blockers, exclusions = [], [], [], []

    def check(check_id, failed_status, reasons, *, passed_reasons=(), blocker=None, exclude=False):
        checks.append(Check(check_id=check_id, version=CHECK_VERSION, status="failed" if failed_status else "passed",
                            reasons=list(reasons) if failed_status else list(passed_reasons)))
        if failed_status:
            statuses.append(failed_status)
            if blocker:
                blockers.append(blocker)
            if exclude:
                exclusions.extend(reasons)

    reserved = [*reservations.reasons(family=candidate.grouping.group_id,
                                      record_id=candidate.identity.source_record_id),
                *(f"matches reserved text {t}" for t in dedup["reserved_text_matches"].get(cid, []))]
    check("not_reserved", "ineligible" if reserved else None, reserved, exclude=True)
    terms_status = {"cleared": None, "pending": config.pending_terms_status, "blocked": "ineligible"}[
        terms.training_use]
    check("terms_cleared", terms_status, terms.blockers, passed_reasons=[terms.licence or "cleared"],
          blocker=f"training_terms_{terms.training_use}")
    check("observations_built", "ineligible" if candidate.observations.packet_status != "built" else None,
          [candidate.observations.not_built_reason or "packet not built"], exclude=True)
    check("family_resolved", "exploratory" if candidate.grouping.group_status != "verified" else None,
          [f"recording family {candidate.grouping.group_id} is {candidate.grouping.group_status}"],
          blocker="recording_family_unresolved")
    check("event_grouped", "ineligible" if candidate.grouping.event_id is None else None,
          ["no event assigned"], exclude=True)
    quarantined = dedup["quarantined"].get(cid, [])
    check("duplicates_resolved", "quarantined" if quarantined else None, quarantined, exclude=True)
    prediction = candidate.prediction_ref
    check("prediction_reference", None, [], passed_reasons=[
        prediction.status if prediction.status == "available" else f"absent: {prediction.absence_reason}"])
    return Eligibility(status=_status(statuses), checks=checks, blockers=sorted(set(blockers)),
                       exclusion_reasons=exclusions)


def evaluate_eligibility(built: Sequence[BuiltCandidate], *, inventory: Mapping[str, SourceInventory],
                         reservations: Reservations, dedup: Mapping, config: CandidateConfig) -> list[BuiltCandidate]:
    """Run every admission check; the most restrictive outcome sets the candidate status."""
    out = []
    for item in built:
        candidate = item.candidate
        eligibility = _eligibility(candidate, inventory[candidate.identity.source_dataset], reservations, dedup, config)
        out.append(item._replace(candidate=candidate.model_copy(update={"eligibility": eligibility})))
    return out


def _origin(item: BuiltCandidate) -> str:
    return str(item.reference.annotation.get("origin") or item.reference.label_status)


def assign_splits(built: Sequence[BuiltCandidate], config: CandidateConfig) -> tuple[list[BuiltCandidate], dict]:
    """Grouped train/validation split over admissible candidates; ineligible ones get no split."""
    admissible = [i for i in built if i.candidate.eligibility.status in ("eligible_pilot", "exploratory")]
    origins: dict[tuple[str, str], set[str]] = defaultdict(set)
    family_of = {}
    for item in admissible:
        family = (item.candidate.identity.source_dataset, item.candidate.grouping.group_id)
        origins[family].add(_origin(item))
        family_of[item.candidate.identity.candidate_id] = family
    family_strata: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for family, values in origins.items():  # a family with mixed origins forms its own stratum
        family_strata[(family[0], "+".join(sorted(values)))].add(family)
    split_of, summary = {}, {}
    for stratum, families in sorted(family_strata.items()):
        ordered = sorted(families, key=lambda f: hashlib.sha256(f"{config.split_seed}:{f[1]}".encode()).hexdigest())
        count = min(round(len(ordered) * config.validation_fraction), len(ordered) - 1) if len(ordered) > 1 else 0
        split_of.update((f, "validation") for f in ordered[:count])
        split_of.update((f, "train") for f in ordered[count:])
        summary[f"{stratum[0]}/{stratum[1]}"] = {"families": len(ordered), "validation_families": count}
    out = []
    for item in built:
        family = family_of.get(item.candidate.identity.candidate_id)
        if family is None:
            out.append(item)
            continue
        eligibility = item.candidate.eligibility.model_copy(update={"split": split_of[family]})
        out.append(item._replace(candidate=item.candidate.model_copy(update={"eligibility": eligibility})))
    rows = defaultdict(int)
    for item in out:
        rows[item.candidate.eligibility.split or "none"] += 1
    return out, {"rule": SPLIT_RULE, "seed": config.split_seed, "validation_fraction": config.validation_fraction,
                 "strata": summary, "rows": dict(sorted(rows.items()))}


# --- Batch selection ---------------------------------------------------------------------------

BATCH_RULE = ("deterministic and stratified: at most one window per event; TelecomTS synthetic faults "
              "up to batch_per_category events per source cause, then TelecomTS real faults (jamming), "
              "labelled-normal windows (distinct families first) and BubbleRAN windows up to their quotas. "
              "Within each stratum round(quota * batch_correction_share) events are cases where the packet "
              "prediction disagrees with the source label and the rest agree; a side that runs short is "
              "filled from the other. Ties follow sha256(seed:id); truncated to batch_size")


def detector_disagrees(item: BuiltCandidate) -> bool:
    """True when the packet prediction names something other than the source label (TelecomTS only)."""
    ref = item.candidate.prediction_ref
    head = ref.heads.get(ref.packet_head)
    if ref.status != "available" or head is None or head.status != "ok":
        return False
    expected = item.reference.causes or ["normal"]
    return head.predicted_label not in expected


def _order(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _pick_events(items: Sequence[BuiltCandidate], limit: int, seed: int, *, correction_share: float,
                 spread_families: bool = False) -> list[BuiltCandidate]:
    """One window per event, at most ``limit`` events, mixing detector corrections with agreements."""
    events = defaultdict(list)
    for item in items:
        events[item.candidate.grouping.event_id].append(item)
    disagree, agree = [], []
    for event in sorted(events, key=lambda e: _order(seed, e or "")):
        members = sorted(events[event], key=lambda i: _order(seed, i.candidate.identity.candidate_id))
        for side, wanted in ((disagree, True), (agree, False)):
            match = next((m for m in members if detector_disagrees(m) == wanted), None)
            if match is not None:
                side.append(match)
    wanted_corrections = round(limit * correction_share)
    order = [*disagree[:wanted_corrections], *agree, *disagree[wanted_corrections:]]
    chosen, seen_events, families = [], set(), set()
    for one_per_family in ((True, False) if spread_families else (False,)):
        for item in order:
            event, family = item.candidate.grouping.event_id, item.candidate.grouping.group_id
            if len(chosen) == limit or event in seen_events or (one_per_family and family in families):
                continue
            chosen.append(item)
            seen_events.add(event)
            families.add(family)
    return chosen


def select_batch(built: Sequence[BuiltCandidate], config: CandidateConfig) -> tuple[list[dict], dict]:
    """The initial review batch and the strata it could not fill."""
    pool = [i for i in built if i.candidate.eligibility.status in config.batch_statuses]
    tele = [i for i in pool if i.candidate.identity.source_dataset == "TelecomTS"]
    strata: list[tuple[str, list[BuiltCandidate], int, bool]] = []
    synthetic = [i for i in tele if i.reference.annotation.get("origin") == "synthetic"]
    for cause in sorted({c for i in synthetic for c in i.reference.causes}):
        strata.append((f"TelecomTS/synthetic/{cause}", [i for i in synthetic if cause in i.reference.causes],
                       config.batch_per_category, False))
    strata.append(("TelecomTS/real", [i for i in tele if i.reference.annotation.get("origin") == "real"],
                   config.batch_jamming, False))
    strata.append(("labelled_normal", [i for i in pool if i.reference.label_status == "labelled_normal"
                                       and i.candidate.identity.source_dataset == "TelecomTS"],
                   config.batch_normal, True))
    strata.append(("BubbleRAN", [i for i in pool if i.candidate.identity.source_dataset == "BubbleRAN"],
                   config.batch_bubbleran, False))
    rows, shortfalls, events = [], {}, set()
    for name, items, quota, spread in strata:
        items = [i for i in items if i.candidate.grouping.event_id not in events]
        picked = _pick_events(items, quota, config.batch_seed, correction_share=config.batch_correction_share,
                              spread_families=spread)
        if len(picked) < quota:
            shortfalls[name] = {"quota": quota, "selected": len(picked), "available_events":
                                len({i.candidate.grouping.event_id for i in items})}
        for item in picked:
            events.add(item.candidate.grouping.event_id)
            candidate = item.candidate
            rows.append({"candidate_id": candidate.identity.candidate_id, "stratum": name,
                         "source_dataset": candidate.identity.source_dataset, "group_id": candidate.grouping.group_id,
                         "event_id": candidate.grouping.event_id, "split": candidate.eligibility.split,
                         "eligibility_status": candidate.eligibility.status,
                         "detector_correction": detector_disagrees(item),
                         "packet_sha256": candidate.observations.packet_sha256})
    return rows[:config.batch_size], shortfalls


# --- Counts, gap report, export and verification -----------------------------------------------

FILES = ("inventory.json", "reservations.json", "config.json", "candidates.jsonl", "references.jsonl",
         "dedup_report.json", "splits.json", "batch.jsonl", "packets.jsonl", "gap_report.md")
CONTRACT_VERSIONS = {"packet": "rca-v1", "candidate": CANDIDATE_VERSION, "reference": REFERENCE_VERSION,
                     "adapter": ADAPTER_VERSION, "calculation": CALCULATION_VERSION, "grouping": GROUPING_VERSION,
                     "dedup": DEDUP_VERSION, "eligibility_checks": CHECK_VERSION}


def _modality(candidate: CandidateRecord) -> str:
    states = candidate.modalities
    included = [name for name in ("kpi", "text", "settings") if getattr(states, name) == "included"]
    return "+".join(included) or "none"


def _prediction_availability(item: BuiltCandidate) -> str:
    """Packet head and its state, e.g. ``diagnosis:ok`` or ``diagnosis:absent``."""
    ref = item.candidate.prediction_ref
    head = ref.heads.get(ref.packet_head)
    return f"{ref.packet_head}:{head.status if head else 'absent'}"


def _tally(items: Iterable[BuiltCandidate], key) -> dict:
    rows, families, events = defaultdict(int), defaultdict(set), defaultdict(set)
    for item in items:
        k, c = key(item), item.candidate
        rows[k] += 1
        families[k].add((c.identity.source_dataset, c.grouping.group_id))
        events[k].add(c.grouping.event_id)
    return {str(k): {"rows": rows[k], "families": len(families[k]), "events": len(events[k])}
            for k in sorted(rows, key=str)}


def count_candidates(built: Sequence[BuiltCandidate], batch_rows: Sequence[Mapping]) -> dict:
    """Rows, independent families and events, reported separately for every breakdown."""
    in_batch = {row["candidate_id"] for row in batch_rows}
    return {
        "total": _tally(built, lambda i: "all")["all"] if built else {"rows": 0, "families": 0, "events": 0},
        "by_source": _tally(built, lambda i: i.candidate.identity.source_dataset),
        "by_source_family": _tally(built, lambda i: f"{i.candidate.identity.source_dataset}/{i.candidate.grouping.group_id}"),
        "by_modality": _tally(built, lambda i: _modality(i.candidate)),
        "by_prediction": _tally(built, _prediction_availability),
        "by_cause_count": _tally(built, lambda i: i.reference.cause_count),
        "by_label_status": _tally(built, lambda i: i.reference.label_status),
        "by_source_cause": _tally(built, lambda i: f"{i.candidate.identity.source_dataset}/"
                                                   f"{'+'.join(i.reference.causes) or i.reference.label_status}"),
        "by_eligibility": _tally(built, lambda i: i.candidate.eligibility.status),
        "by_source_eligibility": _tally(built, lambda i: f"{i.candidate.identity.source_dataset}/"
                                                         f"{i.candidate.eligibility.status}"),
        "by_split": _tally(built, lambda i: i.candidate.eligibility.split or "none"),
        "batch": {
            "by_source": _tally([i for i in built if i.candidate.identity.candidate_id in in_batch],
                                lambda i: i.candidate.identity.source_dataset),
            "by_cause": _tally([i for i in built if i.candidate.identity.candidate_id in in_batch],
                               lambda i: "+".join(i.reference.causes) or i.reference.label_status),
            "by_split": _tally([i for i in built if i.candidate.identity.candidate_id in in_batch],
                               lambda i: i.candidate.eligibility.split or "none"),
            "detector_correction": sum(bool(row["detector_correction"]) for row in batch_rows),
        },
    }


def _table(title: str, rows: Mapping[str, Mapping]) -> list[str]:
    lines = [f"### {title}", "", "| Key | Rows | Families | Events |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['rows']} | {v['families']} | {v['events']} |" for k, v in rows.items()]
    return [*lines, ""]


def gap_report(*, counts: Mapping, inventory: Sequence[SourceInventory], dedup: Mapping, split_summary: Mapping,
               shortfalls: Mapping, batch_size: int) -> str:
    """Markdown summary of counts, blockers and the example types that could not be supplied."""
    lines = ["# Candidate pool gap report", "",
             ("Generated by `clauseguard.cli.build_candidates`. Counts report rows, independent families and "
             "events separately. No target answers are created here."), "", "## Sources", "",
             "| Source | Training use | Local | Blockers |", "|---|---|---|---|"]
    lines += [f"| {s.dataset} | {s.training_use} | {s.local_status} | {'; '.join(s.blockers) or 'none'} |"
              for s in inventory]
    lines += ["", "## Counts", ""]
    for title, key in (("By source", "by_source"), ("By eligibility", "by_eligibility"),
                       ("By source and eligibility", "by_source_eligibility"), ("By modality", "by_modality"),
                       ("By detector prediction (packet head)", "by_prediction"),
                       ("By cause count", "by_cause_count"), ("By label status", "by_label_status"),
                       ("By source and cause", "by_source_cause"), ("By split", "by_split"),
                       ("By source and family", "by_source_family")):
        lines += _table(title, counts[key])
    batch = counts["batch"]
    lines += ["## Initial batch", "", (f"{sum(v['rows'] for v in batch['by_source'].values())} of at most "
              f"{batch_size} candidates; {batch['detector_correction']} where the packet prediction disagrees "
              "with the source label."), ""]
    lines += _table("Batch by source", batch["by_source"]) + _table("Batch by cause", batch["by_cause"])
    lines += _table("Batch by split", batch["by_split"])
    if shortfalls:
        lines += ["Strata the batch could not fill:", ""]
        lines += [f"- {name}: {v['selected']} of {v['quota']} (available events: {v['available_events']})"
                  for name, v in shortfalls.items()]
        lines.append("")
    missing = [name for name in ("text", "kpi+text", "settings") if not any(
        name == key or (name == "settings" and "settings" in key) for key in counts["by_modality"])]
    multi = [k for k in counts["by_cause_count"] if k not in ("0", "1")]
    lines += ["## Missing example types", ""]
    reasons = {"text": "no local source has incident text approved for RCA",
               "kpi+text": "no local source has incident text approved for RCA",
               "settings": "no local source supplies network settings, and rca-v1 has no settings evidence type"}
    lines += [f"- No `{name}` candidates: {reasons[name]}." for name in missing]
    if not multi:
        lines.append("- No multiple-cause candidates: every available source label is single-cause.")
    if "eligible_pilot" not in counts["by_eligibility"]:
        lines.append("- No `eligible_pilot` candidates: every admissible candidate is `exploratory`.")
    lines += ["", "## Duplicates and grouping", "",
              f"- Exact observation duplicate groups: {len(dedup['exact_duplicate_groups'])}",
              f"- Near-duplicate text pairs: {len(dedup['near_duplicate_text'])}",
              f"- Matches with reserved evaluation text: {len(dedup['reserved_text_matches'])}",
              f"- Events that share raw rows across labels: {len(dedup['events_sharing_raw_rows'])}",
              f"- Quarantined candidates: {len(dedup['quarantined'])}", "",
              "## Train and validation", "", f"- Rule: {split_summary['rule']}"]
    lines += [f"- {name}: {v['validation_families']} of {v['families']} families in validation"
              for name, v in split_summary["strata"].items()]
    return "\n".join(lines) + "\n"


def _jsonl(path: Path, rows: Iterable[Mapping]) -> tuple[str, int]:
    content = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest(), content.count("\n")


def export_release(output_dir: Path, *, built: Sequence[BuiltCandidate], inventory: Sequence[SourceInventory],
                   reservations: Reservations, dedup: Mapping, split_summary: Mapping, batch_rows: Sequence[Mapping],
                   shortfalls: Mapping, config: CandidateConfig,
                   rebuild_packet: Callable[[CandidateRecord], EvidencePacket], prediction_files: Mapping[str, str],
                   created_on: str) -> CandidateManifest:
    """Write every release file; batch packets are rebuilt and must match their recorded hashes."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("output exists; choose a new candidate directory")
    output_dir.mkdir(parents=True)
    by_id = {i.candidate.identity.candidate_id: i for i in built}
    packets = []
    for row in batch_rows:
        packet = rebuild_packet(by_id[row["candidate_id"]].candidate)
        if packet_digest(packet) != row["packet_sha256"]:
            raise ValueError(f"rebuilt packet differs from its recorded hash: {row['candidate_id']}")
        packets.append({"candidate_id": row["candidate_id"], "packet": packet.model_dump(mode="json")})
    counts = count_candidates(built, batch_rows)
    files = {}
    for name, value in (("inventory.json", [s.model_dump(mode="json") for s in inventory]),
                        ("reservations.json", reservations.model_dump(mode="json")),
                        ("config.json", config.model_dump(mode="json")), ("dedup_report.json", dedup),
                        ("splits.json", split_summary)):
        files[name] = FileEntry(sha256=write_json(output_dir / name, value))
    for name, rows in (("candidates.jsonl", [i.candidate.model_dump(mode="json") for i in built]),
                       ("references.jsonl", [i.reference.model_dump(mode="json") for i in built]),
                       ("batch.jsonl", batch_rows), ("packets.jsonl", packets)):
        digest, count = _jsonl(output_dir / name, rows)
        files[name] = FileEntry(sha256=digest, rows=count)
    report = gap_report(counts=counts, inventory=inventory, dedup=dedup, split_summary=split_summary,
                        shortfalls=shortfalls, batch_size=config.batch_size)
    (output_dir / "gap_report.md").write_text(report, encoding="utf-8")
    files["gap_report.md"] = FileEntry(sha256=hashlib.sha256(report.encode()).hexdigest())
    file_record = {k: v.model_dump(mode="json") for k, v in sorted(files.items())}
    manifest = CandidateManifest(
        candidate_release_id=canonical_sha256(file_record), created_on=created_on, sources=list(inventory),
        contract_versions={**CONTRACT_VERSIONS, "question_template": config.question_template_version,
                           "packet_head": config.packet_head},
        reservations_sha256=files["reservations.json"].sha256, prediction_files=dict(prediction_files),
        retrieval="not_run: specification snapshots need NVIDIA_API_KEY and the Qdrant index (Task 4.5)",
        configuration=config.model_dump(mode="json"), files=files, counts=counts,
        batch=BatchRecord(selection_rule=BATCH_RULE, seed=config.batch_seed, size_limit=config.batch_size,
                          candidate_ids=[row["candidate_id"] for row in batch_rows],
                          admitted_statuses=list(config.batch_statuses)))
    write_json(output_dir / "manifest.json", manifest.model_dump(mode="json"))
    return manifest


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_release(output_dir: Path) -> dict:
    """Recheck hashes, reference separation, packets, reservations, splits and the batch."""
    output_dir = Path(output_dir)
    manifest = CandidateManifest.model_validate_json((output_dir / "manifest.json").read_bytes())
    if set(manifest.files) != set(FILES):
        raise ValueError("release files differ from the expected set")
    for name, entry in manifest.files.items():
        if hashlib.sha256((output_dir / name).read_bytes()).hexdigest() != entry.sha256:
            raise ValueError(f"release file changed: {name}")
    file_record = {k: v.model_dump(mode="json") for k, v in sorted(manifest.files.items())}
    if canonical_sha256(file_record) != manifest.candidate_release_id:
        raise ValueError("candidate release ID does not match its files")
    reservations = load_reservations(output_dir / "reservations.json", manifest.reservations_sha256)
    candidates = [CandidateRecord.model_validate(r) for r in _read_jsonl(output_dir / "candidates.jsonl")]
    references = {r["candidate_id"]: CandidateReference.model_validate(r)
                  for r in _read_jsonl(output_dir / "references.jsonl")}
    ids = [c.identity.candidate_id for c in candidates]
    if len(ids) != len(set(ids)) or set(ids) != set(references):
        raise ValueError("candidates and references must match one to one")
    splits = defaultdict(set)
    for c in candidates:
        ref = references[c.identity.candidate_id]
        if (c.taxonomy.taxonomy_id, c.taxonomy.cause_count, c.taxonomy.label_status) != (
                ref.taxonomy_id, ref.cause_count, ref.label_status):
            raise ValueError(f"taxonomy differs from its reference: {c.identity.candidate_id}")
        admissible = c.eligibility.status in ("eligible_pilot", "exploratory")
        if admissible and reservations.reasons(family=c.grouping.group_id, record_id=c.identity.source_record_id):
            raise ValueError(f"reserved evaluation data admitted: {c.identity.candidate_id}")
        if c.eligibility.split:
            splits[(c.identity.source_dataset, c.grouping.group_id)].add(c.eligibility.split)
    if any(len(v) > 1 for v in splits.values()):
        raise ValueError("a family appears in both train and validation")
    by_id = dict(zip(ids, candidates, strict=True))
    batch = _read_jsonl(output_dir / "batch.jsonl")
    if [row["candidate_id"] for row in batch] != manifest.batch.candidate_ids:
        raise ValueError("batch file differs from the manifest")
    events = [by_id[row["candidate_id"]].grouping.event_id for row in batch]
    if len(events) != len(set(events)):
        raise ValueError("the batch holds two windows of one event")
    if any(by_id[row["candidate_id"]].eligibility.status not in manifest.batch.admitted_statuses for row in batch):
        raise ValueError("the batch holds a candidate outside its admitted statuses")
    packets = _read_jsonl(output_dir / "packets.jsonl")
    if [p["candidate_id"] for p in packets] != manifest.batch.candidate_ids:
        raise ValueError("packets do not cover the batch")
    for row in packets:
        packet = EvidencePacket.model_validate(row["packet"])  # rejects any reference field
        if packet_digest(packet) != by_id[row["candidate_id"]].observations.packet_sha256:
            raise ValueError(f"packet differs from its candidate: {row['candidate_id']}")
    return {"candidate_release_id": manifest.candidate_release_id, "candidates": len(candidates),
            "batch": len(batch), "packets_checked": len(packets), "files_checked": len(manifest.files)}
