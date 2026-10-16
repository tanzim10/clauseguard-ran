"""One whole-family split across both prediction targets; no acquisition or training."""

import hashlib
import json
import math
import random
from collections import Counter, defaultdict

from clauseguard.datasets.contracts import (
    CATEGORY_LABELS,
    LABEL_MAPPINGS,
    SPLITS,
    TASKS,
    RecordingReview,
)


def family_groups(evidence: dict) -> dict[str, dict]:
    """Union whole traces, parent recordings, candidate links and observed duplicate sources."""
    traces = evidence["traces"]
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]

    def union(left, right):
        a, b = find(left), find(right)
        parents[max(a, b)] = min(a, b)

    scenario_sources = {r["scenario_id"]: r["source_file"] for r in evidence["records"]}
    for source, trace in sorted(traces.items()):
        review = RecordingReview.model_validate(trace["review"])
        token = "source:" + source
        find(token)
        if review.recording_id:
            union(token, "recording:" + review.recording_id)
        for parent in review.parent_recording_ids:
            union(token, "recording:" + parent)
        for recording_id in trace.get("provenance_recording_ids", []):
            union(token, "recording:" + recording_id)
        for parent_id in trace.get("provenance_parent_ids", []):
            if parent_id in scenario_sources:
                union(token, "source:" + scenario_sources[parent_id])
            elif parent_id in traces:
                union(token, "source:" + parent_id)
            else:
                union(token, "recording:" + parent_id)
        for candidate in review.candidate_parent_sources:
            if candidate not in traces:
                raise ValueError(f"unknown candidate parent source: {candidate}")
            union(token, "source:" + candidate)
    for left, right in evidence["duplicate_source_links"]:
        if left not in traces or right not in traces:
            raise ValueError("duplicate link references unknown source")
        union("source:" + left, "source:" + right)
    grouped = defaultdict(list)
    for source in sorted(traces):
        grouped[find("source:" + source)].append(source)
    result = {}
    for sources in grouped.values():
        identity = hashlib.sha256(json.dumps(sources).encode()).hexdigest()
        verified = all(traces[s]["review"]["status"] == "verified" for s in sources)
        group = {"group_id": "family-" + identity, "sources": sources, "verified": verified}
        for source in sources:
            result[source] = group
    return result



def _exclusion_reason(group, members, mode):
    if any(not r["raw_match"] for r in members):
        return "raw_match_failed"
    if mode == "scored" and not group["verified"]:
        return "recording_family_unresolved"
    if any(r["kpi_reason"] for r in members):
        return "unsupported_kpi_contract"
    return ""


def assign_splits(evidence: dict, *, ratios=(0.6, 0.2, 0.2), seed: int = 42,
                  mode: str = "scored", trials: int = 4096) -> dict:
    """Search deterministic group assignments using coverage and size only, never model scores.

    Search is bounded: a missing class may be infeasible or not found by this search.
    Readiness reports the missing support rather than promising an optimal assignment.
    """
    if mode not in ("scored", "exploratory"):
        raise ValueError("mode must be scored or exploratory")
    if (len(ratios) != 3 or any(not math.isfinite(r) or r <= 0 for r in ratios)
            or not math.isclose(sum(ratios), 1.0)):
        raise ValueError("three positive split ratios must sum to one")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(trials, bool) or not isinstance(trials, int) or not 1 <= trials <= 100000:
        raise ValueError("trials must be between 1 and 100000")
    groups = family_groups(evidence)
    records = evidence["records"]
    ids = [r["scenario_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scenario ID in recording evidence")
    if any(r["source_file"] not in groups for r in records):
        raise ValueError("scenario references unknown source")
    family_records = defaultdict(list)
    for record in records:
        family_records[groups[record["source_file"]]["group_id"]].append(record)
    eligible, exclusions = {}, {}
    for gid, members in sorted(family_records.items()):
        group = groups[members[0]["source_file"]]
        reason = _exclusion_reason(group, members, mode)
        if reason:
            exclusions[gid] = reason
        else:
            eligible[gid] = members
    # Strata: normal, abnormal, ten categories, real jamming. Whole families remain indivisible.
    vectors = {}
    for gid, members in eligible.items():
        v = [0] * 13
        for r in members:
            v[int(r["fault_label"] == "abnormal")] += 1
            if not r["category_reason"]:
                v[2 + LABEL_MAPPINGS["category"][r["rca_label"]]] += 1
            v[12] += int(r["origin"] == "real" and r["rca_label"] == "Jamming")
        vectors[gid] = v
    gids = sorted(eligible)
    totals = [sum(v[i] for v in vectors.values()) for i in range(13)]
    def score_assignment(chosen):
        counts = [[0] * 13 for _ in SPLITS]
        for gid, index in zip(gids, chosen):
            for j, value in enumerate(vectors[gid]):
                counts[index][j] += value
        missing = [sum(counts[s][j] == 0 for s in range(3) for j in indices)
                   for indices in (range(2), range(2, 12), (12,))]
        loss = sum((counts[s][j] / total - ratios[s]) ** 2
                   for s in range(3) for j, total in enumerate(totals) if total)
        row_total = sum(totals[:2])
        row_loss = sum(((sum(counts[s][:2]) / row_total) - ratios[s]) ** 2
                       for s in range(3)) if row_total else 0.0
        return (*missing, 8 * row_loss + loss / 13)

    rng = random.Random(seed)
    best, chosen_best = None, []
    for _ in range(trials if gids else 1):
        chosen = rng.choices(range(3), weights=ratios, k=len(gids))
        score = score_assignment(chosen)
        if best is None or score < best:
            best, chosen_best = score, chosen
    # Refine sizes by moving/swapping whole groups, never individual windows.
    # Bounded and deterministic; does not promise a globally optimal assignment.
    for _ in range(10):
        improved, next_choice = best, chosen_best
        for i in range(len(gids)):
            for index in range(3):
                candidate = chosen_best.copy()
                candidate[i] = index
                score = score_assignment(candidate)
                if score < improved:
                    improved, next_choice = score, candidate
            for j in range(i):
                candidate = chosen_best.copy()
                candidate[i], candidate[j] = candidate[j], candidate[i]
                score = score_assignment(candidate)
                if score < improved:
                    improved, next_choice = score, candidate
        if improved == best:
            break
        best, chosen_best = improved, next_choice
    assignment = dict(zip(gids, (SPLITS[i] for i in chosen_best)))
    membership = {}
    for record in sorted(records, key=lambda r: r["scenario_id"]):
        group = groups[record["source_file"]]
        gid = group["group_id"]
        membership[record["scenario_id"]] = record | {
            "group_id": gid, "evidence_status": "verified" if group["verified"] else "unresolved",
            "split": assignment.get(gid, "excluded"), "exclusion_reason": exclusions.get(gid, ""),
            "detection_target": LABEL_MAPPINGS["detection"][record["fault_label"]],
            "category_target": LABEL_MAPPINGS["category"].get(record["rca_label"])
                if not record["category_reason"] else None,
        }
    return {"version": "1", "mode": mode, "algorithm": "family_search_v1",
            "seed": seed, "ratios": list(ratios), "trials": trials,
            "scenarios_sha256": evidence["scenarios_sha256"], "membership": membership}


def select_task_cohort(splits: dict, *, split: str, task: str, modality: str = "kpi") -> list[str]:
    """A filtered view, never a second split. Normal/jammer filtering is task-specific."""
    if split not in SPLITS or task not in TASKS or modality not in ("kpi", "text", "paired"):
        raise ValueError("invalid split, task or modality")
    return [sid for sid, m in sorted(splits["membership"].items())
            if m["split"] == split and not m["kpi_reason"]
            and (task == "detection" or not m["category_reason"])
            and (modality == "kpi" or not m["text_reasons"][task])]


def build_quality(splits: dict) -> dict:
    """Report coverage independently from verified-recording readiness."""
    membership = splits["membership"]
    result = {
        "mode": splits["mode"], "accepted": len(membership),
        "membership_counts": dict(Counter(m["split"] for m in membership.values())),
        "origins": dict(Counter(m["origin"] for m in membership.values())),
        "release_exclusions": dict(Counter(m["exclusion_reason"] for m in membership.values()
                                           if m["exclusion_reason"])),
        "unknown_units": sorted({name for m in membership.values() for name in m["unknown_units"]}),
        "excluded_channels": sorted({name for m in membership.values()
                                     for name in m["excluded_channels"]}),
        "unresolved_records": sum(m["evidence_status"] != "verified" for m in membership.values()),
        "tasks": {},
    }
    for task in TASKS:
        result["tasks"][task] = {}
        expected = {0, 1} if task == "detection" else set(range(len(CATEGORY_LABELS)))
        for modality in ("kpi", "text", "paired"):
            coverage, reasons = {}, []
            for split in SPLITS:
                members = [membership[sid] for sid in select_task_cohort(
                    splits, split=split, task=task, modality=modality)]
                labels = Counter(m[f"{task}_target"] for m in members)
                missing = sorted(expected - set(labels))
                groups_by_label = {str(label): len({m["group_id"] for m in members
                                                   if m[f"{task}_target"] == label})
                                   for label in expected}
                coverage[split] = {
                    "windows": len(members), "groups": len({m["group_id"] for m in members}),
                    "labels": {str(k): v for k, v in sorted(labels.items())},
                    "groups_by_label": groups_by_label, "missing_labels": missing,
                    "origins": dict(Counter(m["origin"] for m in members)),
                }
                if missing:
                    reasons.append(f"{split}: missing labels {missing}")
            coverage_ok = not reasons
            if splits["mode"] != "scored":
                reasons.append("recording independence unverified: exploratory release")
            result["tasks"][task][modality] = {
                "ready": not reasons, "coverage_ready": coverage_ok,
                "reasons": reasons, "splits": coverage,
            }
    result["golden_rca"] = {
        "ready": False, "reason": "reviewed golden answers/citations belong to Task 2.2",
        "eligible_dev_test_scenarios": sum(
            len(select_task_cohort(splits, split=s, task="category")) for s in ("dev", "test")
        ) if splits["mode"] == "scored" else 0,
    }
    result["real_jamming_by_split"] = {
        s: sum(m["split"] == s and m["rca_label"] == "Jamming" and m["origin"] == "real"
               for m in membership.values()) for s in SPLITS
    }
    return result


def validate_membership(splits: dict, evidence: dict) -> None:
    """Validate persisted decisions without running the split search again."""
    if splits.get("version") != "1" or splits.get("mode") not in ("scored", "exploratory"):
        raise ValueError("unsupported split version or mode")
    records = {r["scenario_id"]: r for r in evidence["records"]}
    membership = splits["membership"]
    if len(records) != len(evidence["records"]) or set(records) != set(membership):
        raise ValueError("missing, duplicate or unknown scenario membership")
    groups = family_groups(evidence)
    assignments = {}
    group_records = defaultdict(list)
    for r in records.values():
        group_records[groups[r["source_file"]]["group_id"]].append(r)
    group_reasons = {gid: _exclusion_reason(groups[members[0]["source_file"]], members, splits["mode"])
                     for gid, members in group_records.items()}
    for sid, m in membership.items():
        source = records[sid]["source_file"]
        group = groups[source]
        if any(m.get(k) != v for k, v in records[sid].items()):
            raise ValueError(f"membership changes scenario evidence: {sid}")
        if m["group_id"] != group["group_id"] or m["split"] not in (*SPLITS, "excluded"):
            raise ValueError(f"invalid group or split: {sid}")
        if m["evidence_status"] != ("verified" if group["verified"] else "unresolved"):
            raise ValueError(f"incorrect recording evidence status: {sid}")
        expected_category = LABEL_MAPPINGS["category"].get(m["rca_label"]) if not m["category_reason"] else None
        if (m["detection_target"] != LABEL_MAPPINGS["detection"][m["fault_label"]]
                or m["category_target"] != expected_category):
            raise ValueError(f"incorrect target: {sid}")
        gid = m["group_id"]
        if assignments.setdefault(gid, m["split"]) != m["split"]:
            raise ValueError(f"recording family crosses splits: {gid}")
        required = group_reasons[gid]
        if required:
            if m["split"] != "excluded" or m["exclusion_reason"] != required:
                raise ValueError(f"unsafe membership for excluded family: {sid}")
        elif m["split"] == "excluded" or m["exclusion_reason"]:
            raise ValueError(f"unexpected exclusion: {sid}")
    if splits["scenarios_sha256"] != evidence["scenarios_sha256"]:
        raise ValueError("split and evidence source identities differ")
