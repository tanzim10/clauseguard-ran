"""Audit-only window selections. These do not establish recording independence."""

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass

from clauseguard.datasets.scenarios import ScenarioDataset


@dataclass(frozen=True)
class WindowSpan:
    scenario_id: str
    source: str
    start: int
    end: int
    detection_target: int
    category_target: int | None
    group_id: str
    split: str


def load_window_spans(dataset: ScenarioDataset) -> list[WindowSpan]:
    """Read raw-row intervals from a release already checked by ScenarioDataset."""
    dataset.check_source()
    members = dataset.splits["membership"]
    spans = []
    seen = set()
    with (dataset.release_dir / "window_sources.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            sid = row["scenario_id"]
            if sid in seen or sid not in members:
                raise ValueError("duplicate or unknown window mapping")
            seen.add(sid)
            member = members[sid]
            if row["source_file"] != member["source_file"] or row["raw_match"] != "True":
                raise ValueError(f"unmatched or inconsistent raw window: {sid}")
            start, end = int(row["raw_start_row"]), int(row["raw_end_row"])
            if start < 1 or end - start + 1 != 128:
                raise ValueError(f"invalid raw window interval: {sid}")
            spans.append(WindowSpan(
                sid, row["source_file"], start, end, member["detection_target"],
                member["category_target"], member["group_id"], member["split"],
            ))
    if seen != set(members):
        raise ValueError("missing window mapping")
    dataset.check_source()
    return spans


def select_nonoverlapping(spans: list[WindowSpan], *, preserve_faults: bool,
                          training_only: bool = False) -> list[WindowSpan]:
    """Keep a maximum disjoint set per source; optionally retain every fault window.

    With preserve_faults, the disjoint constraint applies to normal windows only. Fault windows
    may still overlap each other or normal windows. Source windows and split assignments never move.
    """
    if len({span.scenario_id for span in spans}) != len(spans):
        raise ValueError("duplicate scenario ID")
    last_end: dict[str, int] = {}
    selected = []
    for span in sorted(spans, key=lambda item: (item.source, item.end, item.start, item.scenario_id)):
        if span.start < 1 or span.end < span.start:
            raise ValueError("invalid window interval")
        if ((training_only and span.split != "train")
                or (preserve_faults and span.detection_target == 1)):
            selected.append(span)
        elif span.start > last_end.get(span.source, 0):
            selected.append(span)
            last_end[span.source] = span.end
    return selected


def summarize_spans(spans: list[WindowSpan]) -> dict:
    """Count retained windows, labels, and provisional groups without declaring readiness."""
    by_split = defaultdict(list)
    for span in spans:
        by_split[span.split].append(span)
    return {
        "windows": len(spans),
        "sources": len({span.source for span in spans}),
        "groups": len({span.group_id for span in spans}),
        "detection": dict(sorted(Counter(str(span.detection_target) for span in spans).items())),
        "category": dict(sorted(Counter(str(span.category_target) for span in spans
                                        if span.category_target is not None).items())),
        "splits": {split: {
            "windows": len(rows),
            "groups": len({row.group_id for row in rows}),
            "detection": dict(Counter(str(row.detection_target) for row in rows)),
            "category": dict(Counter(str(row.category_target) for row in rows
                                     if row.category_target is not None)),
        } for split, rows in sorted(by_split.items())},
    }
