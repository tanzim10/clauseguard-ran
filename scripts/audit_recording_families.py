"""Find recording links for human review; never modify a release or approve a recording."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from clauseguard.datasets.acquisition import source_path
from clauseguard.datasets.contracts import NUMERIC_CHANNELS, RecordingReview
from clauseguard.datasets.scenarios import ScenarioDataset
from clauseguard.datasets.telecomts import file_sha256


@dataclass
class Trace:
    source: str
    values: np.ndarray
    metadata: dict


def read_traces(dataset: ScenarioDataset, acquisition_dir: Path) -> list[Trace]:
    """Use only checksum-verified raw files tied to the release's acquisition manifests."""
    manifests = {}
    for name, key in (("acquisition.json", "acquisition_sha256"),
                      ("supplements.json", "supplements_sha256")):
        path = acquisition_dir / name
        if file_sha256(path) != dataset.evidence[key]:
            raise ValueError(f"acquisition checksum mismatch: {name}")
        manifest = json.loads(path.read_text())
        if manifest["status"] != "complete" or manifest["revision"] != dataset.evidence["revision"]:
            raise ValueError(f"incomplete or different acquisition: {name}")
        manifests[name] = {f["source_file"]: f for f in manifest["files"]}
        if len(manifests[name]) != len(manifest["files"]):
            raise ValueError(f"duplicate source in {name}")
    splits = defaultdict(set)
    for record in dataset.splits["membership"].values():
        splits[record["source_file"]].add(record["split"])
    traces = []
    for source, evidence in sorted(dataset.evidence["traces"].items()):
        raw_name = evidence["raw_file"]
        description_name = raw_name.removesuffix("metrics.csv") + "description.txt"
        for name, checksum, manifest_name in (
            (source, evidence["processed_sha256"], "acquisition.json"),
            (raw_name, evidence["raw_sha256"], "supplements.json"),
            (description_name, evidence["description_sha256"], "supplements.json"),
        ):
            path = source_path(acquisition_dir / "raw", name, supplemental=True)
            if (manifests[manifest_name][name]["sha256"] != checksum
                    or file_sha256(path) != checksum):
                raise ValueError(f"source checksum mismatch: {name}")
        path = source_path(acquisition_dir / "raw", raw_name, supplemental=True)
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if (not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames))
                    or not set(NUMERIC_CHANNELS).issubset(reader.fieldnames)):
                raise ValueError(f"missing or duplicate raw columns: {source}")
            rows = list(reader)
        if not rows:
            raise ValueError(f"empty raw trace: {source}")
        values = np.array([[float(row[c]) if row[c] else np.nan for c in NUMERIC_CHANNELS]
                           for row in rows], dtype=np.float64)
        if np.isinf(values).any():
            raise ValueError(f"infinite raw value: {source}")
        description = source_path(acquisition_dir / "raw", description_name, supplemental=True)
        metadata = {
            "raw_file": raw_name, "raw_sha256": evidence["raw_sha256"],
            "processed_sha256": evidence["processed_sha256"],
            "description_sha256": evidence["description_sha256"],
            "description": description.read_text(encoding="utf-8").strip(),
            "rows": len(rows), "first_timestamp": rows[0]["timestamp"],
            "last_timestamp": rows[-1]["timestamp"], "splits": sorted(splits[source]),
            "review": evidence["review"],
            "provenance_recording_ids": evidence["provenance_recording_ids"],
            "provenance_parent_ids": evidence["provenance_parent_ids"],
        }
        traces.append(Trace(source, values, metadata))
    return traces


def _fingerprints(values: np.ndarray, length: int, distinct: int):
    """All finite, varied segments; hash hits are always confirmed against the actual values."""
    if len(values) < length:
        return np.array([], dtype=np.uint64), np.array([], dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(values, length)
    ordered = np.sort(windows, axis=1)
    eligible = (np.isfinite(windows).all(axis=1)
                & (1 + (ordered[:, 1:] != ordered[:, :-1]).sum(axis=1) >= distinct))
    starts = np.flatnonzero(eligible)
    selected = windows[starts].copy()
    selected[selected == 0] = 0.0  # Canonicalize signed zero before hashing.
    bits = selected.view(np.uint64)
    hashes = np.zeros(len(starts), dtype=np.uint64)
    for column in range(length):
        hashes = hashes * np.uint64(1099511628211) ^ bits[:, column]
    return hashes, starts


def find_links(traces: list[Trace], *, length: int = 16, distinct: int = 8,
               max_occurrences: int = 64, progress=print) -> tuple[list[dict], list[dict]]:
    """Corroborate exact segments on two channels at the same row offset across any trace pair."""
    if not traces or not 2 <= distinct <= length or max_occurrences < 2:
        raise ValueError("require 2 <= distinct <= length and max_occurrences >= 2")
    alignments, coverage = {}, []
    for channel_index, channel in enumerate(NUMERIC_CHANNELS):
        hashes, owners, positions = [], [], []
        counts = {}
        for owner, trace in enumerate(traces):
            keys, starts = _fingerprints(trace.values[:, channel_index], length, distinct)
            counts[trace.source] = len(starts)
            hashes.append(keys)
            owners.append(np.full(len(keys), owner, dtype=np.int32))
            positions.append(starts)
        keys, owners, positions = map(np.concatenate, (hashes, owners, positions))
        order = np.argsort(keys, kind="stable")
        keys, owners, positions = keys[order], owners[order], positions[order]
        _, starts, sizes = np.unique(keys, return_index=True, return_counts=True)
        cross_source = (np.minimum.reduceat(owners, starts) != np.maximum.reduceat(owners, starts)
                        if len(starts) else np.array([], dtype=bool))
        skipped = cross_source & (sizes > max_occurrences)
        confirmed = 0
        for start, size in zip(starts[cross_source & ~skipped], sizes[cross_source & ~skipped]):
            for left, right in combinations(range(int(start), int(start + size)), 2):
                a, b = int(owners[left]), int(owners[right])
                if a == b:
                    continue
                x, y = int(positions[left]), int(positions[right])
                if a > b:
                    a, b, x, y = b, a, y, x
                if not np.array_equal(traces[a].values[x:x + length, channel_index],
                                      traces[b].values[y:y + length, channel_index]):
                    continue
                confirmed += 1
                channels = alignments.setdefault((a, b, y - x), {})
                anchor = channels.setdefault(channel, {"count": 0, "first": x, "last": x,
                                                       "starts": set()})
                anchor["count"] += 1
                anchor["starts"].add(x)
                anchor["first"], anchor["last"] = min(anchor["first"], x), max(anchor["last"], x)
        coverage.append({"channel": channel, "eligible_segments_by_source": counts,
                         "skipped_repetitive_keys": int(skipped.sum()),
                         "skipped_segment_occurrences": int(sizes[skipped].sum()),
                         "confirmed_cross_source_segment_pairs": confirmed})
        progress(f"{channel}: {sum(counts.values())} eligible segments, {confirmed} matching pairs",
                 flush=True)
    links = []
    for (a, b, offset), channels in sorted(alignments.items()):
        if len(channels) < 2:
            continue
        first, last = max(0, -offset), min(len(traces[a].values), len(traces[b].values) - offset)
        equal = traces[a].values[first:last] == traces[b].values[first + offset:last + offset]
        shared = Counter(x for anchor in channels.values() for x in anchor["starts"])
        corroborated = sorted(x for x, count in shared.items() if count >= 2)
        links.append({
            "source_a": traces[a].source, "source_b": traces[b].source,
            "source_b_minus_source_a_row_offset": offset,
            "status": ("candidate_requires_review" if corroborated else "weak_offset_coincidence"),
            "corroborated_segment_starts": len(corroborated),
            "first_corroborated_start_row": corroborated[0] if corroborated else None,
            "anchors_by_channel": {c: {k: v for k, v in anchor.items() if k != "starts"}
                                   for c, anchor in channels.items()},
            "aligned_rows": last - first,
            "equal_values_by_channel": dict(zip(NUMERIC_CHANNELS, equal.sum(axis=0).tolist())),
            "crosses_current_splits": traces[a].metadata["splits"] != traces[b].metadata["splits"],
        })
    return links, coverage


def audit(release_dir: Path, acquisition_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError("choose a new audit output directory")
    dataset = ScenarioDataset(release_dir, allow_exploratory=True)
    traces = read_traces(dataset, acquisition_dir)
    links, coverage = find_links(traces)
    hints = [{"source_a": a.source, "source_b": b.source,
              "reason": "same description; shared activity/location does not establish lineage"}
             for a, b in combinations(traces, 2)
             if a.metadata["description"] and a.metadata["description"] == b.metadata["description"]]
    reviews = {}
    for trace in traces:
        candidates = set(trace.metadata["review"]["candidate_parent_sources"])
        for link in links:
            if (link["status"] == "candidate_requires_review"
                    and trace.source in (link["source_a"], link["source_b"])):
                candidates.update({link["source_a"], link["source_b"]} - {trace.source})
        # This is a fresh proposal, never an automatic copy/creation of human approvals.
        reviews[trace.source] = RecordingReview(
            candidate_parent_sources=sorted(candidates),
            evidence="See audit.json for local observations; original session/parent needs review.",
        ).model_dump()
    report = {
        "version": "1", "purpose": "recording_review_audit", "release_id": dataset.release_id,
        "revision": dataset.evidence["revision"], "scored_ready": False,
        "method": {"segment_rows": 16, "minimum_distinct_finite_values": 8,
                   "maximum_occurrences_per_key": 64, "minimum_channels_at_same_offset": 2,
                   "corroboration": "two varied channels match the same 16-row segment",
                   "comparison": "exact float64 values; timestamps are not used as session IDs",
                   "scope": "every eligible start in every numeric channel across all trace pairs"},
        "limitations": ["No match does not prove independence or recover an unpublished parent.",
                        "Altered, rounded, resampled or shorter shared sequences may be missed.",
                        "Repetitive keys above the occurrence limit are excluded and counted.",
                        "Descriptions and timestamps cannot verify recording identity.",
                        "Matching segments support review, not automatic parent attribution."],
        "sources": {t.source: t.metadata for t in traces}, "candidate_links": links,
        "description_hints": hints, "coverage": coverage,
        "existing_duplicate_source_links": dataset.evidence["duplicate_source_links"],
    }
    dataset.check_source()
    output_dir.mkdir(parents=True)
    for name, value in (("audit.json", report), ("recording_reviews.proposed.json", reviews)):
        (output_dir / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    supported = sum(link["status"] == "candidate_requires_review" for link in links)
    lines = ["# Recording review audit", "", f"Release: `{dataset.release_id}`", "",
             f"Scanned {len(traces)} traces and {sum(len(t.values) for t in traces):,} raw rows.",
             f"Found {supported} candidate alignments with two channels matching the same segment.",
             f"Found {len(links) - supported} weak offsets matching channels at separate positions.",
             f"Found {len(hints)} pairs with identical descriptions; these are metadata hints only.",
             "", "All proposed reviews remain unresolved. The source release is unchanged.", "",
             "## Candidate alignments", ""]
    for link in links:
        lines.append(f"- `{link['source_a']}` / `{link['source_b']}`: offset "
                     f"{link['source_b_minus_source_a_row_offset']}, "
                     f"{len(link['anchors_by_channel'])} channels; "
                     f"{link['status']}; "
                     f"crosses current splits: {link['crosses_current_splits']}.")
    if not links:
        lines.append("No corroborated exact alignment found under the recorded search rules.")
    lines += ["", "## Limits", "", *[f"- {s}" for s in report["limitations"]], "",
              "## Next decision", "", ("Inspect audit.json and recording_reviews.proposed.json. "
              "Resolve original session IDs and synthetic parent IDs from source evidence before "
              "approving reviews. A broader local search alone cannot certify independence.")]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--acquisition-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit(args.release_dir, args.acquisition_dir, args.output_dir)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Recording audit failed: {exc}\n")
    supported = sum(link["status"] == "candidate_requires_review"
                    for link in report["candidate_links"])
    print(f"Saved audit: {args.output_dir}; {supported} corroborated candidates, "
          f"{len(report['candidate_links']) - supported} weak offsets.")


if __name__ == "__main__":
    main()
