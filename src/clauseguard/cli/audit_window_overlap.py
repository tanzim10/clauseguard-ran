"""Compare reduction policies without changing a release or approving recording groups."""

import argparse
import json
import sys
from pathlib import Path

from clauseguard.datasets.scenarios import ScenarioDataset
from clauseguard.datasets.window_overlap import (
    load_window_spans,
    select_nonoverlapping,
    summarize_spans,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise FileExistsError("output exists; choose a new output directory")
        dataset = ScenarioDataset(args.release_dir)
        spans = load_window_spans(dataset)
        candidates = {
            "train_nonoverlap_normal_keep_faults": select_nonoverlapping(
                spans, preserve_faults=True, training_only=True),
            "nonoverlap_normal_keep_faults": select_nonoverlapping(spans, preserve_faults=True),
            "nonoverlap_all": select_nonoverlapping(spans, preserve_faults=False),
        }
        report = {
            "version": "1", "purpose": "audit_only", "scored_ready": False,
            "release_id": dataset.release_id, "release_mode": dataset.splits["mode"],
            "reason": "window selection does not verify recording independence",
            "original": summarize_spans(spans),
            "policies": {name: summarize_spans(rows) | {
                "removed": len(spans) - len(rows),
                "ids_file": f"{name}.json",
            } for name, rows in candidates.items()},
        }
        args.output_dir.mkdir(parents=True)
        for name, rows in candidates.items():
            selection = {
                "purpose": "audit_only", "release_id": dataset.release_id, "policy": name,
                "scenario_ids": sorted(row.scenario_id for row in rows),
            }
            (args.output_dir / f"{name}.json").write_text(
                json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (args.output_dir / "overlap_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Window audit failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
