"""Build recording evidence and one shared detection/category release from ingested scenarios."""

import argparse
import json
import sys
from pathlib import Path

import httpx

from clauseguard.datasets.acquisition import acquire_supplements, write_manifest
from clauseguard.datasets.evidence import build_recording_evidence
from clauseguard.datasets.release import prepare_release


def _object(path):
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"configuration must be a JSON object: {path}")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=Path("data/scenarios/scenarios.jsonl"))
    parser.add_argument("--acquisition-dir", type=Path, default=Path("data/telecomts"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/scenarios/release"))
    parser.add_argument("--evidence-dir", type=Path, help="Existing evidence for --stage release")
    parser.add_argument("--stage", choices=("evidence", "release", "all"), default="all")
    parser.add_argument("--recording-reviews", type=Path, help="Human review JSON keyed by source path")
    parser.add_argument("--mode", choices=("scored", "exploratory"), default="scored")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.6, 0.2, 0.2))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=4096)
    parser.add_argument("--require-task", action="append", choices=("detection", "category"))
    parser.add_argument("--input-dir", type=Path, help="Offline supplemental source tree")
    parser.add_argument("--checksums", type=Path, help="Offline source-path to SHA256 JSON")
    args = parser.parse_args(argv)
    try:
        if args.stage != "release":
            if (args.output_dir / "evidence.json").exists():
                raise FileExistsError("evidence exists; choose a new output directory")
            if not (args.acquisition_dir / "supplements.json").exists():
                acquisition = _object(args.acquisition_dir / "acquisition.json")
                if acquisition.get("status") != "complete":
                    raise ValueError("processed acquisition is incomplete")
                acquire_supplements(
                    revision=acquisition["revision"], output_dir=args.acquisition_dir,
                    sources=[f["source_file"] for f in acquisition["files"]],
                    local_dir=args.input_dir, checksums=_object(args.checksums),
                    declared_license=acquisition["declared_license"],
                )
            evidence = build_recording_evidence(
                args.scenarios, args.acquisition_dir, args.output_dir,
                reviews=_object(args.recording_reviews),
            )
            if args.stage == "evidence":
                print(json.dumps({"records": len(evidence["records"]),
                                  "traces": len(evidence["traces"]),
                                  "matched_windows": sum(r["raw_match"] for r in evidence["records"])}))
                return 0 if all(r["raw_match"] for r in evidence["records"]) and evidence["records"] else 3
        result = prepare_release(
            args.scenarios, args.evidence_dir or args.output_dir, args.output_dir,
            mode=args.mode, ratios=args.ratios, seed=args.seed, trials=args.trials,
        )
        print(json.dumps(result, sort_keys=True))
        key = "coverage_ready" if args.mode == "exploratory" else "ready"
        return 0 if all(result["quality"]["tasks"][task]["kpi"][key]
                        for task in (args.require_task or ("detection", "category"))) else 3
    except (OSError, ValueError, TypeError, KeyError, httpx.HTTPError) as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(args.output_dir / "preparation_error.json", {"error": str(exc)})
        print(f"Scenario preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
