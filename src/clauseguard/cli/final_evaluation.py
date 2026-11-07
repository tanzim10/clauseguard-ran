"""Freeze the Task 3 protocol on dev results, then score the frozen KPI checkpoints on test."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="record dev-selected settings before test scoring")
    freeze.add_argument("--random-forest", type=Path, required=True)
    freeze.add_argument("--xgboost", type=Path, required=True)
    freeze.add_argument("--timesnet", type=Path, required=True)
    freeze.add_argument("--exposure-dir", type=Path, action="append", default=[],
                        help="extra directories to scan for earlier test-split runs")
    freeze.add_argument("--text-cohort", type=Path,
                        help="Task 3.7 cohort manifest; its readiness is copied into the protocol")
    evaluate = commands.add_parser("evaluate", help="predict test with the frozen checkpoints and "
                                   "write the report; saved runs are reused")
    evaluate.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu",
                          help="TimesNet device; only CPU is bit-reproducible")
    evaluate.add_argument("--bootstrap-replicates", type=int, default=1000)
    for command in (freeze, evaluate):
        command.add_argument("--release-dir", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--allow-exploratory", action="store_true",
                             help="allow provisional results when recording independence is unresolved")
    args = parser.parse_args(argv)
    try:
        from clauseguard.baselines.final_evaluation import (
            freeze_protocol,
            predict_test,
            write_report,
        )
        from clauseguard.datasets.scenarios import ScenarioDataset

        dataset = ScenarioDataset(args.release_dir, allow_exploratory=args.allow_exploratory)
        if dataset.splits["mode"] != "scored" and not args.allow_exploratory:
            raise ValueError("exploratory release: pass --allow-exploratory")
        if args.command == "freeze":
            result = freeze_protocol(
                dataset, {"random_forest": args.random_forest, "xgboost": args.xgboost,
                          "timesnet": args.timesnet},
                args.output_dir, frozen_on=datetime.now(UTC).date().isoformat(),
                exposure_dirs=args.exposure_dir, text_cohort=args.text_cohort)
            result = {key: result[key] for key in ("protocol_sha256", "release_id", "evaluation_status",
                                                   "dev_comparators", "test_cohorts", "text")}
        else:
            if args.bootstrap_replicates < 0:
                raise ValueError("bootstrap count must be nonnegative")
            runs = predict_test(dataset, args.output_dir, allow_exploratory=args.allow_exploratory,
                                device=args.device, bootstrap_replicates=args.bootstrap_replicates)
            tables = write_report(dataset, args.output_dir, runs,
                                  bootstrap_replicates=args.bootstrap_replicates)["tables"]
            result = {"report": str(args.output_dir / "report.md"),
                      "seed_summary": tables["seed_summary"]}
    except ImportError as exc:
        print(f"Install the model extras with: uv sync --extra baselines --extra timesnet ({exc})",
              file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"final evaluation {args.command} failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
