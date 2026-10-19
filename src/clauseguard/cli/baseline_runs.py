"""Rescore or compare saved baseline runs without loading model code."""

import argparse
import json
import sys
from pathlib import Path

from clauseguard.baselines.artifacts import compare_runs, load_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score_parser = subparsers.add_parser("score", help="recalculate a saved run")
    score_parser.add_argument("run_dir", type=Path)
    compare_parser = subparsers.add_parser("compare", help="compare compatible saved runs")
    compare_parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args(argv)
    try:
        result = (load_run(args.run_dir)["metrics"] if args.command == "score"
                  else compare_runs(args.run_dirs))
    except (OSError, ValueError) as exc:
        print(f"baseline {args.command} failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
