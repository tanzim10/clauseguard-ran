"""CLI: build or check the golden release manifest (Week 5).

    python -m clauseguard.cli.build_golden build [--strict]
    python -m clauseguard.cli.build_golden check [--strict]

Exit status is 1 on validation problems, or with `--strict` when the release is incomplete.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from clauseguard.eval.release import ReleasePaths, build_release, check_release


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or check the golden release manifest.")
    parser.add_argument("command", choices=("build", "check"))
    parser.add_argument("--golden-dir", type=Path, default=ReleasePaths.golden_dir)
    parser.add_argument("--corpus-manifest", type=Path, default=ReleasePaths.corpus_manifest)
    parser.add_argument("--scenario-release", type=Path, default=ReleasePaths.scenario_release)
    parser.add_argument(
        "--strict", action="store_true", help="also fail when the release is incomplete"
    )
    args = parser.parse_args(argv)

    paths = ReleasePaths(args.golden_dir, args.corpus_manifest, args.scenario_release)
    report = build_release(paths) if args.command == "build" else check_release(paths)

    for problem in report.problems:
        print(f"error: {problem}", file=sys.stderr)
    if not report.ok:
        return 1
    for gap in report.incomplete:
        print(f"incomplete: {gap}")
    status = "complete" if report.complete else "incomplete"
    action = "written to" if args.command == "build" else "matches"
    print(f"golden release {report.manifest['release_id']} ({status}) {action} {paths.manifest}")
    return 1 if args.strict and not report.complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
