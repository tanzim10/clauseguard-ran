"""CLI for parsing the manifest-validated corpus into plain text."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from clauseguard.corpus.parser import ParseSummary, parse_corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and parse acquired corpus documents into plain text."
    )
    parser.add_argument("--manifest", dest="manifest_path")
    parser.add_argument("--raw-dir", dest="raw_dir")
    parser.add_argument("--output-dir", dest="output_dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run corpus parsing and return a shell-friendly exit status."""
    args = _parser().parse_args(argv)
    try:
        summary: ParseSummary = parse_corpus(
            manifest_path=args.manifest_path,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
        )
    except Exception as error:  # noqa: BLE001 - convert operational failures to CLI status
        print(f"parsing failed: {error}", file=sys.stderr)
        return 1

    print(f"Parsed {summary.documents} documents into {len(summary.outputs)} text files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
