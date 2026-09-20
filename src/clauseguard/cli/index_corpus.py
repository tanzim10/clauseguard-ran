"""CLI for rebuilding the Qdrant index from parsed corpus text."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from clauseguard.retrieval.indexer import IndexSummary, index_corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chunk parsed corpus text, embed it, and upsert it into Qdrant."
    )
    parser.add_argument("--manifest", dest="manifest_path")
    parser.add_argument("--parsed-dir", dest="parsed_dir")
    parser.add_argument("--collection")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run indexing and return a shell-friendly exit status."""
    args = _parser().parse_args(argv)
    try:
        summary: IndexSummary = index_corpus(
            manifest_path=args.manifest_path,
            parsed_dir=args.parsed_dir,
            collection=args.collection,
        )
    except Exception as error:  # noqa: BLE001 - convert operational failures to CLI status
        print(f"indexing failed: {error}", file=sys.stderr)
        return 1

    print(f"Indexed {summary.documents} documents into {summary.chunks} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
