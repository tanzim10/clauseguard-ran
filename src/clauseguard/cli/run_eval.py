"""CLI: run golden evaluation (Week 5).

    python -m clauseguard.cli.run_eval retrieval [--split dev|test] [--top-k 10]
    python -m clauseguard.cli.run_eval query     [--split dev|test] [--top-k 10]
    python -m clauseguard.cli.run_eval replay --from artifacts/eval/<run_id>
    python -m clauseguard.cli.run_eval compare artifacts/eval/<new> artifacts/eval/<old>

`retrieval` and `query` call the embedding service and Qdrant (and the generation models for
`query`); `replay` and `compare` make no network calls. The three run commands write a new
directory under `artifacts/eval/`; `--compare-to <run>` also writes `comparison.json` into it.
`compare` prints a comparison of two existing runs without modifying either.

Exit status: 0 when every row succeeded, 1 when some rows failed (artifacts are kept),
2 when the run could not start (configuration, release or replay checks).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from clauseguard.api.query_service import QueryService, _grounded_prompt
from clauseguard.api.search_service import SearchService
from clauseguard.config import Settings
from clauseguard.eval.artifacts import COMPARISON_FILE, ArtifactError
from clauseguard.eval.compare import compare_runs
from clauseguard.eval.metrics import DEFAULT_KS, MetricPolicy
from clauseguard.eval.release import ReleasePaths
from clauseguard.eval.runner import (
    DEFAULT_OUTPUT_ROOT,
    EvalConfigError,
    RunResult,
    replay,
    run_live,
)
from clauseguard.llm.clients import GROUNDED_QUERY_MODELS, LLMRouter


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "compare":
        try:
            comparison = compare_runs(args.current, args.previous)
        except ArtifactError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(comparison, indent=2, ensure_ascii=False))
        return 0
    try:
        policy = MetricPolicy(ks=tuple(args.k))
        paths = ReleasePaths(args.golden_dir, args.corpus_manifest, args.scenario_release)
        if args.command == "replay":
            result = replay(
                args.source,
                paths=paths,
                output_root=args.output_root,
                policy=policy,
                compare_to=args.compare_to,
            )
        else:
            result = _run_live(args, paths, policy)
    except (EvalConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _report(result)
    return 0 if result.status == "complete" else 1


def _run_live(args: argparse.Namespace, paths: ReleasePaths, policy: MetricPolicy) -> RunResult:
    settings = Settings()
    if not settings.nvidia_api_key.strip():
        raise EvalConfigError("NVIDIA_API_KEY is not set; it is needed for query embeddings")
    search_service = SearchService(settings=settings)
    query_service = None
    if args.command == "query":
        query_service = QueryService(search_service, LLMRouter(settings=settings))
    return run_live(
        args.command,
        search_service=search_service,
        query_service=query_service,
        paths=paths,
        output_root=args.output_root,
        policy=policy,
        split=args.split,
        top_k=args.top_k,
        settings=settings_record(settings, args.command),
        compare_to=args.compare_to,
    )


def settings_record(settings: Settings, mode: str) -> dict[str, Any]:
    """Retrieval and model settings for `run.json`; credentials are never included."""
    record: dict[str, Any] = {
        "retrieval": {
            "qdrant_url": settings.qdrant_url,
            "collection": settings.qdrant_collection,
            "distance": settings.qdrant_distance,
            "embedding_model": settings.nvidia_embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
            "index_snapshot": "unknown: the index does not record a snapshot identity",
        }
    }
    if mode == "query":
        prompt = _grounded_prompt("<question>", {})
        record["generation"] = {
            "configured_models": list(GROUNDED_QUERY_MODELS),
            "actual_model": "unknown: the router does not report which model answered",
            "fallback": "unknown per row: the router fails over on transient errors",
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or replay a golden evaluation.")
    commands = parser.add_subparsers(dest="command", required=True)

    live = []
    for name, text in (
        ("retrieval", "search every golden row and score the rankings"),
        ("query", "also answer specification rows and score abstention"),
    ):
        sub = commands.add_parser(name, help=text)
        sub.add_argument("--split", choices=("dev", "test"))
        sub.add_argument("--top-k", type=int, help="retrieval depth (default: the largest k)")
        live.append(sub)

    replay_parser = commands.add_parser("replay", help="re-score a saved run without network calls")
    replay_parser.add_argument("--from", dest="source", type=Path, required=True)

    for sub in (*live, replay_parser):
        sub.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS))
        sub.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        sub.add_argument("--golden-dir", type=Path, default=ReleasePaths.golden_dir)
        sub.add_argument("--corpus-manifest", type=Path, default=ReleasePaths.corpus_manifest)
        sub.add_argument("--scenario-release", type=Path, default=ReleasePaths.scenario_release)
        sub.add_argument("--compare-to", type=Path, help="previous run to compare against")

    compare_parser = commands.add_parser("compare", help="print a comparison of two saved runs")
    compare_parser.add_argument("current", type=Path)
    compare_parser.add_argument("previous", type=Path)
    return parser


def _report(result: RunResult) -> None:
    retrieval = result.metrics["retrieval"]
    denominators = retrieval["denominators"]
    print(f"run written to {result.path} ({result.status})")
    if (result.path / COMPARISON_FILE).exists():
        print(f"comparison written to {result.path / COMPARISON_FILE}")
    print(
        f"answerable rows: {denominators['answerable']} "
        f"(failed {denominators['failed']}, no-answer excluded {denominators['no_answer_excluded']})"
    )
    headline = retrieval["headline"] or {}
    for name, value in headline.items():
        print(f"  {name}: {value:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
