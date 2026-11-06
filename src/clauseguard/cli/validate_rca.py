"""CLI: have an independent model review golden RCA rows (Task 4.3).

    python -m clauseguard.cli.validate_rca --provider gemini --list-models
    python -m clauseguard.cli.validate_rca --provider gemini --model gemini-3.8-flash \\
        --rows data/golden/rca.jsonl --annotations data/golden/annotations.jsonl \\
        --out-dir artifacts/eval/rca_review

Providers use OpenAI-compatible endpoints; the key comes from the environment or .env:
NVIDIA_API_KEY (nvidia, build.nvidia.com), GEMINI_API_KEY (gemini, aistudio.google.com),
GROQ_API_KEY (groq) or OPENROUTER_API_KEY (openrouter). Each row's verdict is saved as <out-dir>/<row id>.json with summary.json
beside them; rerunning skips rows already reviewed by the same provider, model and prompt.
Exit status is 1 when any review failed.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from openai import OpenAI

from clauseguard.config import Settings, get_settings
from clauseguard.corpus.manifest import load_manifest
from clauseguard.eval.annotations import load_annotations
from clauseguard.eval.golden import load_rca
from clauseguard.eval.rca_validation import ReviewSettings, pdf_passage_reader, review_rows

# provider -> (base URL, Settings attribute holding the key, env var name, default model)
PROVIDERS = {
    "nvidia": (None, "nvidia_api_key", "NVIDIA_API_KEY", "openai/gpt-oss-120b"),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini_api_key",
        "GEMINI_API_KEY",
        "gemini-flash-latest",
    ),
    "groq": ("https://api.groq.com/openai/v1", "groq_api_key", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "openrouter": (
        "https://openrouter.ai/api/v1",
        "openrouter_api_key",
        "OPENROUTER_API_KEY",
        "openai/gpt-oss-120b:free",
    ),
}


def build_client(provider: str, settings: Settings, timeout: float = 300.0) -> OpenAI | None:
    """Return an OpenAI-compatible client for the provider, or None when its key is missing."""
    base_url, key_attr, _, _ = PROVIDERS[provider]
    api_key = getattr(settings, key_attr).strip()
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url=base_url or settings.nvidia_base_url,
        timeout=timeout,
        max_retries=0,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review golden RCA rows with an independent model.")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="gemini")
    parser.add_argument("--model", help="defaults to the provider's default reviewer model")
    parser.add_argument("--rows", type=Path, default=Path("data/golden/rca.jsonl"))
    parser.add_argument("--annotations", type=Path, default=Path("data/golden/annotations.jsonl"))
    parser.add_argument("--corpus-manifest", type=Path, default=Path("data/corpus/manifest.csv"))
    parser.add_argument("--raw-corpus-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/eval/rca_review"))
    parser.add_argument("--ids", nargs="*", help="review only these row ids")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between requests")
    parser.add_argument("--max-wait", type=float, default=120.0, help="longest rate-limit wait")
    parser.add_argument("--max-tokens", type=int, default=8000, help="reply budget incl. reasoning")
    parser.add_argument("--timeout", type=float, default=300.0, help="per-request timeout (s)")
    parser.add_argument("--list-models", action="store_true", help="list the provider's models")
    args = parser.parse_args(argv)

    settings = get_settings()
    client = build_client(args.provider, settings, args.timeout)
    if client is None:
        env_var = PROVIDERS[args.provider][2]
        print(f"error: {env_var} is not set for provider {args.provider}", file=sys.stderr)
        return 1
    if args.list_models:
        for model in sorted(m.id for m in client.models.list()):
            print(model)
        return 0

    manifest = load_manifest(args.corpus_manifest)
    spec_ids = {row["spec_id"] for row in manifest if row.get("spec_id")}
    rows = load_rca(args.rows, known_spec_ids=spec_ids)
    annotations = {a.id: a for a in load_annotations(args.annotations, manifest)}
    if args.ids:
        rows = [row for row in rows if row.id in set(args.ids)]
    missing = [row.id for row in rows if row.id not in annotations]
    if missing:
        print(f"error: no annotation for {', '.join(missing)}", file=sys.stderr)
        return 1

    review_settings = ReviewSettings(
        model=args.model or PROVIDERS[args.provider][3],
        provider=args.provider,
        delay_seconds=args.delay,
        max_wait_seconds=args.max_wait,
        max_tokens=args.max_tokens,
    )
    raw_dir = args.raw_corpus_dir or Path(settings.raw_corpus_dir)
    summary = review_rows(
        client, rows, annotations, pdf_passage_reader(manifest, raw_dir), args.out_dir, review_settings
    )
    print(
        f"{summary['rows']} rows reviewed by {summary['provider']}/{summary['model']}: "
        f"{summary['verdicts']}; failed {summary['failed'] or 'none'}; "
        f"question leaks {summary['question_leaks'] or 'none'}"
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
