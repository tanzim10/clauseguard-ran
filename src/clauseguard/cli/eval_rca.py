"""CLI: score saved base-LLM RCA runs (Task 4.8).

    python -m clauseguard.cli.eval_rca --runs artifacts/rca/runs-a [--runs artifacts/rca/runs-b ...] \\
        --output artifacts/rca_eval/<name> [--no-judge] [--judge-model ID] [--reuse-judgments FILE]

Reads saved `rca-run-v1` records (no generator is called), loads references from the golden
files, optionally runs the separate judge, and writes a new directory with `policy.json`,
`judgments.jsonl`, `metrics.json` (deterministic), `summary.md` and `eval.json` (run details).

Results are provisional: `human_review_status` stays `pending`. Exit status: 0 on success,
1 when some judge decisions are missing, 2 when the evaluation could not start.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clauseguard.eval.rca_eval import (
    EvalPolicy,
    EvaluationInputs,
    evaluate,
    load_references,
    resources,
)
from clauseguard.eval.rca_judge import (
    DEFAULT_JUDGE_MODEL,
    ClaimJudgment,
    JudgePolicy,
    OpenAIJudgeClient,
    judge_runs,
)
from clauseguard.eval.rca_runs import RunRecordError, RunView, load_run_records


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("output directory exists; an evaluation goes to a new directory")
        policy = EvalPolicy(category_mappings=json.loads(args.category_mappings.read_text()) if args.category_mappings else {})
        views = load_run_records(args.runs)
        refs = load_references(args.rca_rows, args.annotations, args.policy_file, args.overlay,
                               args.abstention_references, policy)
        judge_policy = JudgePolicy(model=args.judge_model)
        _check_judge_distinct(views, judge_policy)
        judgments = _judge(args, views, judge_policy)
    except (FileExistsError, FileNotFoundError, RunRecordError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = evaluate(EvaluationInputs(views, refs, judgments, policy, None if args.no_judge else judge_policy, args.retrieval_eval))
    report["resources"] = resources(views, judgments, _settings(views))
    _write(args.output, policy, judge_policy, judgments, report, args, views)
    print(_summary(report))
    pending = report["support"]["claims_submitted"] - report["support"]["status_counts"].get("judged", 0)
    return 1 if not args.no_judge and pending else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score saved base-LLM RCA runs with automatic metrics and a separate judge.")
    parser.add_argument("--runs", type=Path, action="append", required=True,
                        help="run_rca output directory; repeat to merge reruns (a later good run replaces a generation failure)")
    parser.add_argument("--output", type=Path, required=True, help="new directory (use artifacts/rca_eval/<name>)")
    parser.add_argument("--rca-rows", type=Path, default=Path("data/golden/rca.jsonl"))
    parser.add_argument("--annotations", type=Path, default=Path("data/golden/annotations.jsonl"))
    parser.add_argument("--policy-file", type=Path, default=Path("data/golden/rca_policy.json"))
    parser.add_argument("--overlay", type=Path, help="alternative acceptable sets and corpus-gap clauses per case")
    parser.add_argument("--abstention-references", type=Path, help="separate file of unanswerable fixtures")
    parser.add_argument("--category-mappings", type=Path, help="JSON object mapping aliases to canonical categories")
    parser.add_argument("--retrieval-eval", type=Path, help="existing run_eval directory for spec_qa Recall@k and MRR")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--no-judge", action="store_true", help="structural and identification metrics only")
    parser.add_argument("--reuse-judgments", type=Path, help="judgments.jsonl from an earlier evaluation of the same runs")
    return parser


def _check_judge_distinct(views: Sequence[RunView], policy: JudgePolicy) -> None:
    generators = {m for v in views for m in (v.requested_model, v.served_model) if m}
    if policy.model in generators:
        raise ValueError(f"the judge model {policy.model} is also a generator model; choose a separate judge")


def _judge(args: argparse.Namespace, views: Sequence[RunView], policy: JudgePolicy) -> list[ClaimJudgment]:
    if args.no_judge:
        return []
    from openai import OpenAI

    from clauseguard.config import get_settings

    settings = get_settings()
    if not settings.nvidia_api_key.strip():
        raise RuntimeError("judging needs NVIDIA_API_KEY; use --no-judge or --reuse-judgments")
    reuse = {}
    if args.reuse_judgments:
        for line in args.reuse_judgments.read_text().splitlines():
            if line.strip():
                j = ClaimJudgment.model_validate_json(line)
                reuse[(j.run_id, j.claim_id)] = j
    client = OpenAIJudgeClient(
        OpenAI(api_key=settings.nvidia_api_key, base_url=settings.nvidia_base_url, timeout=600.0, max_retries=0), policy)
    return judge_runs(client, policy, views, reuse)


def _settings(views: Sequence[RunView]) -> dict[str, Any]:
    return {
        "generation_settings": sorted({json.dumps(v.generation_settings, sort_keys=True) for v in views}),
        "generator_models": sorted({v.served_model or v.requested_model or "unknown" for v in views}),
        "prompt_versions": sorted({v.prompt_version for v in views}),
        "warm_up": "none: every run is timed as recorded, including the first request",
        "concurrency": "runs were recorded as saved; judging is sequential, one request per run",
    }


def _write(output: Path, policy: EvalPolicy, judge: JudgePolicy, judgments: Sequence[ClaimJudgment],
           report: dict[str, Any], args: argparse.Namespace, views: Sequence[RunView]) -> None:
    output.mkdir(parents=True)
    (output / "policy.json").write_text(json.dumps(
        {"evaluation": policy.model_dump(mode="json"), "evaluation_sha256": policy.sha256,
         "judge": judge.identity() if not args.no_judge else None}, indent=2, sort_keys=True))
    (output / "judgments.jsonl").write_text("".join(j.model_dump_json() + "\n" for j in judgments))
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (output / "summary.md").write_text(_summary(report) + "\n")
    (output / "eval.json").write_text(json.dumps({
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_ids": [v.run_id for v in views], "superseded_run_ids": [s for v in views for s in v.superseded],
        "run_directories": [str(p) for p in args.runs],
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "python": platform.python_version(), "cpus": os.cpu_count()},
        "command": " ".join(sys.argv),
    }, indent=2, sort_keys=True))


def _fmt(item: dict[str, Any]) -> str:
    return "unavailable" if item["value"] is None else f"{item['value']:.3f} ({item['numerator']}/{item['denominator']})"


def _summary(report: dict[str, Any]) -> str:
    status, support, structure = report["status"], report["support"], report["structure"]["successful_only"]
    scored, exploratory = report["identification"]["scored"], report["identification"]["exploratory"]
    lines = [
        f"RCA evaluation ({report['evaluation_status']}, human review {report['human_review_status']}), policy {report['policy_version']}",
        f"runs {status['runs']}: answered {status['answered']}, abstained {status['abstained']}, failed {status['failed']} {status['failed_by_stage']}",
        f"identification scored: {scored['status']}" + (f" ({scored['reason']})" if scored["status"] == "blocked" else ""),
        f"identification exploratory: {exploratory['status']}" + (f" ({exploratory['reason']})" if exploratory["status"] == "blocked" else ""),
        f"citation coverage {_fmt(structure['citation_coverage'])}; cause-link coverage {_fmt(structure['cause_link_coverage'])}",
        f"judgment coverage {_fmt(support['judgment_coverage'])}; provisional faithfulness {_fmt(support['provisional_faithfulness'])}",
        f"decisions {support['decision_counts']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
