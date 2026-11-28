"""Task 4.10 QLoRA: release check (no GPU), memory trial and training. Reproduction on EC2 g6.xlarge (L4, CUDA):

    uv sync --extra slm
    R=<path>/release-final   # train.jsonl, validation.jsonl, manifest.json (sample_approved)
    uv run python -m clauseguard.cli.train_lora check --release-dir $R --out runs/410/check
    uv run python -m clauseguard.cli.train_lora memory-trial --release-dir $R --out runs/410/trial
    uv run python -m clauseguard.cli.train_lora train --release-dir $R --out runs/410/train
    # untuned and adapted runs differ only by --adapter (same evidence, prompts, quantization, settings)
    uv run python -m clauseguard.cli.run_rca <evidence args> --run-kind scored \\
        --local-model Qwen/Qwen3.5-4B --output runs/410/base
    uv run python -m clauseguard.cli.run_rca <evidence args> --run-kind scored \\
        --local-model Qwen/Qwen3.5-4B --adapter runs/410/train/selected_adapter --output runs/410/adapted

Selection rule (fixed before training): lowest mean validation loss on assistant tokens at the end of each
epoch; ties at 4 decimals go to the earlier epoch. Rows over --max-length are rejected, never truncated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def check(release_dir: Path, out: Path, *, max_length: int, allow_exploratory: bool, model: str | None = None) -> dict:
    """Verify hashes/status, then render and mask-check every row with the real tokenizer if available."""
    from clauseguard.llm.lora import (
        HFTrainTokenizer,
        QLoraConfig,
        ReleaseError,
        load_rows,
        verify_release,
        write_json,
    )
    from clauseguard.rca.training_release import check_row

    config = QLoraConfig(max_length=max_length, **({"base_model": model} if model else {}))
    release = verify_release(release_dir, allow_exploratory=allow_exploratory)
    report: dict[str, Any] = {"release": release, "config_sha256": config.sha256(), "max_length": max_length,
                              "rows": {}}
    try:
        tok = HFTrainTokenizer(config.base_model)
    except (RuntimeError, OSError) as exc:
        report["tokenizer"] = {"status": "skipped", "reason": str(exc)[:200]}
        write_json(out / "check_report.json", report)
        return report
    report["tokenizer"] = {"status": "checked", "name": config.base_model}
    problems = 0
    for name in ("train.jsonl", "validation.jsonl"):
        checks = [check_row(str(i), str(i), messages, tok, max_length)
                  for i, messages in enumerate(load_rows(release_dir / name), start=1)]
        totals = [c.total_tokens for c in checks]
        problems += sum(1 for c in checks if c.problems)
        report["rows"][name] = {
            "rows": len(checks), "tokens_min": min(totals, default=0), "tokens_max": max(totals, default=0),
            "assistant_tokens_min": min((c.assistant_tokens for c in checks), default=0),
            "assistant_tokens_max": max((c.assistant_tokens for c in checks), default=0),
            "think_block_present": sum(c.think_block_present for c in checks),
            "mask_aligned": sum(c.mask_aligned for c in checks),
            "ends_with_im_end": sum(c.ends_with_im_end for c in checks),
            "rows_with_problems": [{"row": c.record_id, "problems": c.problems} for c in checks if c.problems]}
    report["passed"] = problems == 0
    write_json(out / "check_report.json", report)
    if problems:
        raise ReleaseError(f"{problems} rows failed the check; see {out / 'check_report.json'}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "memory-trial", "train"):
        p = sub.add_parser(name)
        p.add_argument("--release-dir", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--max-length", type=int, default=40960)
        p.add_argument("--allow-exploratory", action="store_true", help="dry runs only: accept any release status")
        p.add_argument("--model", help="override the base model id (default Qwen/Qwen3.5-4B)")
        if name != "check":
            p.add_argument("--no-liger", action="store_true", help="disable the fused linear cross entropy")
    args = parser.parse_args(argv)
    try:
        from clauseguard.llm import lora

        if args.command == "check":
            report = check(args.release_dir, args.out, max_length=args.max_length,
                           allow_exploratory=args.allow_exploratory, model=args.model)
            print(json.dumps({"release_status": report["release"]["release_status"], "tokenizer": report["tokenizer"],
                              "rows": {k: {x: v for x, v in r.items() if x != "rows_with_problems"}
                                       for k, r in report["rows"].items()}, "passed": report.get("passed")},
                             indent=2))
            return 0
        overrides: dict[str, Any] = {"max_length": args.max_length, "use_liger_kernel": not args.no_liger}
        if args.model:
            overrides["base_model"] = args.model
        config = lora.QLoraConfig(**overrides)
        run = lora.memory_trial if args.command == "memory-trial" else lora.train
        record = run(config, args.release_dir, args.out, allow_exploratory=args.allow_exploratory)
        print(json.dumps(record["results"], indent=2, default=str))
        return 0 if record["results"].get("passed", True) else 1
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        print(f"train_lora failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
