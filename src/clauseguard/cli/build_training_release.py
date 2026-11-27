"""Build the cited SLM training release from judged RCA runs (Task 4.9)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _file_sha(path: Path) -> str:
    from clauseguard.rca.training_release import sha256_bytes

    return sha256_bytes(path.read_bytes())


def build(args: argparse.Namespace) -> dict:
    from clauseguard.rca.candidates import CandidateReference
    from clauseguard.rca.training_inputs import TrainingSelection
    from clauseguard.rca.training_release import (
        DEFAULT_TOKENIZER,
        HFChatTokenizer,
        HumanReview,
        MachineAcceptance,
        assemble_release,
        load_run_records,
    )

    if args.output_dir.exists():
        raise FileExistsError("output exists; choose a new directory")
    selection = TrainingSelection.model_validate_json(args.selection.read_text(encoding="utf-8"))
    references = {r.candidate_id: r for r in (
        CandidateReference.model_validate_json(line)
        for line in args.references.read_text(encoding="utf-8").splitlines() if line.strip())}
    acceptance = MachineAcceptance.model_validate_json(args.acceptance.read_text(encoding="utf-8"))
    human = HumanReview.model_validate_json(args.human_review.read_text(encoding="utf-8")) if args.human_review else None
    records, run_hashes = load_run_records(args.runs)
    tokenizer_name = None if args.no_tokenizer else (args.tokenizer or DEFAULT_TOKENIZER)
    tokenizer = HFChatTokenizer(tokenizer_name) if tokenizer_name else None
    input_hashes = {"runs": run_hashes, "selection": _file_sha(args.selection),
                    "references": _file_sha(args.references), "acceptance": _file_sha(args.acceptance),
                    "human_review": _file_sha(args.human_review) if args.human_review else None}
    files = assemble_release(selection=selection, references=references, records=records, acceptance=acceptance,
                             human_review=human, tokenizer=tokenizer, tokenizer_name=tokenizer_name,
                             max_tokens=args.max_tokens, created_on=args.created_on, input_hashes=input_hashes)
    if args.sources:
        # Issue #75: all five sources with eligibility, gaps and the selected coverage, recorded with the release.
        report = json.loads(args.sources.read_text(encoding="utf-8"))
        files["sources.json"] = json.dumps(report, indent=2, sort_keys=True) + "\n"
        manifest = json.loads(files["manifest.json"])
        manifest["sources"] = report
        manifest["inputs"]["sources"] = _file_sha(args.sources)
        manifest["outputs"]["sources.json"] = hashlib.sha256(files["sources.json"].encode()).hexdigest()
        files["manifest.json"] = json.dumps(manifest, indent=2) + "\n"
    args.output_dir.mkdir(parents=True)
    for name, text in files.items():
        (args.output_dir / name).write_text(text, encoding="utf-8")
    manifest = json.loads(files["manifest.json"])
    qwen = json.loads(files["qwen_check.json"])
    return {"release_status": manifest["release_status"], "counts": manifest["counts"],
            "exclusions": len(manifest["exclusions"]), "qwen_check": qwen.get("summary", qwen),
            "output_dir": str(args.output_dir)}


def main(argv: list[str] | None = None) -> int:
    from clauseguard.rca.training_release import DEFAULT_MAX_TOKENS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, action="append", required=True, help="run_rca output dir (repeatable)")
    parser.add_argument("--selection", type=Path, required=True, help="selection.json from build_training_inputs")
    parser.add_argument("--references", type=Path, required=True, help="candidate references.jsonl")
    parser.add_argument("--acceptance", type=Path, required=True, help="machine acceptance.json")
    parser.add_argument("--human-review", type=Path, help="human_review.json; unlisted cases stay pending")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tokenizer", help="local Hugging Face tokenizer (default Qwen/Qwen3.5-4B)")
    group.add_argument("--no-tokenizer", action="store_true", help="skip the Qwen template and token check")
    parser.add_argument("--sources", type=Path, help="source coverage report (JSON) copied into the release")
    parser.add_argument("--created-on", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build(args)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"training release failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
