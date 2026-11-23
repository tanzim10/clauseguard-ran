"""Run the shared RCA core on saved evidence packets and retrieval snapshots (Task 4.7).

Generation is live through the NVIDIA router, or replayed from recorded raw outputs keyed by the
exact prompt hash (``--replay``). Replayed runs are labelled ``generator_kind: replay``.
"""

import argparse
import json
import sys
from pathlib import Path


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_cases(*, packets: Path, audits: Path, run_dir: Path, manifest: dict) -> list:
    """Pair each packet with its audit group (file order, as Task 4.4 writes them) and snapshot."""
    from clauseguard.rca.contracts import EvidencePacket
    from clauseguard.rca.evidence import EvidenceAudit
    from clauseguard.rca.pipeline import RcaCase

    groups: dict[str, list] = {}
    for row in _jsonl(audits):
        case_id = row.pop("case_id")
        groups.setdefault(case_id, []).append(EvidenceAudit.model_validate(row))
    rows = _jsonl(packets)
    if len(rows) != len(groups):
        raise ValueError(f"{len(rows)} packets but {len(groups)} audit groups")
    index = json.loads((Path(run_dir) / "index.json").read_text(encoding="utf-8"))
    if index["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("snapshot run belongs to a different retrieval manifest")
    snapshots = {row["scenario_id"]: row["snapshot"] for row in index["snapshots"]}
    cases = []
    for row, (case_id, case_audits) in zip(rows, groups.items(), strict=True):
        packet = EvidencePacket.model_validate(row)
        name = snapshots.get(packet.source.case_id)
        cases.append(RcaCase(case_id=case_id, packet=packet, audits=case_audits, manifest=manifest,
                             snapshot_path=Path(run_dir) / "snapshots" / name if name else None))
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="frozen Task 4.5 retrieval manifest")
    parser.add_argument("--packets", type=Path, required=True, help="Task 4.4 packets.jsonl")
    parser.add_argument("--audits", type=Path, required=True, help="Task 4.4 audits.jsonl")
    parser.add_argument("--run-dir", type=Path, required=True, help="Task 4.5 snapshot run directory")
    parser.add_argument("--output", type=Path, required=True, help="new directory for run records")
    parser.add_argument("--configuration", default="base_llm_full",
                        choices=("base_llm_full", "adapted_slm_full", "text_rag", "kpi_summary_rag"))
    parser.add_argument("--run-kind", default="exploratory", choices=("exploratory", "scored"))
    parser.add_argument("--model", help="fixed generation model (default: RCA_MODEL_ID setting)")
    parser.add_argument("--fallback", action="append", default=[], help="exploratory fallback model")
    parser.add_argument("--budget-chars", type=int, help="context budget in characters")
    parser.add_argument("--max-tokens", type=int,
                        help="completion token cap (default: RCA_MAX_TOKENS, 16384); reasoning models spend "
                             "part of it before the answer")
    parser.add_argument("--prompt-version", default=None, choices=("rca-prompt-v1", "rca-prompt-v2"),
                        help="prompt version (default: current)")
    parser.add_argument("--numeric-policy", default="fail", choices=("fail", "record"))
    parser.add_argument("--disallow-normal", action="store_true", help="taxonomy without normal")
    parser.add_argument("--policy", type=Path, default=Path("data/golden/rca_policy.json"),
                        help="source of the minimal question for kpi_summary_rag")
    parser.add_argument("--replay", type=Path, help="directory of recorded raw outputs (fixtures)")
    parser.add_argument("--case", action="append", default=[], help="only these case IDs")
    args = parser.parse_args(argv)
    try:
        from clauseguard.config import get_settings
        from clauseguard.llm.clients import LLMRouter
        from clauseguard.rca import pipeline
        from clauseguard.rca.checks import TELECOMTS_TAXONOMY
        from clauseguard.rca.retrieval import load_manifest_record

        settings = get_settings()
        if args.output.exists():
            raise FileExistsError("output directory exists; a new run goes to a new directory")
        args.output.mkdir(parents=True)
        minimal = None
        if args.configuration == "kpi_summary_rag":
            minimal = json.loads(args.policy.read_text(encoding="utf-8"))["minimal_question"]
        config = pipeline.ExperimentConfig(
            configuration=args.configuration, run_kind=args.run_kind,
            model_id=args.model or settings.rca_model_id, fallback_models=tuple(args.fallback),
            context_budget_chars=args.budget_chars or settings.rca_context_budget_chars,
            numeric_policy=args.numeric_policy,
            **({"prompt_version": args.prompt_version} if args.prompt_version else {}),
            generation=pipeline.GenerationSettings(
                max_tokens=args.max_tokens or settings.rca_max_tokens,
                timeout_seconds=settings.rca_timeout_seconds), minimal_question=minimal,
            taxonomy=TELECOMTS_TAXONOMY.model_copy(update={"allows_normal": not args.disallow_normal}))
        manifest = load_manifest_record(args.manifest)
        cases = load_cases(packets=args.packets, audits=args.audits, run_dir=args.run_dir, manifest=manifest)
        if args.case:
            cases = [case for case in cases if case.case_id in set(args.case)]
        generator = (pipeline.ReplayGenerator.from_directory(args.replay) if args.replay
                     else LLMRouter(settings=settings))
        rca = pipeline.RcaPipeline(generator=generator)
        rows = []
        for case in cases:
            record = rca.run_rca(case, config)
            pipeline.save_run_record(record, args.output / "runs")
            response = record.response
            rows.append({
                "case_id": case.case_id, "run_id": record.run_id, "result_sha256": record.result_sha256,
                "generator_kind": record.generator_kind, "generator_called": record.generator_called,
                "execution_status": response.execution_status, "execution_stage": response.execution_stage,
                "execution_reason": response.execution_reason,
                "result_status": response.result.status if response.result else None,
                "categories": [c.category for c in response.result.final_fault_categories]
                if response.result else [],
                "abstention_reason": response.result.abstention_reason if response.result else None,
                "detector_label": response.detector_prediction.predicted_label
                if response.detector_prediction else None,
                "human_review_status": response.human_review_status,
                "factual_support": response.factual_support})
        (args.output / "index.json").write_text(json.dumps(
            {"config": config.model_dump(mode="json"), "config_sha256": config.sha256(),
             "generator": "replay" if args.replay else "live", "runs": rows},
            indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = {"runs": len(rows), "output": str(args.output),
                  "execution_status": {s: sum(r["execution_status"] == s for r in rows)
                                       for s in ("complete", "incomplete", "failed")},
                  "result_status": {s: sum(r["result_status"] == s for r in rows)
                                    for s in ("answered", "abstained")}}
    except (OSError, ValueError, KeyError) as exc:
        print(f"run_rca failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
