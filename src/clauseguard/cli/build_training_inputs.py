"""Select one window per eligible event from a candidate release and write run_rca inputs (Task 4.9)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def build(args: argparse.Namespace) -> dict:
    from clauseguard.datasets.scenarios import ScenarioDataset
    from clauseguard.rca import candidate_pool as pool
    from clauseguard.rca.candidate_release import verify_release
    from clauseguard.rca.candidates import CandidateRecord, CandidateReference
    from clauseguard.rca.training_inputs import TrainingSelection, rebuild_inputs, select_events

    if args.output_dir.exists():
        raise FileExistsError("output exists; choose a new directory")
    release = verify_release(args.candidates_dir)
    manifest = json.loads((args.candidates_dir / "manifest.json").read_text(encoding="utf-8"))
    config = pool.CandidateConfig.model_validate(manifest["configuration"])
    candidates = {c.identity.candidate_id: c for c in
                  (CandidateRecord.model_validate(r) for r in _jsonl(args.candidates_dir / "candidates.jsonl"))}
    references = {r.candidate_id: r for r in
                  (CandidateReference.model_validate(r) for r in _jsonl(args.candidates_dir / "references.jsonl"))}
    selected = select_events(list(candidates.values()), references, seed=args.seed,
                             normal_share=args.normal_share, pilot_size=args.pilot_size)

    dataset = ScenarioDataset(args.release_dir, allow_exploratory=args.allow_exploratory)
    folds = args.folds or args.predictions.parent / "folds.json"
    cases, digest = pool.load_predictions(args.predictions, folds, release_id=dataset.release_id)
    if digest != manifest["prediction_files"]["cross_fit_predictions"]:
        raise ValueError("predictions differ from the ones the candidate release used")
    csv_paths = {name: {p.name: p for p in sorted(d.glob("*.csv"))} if d else {}
                 for name, d in (("BubbleRAN", args.bubbleran_dir), ("SpotLight", args.spotlight_dir))}

    def rebuild(candidate):
        return pool.rebuild_packet(candidate, dataset=dataset, cases=cases, csv_paths=csv_paths, config=config)

    from clauseguard.rca.training_baselines import (
        BASELINE_SELECTION_VERSION,
        apply_baseline,
        plan_baselines,
    )

    plans = {p.candidate_id: p for p in plan_baselines([s.candidate_id for s in selected], candidates,
                                                        seed=args.seed)}
    links: dict[str, int] = {}

    def augment(item, packet, evidence):
        plan = plans[item.candidate_id]
        if plan.rule == "none":
            links[item.candidate_id] = 0
            return packet
        if plan.rule == "preceding_window":
            loc = plan.baseline_location
            _, baseline = pool.bubbleran_packet(csv_paths["BubbleRAN"][loc["file"]], loc["first_row"],
                                                loc["last_row"], config)
        else:
            _, baseline = rebuild(candidates[plan.baseline_candidate_id])
        packet, links[item.candidate_id] = apply_baseline(packet, evidence, baseline, plan)
        return packet

    args.output_dir.mkdir(parents=True)
    selection = TrainingSelection(candidate_release_id=release["candidate_release_id"], seed=args.seed,
                                  normal_share=args.normal_share, pilot_size=args.pilot_size, events=selected)
    (args.output_dir / "selection.json").write_text(selection.model_dump_json(indent=2) + "\n", encoding="utf-8")
    for name, subset in (("all", selected), ("pilot", [s for s in selected if s.pilot])):
        packets, audits = rebuild_inputs(subset, candidates, rebuild, augment if args.baselines else None)
        _write_jsonl(args.output_dir / f"packets.{name}.jsonl", packets)
        _write_jsonl(args.output_dir / f"audits.{name}.jsonl", audits)

    if args.baselines:
        record = {"version": BASELINE_SELECTION_VERSION, "seed": args.seed,
                  "plans": [p.model_dump(mode="json") | {"links": links.get(p.candidate_id, 0)}
                            for p in plans.values()]}
        (args.output_dir / "baselines.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    def tally(items):
        return dict(sorted(Counter(f"{s.source_dataset}:{s.stratum}:{s.split}" for s in items).items()))

    rules = Counter(p.rule for p in plans.values()) if args.baselines else {}
    return {"candidate_release_id": release["candidate_release_id"], "selected": len(selected),
            "baselines": dict(rules), "linked_windows": sum(1 for v in links.values() if v),
            "all": tally(selected), "pilot": tally([s for s in selected if s.pilot]),
            "pilot_causes": dict(sorted(Counter("+".join(s.causes) or "normal"
                                                for s in selected if s.pilot).items()))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-dir", type=Path, required=True, help="verified Task 4.6 candidate release")
    parser.add_argument("--release-dir", type=Path, required=True, help="scenario release the candidates used")
    parser.add_argument("--allow-exploratory", action="store_true")
    parser.add_argument("--predictions", type=Path, required=True, help="cross_fit/predictions.jsonl")
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--bubbleran-dir", type=Path, default=Path("data/bubbleran/raw"))
    parser.add_argument("--spotlight-dir", type=Path, default=Path("data/spotlight/raw"))
    parser.add_argument("--normal-share", type=float, default=0.12, help="normal events as a share of all (3 in 25)")
    parser.add_argument("--pilot-size", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baselines", action="store_true",
                        help="add label-blind baselines (BubbleRAN preceding window, TelecomTS same zone/app normal)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"training inputs failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
