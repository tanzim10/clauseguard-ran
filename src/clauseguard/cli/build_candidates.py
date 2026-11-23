"""Inventory sources, reserve evaluation data, build the SLM candidate pool, or verify a release."""

import argparse
import json
import sys
from pathlib import Path


def _bubbleran_family(paths: list[Path]) -> str:
    """One recording family for every capture of one UE on one gNB (both files share a session)."""
    from clauseguard.rca.sources import _read_rows

    identities = {(row["ran_ue_id"], row["e2node_nb_id"]) for path in paths for row in _read_rows(path)[1]}
    if len(identities) != 1:
        raise ValueError("BubbleRAN files cover more than one UE/gNB; families need review")
    ue, gnb = identities.pop()
    return f"BubbleRAN:ran_ue_id={ue}:e2node_nb_id={gnb}"


def build(args) -> dict:
    from clauseguard.datasets.scenarios import ScenarioDataset
    from clauseguard.datasets.window_overlap import load_window_spans
    from clauseguard.rca import candidate_pool as pool
    from clauseguard.rca import candidate_release as release

    if args.output_dir.exists():
        raise FileExistsError("output exists; choose a new candidate directory")
    config = (pool.CandidateConfig.model_validate_json(args.config.read_bytes()) if args.config
              else pool.CandidateConfig())
    dataset = ScenarioDataset(args.release_dir, allow_exploratory=args.allow_exploratory)
    inventory = pool.build_inventory(telecomts_acquisition=args.acquisition, bubbleran_dir=args.bubbleran_dir,
                                     spotlight_dir=args.spotlight_dir)
    reservations = pool.build_reservations(dataset, args.golden_dir)
    folds = args.folds or args.predictions.parent / "folds.json"
    cases, predictions_sha256 = pool.load_predictions(args.predictions, folds, release_id=dataset.release_id)
    events, shared = release.telecomts_events(load_window_spans(dataset))
    built = release.assign_events(pool.telecomts_candidates(dataset, cases, predictions_sha256=predictions_sha256,
                                                            config=config), events)
    bubbleran = sorted(args.bubbleran_dir.glob("*.csv")) if args.bubbleran_dir else []
    if bubbleran:
        built += pool.bubbleran_candidates(bubbleran, config=config, family_id=_bubbleran_family(bubbleran))
    spotlight = sorted(args.spotlight_dir.glob("*.csv")) if args.spotlight_dir else []
    if spotlight:  # the local sample is one experiment, so one family covers both streams and captures
        revision = next(s.revision for s in inventory if s.dataset == "SpotLight")
        built += pool.spotlight_candidates(spotlight, label_path=args.spotlight_dir / "label", config=config,
                                           family_id=f"SpotLight:experiment:{revision}")
    dedup = release.duplicate_checks(built, reservations, config, shared_row_events=shared)
    built = release.evaluate_eligibility(built, inventory={s.dataset: s for s in inventory},
                                         reservations=reservations, dedup=dedup, config=config)
    built, split_summary = release.assign_splits(built, config)
    rows, shortfalls = release.select_batch(built, config)
    membership = dataset.splits["membership"]
    csv_paths = {"BubbleRAN": {p.name: p for p in bubbleran}, "SpotLight": {p.name: p for p in spotlight}}

    def rebuild(candidate):
        source, location = candidate.identity.source_dataset, candidate.identity.source_location
        if source == "TelecomTS":
            sid = candidate.identity.source_record_id
            prediction = getattr(cases[sid], config.packet_head).prediction if sid in cases else None
            return pool.telecomts_packet(dataset.load_scenario(sid), membership[sid], prediction, config)[0]
        path = csv_paths[source][location["file"]]
        if source == "SpotLight":
            return pool.spotlight_packet(path, location["start_time"], location["end_time"], config)[0]
        return pool.bubbleran_packet(path, location["first_row"], location["last_row"], config)[0]

    manifest = release.export_release(
        args.output_dir, built=built, inventory=inventory, reservations=reservations, dedup=dedup,
        split_summary=split_summary, batch_rows=rows, shortfalls=shortfalls, config=config, rebuild_packet=rebuild,
        prediction_files={"cross_fit_predictions": predictions_sha256}, created_on=args.created_on)
    dataset.check_source()
    counts = manifest.counts
    return {"candidate_release_id": manifest.candidate_release_id, "candidates": counts["total"],
            "by_source": counts["by_source"], "by_eligibility": counts["by_eligibility"],
            "batch": len(manifest.batch.candidate_ids), "batch_shortfalls": shortfalls}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory", help="terms, access and local state of all five sources")
    reserve = commands.add_parser("reserve", help="golden cases, families and texts excluded from training")
    build_cmd = commands.add_parser("build", help="build candidates, references, batch, manifest and gap report")
    verify = commands.add_parser("verify", help="recheck a candidate release")
    verify.add_argument("--output-dir", type=Path, required=True)
    for command in (inventory, build_cmd):
        command.add_argument("--acquisition", type=Path, default=Path("data/telecomts/acquisition.json"))
        command.add_argument("--bubbleran-dir", type=Path, default=Path("data/bubbleran/raw"))
        command.add_argument("--spotlight-dir", type=Path, default=Path("data/spotlight/raw"))
    for command in (reserve, build_cmd):
        command.add_argument("--release-dir", type=Path, required=True)
        command.add_argument("--allow-exploratory", action="store_true")
        command.add_argument("--golden-dir", type=Path, default=Path("data/golden"))
    reserve.add_argument("--output", type=Path, help="write reservations.json here")
    build_cmd.add_argument("--predictions", type=Path, required=True,
                           help="Task 4.2 cross_fit/predictions.jsonl (out-of-fold training predictions)")
    build_cmd.add_argument("--folds", type=Path, help="cross_fit/folds.json (default: beside --predictions)")
    build_cmd.add_argument("--config", type=Path, help="CandidateConfig JSON; defaults are documented in code")
    build_cmd.add_argument("--created-on", required=True)
    build_cmd.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            from clauseguard.rca.candidate_pool import build_inventory

            result = [s.model_dump(mode="json", exclude={"local_files"}) | {"local_files": len(s.local_files)}
                      for s in build_inventory(telecomts_acquisition=args.acquisition,
                                               bubbleran_dir=args.bubbleran_dir, spotlight_dir=args.spotlight_dir)]
        elif args.command == "reserve":
            from clauseguard.datasets.scenarios import ScenarioDataset
            from clauseguard.rca.candidate_pool import build_reservations, write_json

            dataset = ScenarioDataset(args.release_dir, allow_exploratory=args.allow_exploratory)
            reservations = build_reservations(dataset, args.golden_dir)
            result = {"families": len(reservations.families), "scenario_ids": len(reservations.scenario_ids),
                      "texts": len(reservations.texts), "external": reservations.external}
            if args.output:
                result["sha256"] = write_json(args.output, reservations.model_dump(mode="json"))
        elif args.command == "build":
            result = build(args)
        else:
            from clauseguard.rca.candidate_release import verify_release

            result = verify_release(args.output_dir)
    except (OSError, ValueError, KeyError) as exc:
        print(f"candidates {args.command} failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
