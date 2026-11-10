"""Freeze retrieval settings, snapshot specification passages per RCA case, or replay them."""

import argparse
import json
import sys
from pathlib import Path


def check_settings(manifest: dict, settings, top_k: int | None = None) -> None:
    """Live retrieval must use the frozen settings; a change needs a new manifest."""
    frozen = {"collection": manifest["index"]["collection"], "model": manifest["embeddings"]["model"],
              "dimensions": manifest["embeddings"]["dimensions"],
              "chunk_size_tokens": manifest["chunks"]["chunk_size_tokens"]}
    current = {"collection": settings.qdrant_collection, "model": settings.nvidia_embedding_model,
               "dimensions": settings.embedding_dimensions, "chunk_size_tokens": settings.chunk_size_tokens}
    changed = sorted(key for key in frozen if frozen[key] != current[key])
    if changed or (top_k is not None and top_k != manifest["search"]["top_k"]):
        raise ValueError(f"settings differ from the frozen manifest: {changed or ['top_k']}; freeze a new one")


def _packets(path: Path):
    from clauseguard.rca.contracts import EvidencePacket

    return [EvidencePacket.model_validate_json(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="seal corpus, chunk, index, embedding and search settings")
    freeze.add_argument("--corpus-manifest", type=Path, default=Path("data/corpus/manifest.csv"))
    freeze.add_argument("--parsed-dir", type=Path)
    freeze.add_argument("--top-k", type=int)
    freeze.add_argument("--frozen-on", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="retrieve once per evidence packet and save snapshots")
    replay_cmd = commands.add_parser("replay", help="attach saved passages to packets without live retrieval")
    for command in (run, replay_cmd):
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--packets", type=Path, required=True, help="Task 4.4 packets.jsonl")
        command.add_argument("--run-dir", type=Path, required=True)
    replay_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from clauseguard.config import get_settings
        from clauseguard.rca import retrieval

        settings = get_settings()
        if args.command == "freeze":
            result = retrieval.freeze_manifest(
                settings=settings, manifest_path=args.corpus_manifest,
                parsed_dir=args.parsed_dir or Path(settings.parsed_corpus_dir),
                top_k=args.top_k or settings.retrieval_top_k, output=args.output, frozen_on=args.frozen_on)
            result = {key: result[key] for key in ("manifest_sha256", "search", "embeddings")} | {
                "chunks": result["chunks"]["count"], "chunk_inventory_sha256": result["chunks"]["inventory_sha256"]}
        elif args.command == "run":
            from clauseguard.api.search_service import SearchService

            manifest = retrieval.load_manifest_record(args.manifest)
            check_settings(manifest, settings)
            if args.run_dir.exists():
                raise FileExistsError("run directory exists; snapshots of a new run go to a new directory")
            service, index = SearchService(settings=settings), []
            for packet in _packets(args.packets):
                snapshot = retrieval.snapshot_case(packet, manifest, service, case_id=packet.source.case_id)
                path = retrieval.save_snapshot(snapshot, args.run_dir / "snapshots")
                index.append({"scenario_id": packet.source.case_id, "snapshot": path.name,
                              "state": snapshot.state, "stage": snapshot.stage, "passages": len(snapshot.passages)})
            (args.run_dir / "index.json").write_text(json.dumps(
                {"manifest_sha256": manifest["manifest_sha256"], "snapshots": index}, indent=2) + "\n")
            result = {"snapshots": len(index), "states": {s: sum(r["state"] == s for r in index)
                                                          for s in ("completed", "empty", "not_run", "failed")}}
        else:
            manifest = retrieval.load_manifest_record(args.manifest)
            index = json.loads((args.run_dir / "index.json").read_text(encoding="utf-8"))
            if index["manifest_sha256"] != manifest["manifest_sha256"]:
                raise ValueError("run belongs to a different retrieval manifest")
            by_case = {row["scenario_id"]: row["snapshot"] for row in index["snapshots"]}
            replayed, timings = [], []
            for packet in _packets(args.packets):
                if packet.source.case_id not in by_case:
                    raise ValueError(f"no snapshot for {packet.source.case_id}")
                snapshot = retrieval.load_snapshot(args.run_dir / "snapshots" / by_case[packet.source.case_id],
                                                   manifest)
                filled, timing = retrieval.replay(packet, snapshot)
                shared = retrieval.specification_for_configurations(packet, snapshot)
                if len({json.dumps([s.model_dump() for s in v]) for v in shared.values()}) != 1:
                    raise ValueError("configurations received different passages")
                replayed.append(filled)
                timings.append(timing)
            args.output.write_text("".join(p.model_dump_json() + "\n" for p in replayed), encoding="utf-8")
            result = {"replayed": len(replayed), "identical_across_configurations": True,
                      "replay_seconds_total": sum(t["replay_seconds"] for t in timings),
                      "retrieval_status": {s: sum(p.retrieval_status == s for p in replayed)
                                           for s in ("completed", "not_run", "failed")}}
    except (OSError, ValueError, KeyError) as exc:
        print(f"retrieval {args.command} failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
