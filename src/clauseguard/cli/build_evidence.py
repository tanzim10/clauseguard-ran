"""Build traceable RCA evidence packets, check them against raw sources, or read a table window."""

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

RAW_TOLERANCE = 1e-9  # the release matched every sample to raw metrics.csv within this bound


def _jsonl(path: Path, rows) -> str:
    content = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest()


def build_telecomts(dataset, case_rows, output_dir: Path, *, predictions: dict | None) -> dict:
    from clauseguard.rca.evidence import check_audits
    from clauseguard.rca.sources import UNAVAILABLE_SOURCES, telecomts_evidence

    output_dir.mkdir(parents=True)
    membership = dataset.splits["membership"]
    packets, audits, rejections = [], [], []
    for row in case_rows:
        record = dataset.load_scenario(row.scenario_id)
        built = telecomts_evidence(record, family_id=membership[row.scenario_id]["group_id"],
                                   family_status=membership[row.scenario_id]["evidence_status"])
        prediction = predictions[row.scenario_id].diagnosis.prediction if predictions else None
        packet = built.registry.packet(source=built.source, question=row.question, kpi_state=built.kpi_state,
                                       text_state=built.text_state, retrieval_status="not_run",
                                       detector_prediction=prediction)
        check_audits(packet, built.registry.audits())
        packets.append(packet.model_dump(mode="json"))
        audits.extend({"case_id": row.id, **a.model_dump(mode="json")} for a in built.registry.audits())
        rejections.extend({"case_id": row.id, "scenario_id": row.scenario_id, **r.model_dump(mode="json")}
                          for r in built.rejections)
    manifest = {
        "version": "rca-evidence-v1", "release_id": dataset.release_id, "cases": len(packets),
        "packets_sha256": _jsonl(output_dir / "packets.jsonl", packets),
        "audits_sha256": _jsonl(output_dir / "audits.jsonl", audits),
        "rejections_sha256": _jsonl(output_dir / "rejections.jsonl", rejections),
        "kpi_records": sum(len(p["kpi_evidence"]) for p in packets), "rejected": len(rejections),
        "detector_predictions": "attached" if predictions else "absent",
        "retrieval_status": "not_run; specification excerpts are added by Task 4.5 snapshots",
        "unavailable_sources": UNAVAILABLE_SOURCES,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_raw(dataset, output_dir: Path, raw_dir: Path) -> dict:
    """Recompute every statistic from raw metrics.csv rows with independent code."""
    packets = [json.loads(line) for line in (output_dir / "packets.jsonl").read_text().splitlines()]
    checked, worst = 0, 0.0
    for packet in packets:
        record = dataset.load_scenario(packet["source"]["case_id"])
        family = record.provenance.source_file.removesuffix("/processed/chunked.jsonl")
        with (raw_dir / family / "raw" / "metrics.csv").open(encoding="utf-8", newline="") as stream:
            rows = {row["timestamp"]: row for row in csv.DictReader(stream)}
        for item in packet["kpi_evidence"]:
            start = datetime.fromisoformat(item["time_start"])
            stamps = [s for s in rows if start <= datetime.fromisoformat(s) <= datetime.fromisoformat(item["time_end"])]
            values = [float(rows[s][item["channel"]]) for s in stamps if rows[s][item["channel"]] != ""]
            if len(stamps) != item["sample_count"] or len(values) != item["valid_count"]:
                raise ValueError(f"{item['evidence_id']} sample counts differ from the raw file")
            expected = {"mean": sum(values) / len(values), "minimum": min(values),
                        "maximum": max(values)}[item["statistic"]]
            difference = abs(expected - item["value"])
            if not math.isclose(expected, item["value"], rel_tol=1e-12, abs_tol=RAW_TOLERANCE):
                raise ValueError(f"{item['evidence_id']} differs from raw by {difference}")
            worst, checked = max(worst, difference), checked + 1
    return {"statistics_checked": checked, "max_abs_difference": worst, "tolerance": RAW_TOLERANCE}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    tele = commands.add_parser("telecomts", help="evidence packets for the golden RCA cases")
    tele.add_argument("--rca", type=Path, default=Path("data/golden/rca.jsonl"))
    tele.add_argument("--detector-predictions", type=Path,
                      help="Task 4.2 rca_cases.jsonl; its diagnosis prediction enters each packet")
    tele.add_argument("--output-dir", type=Path, required=True)
    raw = commands.add_parser("verify-raw", help="recompute packet statistics from raw metrics.csv")
    raw.add_argument("--output-dir", type=Path, required=True)
    raw.add_argument("--raw-dir", type=Path, default=Path("data/telecomts/raw"))
    for command in (tele, raw):
        command.add_argument("--release-dir", type=Path, required=True)
        command.add_argument("--allow-exploratory", action="store_true")
    table = commands.add_parser("table", help="evidence for one BubbleRAN or SpotLight window")
    table.add_argument("--dataset", choices=("bubbleran", "spotlight"), required=True)
    table.add_argument("--csv", type=Path, required=True)
    table.add_argument("--start", required=True, help="first row (BubbleRAN) or start time (SpotLight)")
    table.add_argument("--end", required=True, help="last row (BubbleRAN) or end time (SpotLight)")
    args = parser.parse_args(argv)
    try:
        if args.command == "table":
            from clauseguard.rca.sources import bubbleran_evidence, spotlight_evidence

            built = (bubbleran_evidence(args.csv, start_row=int(args.start), end_row=int(args.end))
                     if args.dataset == "bubbleran"
                     else spotlight_evidence(args.csv, start_time=args.start, end_time=args.end))
            result = {"source": built.source.model_dump(mode="json"), "time": built.time,
                      "kpi_evidence": [r.model_dump(mode="json") for r in built.registry.records("kpi")],
                      "rejections": [r.model_dump(mode="json") for r in built.rejections]}
        else:
            from clauseguard.datasets.scenarios import ScenarioDataset

            dataset = ScenarioDataset(args.release_dir, allow_exploratory=args.allow_exploratory)
            if args.command == "telecomts":
                from clauseguard.eval.golden import load_rca
                from clauseguard.rca.detector import read_cases

                if args.output_dir.exists():
                    raise FileExistsError("output exists; choose a new evidence directory")
                predictions = None
                if args.detector_predictions:
                    predictions = {r.scenario_id: r for r in read_cases(args.detector_predictions)}
                    if any(r.release_id != dataset.release_id for r in predictions.values()):
                        raise ValueError("detector predictions belong to a different release")
                rows = load_rca(args.rca)
                if predictions is not None and {r.scenario_id for r in rows} - set(predictions):
                    raise ValueError("detector predictions do not cover every RCA case")
                result = build_telecomts(dataset, rows, args.output_dir, predictions=predictions)
            else:
                result = verify_raw(dataset, args.output_dir, args.raw_dir)
    except (OSError, ValueError, KeyError) as exc:
        print(f"evidence {args.command} failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
