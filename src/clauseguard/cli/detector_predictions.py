"""Freeze the Task 4.2 detector selection, save its predictions, or reproduce them."""

import argparse
import json
import sys
from pathlib import Path


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="choose and seal the detector from development results")
    freeze.add_argument("--candidate", action="append", required=True, metavar="NAME=DIR",
                        help="repeatable Task 3 experiment, e.g. xgboost=artifacts/experiments/xgboost-full-data")
    freeze.add_argument("--exposure-dir", action="append", type=Path, default=[],
                        help="directories searched for earlier test-split runs to disclose")
    freeze.add_argument("--frozen-on", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    predict = commands.add_parser("predict", help="RCA case predictions and out-of-fold SLM candidates")
    predict.add_argument("--rca", type=Path, default=Path("data/golden/rca.jsonl"))
    predict.add_argument("--folds", type=int, default=5)
    predict.add_argument("--skip-cross-fit", action="store_true")
    predict.add_argument("--output-dir", type=Path, required=True)
    reproduce = commands.add_parser("reproduce", help="re-predict from saved checkpoints and compare")
    reproduce.add_argument("--output-dir", type=Path, required=True)
    for command in (freeze, predict, reproduce):
        command.add_argument("--release-dir", type=Path, required=True)
        command.add_argument("--allow-exploratory", action="store_true",
                             help="run provisionally when recording independence is unresolved")
    for command in (predict, reproduce):
        command.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from clauseguard.baselines.tree_data import require_ready
        from clauseguard.datasets.scenarios import ScenarioDataset
        from clauseguard.eval.golden import load_rca
        from clauseguard.rca import cross_fit, detector, detector_report

        dataset = ScenarioDataset(args.release_dir, allow_exploratory=args.allow_exploratory)
        require_ready(dataset, ("detection", "category"), allow_exploratory=args.allow_exploratory)
        if args.command == "freeze":
            candidates = dict(item.split("=", 1) for item in args.candidate)
            result = detector.freeze_selection(dataset, {k: Path(v) for k, v in candidates.items()},
                                               args.output, frozen_on=args.frozen_on,
                                               exposure_dirs=args.exposure_dir)
        elif args.command == "predict":
            selection = detector.load_selection(args.selection)
            if args.output_dir.exists():
                raise FileExistsError("output exists; choose a new prediction directory")
            rows = load_rca(args.rca)
            case_ids = [row.scenario_id for row in rows]
            records = detector.predict_cases(dataset, selection, case_ids)
            args.output_dir.mkdir(parents=True)
            rca = {"cases": "rca_cases.jsonl",
                   "cases_sha256": detector.write_jsonl(args.output_dir / "rca_cases.jsonl", records),
                   "summary": detector_report.summarize_cases(records),
                   "saved_run_agreement": detector.saved_run_agreement(records, selection),
                   "case_table": detector_report.case_table(
                       dataset, selection, records, {row.scenario_id: row.id for row in rows})}
            evaluation = {"rca_cases": detector_report.evaluate_against_release(dataset, records)}
            cross = None
            if not args.skip_cross_fit:
                cross = cross_fit.cross_fit(dataset, selection, args.output_dir / "cross_fit",
                                            folds=args.folds, case_ids=case_ids)
                candidates = detector.read_cases(args.output_dir / "cross_fit" / cross["predictions"])
                cross["summary"] = detector_report.summarize_cases(candidates)
                evaluation["slm_candidates"] = detector_report.evaluate_against_release(dataset, candidates)
            result = {"selection_sha256": selection["manifest_sha256"], "rca": rca, "cross_fit": cross,
                      "evaluation": evaluation}
            _write_json(args.output_dir / "report.json", result)
            (args.output_dir / "report.md").write_text(
                detector_report.render_report(selection, rca, cross, None), encoding="utf-8")
        else:
            selection = detector.load_selection(args.selection)
            report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
            if report["selection_sha256"] != selection["manifest_sha256"]:
                raise ValueError("predictions belong to a different selection")
            saved = detector.read_cases(args.output_dir / report["rca"]["cases"])
            again = detector.predict_cases(dataset, selection, [r.scenario_id for r in saved])
            if again != saved:
                raise ValueError("RCA case predictions differ after reload")
            result = {"rca_cases_reproduced": len(again)}
            if report["cross_fit"] is not None:
                result["cross_fit"] = cross_fit.reproduce_cross_fit(dataset, selection,
                                                                    args.output_dir / "cross_fit")
    except ImportError:
        print("Install detector dependencies: uv sync --extra dev --extra baselines", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"detector {args.command} failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result if args.command != "predict" else
                     {"rca": result["rca"]["summary"], "cross_fit_windows":
                      result["cross_fit"]["windows"] if result["cross_fit"] else 0},
                     indent=2, sort_keys=True, allow_nan=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
