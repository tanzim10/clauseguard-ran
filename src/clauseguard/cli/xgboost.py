"""Train XGBoost on the RF feature contract or reload a selected checkpoint."""

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="fit on train and select on dev; test is untouched")
    train.add_argument("--task", choices=("detection", "category", "both"), default="both")
    train.add_argument("--training-selection", type=Path)
    train.add_argument("--rf-reference", type=Path, required=True)
    train.add_argument("--allow-exploratory", action="store_true",
                       help="run provisionally when recording independence is unresolved")
    predict = commands.add_parser("predict", help="score a checkpoint without refitting")
    predict.add_argument("--model-run-dir", type=Path, required=True)
    predict.add_argument("--split", choices=("dev", "test"), default="test")
    predict.add_argument("--allow-exploratory", action="store_true",
                         help="score provisionally when recording independence is unresolved")
    for command in (train, predict):
        command.add_argument("--release-dir", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args(argv)
    try:
        from clauseguard.baselines.xgboost import predict_saved_model, run_xgboost
        from clauseguard.datasets.scenarios import ScenarioDataset

        if args.bootstrap_replicates < 0:
            raise ValueError("bootstrap replicates must be nonnegative")
        if args.output_dir.exists():
            raise FileExistsError("output exists; choose a new experiment directory")
        dataset = ScenarioDataset(args.release_dir, allow_exploratory=args.allow_exploratory)
        if args.command == "train":
            tasks = ("detection", "category") if args.task == "both" else (args.task,)
            result = run_xgboost(
                dataset, args.output_dir, tasks=tasks, rf_reference=args.rf_reference,
                training_selection=args.training_selection,
                bootstrap_replicates=args.bootstrap_replicates,
                allow_exploratory=args.allow_exploratory,
            )
        else:
            result = predict_saved_model(
                dataset, args.model_run_dir, args.output_dir, split=args.split,
                bootstrap_replicates=args.bootstrap_replicates,
                allow_exploratory=args.allow_exploratory,
            )
    except ImportError:
        print("Install XGBoost dependencies: uv sync --extra dev --extra baselines", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"XGBoost {args.command} failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
