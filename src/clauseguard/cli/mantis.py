"""Train or replay the frozen MantisV2 TelecomTS baseline."""

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="fit both heads on train and select on dev")
    train.add_argument("--task", choices=("detection", "category", "both"), default="both")
    train.add_argument("--rf-reference", type=Path, required=True)
    train.add_argument("--xgboost-reference", type=Path, required=True)
    train.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache/mantis"))
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--allow-exploratory", action="store_true",
                       help="allow a provisional run when recording independence is unresolved")
    predict = commands.add_parser("predict", help="replay a saved head on one release split")
    predict.add_argument("--model-run-dir", type=Path, required=True)
    predict.add_argument("--split", choices=("dev", "test"), default="test")
    predict.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache/mantis"))
    predict.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    predict.add_argument("--batch-size", type=int, default=32)
    predict.add_argument("--allow-exploratory", action="store_true",
                         help="allow provisional scoring when recording independence is unresolved")
    for command in (train, predict):
        command.add_argument("--release-dir", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args(argv)
    try:
        from clauseguard.baselines.mantis import predict_saved_model, run_mantis
        from clauseguard.datasets.scenarios import ScenarioDataset

        if args.output_dir.exists():
            raise FileExistsError("output exists; choose a new experiment directory")
        if args.batch_size <= 0 or args.bootstrap_replicates < 0:
            raise ValueError("batch size must be positive and bootstrap count must be nonnegative")
        dataset = ScenarioDataset(args.release_dir, allow_exploratory=args.allow_exploratory)
        if args.command == "train":
            tasks = ("detection", "category") if args.task == "both" else (args.task,)
            result = run_mantis(
                dataset, args.output_dir, tasks=tasks, rf_reference=args.rf_reference,
                xgboost_reference=args.xgboost_reference, cache_dir=args.cache_dir,
                bootstrap_replicates=args.bootstrap_replicates,
                allow_exploratory=args.allow_exploratory, device=args.device,
                batch_size=args.batch_size)
            if any(row.get("status") == "failed" for row in result["tasks"].values()):
                print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
                return 1
        else:
            result = predict_saved_model(
                dataset, args.model_run_dir, args.output_dir, split=args.split,
                cache_dir=args.cache_dir, bootstrap_replicates=args.bootstrap_replicates,
                allow_exploratory=args.allow_exploratory, device=args.device,
                batch_size=args.batch_size)
    except ImportError:
        print("Install Mantis with: uv sync --extra dev --extra baselines --extra mantis",
              file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"Mantis {args.command} failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
