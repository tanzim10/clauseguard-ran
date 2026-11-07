"""Train PatchTST on the shared KPI windows, reload a checkpoint, or compare dev results."""

import argparse
import json
import sys
from pathlib import Path


def _progress(task, run, row):
    print(f"[{task}/{run}] epoch {row['epoch']:>3} train {row['train_loss']:.4f} "
          f"dev {row['dev_loss']:.4f} F1 {row['dev_macro_f1']:.3f} ({row['seconds']:.1f}s)",
          file=sys.stderr, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="fit on train and select on dev; test is untouched")
    train.add_argument("--task", choices=("detection", "category", "both"), default="both")
    train.add_argument("--xgboost-reference", type=Path, required=True)
    train.add_argument("--max-epochs", type=int, default=100)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu",
                       help="only cpu is bit-reproducible; mps/cuda runs are recorded as such")
    train.add_argument("--skip-seed-repeats", action="store_true",
                       help="tune with seed 42 only; seeds 43/44 are left unfinished")
    predict = commands.add_parser("predict", help="score a checkpoint without refitting")
    predict.add_argument("--model-run-dir", type=Path, required=True)
    predict.add_argument("--split", choices=("dev", "test"), default="test")
    predict.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    compare = commands.add_parser("compare", help="dev-only table against earlier baselines")
    compare.add_argument("--experiment", action="append", required=True, metavar="NAME=DIR",
                         help="repeatable, e.g. xgboost=artifacts/experiments/xgboost-full-data")
    compare.add_argument("--output-dir", type=Path, required=True)
    for command in (train, predict):
        command.add_argument("--release-dir", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--bootstrap-replicates", type=int, default=1000)
        command.add_argument("--allow-exploratory", action="store_true",
                             help="run provisionally when recording independence is unresolved")
    args = parser.parse_args(argv)
    try:
        from clauseguard.baselines.patchtst import (
            compare_with_baselines,
            predict_saved_model,
            run_patchtst,
        )
        from clauseguard.datasets.scenarios import ScenarioDataset

        if args.output_dir.exists():
            raise FileExistsError("output exists; choose a new experiment directory")
        if args.command == "compare":
            experiments = dict(item.split("=", 1) for item in args.experiment)
            result = compare_with_baselines({k: Path(v) for k, v in experiments.items()},
                                            args.output_dir)
        else:
            if args.bootstrap_replicates < 0:
                raise ValueError("bootstrap replicates must be nonnegative")
            dataset = ScenarioDataset(args.release_dir, allow_exploratory=args.allow_exploratory)
            if args.command == "train":
                tasks = ("detection", "category") if args.task == "both" else (args.task,)
                result = run_patchtst(
                    dataset, args.output_dir, tasks=tasks, xgboost_reference=args.xgboost_reference,
                    bootstrap_replicates=args.bootstrap_replicates,
                    allow_exploratory=args.allow_exploratory, max_epochs=args.max_epochs,
                    patience=args.patience, device=args.device,
                    repeat_seeds=not args.skip_seed_repeats, progress=_progress)
            else:
                result = predict_saved_model(
                    dataset, args.model_run_dir, args.output_dir, split=args.split,
                    device=args.device, bootstrap_replicates=args.bootstrap_replicates,
                    allow_exploratory=args.allow_exploratory)
    except ImportError:
        print("Install PatchTST dependencies: uv sync --extra dev --extra baselines --extra timesnet",
              file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"PatchTST {args.command} failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
