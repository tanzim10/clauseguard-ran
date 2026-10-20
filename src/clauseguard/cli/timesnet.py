"""Train TimesNet on the shared KPI windows or reload a selected checkpoint."""

import argparse
import json
import sys
from pathlib import Path


def _progress(task, run, row):
    groups = " ".join(f"{group[-6:]}={stats['accuracy']:.2f}"
                      for group, stats in row["per_group"].items())
    print(f"[{task}/{run}] epoch {row['epoch']:>3} train {row['train_loss']:.4f} "
          f"dev {row['dev_loss']:.4f} sel {row['dev_selection_loss']:.4f} "
          f"F1 {row['dev_macro_f1']:.3f} ({row['seconds']:.1f}s) acc[{groups}]",
          file=sys.stderr, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="fit on train and select on dev; test is untouched")
    train.add_argument("--task", choices=("detection", "category", "both"), default="both")
    train.add_argument("--training-selection", type=Path)
    train.add_argument("--xgboost-reference", type=Path, required=True)
    train.add_argument("--max-epochs", type=int, default=100)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu",
                       help="only cpu is bit-reproducible; mps/cuda runs are recorded as such")
    train.add_argument("--skip-seed-repeats", action="store_true",
                       help="tune with seed 42 only; seeds 43/44 are left unfinished")
    train.add_argument("--recipe", choices=("issue33", "v2"), default="issue33",
                       help="v2: prior-corrected stopping, cosine LR decay, 3 inception kernels")
    train.add_argument("--head", choices=("flatten", "pooled"), default="flatten",
                       help="pooled: average over time before the classifier (position-independent)")
    train.add_argument("--recording-weights", action="store_true",
                       help="weight windows so every recording counts equally within its class")
    train.add_argument("--instance-norm", action="store_true",
                       help="standardize each window per channel (drops absolute KPI levels)")
    train.add_argument("--allow-exploratory", action="store_true",
                       help="run provisionally when recording independence is unresolved")
    predict = commands.add_parser("predict", help="score a checkpoint without refitting")
    predict.add_argument("--model-run-dir", type=Path, required=True)
    predict.add_argument("--split", choices=("dev", "test"), default="test")
    predict.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    predict.add_argument("--allow-exploratory", action="store_true",
                         help="score provisionally when recording independence is unresolved")
    for command in (train, predict):
        command.add_argument("--window-selection", type=Path,
                             help="audit selection applied to train, dev and test "
                                  "(e.g. nonoverlap_normal_keep_faults.json)")
        command.add_argument("--release-dir", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args(argv)
    try:
        from clauseguard.baselines.timesnet import predict_saved_model, run_timesnet
        from clauseguard.datasets.scenarios import ScenarioDataset

        if args.bootstrap_replicates < 0:
            raise ValueError("bootstrap replicates must be nonnegative")
        if args.output_dir.exists():
            raise FileExistsError("output exists; choose a new experiment directory")
        dataset = ScenarioDataset(args.release_dir, allow_exploratory=args.allow_exploratory)
        if args.command == "train":
            tasks = ("detection", "category") if args.task == "both" else (args.task,)
            result = run_timesnet(
                dataset, args.output_dir, tasks=tasks, xgboost_reference=args.xgboost_reference,
                training_selection=args.training_selection,
                bootstrap_replicates=args.bootstrap_replicates,
                allow_exploratory=args.allow_exploratory, max_epochs=args.max_epochs,
                patience=args.patience, device=args.device,
                repeat_seeds=not args.skip_seed_repeats, progress=_progress,
                recipe=args.recipe, instance_norm=args.instance_norm, head=args.head,
                recording_weights=args.recording_weights,
                window_selection=args.window_selection)
        else:
            result = predict_saved_model(
                dataset, args.model_run_dir, args.output_dir, split=args.split, device=args.device,
                bootstrap_replicates=args.bootstrap_replicates,
                allow_exploratory=args.allow_exploratory,
                window_selection=args.window_selection)
    except ImportError:
        print("Install TimesNet dependencies: uv sync --extra dev --extra baselines --extra timesnet",
              file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"TimesNet {args.command} failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
