"""Select or replay the frozen Toto 2.0 forecast-error detector on TelecomTS."""

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="fit scales on normal train windows; pick thresholds on dev")
    train.add_argument("--xgboost-reference", type=Path, required=True)
    predict = commands.add_parser("predict", help="replay a saved detector on one release split")
    predict.add_argument("--detector-run-dir", type=Path, required=True)
    predict.add_argument("--split", choices=("dev", "test"), default="test")
    for command in (train, predict):
        command.add_argument("--release-dir", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache/toto"))
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--batch-size", type=int, default=64)
        command.add_argument("--bootstrap-replicates", type=int, default=1000)
        command.add_argument("--allow-exploratory", action="store_true",
                             help="allow a provisional run when recording independence is unresolved")
    args = parser.parse_args(argv)
    try:
        from clauseguard.baselines.toto import predict_saved_detector, run_toto
        from clauseguard.datasets.scenarios import ScenarioDataset

        if args.output_dir.exists():
            raise FileExistsError("output exists; choose a new experiment directory")
        if args.batch_size <= 0 or args.bootstrap_replicates < 0:
            raise ValueError("batch size must be positive and bootstrap count must be nonnegative")
        dataset = ScenarioDataset(args.release_dir, allow_exploratory=args.allow_exploratory)
        options = {"cache_dir": args.cache_dir, "device": args.device, "batch_size": args.batch_size,
                   "bootstrap_replicates": args.bootstrap_replicates,
                   "allow_exploratory": args.allow_exploratory}
        if args.command == "train":
            result = run_toto(dataset, args.output_dir, xgboost_reference=args.xgboost_reference,
                              **options)
        else:
            result = predict_saved_detector(dataset, args.detector_run_dir, args.output_dir,
                                            split=args.split, **options)
    except ImportError:
        print("Install Toto on Python 3.12+ with: uv sync --python 3.12 --extra dev --extra baselines "
              "--extra toto", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"Toto {args.command} failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
