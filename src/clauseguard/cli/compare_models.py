"""Compare saved KPI model experiments on the development split; never touches test data."""

import argparse
import json
import sys
from pathlib import Path

from clauseguard.baselines.comparison import compare_experiments, render_markdown


def _model(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("use NAME=EXPERIMENT_DIR")
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=_model, required=True,
                        help="NAME=EXPERIMENT_DIR containing selection.json; repeat per model")
    parser.add_argument("--task", action="append", choices=("detection", "category"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        models = dict(args.model)
        if len(models) != len(args.model):
            raise ValueError("model names must be unique")
        if args.output_dir.exists():
            raise FileExistsError("output exists; choose a new output directory")
        tasks = sorted(set(args.task or ("detection", "category")))
        reports = [compare_experiments(models, task) for task in tasks]
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "comparison.json").write_text(
            json.dumps(reports, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        (args.output_dir / "comparison.md").write_text(render_markdown(reports), encoding="utf-8")
    except (OSError, ValueError, KeyError) as exc:
        print(f"Model comparison failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"tasks": tasks, "models": list(models), "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
