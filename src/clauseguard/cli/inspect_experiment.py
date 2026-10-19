"""Validate an experiment release and save cohort identities and readiness diagnostics."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from clauseguard.baselines.data import combined_diagnosis_ids, experiment_cohort
from clauseguard.datasets.acquisition import write_manifest
from clauseguard.datasets.contracts import SPLITS
from clauseguard.datasets.scenarios import DEFAULT_RELEASE, ScenarioDataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", action="append", choices=("detection", "category"))
    parser.add_argument("--modality", action="append", choices=("kpi", "text", "paired"))
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise FileExistsError("output exists; choose a new output directory")
        dataset = ScenarioDataset(args.release_dir)
        tasks = sorted(set(args.task or ("detection", "category")))
        modalities = sorted(set(args.modality or ("kpi",)))
        cohorts = {
            f"{task}.{modality}.{split}": experiment_cohort(
                dataset, task=task, modality=modality, split=split,
            ) for task in tasks for modality in modalities for split in SPLITS
        }
        report = {
            "version": "1", "release_validated": True, "release_id": dataset.release_id,
            "source_hash": dataset.manifest["scenarios_sha256"], "mode": dataset.splits["mode"],
            "ready": all(cohort.ready for cohort in cohorts.values()),
            "quality": dataset.quality,
            "cohorts": {
                name: {
                    "file": f"{name}.json", "cohort_id": cohort.cohort_id,
                    "input_hash": cohort.input_hash, "split_hash": cohort.split_hash,
                    "ready": cohort.ready, "reasons": cohort.reasons,
                    "coverage": cohort.coverage[cohort.split],
                    "exclusions": dict(Counter(cohort.exclusions.values())),
                } for name, cohort in cohorts.items()
            },
            "combined_diagnosis_ids": {
                f"{modality}.{split}": combined_diagnosis_ids(dataset, split=split, modality=modality)
                for modality in modalities for split in SPLITS
            },
        }
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, cohort in cohorts.items():
            write_manifest(args.output_dir / f"{name}.json", cohort.model_dump(mode="json"))
        write_manifest(args.output_dir / "readiness.json", report)
        print(json.dumps({"release_validated": True, "release_id": dataset.release_id,
                          "ready": report["ready"], "report": str(args.output_dir / "readiness.json")}))
        return 0 if report["ready"] else 3
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Experiment inspection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
