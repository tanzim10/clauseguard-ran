"""Persist one task-aware release, with separate inference inputs and evaluation targets."""

import csv
import json
import os
import shutil
from pathlib import Path

from clauseguard.datasets.acquisition import write_manifest
from clauseguard.datasets.contracts import (
    LABEL_MAPPINGS,
    NUMERIC_CHANNELS,
    ModelInputs,
    NumericInputs,
)
from clauseguard.datasets.csv_export import iter_analysis_scenarios
from clauseguard.datasets.evidence import summarize_record, text_eligibility
from clauseguard.datasets.splits import assign_splits, build_quality, validate_membership
from clauseguard.datasets.telecomts import file_sha256

EVIDENCE_FILES = ("evidence.json", "recording_evidence.csv", "window_sources.csv")
RELEASE_FILES = ("splits.json", "quality.json", "membership.csv", "labels.json",
                 "inputs.kpi.jsonl", "inputs.text.jsonl", "targets.jsonl")


def project_inputs(record, *, modality="kpi", task=None) -> ModelInputs:
    """Only explicit observations enter the predictor payload. No source serialization."""
    if modality not in ("kpi", "text", "paired"):
        raise ValueError("invalid modality")
    if modality != "kpi" and (task not in ("detection", "category")
                              or text_eligibility(record, task)):
        raise ValueError("text is not approved for this task")
    numeric = None
    if modality != "text":
        channels = record.kpi_window.channels
        if record.kpi_window.sample_count != 128 or any(
            name not in channels or channels[name].kind != "numeric" for name in NUMERIC_CHANNELS
        ):
            raise ValueError("requires 128 steps and 16 named numeric channels")
        numeric = NumericInputs(
            channel_names=list(NUMERIC_CHANNELS), sampling_rate_hz=record.kpi_window.sampling_rate_hz,
            values=[[channels[name].values[i] for name in NUMERIC_CHANNELS] for i in range(128)],
        )
    return ModelInputs(kpi=numeric, text=record.nl_context if modality != "kpi" else None)


def _identity(manifest: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare_release(scenarios_path: Path, evidence_dir: Path, output_dir: Path, **split_settings) -> dict:
    """Keep blocked diagnostics; callers decide which task readiness is required."""
    if any((output_dir / name).exists() for name in (*RELEASE_FILES, "release.json")):
        raise FileExistsError("release exists; choose a new output directory")
    evidence = json.loads((evidence_dir / "evidence.json").read_text(encoding="utf-8"))
    if file_sha256(scenarios_path) != evidence["scenarios_sha256"]:
        raise ValueError("scenario checksum differs from recording evidence")
    if evidence.get("version") != "1":
        raise ValueError("unsupported evidence version")
    for name in EVIDENCE_FILES[1:]:
        if file_sha256(evidence_dir / name) != evidence["files"][name]:
            raise ValueError(f"evidence checksum mismatch: {name}")
    splits = assign_splits(evidence, **split_settings)
    validate_membership(splits, evidence)
    quality = build_quality(splits)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in EVIDENCE_FILES:
        if (evidence_dir / name).resolve() != (output_dir / name).resolve():
            if (output_dir / name).exists():
                raise FileExistsError(f"evidence destination exists: {name}")
            shutil.copyfile(evidence_dir / name, output_dir / name)
    write_manifest(output_dir / "splits.json", splits)
    write_manifest(output_dir / "quality.json", quality)
    write_manifest(output_dir / "labels.json", LABEL_MAPPINGS)
    columns = ["scenario_id", "group_id", "split", "evidence_status", "exclusion_reason",
               "source_file", "origin", "fault_label", "rca_label", "detection_eligible",
               "category_eligible", "category_reason", "detection_text_reason", "category_text_reason"]
    with (output_dir / "membership.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for sid, m in splits["membership"].items():
            writer.writerow({k: m[k] for k in columns if k in m} | {
                "detection_eligible": m["split"] != "excluded" and not m["kpi_reason"],
                "category_eligible": m["split"] != "excluded" and not m["kpi_reason"]
                    and not m["category_reason"],
                "detection_text_reason": m["text_reasons"]["detection"],
                "category_text_reason": m["text_reasons"]["category"],
            })
    seen = set()
    with (output_dir / "inputs.kpi.jsonl").open("w", encoding="utf-8") as kpi, \
            (output_dir / "inputs.text.jsonl").open("w", encoding="utf-8") as text, \
            (output_dir / "targets.jsonl").open("w", encoding="utf-8") as targets:
        for record in iter_analysis_scenarios(scenarios_path):
            sid = record.scenario_id
            if sid in seen or sid not in splits["membership"]:
                raise ValueError(f"duplicate or unknown scenario: {sid}")
            seen.add(sid)
            m = splits["membership"][sid]
            if any(m[k] != v for k, v in summarize_record(record).items()):
                raise ValueError(f"scenario differs from evidence: {sid}")
            targets.write(json.dumps({"scenario_id": sid, "detection": m["detection_target"],
                                      "category": m["category_target"]}) + "\n")
            if m["split"] == "excluded":
                continue
            kpi.write(json.dumps({"scenario_id": sid, "inputs": project_inputs(record).model_dump(
                exclude_none=True)}) + "\n")
            for task in ("detection", "category"):
                if not m["text_reasons"][task] and (task == "detection" or not m["category_reason"]):
                    text.write(json.dumps({"scenario_id": sid, "task": task,
                                           "inputs": {"text": record.nl_context}}) + "\n")
    if seen != set(splits["membership"]):
        raise ValueError("membership references scenarios absent from source")
    manifest = {
        "version": "1", "scenarios_path": os.path.relpath(scenarios_path.resolve(), output_dir.resolve()),
        "scenarios_sha256": evidence["scenarios_sha256"], "revision": evidence["revision"],
        "mode": splits["mode"], "numeric_channels": list(NUMERIC_CHANNELS),
        "files": {name: file_sha256(output_dir / name) for name in (*EVIDENCE_FILES, *RELEASE_FILES)},
    }
    manifest["release_id"] = _identity(manifest)
    write_manifest(output_dir / "release.json", manifest)
    return {"release_id": manifest["release_id"], "quality": quality}
