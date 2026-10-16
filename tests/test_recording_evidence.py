import csv
import json

import pytest

from clauseguard.datasets.evidence import build_recording_evidence
from clauseguard.datasets.telecomts import file_sha256


def test_raw_mapping_preserves_separate_recording_approval(release_sources):
    s = release_sources
    evidence = build_recording_evidence(s["scenarios"], s["acquired"], s["output"])
    assert len(evidence["records"]) == 39
    assert all(r["raw_match"] for r in evidence["records"])
    assert all(t["review"]["status"] == "unresolved" for t in evidence["traces"].values())
    assert sum(r["fault_label"] == "normal" for r in evidence["records"]) == 6
    assert sum(not r["category_reason"] for r in evidence["records"]) == 30
    mapping = list(csv.DictReader((s["output"] / "window_sources.csv").open()))
    assert len(mapping) == 39
    assert all(int(r["raw_end_row"]) - int(r["raw_start_row"]) == 127 for r in mapping)


def test_checksum_failure_is_not_a_recording_match(release_sources):
    s = release_sources
    path = next((s["acquired"] / "raw").rglob("metrics.csv"))
    path.write_text(path.read_text().replace("0.1", "0.2", 1))
    with pytest.raises(ValueError, match="checksum"):
        build_recording_evidence(s["scenarios"], s["acquired"], s["output"])


@pytest.mark.parametrize("mutation, expected", [
    ("numeric", "raw numeric mismatch"), ("categorical", "raw categorical mismatch"),
    ("timestamp", "duplicate raw CSV timestamp"), ("rounding", None),
])
def test_raw_differences_are_reported_with_numeric_tolerance(release_sources, mutation, expected):
    s = release_sources
    path = next((s["acquired"] / "raw").rglob("metrics.csv"))
    rows = list(csv.DictReader(path.open()))
    if mutation == "numeric":
        rows[0]["DL_BLER"] = "999999"
    elif mutation == "categorical":
        rows[0]["UL_Protocol"] = "UDP"
    elif mutation == "timestamp":
        rows[0]["timestamp"] = rows[1]["timestamp"]
    else:
        rows[0]["DL_BLER"] = str(float(rows[0]["DL_BLER"]) + 1e-10)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = s["acquired"] / "supplements.json"
    manifest = json.loads(manifest_path.read_text())
    name = str(path.relative_to(s["acquired"] / "raw"))
    for entry in manifest["files"]:
        if entry["source_file"] == name:
            entry["sha256"] = file_sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    evidence = build_recording_evidence(s["scenarios"], s["acquired"], s["output"])
    if expected is None:
        assert all(r["raw_match"] for r in evidence["records"])
    else:
        assert any(expected in r["raw_reason"] for r in evidence["records"])
    assert all(t["review"]["status"] == "unresolved" for t in evidence["traces"].values())


def test_synthetic_verification_requires_parent_evidence(release_sources):
    s = release_sources
    name = next(n for n in s["reviews"] if "/synthetic/" in n)
    s["reviews"][name]["parent_recording_ids"] = []
    with pytest.raises(ValueError, match="parent evidence"):
        build_recording_evidence(s["scenarios"], s["acquired"], s["output"], reviews=s["reviews"])


@pytest.mark.parametrize("malformed", [False, True])
def test_duplicate_or_malformed_scenarios_fail_with_context(release_sources, malformed):
    s = release_sources
    if malformed:
        s["scenarios"].write_text("not json\n")
    else:
        first = s["scenarios"].open().readline()
        with s["scenarios"].open("a") as stream:
            stream.write(first)
    with pytest.raises(ValueError, match="scenarios.jsonl:1" if malformed else "duplicate scenario ID"):
        build_recording_evidence(s["scenarios"], s["acquired"], s["output"])
