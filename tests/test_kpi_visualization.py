"""Plotting checks use tiny offline observations and reject mismatched saved metadata."""

import json

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from clauseguard.datasets.telecomts import file_sha256
from clauseguard.kpi.visualization import visualize_release


@pytest.fixture
def plot_release(tmp_path):
    release = tmp_path / "release"
    raw = tmp_path / "raw"
    release.mkdir()
    members, traces = {}, {}
    for index, origin in enumerate(("normal", "synthetic", "real")):
        source = f"{origin}/processed/chunked.jsonl"
        path = raw / source
        path.parent.mkdir(parents=True)
        observation = {"sampling_rate": 10, "start_time": "2025-01-01 00:00:00",
                       "end_time": "2025-01-01 00:00:12.700",
                       "KPIs": {name: list(range(128)) for name in ("RSRP", "DL_BLER", "UL_SNR")}}
        path.write_text(json.dumps(observation) + "\n")
        traces[source] = {"processed_sha256": file_sha256(path)}
        members[str(index)] = {
            "source_file": source, "source_row": 1, "origin": origin, "split": "train",
            "start_time": observation["start_time"], "end_time": observation["end_time"],
            "detection_target": int(index != 0), "category_target": 0 if index == 1 else None,
            "rca_label": ("normal", "Antenna Failure", "Jamming")[index],
        }
    data = {"splits.json": {"membership": members},
            "labels.json": {"category": {"Antenna Failure": 0}},
            "evidence.json": {"traces": traces}}
    for name, payload in data.items():
        (release / name).write_text(json.dumps(payload))
    (release / "release.json").write_text(json.dumps({
        "release_id": "test-release", "files": {name: file_sha256(release / name) for name in data},
    }))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"release_id": "test-release", "scenario_ids": ["1", "2"]}))
    return release, raw, selection


def test_visualize_saved_release_without_altering_inputs(plot_release, tmp_path):
    release, raw, selection = plot_release
    before = {path: file_sha256(path) for root in (release, raw) for path in root.rglob("*")
              if path.is_file()}
    paths = visualize_release(release, tmp_path / "plots", selection_path=selection, raw_dir=raw)
    assert len(paths) == 8
    assert all(path.stat().st_size > 100 for path in paths.values())
    assert all(file_sha256(path) == digest for path, digest in before.items())


def test_visualization_rejects_changed_metadata(plot_release, tmp_path):
    release, raw, selection = plot_release
    (release / "splits.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        visualize_release(release, tmp_path / "plots", selection_path=selection, raw_dir=raw)


def test_visualization_rejects_selection_that_drops_faults(plot_release, tmp_path):
    release, raw, selection = plot_release
    selection.write_text(json.dumps({"release_id": "test-release", "scenario_ids": ["1"]}))
    with pytest.raises(ValueError, match="removed evaluation cases or faults"):
        visualize_release(release, tmp_path / "plots", selection_path=selection, raw_dir=raw)
