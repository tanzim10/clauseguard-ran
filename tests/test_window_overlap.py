import json
from dataclasses import replace

import pytest

from clauseguard.cli.audit_window_overlap import main
from clauseguard.datasets.release import prepare_release
from clauseguard.datasets.window_overlap import WindowSpan, select_nonoverlapping, summarize_spans


def span(sid, start, end, *, source="trace", fault=0, category=None):
    return WindowSpan(sid, source, start, end, fault, category, "group-" + source, "train")


def test_earliest_end_selection_is_disjoint_and_deterministic():
    rows = [span("long", 1, 300), span("first", 1, 128), span("overlap", 33, 160),
            span("touching", 128, 255), span("next", 129, 256),
            span("other-source", 1, 128, source="other")]
    kept = select_nonoverlapping(rows, preserve_faults=False)
    assert {row.scenario_id for row in kept} == {"first", "next", "other-source"}
    assert kept == select_nonoverlapping(list(reversed(rows)), preserve_faults=False)
    assert all(row in rows for row in kept)


def test_preserve_faults_keeps_every_rare_case_and_original_split():
    rows = [span("normal1", 1, 128), span("normal2", 33, 160),
            span("fault1", 33, 160, fault=1, category=2),
            span("fault2", 65, 192, fault=1, category=3)]
    kept = select_nonoverlapping(rows, preserve_faults=True)
    assert {row.scenario_id for row in kept} == {"normal1", "fault1", "fault2"}
    summary = summarize_spans(kept)
    assert summary["windows"] == 3
    assert summary["category"] == {"2": 1, "3": 1}
    assert summary["splits"]["train"]["windows"] == 3


def test_selection_rejects_duplicate_ids_and_invalid_intervals():
    with pytest.raises(ValueError, match="duplicate"):
        select_nonoverlapping([span("a", 1, 128), span("a", 129, 256)], preserve_faults=True)
    with pytest.raises(ValueError, match="invalid"):
        select_nonoverlapping([span("a", 129, 128)], preserve_faults=False)


def test_training_only_selection_keeps_every_evaluation_window():
    rows = [span("train1", 1, 128), span("train2", 33, 160),
            replace(span("dev1", 1, 128), split="dev"),
            replace(span("dev2", 33, 160), split="dev"),
            replace(span("test1", 1, 128), split="test")]
    kept = select_nonoverlapping(rows, preserve_faults=True, training_only=True)
    assert {row.scenario_id for row in kept} == {"train1", "dev1", "dev2", "test1"}


def test_audit_saves_ids_without_changing_release_and_refuses_overwrite(
        release_sources, recording_evidence, tmp_path):
    source = release_sources
    prepare_release(source["scenarios"], source["output"], source["output"], trials=128)
    before = (source["output"] / "splits.json").read_bytes()
    output = tmp_path / "audit"
    args = ["--release-dir", str(source["output"]), "--output-dir", str(output)]
    assert main(args) == 0
    report = json.loads((output / "overlap_report.json").read_text())
    assert report["purpose"] == "audit_only" and not report["scored_ready"]
    selection = json.loads((output / "train_nonoverlap_normal_keep_faults.json").read_text())
    assert selection["release_id"] == report["release_id"]
    assert len(selection["scenario_ids"]) == len(set(selection["scenario_ids"]))
    assert (source["output"] / "splits.json").read_bytes() == before
    assert main(args) == 1
