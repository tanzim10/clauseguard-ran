from copy import deepcopy

import pytest

from clauseguard.datasets.splits import assign_splits, build_quality, select_task_cohort


def test_one_family_membership_for_both_targets(recording_evidence):
    splits = assign_splits(recording_evidence, trials=256)
    assert splits == assign_splits(recording_evidence, trials=256)
    quality = build_quality(splits)
    assert quality["tasks"]["detection"]["kpi"]["ready"]
    assert quality["tasks"]["category"]["kpi"]["ready"]
    assert sum(quality["membership_counts"].values()) == 39
    seen = {}
    for m in splits["membership"].values():
        assert seen.setdefault(m["group_id"], m["split"]) == m["split"]
    for split in ("train", "dev", "test"):
        detection = set(select_task_cohort(splits, split=split, task="detection"))
        category = set(select_task_cohort(splits, split=split, task="category"))
        assert len(detection) == 13 and len(category) == 10
        assert category < detection
        assert any(splits["membership"][sid]["origin"] == "real" for sid in detection)
        assert not select_task_cohort(splits, split=split, task="detection", modality="text")


def test_unresolved_parent_excludes_whole_family(recording_evidence):
    evidence = deepcopy(recording_evidence)
    source = next(s for s in evidence["traces"] if "/synthetic/" in s)
    evidence["traces"][source]["review"]["status"] = "unresolved"
    splits = assign_splits(evidence, trials=128)
    excluded = [m for m in splits["membership"].values() if m["split"] == "excluded"]
    assert len(excluded) == 13
    assert {m["exclusion_reason"] for m in excluded} == {"recording_family_unresolved"}
    assert not build_quality(splits)["tasks"]["detection"]["kpi"]["ready"]
    exploratory = assign_splits(evidence, mode="exploratory", trials=128)
    quality = build_quality(exploratory)
    assert quality["tasks"]["detection"]["kpi"]["coverage_ready"]
    assert not quality["tasks"]["detection"]["kpi"]["ready"]


def test_raw_failure_does_not_create_exploratory_eligibility(recording_evidence):
    recording_evidence["records"][0]["raw_match"] = False
    splits = assign_splits(recording_evidence, mode="exploratory", trials=64)
    assert sum(m["split"] == "excluded" for m in splits["membership"].values()) == 13


def test_one_class_detection_remains_unready(recording_evidence):
    recording_evidence["records"] = [r for r in recording_evidence["records"]
                                       if r["fault_label"] == "abnormal"]
    quality = build_quality(assign_splits(recording_evidence, trials=128))
    assert not quality["tasks"]["detection"]["kpi"]["ready"]
    assert all(0 in c["missing_labels"] for c in
               quality["tasks"]["detection"]["kpi"]["splits"].values())


def test_invalid_settings_and_task_fail_clearly(recording_evidence):
    with pytest.raises(ValueError, match="ratios"):
        assign_splits(recording_evidence, ratios=(0.8, 0.2, 0))
    with pytest.raises(ValueError, match="split, task"):
        select_task_cohort({}, split="train", task="guess")


def test_unrelated_source_clocks_do_not_merge_families(recording_evidence):
    for record in recording_evidence["records"]:
        record["start_time"] = "2025-01-01 00:00:00"
        record["end_time"] = "2025-01-01 00:00:12.700"
    splits = assign_splits(recording_evidence, trials=128)
    assert len({m["group_id"] for m in splits["membership"].values()}) == 3
    assert build_quality(splits)["tasks"]["detection"]["kpi"]["ready"]
