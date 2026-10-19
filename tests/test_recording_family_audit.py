import importlib.util
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from clauseguard.datasets.contracts import NUMERIC_CHANNELS, RecordingReview
from clauseguard.datasets.release import prepare_release
from clauseguard.datasets.telecomts import file_sha256

spec = importlib.util.spec_from_file_location(
    "recording_audit", Path(__file__).parents[1] / "scripts/audit_recording_families.py")
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def trace(name, values, split="train"):
    return audit.Trace(name, values, {"splits": [split]})


def test_finds_late_shifted_copy_and_reports_channels_and_split():
    rng = np.random.default_rng(42)
    a, b = (rng.normal(size=(300, len(NUMERIC_CHANNELS))) for _ in range(2))
    b[213:253, :2] = a[173:213, :2]
    links, coverage = audit.find_links([trace("original", a), trace("copy", b, "test")])
    assert len(links) == 1
    link = links[0]
    assert link["source_b_minus_source_a_row_offset"] == 40
    assert set(link["anchors_by_channel"]) == set(NUMERIC_CHANNELS[:2])
    assert link["anchors_by_channel"][NUMERIC_CHANNELS[0]]["first"] == 173
    assert link["anchors_by_channel"][NUMERIC_CHANNELS[0]]["last"] == 197
    assert link["equal_values_by_channel"][NUMERIC_CHANNELS[0]] == 40
    assert link["crosses_current_splits"]
    assert all(c["eligible_segments_by_source"] == {"original": 285, "copy": 285}
               for c in coverage)


def test_constants_and_single_channel_copy_do_not_create_links():
    rng = np.random.default_rng(7)
    a, b = (np.zeros((80, len(NUMERIC_CHANNELS))) for _ in range(2))
    a[:, 0] = b[:, 0] = rng.normal(size=80)
    links, coverage = audit.find_links([trace("a", a), trace("b", b)])
    assert links == []
    assert coverage[0]["confirmed_cross_source_segment_pairs"] == 65
    assert coverage[1]["eligible_segments_by_source"] == {"a": 0, "b": 0}


def test_repetitive_keys_are_counted_instead_of_expanding_all_pairs():
    values = np.tile(np.arange(16), 20).astype(float)
    a = np.repeat(values[:, None], len(NUMERIC_CHANNELS), axis=1)
    links, coverage = audit.find_links([trace("a", a), trace("b", a)], max_occurrences=4)
    assert links == []
    assert all(c["skipped_repetitive_keys"] == 16 for c in coverage)
    assert all(c["skipped_segment_occurrences"] == 610 for c in coverage)


def test_signed_zero_matches_but_missing_values_do_not():
    a = np.arange(32, dtype=float)
    b = a.copy()
    b[0] = -0.0
    ka, positions = audit._fingerprints(a, 16, 8)
    kb, _ = audit._fingerprints(b, 16, 8)
    assert np.array_equal(ka, kb)
    a[16] = np.nan
    _, positions = audit._fingerprints(a, 16, 8)
    assert positions.tolist() == [0]


def test_audit_writes_unresolved_proposals_and_rejects_tampered_sources(
    release_sources, recording_evidence, tmp_path,
):
    s = release_sources
    prepare_release(s["scenarios"], s["output"], s["output"], trials=64)
    before = {p.name: file_sha256(p) for p in s["output"].iterdir()}
    report = audit.audit(s["output"], s["acquired"], tmp_path / "audit")
    assert report["candidate_links"]
    assert len(report["sources"]) == 9
    assert not report["scored_ready"]
    proposals = json.loads((tmp_path / "audit/recording_reviews.proposed.json").read_text())
    assert all(RecordingReview.model_validate(r).status == "unresolved" for r in proposals.values())
    assert all(not r["reviewer"] and r["recording_id"] is None for r in proposals.values())
    assert before == {p.name: file_sha256(p) for p in s["output"].iterdir()}
    with pytest.raises(FileExistsError):
        audit.audit(s["output"], s["acquired"], tmp_path / "audit")
    raw = next((s["acquired"] / "raw").rglob("metrics.csv"))
    with raw.open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="source checksum mismatch"):
        audit.audit(s["output"], s["acquired"], tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()
