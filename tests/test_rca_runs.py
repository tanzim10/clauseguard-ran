"""Task 4.8: reading saved run records into evaluator views."""

import pytest
from rca_eval_helpers import make_record, write_runs

from clauseguard.eval.rca_runs import RunRecordError, load_run_records, view_from_record


def test_answered_view_exposes_causes_claims_and_packet_ids():
    view = view_from_record(make_record())
    assert view.answered and view.complete and not view.abstained
    assert view.category_set == {"Co-Channel Interference (Severe)"}
    assert [c["claim_id"] for c in view.claims] == ["C1", "C2", "C3"]
    assert view.packet_evidence_ids == {"K1", "K2", "S1", "S2"}


def test_abstained_and_failed_views_have_no_categories():
    abstained = view_from_record(make_record(outcome="abstained"))
    failed = view_from_record(make_record(status="failed", stage="generation", reason="down"))
    assert abstained.abstained and abstained.category_set == frozenset() and abstained.claims == ()
    assert not failed.complete and failed.category_set == frozenset()


def test_validation_failure_keeps_parsed_claims_but_no_categories():
    view = view_from_record(make_record(status="failed", stage="validation", reason="numeric.v2"))
    assert not view.answered and len(view.claims) == 3 and view.category_set == frozenset()


def test_rejects_other_record_versions_and_missing_fields():
    record = make_record()
    with pytest.raises(RunRecordError, match="not an rca-run-v1"):
        view_from_record({**record, "version": "other"})
    del record["response"]
    with pytest.raises(RunRecordError, match="malformed"):
        view_from_record(record)


def test_later_directory_replaces_generation_failure_and_records_superseded(tmp_path):
    failed = make_record("rca_001", status="failed", stage="generation", reason="down", run_id="r-failed")
    good = make_record("rca_001", run_id="r-good")
    first = write_runs(tmp_path / "a", [failed, make_record("rca_002", run_id="r-two")])
    second = write_runs(tmp_path / "b", [good])
    views = load_run_records([first, second])
    assert [v.case_id for v in views] == ["rca_001", "rca_002"]
    assert views[0].run_id == "r-good" and views[0].superseded == ("r-failed",)


def test_earlier_good_run_is_kept_over_later_generation_failure(tmp_path):
    good = make_record("rca_001", run_id="r-good")
    failed = make_record("rca_001", status="failed", stage="generation", reason="down", run_id="r-failed")
    views = load_run_records([write_runs(tmp_path / "a", [good]), write_runs(tmp_path / "b", [failed])])
    assert views[0].run_id == "r-good" and views[0].superseded == ("r-failed",)


def test_empty_directory_is_an_error(tmp_path):
    (tmp_path / "runs").mkdir()
    with pytest.raises(RunRecordError, match="no run records"):
        load_run_records([tmp_path])
