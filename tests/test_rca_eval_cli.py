"""Task 4.8: evaluation CLI over saved run records and the real golden files (no live calls)."""

import json
from pathlib import Path

import pytest
from rca_eval_helpers import CLAIMS, make_record, write_runs

from clauseguard.cli import eval_rca
from clauseguard.eval.rca_eval import EvalPolicy, load_references
from clauseguard.eval.rca_judge import ClaimJudgment, claim_sha256

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "golden"


def args(tmp_path, *extra, name="out", runs=None):
    return ["--runs", str(runs or tmp_path / "runs"), "--output", str(tmp_path / name),
            "--rca-rows", str(GOLDEN / "rca.jsonl"), "--annotations", str(GOLDEN / "annotations.jsonl"),
            "--policy-file", str(GOLDEN / "rca_policy.json"), *extra]


@pytest.fixture
def runs(tmp_path):
    write_runs(tmp_path / "runs", [make_record("rca_001"), make_record("rca_002", outcome="abstained"),
                                   make_record("rca_003", status="failed", stage="generation", reason="down")])
    return tmp_path / "runs"


def test_golden_references_are_unapproved_and_exploratory_so_scored_results_are_blocked():
    refs = load_references(GOLDEN / "rca.jsonl", GOLDEN / "annotations.jsonl", GOLDEN / "rca_policy.json")
    assert len(refs) == 12
    assert not any(r.approved for r in refs.values()) and all(r.exploratory for r in refs.values())
    assert refs["rca_001"].source_label == "Co-Channel Interference (Severe)"
    assert ("3GPP TS 32.425", "A.9") in refs["rca_010"].expected_clauses


def test_overlay_adds_alternative_sets_and_abstention_file_adds_fixtures(tmp_path):
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps({"rca_001": {"acceptable_sets": [["Co-Channel Interference (Severe)"], ["Antenna Failure"]]}}))
    fixtures = tmp_path / "abstain.json"
    fixtures.write_text(json.dumps({"abstain_fx_1": {"answerable": False}}))
    refs = load_references(GOLDEN / "rca.jsonl", GOLDEN / "annotations.jsonl", GOLDEN / "rca_policy.json", overlay, fixtures)
    assert refs["rca_001"].approved and len(refs["rca_001"].acceptable_sets) == 2
    assert refs["abstain_fx_1"].fixture and refs["abstain_fx_1"].answerable is False


def test_no_judge_run_writes_all_files_and_is_reproducible(tmp_path, runs, capsys):
    assert eval_rca.main(args(tmp_path, "--no-judge", name="a", runs=runs)) == 0
    assert eval_rca.main(args(tmp_path, "--no-judge", name="b", runs=runs)) == 0
    for name in ("policy.json", "judgments.jsonl", "metrics.json", "summary.md", "eval.json"):
        assert (tmp_path / "a" / name).is_file()
    assert (tmp_path / "a" / "metrics.json").read_text() == (tmp_path / "b" / "metrics.json").read_text()
    metrics = json.loads((tmp_path / "a" / "metrics.json").read_text())
    assert metrics["human_review_status"] == "pending" and metrics["identification"]["scored"]["status"] == "blocked"
    assert metrics["status"]["failed_by_stage"] == {"generation": 1}
    assert json.loads((tmp_path / "a" / "policy.json").read_text())["evaluation_sha256"] == EvalPolicy().sha256
    assert "identification scored: blocked" in capsys.readouterr().out


def test_existing_output_directory_is_never_overwritten(tmp_path, runs):
    (tmp_path / "out").mkdir()
    assert eval_rca.main(args(tmp_path, "--no-judge", runs=runs)) == 2


def test_judge_equal_to_generator_is_refused(tmp_path, runs):
    assert eval_rca.main(args(tmp_path, "--judge-model", "gen/model-a", runs=runs)) == 2


def test_judging_without_a_key_is_a_start_error(tmp_path, runs, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    monkeypatch.chdir(tmp_path)
    assert eval_rca.main(args(tmp_path, runs=runs)) == 2


def test_judged_run_reports_support_and_exit_status_reflects_missing_decisions(tmp_path, runs, monkeypatch):
    def fake_judge(_args, views, policy):
        v = next(v for v in views if v.case_id == "rca_001")
        decisions = ["fully_supported", "unsupported"]
        return [ClaimJudgment(run_id=v.run_id, case_id=v.case_id, claim_id=c["claim_id"], claim_sha256=claim_sha256(c),
                              status="judged" if i < 2 else "judge_failed", decision=decisions[i] if i < 2 else None,
                              judge_model=policy.model, prompt_sha256=policy.prompt_sha256)
                for i, c in enumerate(CLAIMS)]

    monkeypatch.setattr(eval_rca, "_judge", fake_judge)
    assert eval_rca.main(args(tmp_path, runs=runs)) == 1
    support = json.loads((tmp_path / "out" / "metrics.json").read_text())["support"]
    assert support["status_counts"] == {"judge_failed": 1, "judged": 2}
    assert support["provisional_faithfulness"]["value"] == 0.5
    assert support["provisional_supported_coverage"]["value"] is None
    assert len((tmp_path / "out" / "judgments.jsonl").read_text().splitlines()) == 3


def test_unreadable_runs_are_a_start_error(tmp_path):
    (tmp_path / "empty" / "runs").mkdir(parents=True)
    assert eval_rca.main(args(tmp_path, "--no-judge", runs=tmp_path / "empty")) == 2
