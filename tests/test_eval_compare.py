from __future__ import annotations

import json

import pytest

from clauseguard.api.query_service import QueryService
from clauseguard.api.schemas import SearchHit
from clauseguard.api.search_service import SearchDependencyError
from clauseguard.cli.run_eval import main
from clauseguard.eval.compare import _classify, compare_runs
from clauseguard.eval.metrics import MetricPolicy
from clauseguard.eval.runner import EvalConfigError, replay, run_live


def _hit(rank: int, spec_id: str, section: str) -> SearchHit:
    return SearchHit(
        id=f"{spec_id}-{section}",
        score=1.0 / rank,
        text="Passage.",
        spec_id=spec_id,
        section=section,
        source_file="spec.pdf",
        chunk_id=f"{spec_id}::{section}",
    )


OTHER = ("3GPP TS 38.331", "9.9 Other")
Q1_GOLD = ("3GPP TS 28.552", "5.1 Title")
Q3_GOLD = ("3GPP TS 38.331", "5.1 Title")


class Search:
    """Returns fixed rankings per question; a question mapped to an exception raises it."""

    def __init__(self, rankings):
        self.rankings = rankings
        self.calls = 0

    def search(self, query, top_k):
        self.calls += 1
        result = self.rankings[query]
        if isinstance(result, Exception):
            raise result
        return [_hit(rank, *hit) for rank, hit in enumerate(result, start=1)]


BASELINE = {
    "Question one?": [OTHER, Q1_GOLD],  # gold at rank 2
    "Question two?": [OTHER],
    "Question three?": [Q3_GOLD],
}


def _run(tmp_path, golden, rankings, **kwargs):
    return run_live(
        "retrieval",
        search_service=Search(rankings),
        paths=golden,
        output_root=tmp_path / "runs",
        **kwargs,
    )


def test_improved_row_and_settings_change(tmp_path, golden):
    before = _run(tmp_path, golden, BASELINE, settings={"retrieval": {"collection": "dense"}})
    after = _run(
        tmp_path,
        golden,
        BASELINE | {"Question one?": [Q1_GOLD, OTHER]},  # gold moves to rank 1
        settings={"retrieval": {"collection": "hybrid"}},
    )

    comparison = compare_runs(after.path, before.path)

    assert comparison["equivalent"] is True
    assert comparison["non_equivalence"] == []
    assert comparison["experiment_variables"] == {
        "settings.retrieval.collection": {"previous": "dense", "current": "hybrid"}
    }
    assert comparison["rows"] == {
        "common": 3,
        "compared": 3,
        "added": [],
        "removed": [],
        "input_changed": [],
    }
    assert comparison["summary"] == {"improved": 1, "worsened": 0, "mixed": 0, "unchanged": 2}

    [row] = comparison["per_row"]
    assert row["id"] == "q_001"
    assert row["change"] == "improved"
    assert row["outputs_changed"] == ["ranking"]
    assert row["deltas"] == {
        "clause_recall@1": 1.0,
        "complete_reference_match@1": 1.0,
        "group_coverage@1": 1.0,
        "mrr": 0.5,
        "precision@1": 1.0,
    }
    assert comparison["metric_changes"]["mrr"] == {"improved": 1, "worsened": 0, "unchanged": 1}
    assert comparison["headline_delta"]["mrr"] == pytest.approx(0.25)  # (1+1)/2 - (0.5+1)/2


def test_new_failure_is_a_worsened_row(tmp_path, golden):
    before = _run(tmp_path, golden, BASELINE)
    after = _run(tmp_path, golden, BASELINE | {"Question three?": SearchDependencyError()})

    comparison = compare_runs(after.path, before.path)

    [row] = comparison["per_row"]
    assert (row["id"], row["change"]) == ("q_003", "worsened")
    assert row["deltas"]["mrr"] == -1.0
    assert row["outputs_changed"] == ["status", "ranking"]


def test_different_split_lists_removed_rows_and_is_not_equivalent(tmp_path, golden):
    full = _run(tmp_path, golden, BASELINE)
    dev = _run(tmp_path, golden, BASELINE, split="dev")

    comparison = compare_runs(dev.path, full.path)

    assert comparison["rows"]["removed"] == ["q_003"]
    assert comparison["rows"]["added"] == []
    assert comparison["equivalent"] is False
    assert comparison["non_equivalence"] == ["evaluated split changed", "row inputs changed"]


def test_changed_inputs_are_not_scored_against_each_other(tmp_path, golden):
    shallow = _run(tmp_path, golden, BASELINE, top_k=10)
    deep = _run(tmp_path, golden, BASELINE, top_k=20)

    comparison = compare_runs(deep.path, shallow.path)

    assert comparison["rows"]["input_changed"] == ["q_001", "q_002", "q_003"]
    assert comparison["rows"]["compared"] == 0
    assert "row inputs changed" in comparison["non_equivalence"]


def test_policy_change_is_not_equivalent_and_compares_common_cutoffs(tmp_path, golden):
    before = _run(tmp_path, golden, BASELINE)
    after = replay(
        before.path, paths=golden, output_root=tmp_path / "runs", policy=MetricPolicy(ks=(1, 10))
    )

    comparison = compare_runs(after.path, before.path)

    assert comparison["non_equivalence"] == ["metric policy changed"]
    assert comparison["experiment_variables"] == {
        "mode": {"previous": "retrieval", "current": "replay"}
    }
    assert set(comparison["metric_changes"]) == {
        "mrr",
        *(
            f"{m}@{k}"
            for k in (1, 10)
            for m in ("clause_recall", "group_coverage", "precision", "complete_reference_match")
        ),
    }


def test_golden_release_change_is_not_equivalent(tmp_path, golden):
    before = _run(tmp_path, golden, BASELINE)
    run = json.loads((before.path / "run.json").read_text())
    run["identities"]["golden_release"] = "0" * 16
    (before.path / "run.json").write_text(json.dumps(run))
    after = _run(tmp_path, golden, BASELINE)

    assert "golden release changed" in compare_runs(after.path, before.path)["non_equivalence"]


def test_abstention_changes_are_compared(tmp_path, golden):
    class Router:
        def __init__(self, abstain):
            self.abstain = abstain

        def generate(self, prompt, *, task):
            if "Question two?" in prompt and not self.abstain:
                return json.dumps({"answer": "Invented.", "evidence_ids": ["E1"]})
            if "Question two?" in prompt:
                return json.dumps({"answer": "not found", "evidence_ids": []})
            return json.dumps({"answer": "Answer.", "evidence_ids": ["E1"]})

    def query_run(abstain):
        search = Search(BASELINE)
        return run_live(
            "query",
            search_service=search,
            query_service=QueryService(search, Router(abstain)),
            paths=golden,
            output_root=tmp_path / "runs",
        )

    before, after = query_run(abstain=True), query_run(abstain=False)

    comparison = compare_runs(after.path, before.path)

    [row] = comparison["per_row"]
    assert (row["id"], row["change"], row["deltas"]) == (
        "q_002",
        "worsened",
        {"correct_abstention": -1.0},
    )
    assert row["outputs_changed"] == ["answer"]


def test_classify_reports_mixed_changes():
    assert _classify({"mrr": 0.5, "precision@1": -1.0}) == "mixed"
    assert _classify({}) == "unchanged"


def test_compare_to_writes_comparison_into_the_new_run(tmp_path, golden):
    before = _run(tmp_path, golden, BASELINE)

    after = _run(tmp_path, golden, BASELINE, compare_to=before.path)

    comparison = json.loads((after.path / "comparison.json").read_text())
    assert comparison["previous"]["run_id"] == before.path.name
    assert not (before.path / "comparison.json").exists()  # the previous run is untouched


def test_unusable_compare_to_fails_before_any_call(tmp_path, golden):
    search = Search(BASELINE)

    with pytest.raises(EvalConfigError, match="cannot compare"):
        run_live(
            "retrieval",
            search_service=search,
            paths=golden,
            output_root=tmp_path / "runs",
            compare_to=tmp_path / "missing",
        )
    assert search.calls == 0
    assert not (tmp_path / "runs").exists()


def test_cli_compare_prints_json(tmp_path, golden, capsys):
    before = _run(tmp_path, golden, BASELINE)
    after = _run(tmp_path, golden, BASELINE)

    assert main(["compare", str(after.path), str(before.path)]) == 0
    assert json.loads(capsys.readouterr().out)["summary"]["unchanged"] == 3
    assert main(["compare", str(after.path), str(tmp_path / "missing")]) == 2


def test_runs_without_metrics_cannot_be_compared(tmp_path, golden):
    # A source-only run (like the documented fixture) has rows but no metrics.json.
    run = _run(tmp_path, golden, BASELINE)
    (run.path / "metrics.json").unlink()
    search = Search(BASELINE)

    with pytest.raises(EvalConfigError, match="no metrics.json"):
        run_live(
            "retrieval",
            search_service=search,
            paths=golden,
            output_root=tmp_path / "new",
            compare_to=run.path,
        )
    assert search.calls == 0
    assert main(["compare", str(run.path), str(run.path)]) == 2
