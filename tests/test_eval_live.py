from __future__ import annotations

import json

import pytest

from clauseguard.api.query_service import QueryService
from clauseguard.api.schemas import SearchHit
from clauseguard.api.search_service import SearchDependencyError
from clauseguard.cli.run_eval import main, settings_record
from clauseguard.config import Settings
from clauseguard.eval import release
from clauseguard.eval.release import ScenarioMembership, build_release
from clauseguard.eval.runner import EvalConfigError, replay, run_live
from clauseguard.llm.clients import GenerationUnavailable


def _hit(rank: int, spec_id: str, section: str) -> SearchHit:
    return SearchHit(
        id=f"point-{spec_id}-{rank}",
        score=1.0 / rank,
        text=f"Passage {rank}.",
        spec_id=spec_id,
        section=section,
        source_file="spec.pdf",
        chunk_id=f"{spec_id}::{section}::{rank}",
    )


RANKINGS = {
    "Question one?": [
        _hit(1, "3GPP TS 38.331", "9.9 Other"),
        _hit(2, "3GPP TS 28.552", "5.1 Title"),
    ],
    "Question two?": [_hit(1, "3GPP TS 38.331", "9.9 Other")],
    "Question three?": [_hit(1, "3GPP TS 38.331", "5.1 Title")],
}


class FakeSearch:
    def __init__(self, fail_on: dict[str, Exception] | None = None):
        self.calls: list[tuple[str, int]] = []
        self.fail_on = fail_on or {}

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        self.calls.append((query, top_k))
        if query in self.fail_on:
            raise self.fail_on[query]
        return RANKINGS[query]


class FakeRouter:
    """Answers question one from E2, abstains on question two, is unavailable for three."""

    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, task: str) -> str:
        self.prompts.append(prompt)
        if "Question one?" in prompt:
            return json.dumps({"answer": "One.", "evidence_ids": ["E2"]})
        if "Question two?" in prompt:
            return json.dumps({"answer": "not found", "evidence_ids": []})
        raise GenerationUnavailable("all configured generation models are unavailable")


def _rows(result) -> list[dict]:
    return [json.loads(line) for line in (result.path / "rows.jsonl").read_text().splitlines()]


def test_retrieval_mode_saves_raw_rankings_and_scores(tmp_path, golden):
    search = FakeSearch()

    result = run_live(
        "retrieval",
        search_service=search,
        paths=golden,
        output_root=tmp_path / "out",
        settings={"retrieval": {"collection": "fixture"}},
    )

    assert search.calls == [("Question one?", 10), ("Question two?", 10), ("Question three?", 10)]
    assert result.status == "complete"
    rows = _rows(result)
    assert rows[0]["input"] == {"question": "Question one?", "top_k": 10}
    assert rows[0]["retrieval"]["passages"][1] == {
        "rank": 2,
        "id": "point-3GPP TS 28.552-2",
        "score": 0.5,
        "spec_id": "3GPP TS 28.552",
        "section": "5.1 Title",
        "source_file": "spec.pdf",
        "chunk_id": "3GPP TS 28.552::5.1 Title::2",
        "text": "Passage 2.",
    }
    assert rows[0]["answer"] is None
    assert "search" in rows[0]["timings_ms"]
    assert "expected_answer" not in json.dumps(rows)  # golden answers never enter outputs

    run = json.loads((result.path / "run.json").read_text())
    assert (run["mode"], run["status"], run["answers_captured"]) == ("retrieval", "complete", False)
    assert run["settings"] == {"retrieval": {"collection": "fixture"}}
    assert run["rows"] == {"total": 3, "ok": 3, "failed": 0, "excluded": []}

    headline = result.metrics["retrieval"]["headline"]
    assert headline["mrr"] == 0.75  # (1/2 for q_001 + 1 for q_003) / 2
    assert result.metrics["abstention"]["status"] == "unavailable"


def test_query_mode_reuses_the_recorded_search(tmp_path, golden):
    search, router = FakeSearch(), FakeRouter()

    result = run_live(
        "query",
        search_service=search,
        query_service=QueryService(search, router),
        paths=golden,
        output_root=tmp_path / "out",
    )

    assert len(search.calls) == 3  # one search per row; answers reuse it
    assert len(router.prompts) == 3
    assert "Passage 2." in router.prompts[0]  # the recorded hits are what the model saw

    rows = _rows(result)
    assert rows[0]["answer"] == {
        "answer": "One.",
        "citations": [
            {
                "spec_id": "3GPP TS 28.552",
                "section": "5.1 Title",
                "source_file": "spec.pdf",
                "chunk_id": "3GPP TS 28.552::5.1 Title::2",
            }
        ],
    }
    assert rows[1]["answer"] == {"answer": "not found", "citations": []}
    assert rows[2]["status"] == "failed"
    assert rows[2]["error"] == {
        "type": "GenerationUnavailable",
        "message": "all configured generation models are unavailable",
    }
    assert rows[2]["retrieval"]["passages"]  # the ranking is kept even when answering fails

    assert result.status == "row_failures"
    abstention = result.metrics["abstention"]
    assert abstention["correct_abstention"] == {
        "count": 1,
        "denominator": 1,
        "rate": 1.0,
        "failed": 0,
    }
    assert abstention["false_abstention"] == {"count": 0, "denominator": 1, "rate": 0.0}
    assert result.metrics["retrieval"]["headline"]["mrr"] == 0.25  # q_003 failed -> 0


def test_query_mode_excludes_rca_rows_with_a_reason(tmp_path, golden, monkeypatch):
    rca_row = {
        "id": "rca_001",
        "scenario_id": "scn_001",
        "origin": "real",
        "inputs": "kpi_only",
        "question": "Question three?",
        "kpi_evidence": [
            {
                "channel": "RRU.PrbUsedDl",
                "baseline": {"start": "0", "end": "1"},
                "comparison": {"start": "2", "end": "3"},
                "baseline_value": 1.0,
                "comparison_value": 2.0,
                "direction": "increase",
            }
        ],
        "source_labels": {"fault_label": "congestion"},
        "expected_conclusion": "Saturation.",
        "uncertainty": "Small sample.",
        "expected_citations": [
            {
                "spec_id": "3GPP TS 38.331",
                "section": "5.1 Title",
                "role": "mechanism",
                "explanation": "E.",
            }
        ],
    }
    (golden.golden_dir / "rca.jsonl").write_text(json.dumps(rca_row) + "\n")
    with open(golden.golden_dir / "annotations.jsonl", "a") as f:
        annotation = {
            "id": "rca_001",
            "status": "approved",
            "reviewer": "Reviewer",
            "reviewed_on": "2026-10-16",
            "membership": "test",
            "families": ["direct"],
            "evidence": [
                {
                    "claim": "C.",
                    "alternatives": [
                        {
                            "spec_id": "3GPP TS 38.331",
                            "version": "18.0.0",
                            "sha256": "b" * 64,
                            "clause": "5.1",
                            "pdf_page": 1,
                        }
                    ],
                }
            ],
        }
        f.write(json.dumps(annotation) + "\n")
    # Stand-in for a validated issue #3 scenario release (see tests/test_golden_release.py).
    golden.scenario_release.mkdir()
    (golden.scenario_release / "release.json").write_text("{}")
    scenario = {
        "split": "test",
        "kpi_reason": "",
        "origin": "real",
        "text_reasons": {"detection": "no_approved_text", "category": "no_approved_text"},
    }
    membership = ScenarioMembership("fake-release", "scored", {"scn_001": scenario})
    monkeypatch.setattr(release, "load_scenario_membership", lambda _: membership)
    assert build_release(golden).ok

    search = FakeSearch()
    result = run_live(
        "query",
        search_service=search,
        query_service=QueryService(search, FakeRouter()),
        paths=golden,
        output_root=tmp_path / "out",
    )

    run = json.loads((result.path / "run.json").read_text())
    assert run["rows"]["excluded"] == [
        {"id": "rca_001", "reason": "query generation is specification-only"}
    ]
    assert len(search.calls) == 3  # the excluded RCA row is never searched
    assert result.metrics["retrieval"]["denominators"]["answerable"] == 2


def test_row_failures_are_saved_and_the_run_continues(tmp_path, golden):
    error = SearchDependencyError()
    error.__cause__ = ConnectionError("https://user:secret@host refused")
    search = FakeSearch(fail_on={"Question one?": error})

    result = run_live(
        "retrieval", search_service=search, paths=golden, output_root=tmp_path / "out"
    )

    rows = _rows(result)
    assert rows[0]["status"] == "failed"
    assert rows[0]["error"] == {
        "type": "SearchDependencyError",
        "message": "caused by ConnectionError",
    }
    assert "secret" not in (result.path / "rows.jsonl").read_text()
    assert [row["status"] for row in rows[1:]] == ["ok", "ok"]
    assert result.status == "row_failures"
    denominators = result.metrics["retrieval"]["denominators"]
    assert (denominators["failed"], denominators["successful"]) == (1, 1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": 5}, "top_k must be between k=10 and 50"),
        ({"top_k": 51}, "top_k must be between"),
        ({"split": "train"}, "split"),
    ],
)
def test_invalid_configuration_fails_before_any_call(tmp_path, golden, kwargs, message):
    search = FakeSearch()

    with pytest.raises(EvalConfigError, match=message):
        run_live(
            "retrieval", search_service=search, paths=golden, output_root=tmp_path / "out", **kwargs
        )
    assert search.calls == []
    assert not (tmp_path / "out").exists()


def test_query_mode_needs_a_query_service(tmp_path, golden):
    with pytest.raises(EvalConfigError, match="query service"):
        run_live("query", search_service=FakeSearch(), paths=golden, output_root=tmp_path / "out")


def test_failing_release_check_stops_before_any_call(tmp_path, golden):
    golden.manifest.unlink()
    search = FakeSearch()

    with pytest.raises(EvalConfigError, match="golden release check failed"):
        run_live("retrieval", search_service=search, paths=golden, output_root=tmp_path / "out")
    assert search.calls == []


def test_unexpected_errors_leave_a_partial_unfinished_run(tmp_path, golden):
    search = FakeSearch(fail_on={"Question two?": RuntimeError("bug")})

    with pytest.raises(RuntimeError):
        run_live("retrieval", search_service=search, paths=golden, output_root=tmp_path / "out")

    [run_dir] = (tmp_path / "out").iterdir()
    assert json.loads((run_dir / "run.json").read_text())["status"] == "running"
    assert len((run_dir / "rows.jsonl").read_text().splitlines()) == 1  # q_001 was kept
    with pytest.raises(EvalConfigError, match="did not finish"):
        replay(run_dir, paths=golden, output_root=tmp_path / "replays")


def test_live_run_replays_to_the_same_scores(tmp_path, golden):
    search = FakeSearch()
    live = run_live(
        "query",
        search_service=search,
        query_service=QueryService(search, FakeRouter()),
        paths=golden,
        output_root=tmp_path / "out",
    )

    replayed = replay(live.path, paths=golden, output_root=tmp_path / "out")

    assert replayed.metrics == live.metrics


def test_settings_record_never_contains_credentials():
    settings = Settings(nvidia_api_key="nvapi-secret-value", _env_file=None)

    record = settings_record(settings, "query")

    assert "nvapi-secret-value" not in json.dumps(record)
    assert record["generation"]["actual_model"].startswith("unknown")
    assert len(record["generation"]["prompt_sha256"]) == 64


def test_cli_refuses_live_runs_without_an_api_key(tmp_path, golden, monkeypatch, capsys):
    monkeypatch.setenv("NVIDIA_API_KEY", "")

    code = main(
        ["retrieval", f"--golden-dir={golden.golden_dir}", f"--output-root={tmp_path / 'out'}"]
    )

    assert code == 2
    assert "NVIDIA_API_KEY is not set" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_metrics_count_failures_by_error_type(tmp_path, golden):
    search = FakeSearch(fail_on={"Question one?": SearchDependencyError()})

    result = run_live(
        "retrieval", search_service=search, paths=golden, output_root=tmp_path / "out"
    )

    assert result.metrics["errors"] == {"SearchDependencyError": 1}
    assert result.metrics["cohorts"]["split"]["dev"]["errors"] == {"SearchDependencyError": 1}
    assert result.metrics["cohorts"]["split"]["test"]["errors"] == {}
