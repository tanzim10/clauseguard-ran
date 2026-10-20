from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from clauseguard.cli.run_eval import main
from clauseguard.eval.artifacts import RowRecord, RunDirectory
from clauseguard.eval.release import ReleasePaths
from clauseguard.eval.runner import EvalConfigError, load_golden_set, replay


def _passage(rank: int, spec_id: str, section: str) -> dict:
    return {
        "rank": rank,
        "id": f"p{rank}",
        "score": 1.0 / rank,
        "spec_id": spec_id,
        "section": section,
        "source_file": "spec.pdf",
        "chunk_id": f"{spec_id}::{section}::{rank}",
        "text": "Passage text.",
    }


def _records(top_k: int = 10, with_answers: bool = False) -> list[dict]:
    q1 = {
        "id": "q_001",
        "type": "spec_qa",
        "split": "dev",
        "families": ["direct"],
        "answerable": True,
        "input": {"question": "Question one?", "top_k": top_k},
        "status": "ok",
        "retrieval": {
            "requested_top_k": top_k,
            "passages": [
                _passage(1, "3GPP TS 38.331", "9.9 Other"),
                _passage(2, "3GPP TS 28.552", "5.1 Title"),
            ],
        },
    }
    q2 = {
        "id": "q_002",
        "type": "spec_qa",
        "split": "dev",
        "families": ["no_answer"],
        "answerable": False,
        "input": {"question": "Question two?", "top_k": top_k},
        "status": "ok",
        "retrieval": {"requested_top_k": top_k, "passages": []},
    }
    q3 = {
        "id": "q_003",
        "type": "spec_qa",
        "split": "test",
        "families": ["difficult"],
        "answerable": True,
        "input": {"question": "Question three?", "top_k": top_k},
        "status": "failed",
        "error": {"type": "SearchDependencyError", "message": "qdrant unavailable"},
    }
    if with_answers:
        q1["answer"] = {
            "answer": "One.",
            "citations": [
                {
                    "spec_id": "3GPP TS 28.552",
                    "section": "5.1 Title",
                    "source_file": "spec.pdf",
                    "chunk_id": "c",
                }
            ],
        }
        q2["answer"] = {"answer": "not found", "citations": []}
    return [q1, q2, q3]


def _source_run(tmp_path, golden: ReleasePaths, records=None, **run_overrides) -> Path:
    records = _records() if records is None else records
    directory = RunDirectory.create(tmp_path / "source", "retrieval")
    for record in records:
        directory.append_row(RowRecord.model_validate(record))
    release_id = json.loads(golden.manifest.read_text())["release_id"]
    run = {
        "run_id": directory.path.name,
        "mode": "retrieval",
        "status": "row_failures",
        "identities": {"golden_release": release_id, "split": None},
        "settings": {"retrieval": {"collection": "fixture"}},
        "answers_captured": False,
        "rows": {"total": len(records)},
    } | run_overrides
    directory.write_run(run)
    return directory.path


def test_replay_writes_a_new_complete_run(tmp_path, golden):
    source = _source_run(tmp_path, golden)
    before = sorted(p.read_bytes() for p in source.iterdir())

    result = replay(source, paths=golden, output_root=tmp_path / "out")

    assert result.path.parent == tmp_path / "out"
    assert sorted(p.name for p in result.path.iterdir()) == [
        "metrics.json",
        "rows.jsonl",
        "run.json",
    ]
    assert sorted(p.read_bytes() for p in source.iterdir()) == before

    run = json.loads((result.path / "run.json").read_text())
    assert (run["mode"], run["status"], result.status) == ("replay", "row_failures", "row_failures")
    assert run["source_run"]["mode"] == "retrieval"
    assert run["settings"] == {"retrieval": {"collection": "fixture"}}
    assert run["rows"] == {"total": 3, "ok": 2, "failed": 1, "excluded": []}
    assert run["identities"]["policy"] and run["identities"]["inputs"]

    retrieval = result.metrics["retrieval"]
    assert retrieval["denominators"] == {
        "answerable": 2,
        "successful": 1,
        "failed": 1,
        "no_answer_excluded": 1,
    }
    assert retrieval["headline"]["mrr"] == 0.25  # q_001 ranks its clause 2nd; q_003 failed
    assert retrieval["successful_only"]["mrr"] == 0.5
    assert result.metrics["abstention"]["status"] == "unavailable"
    assert result.metrics["cohorts"]["split"]["test"]["retrieval"]["headline"]["mrr"] == 0.0

    rows = [json.loads(line) for line in (result.path / "rows.jsonl").read_text().splitlines()]
    assert rows[0]["scores"]["retrieval"]["first_relevant_rank"] == 2
    assert rows[0]["retrieval"]["passages"][0]["section"] == "9.9 Other"  # outputs kept as saved
    assert rows[1]["scores"]["retrieval"] is None  # no-answer row
    assert rows[2]["scores"] == {"failed": True}


def test_replay_reproduces_scores(tmp_path, golden):
    first = replay(_source_run(tmp_path, golden), paths=golden, output_root=tmp_path / "out")
    second = replay(first.path, paths=golden, output_root=tmp_path / "out")

    assert second.metrics == first.metrics
    assert second.path != first.path


def test_replay_makes_no_network_calls(tmp_path, golden, monkeypatch):
    source = _source_run(tmp_path, golden)

    def refuse(*args, **kwargs):
        raise AssertionError("replay opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    replay(source, paths=golden, output_root=tmp_path / "out")


def test_replay_scores_captured_answers(tmp_path, golden):
    source = _source_run(
        tmp_path, golden, _records(with_answers=True), answers_captured=True, mode="query"
    )

    abstention = replay(source, paths=golden, output_root=tmp_path / "out").metrics["abstention"]

    assert abstention["correct_abstention"] == {
        "count": 1,
        "denominator": 1,
        "rate": 1.0,
        "failed": 0,
    }
    assert abstention["false_abstention"] == {"count": 0, "denominator": 1, "rate": 0.0}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda run, rows: run.update(status="running"), "did not finish"),
        (lambda run, rows: run.update(rows={"total": 5}), "lists 5 rows but rows.jsonl has 3"),
        (lambda run, rows: run["identities"].update(golden_release="0" * 16), "golden release"),
        (lambda run, rows: rows[0].update(id="q_999"), "rows not in the golden set: q_999"),
        (lambda run, rows: rows[0]["input"].update(question="Edited?"), "q_001 input differs"),
    ],
)
def test_replay_rejects_invalid_sources(tmp_path, golden, change, message):
    records = _records()
    run_overrides = {
        "identities": {
            "golden_release": json.loads(golden.manifest.read_text())["release_id"],
            "split": None,
        },
        "rows": {"total": 3},
        "status": "row_failures",
    }
    change(run_overrides, records)
    source = _source_run(tmp_path, golden, records, **run_overrides)

    with pytest.raises(EvalConfigError, match=message):
        replay(source, paths=golden, output_root=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_replay_rejects_unknown_truncated_depth(tmp_path, golden):
    source = _source_run(tmp_path, golden, _records(top_k=5))

    with pytest.raises(EvalConfigError, match=r"q_001: ranking depth 5 .* k=10"):
        replay(source, paths=golden, output_root=tmp_path / "out")


def test_replay_rejects_malformed_and_duplicate_rows(tmp_path, golden):
    source = _source_run(tmp_path, golden)
    rows = (source / "rows.jsonl").read_text().splitlines()

    (source / "rows.jsonl").write_text(rows[0] + "\n" + rows[0] + "\n" + rows[2] + "\n")
    with pytest.raises(EvalConfigError, match="duplicate row id 'q_001'"):
        replay(source, paths=golden, output_root=tmp_path / "out")

    broken = json.loads(rows[0]) | {"retrieval": None}
    (source / "rows.jsonl").write_text(json.dumps(broken) + "\n" + rows[1] + "\n" + rows[2] + "\n")
    with pytest.raises(EvalConfigError, match="ok rows need retrieval output"):
        replay(source, paths=golden, output_root=tmp_path / "out")


def test_replay_fails_before_writing_when_the_release_check_fails(tmp_path, golden):
    source = _source_run(tmp_path, golden)
    golden.manifest.unlink()

    with pytest.raises(EvalConfigError, match="golden release check failed"):
        replay(source, paths=golden, output_root=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_run_directories_are_never_reused(tmp_path):
    first = RunDirectory.create(tmp_path, "replay")
    second = RunDirectory.create(tmp_path, "replay")

    assert first.path != second.path
    with pytest.raises(FileExistsError):
        first.path.mkdir()


def test_golden_set_filters_by_split(golden):
    assert [item.id for item in load_golden_set(golden, "test").items] == ["q_003"]
    with pytest.raises(EvalConfigError, match="split"):
        load_golden_set(golden, "train")


def _cli_args(golden: ReleasePaths, tmp_path) -> list[str]:
    return [
        f"--golden-dir={golden.golden_dir}",
        f"--corpus-manifest={golden.corpus_manifest}",
        f"--scenario-release={golden.scenario_release}",
        f"--output-root={tmp_path / 'out'}",
    ]


def test_cli_replay_exit_codes(tmp_path, golden, capsys):
    with_failure = _source_run(tmp_path, golden)
    records = _records()
    records[2] = records[0] | {
        "id": "q_003",
        "split": "test",
        "input": {"question": "Question three?", "top_k": 10},
    }
    all_ok = _source_run(tmp_path, golden, records, status="complete")

    assert main(["replay", f"--from={with_failure}", *_cli_args(golden, tmp_path)]) == 1
    assert main(["replay", f"--from={all_ok}", *_cli_args(golden, tmp_path)]) == 0
    assert "clause_recall@10" in capsys.readouterr().out
    assert main(["replay", f"--from={tmp_path / 'missing'}", *_cli_args(golden, tmp_path)]) == 2
    assert main(["replay", f"--from={all_ok}", "--k", "5", "1", *_cli_args(golden, tmp_path)]) == 2


def test_documented_fixture_run_replays_offline(tmp_path):
    # The README walks through this example; rebuild it with
    # tests/fixtures/eval/make_fixture_run.py whenever the golden release changes.
    source = Path("tests/fixtures/eval/fixture-retrieval-run")

    result = replay(source, output_root=tmp_path)

    headline = result.metrics["retrieval"]["headline"]
    assert result.status == "complete"
    assert result.metrics["retrieval"]["denominators"]["answerable"] == 20
    # Gold sits at rank 1, 2, 3, 4 or is missing, cycling over the 20 answerable rows.
    assert headline["mrr"] == pytest.approx(4 * (1 + 1 / 2 + 1 / 3 + 1 / 4) / 20)
    assert headline["precision@1"] == pytest.approx(4 / 20)
