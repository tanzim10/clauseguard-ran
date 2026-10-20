"""Rebuild the documented fixture run (hand-made rankings, not real retrieval).

Run from the repository root after the golden release changes:

    uv run python tests/fixtures/eval/make_fixture_run.py
"""

import shutil
from pathlib import Path

from clauseguard.eval.artifacts import Passage, RetrievalOutput, RowInput, RowRecord, RunDirectory
from clauseguard.eval.runner import DEFAULT_PATHS, load_golden_set

out = Path("tests/fixtures/eval/fixture-retrieval-run")
if out.exists():
    shutil.rmtree(out)
golden = load_golden_set(DEFAULT_PATHS)
directory = RunDirectory(out)
out.mkdir(parents=True)
(out / "rows.jsonl").touch()


def passage(rank, spec_id, clause):
    return Passage(
        rank=rank,
        id=f"fixture-{rank}",
        score=round(1 - rank / 20, 3),
        spec_id=spec_id,
        section=f"{clause} (fixture heading)",
        source_file="fixture",
        chunk_id=f"{spec_id}::{clause}::fixture",
        text="Fixture passage text.",
    )


answerable_index = 0
for item in golden.items:
    filler = [("fixture-filler", f"0.{n}") for n in range(1, 11)]
    ranking = filler[:10]
    if item.reference.answerable:
        gold = min(item.reference.groups[0])
        position = answerable_index % 5  # gold at rank 1..4, or missing on every 5th row
        answerable_index += 1
        if position < 4:
            ranking = filler[:position] + [gold] + filler[position:9]
    record = RowRecord(
        id=item.id,
        type=item.row.type,
        split=item.split,
        families=item.families,
        answerable=item.reference.answerable,
        input=RowInput(question=item.row.question, top_k=10),
        status="ok",
        retrieval=RetrievalOutput(
            requested_top_k=10, passages=[passage(r, *c) for r, c in enumerate(ranking, 1)]
        ),
    )
    directory.append_row(record)

directory.write_run(
    {
        "run_id": "fixture-retrieval-run",
        "label": "FIXTURE: hand-made rankings for documentation and tests, not a real retrieval run",
        "mode": "retrieval",
        "status": "complete",
        "identities": {"golden_release": golden.release["release_id"], "split": None},
        "settings": {"retrieval": {"collection": "fixture"}},
        "answers_captured": False,
        "rows": {"total": len(golden.items)},
    }
)
print(golden.release["release_id"], len(golden.items))
