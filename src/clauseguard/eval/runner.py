"""Golden evaluation runner (Week 5): score golden rows and write a run directory.

Three modes share one scorer and one artifact format:

- `retrieval`: search every golden row through `SearchService` and save the raw ranking.
- `query`: also answer specification rows through `QueryService.answer`, reusing that same
  ranking, and save the answer and citations.
- `replay`: re-score a finished run offline from its saved passages and answers. No service is
  called, and every check runs before the new directory is created.

Configuration and the golden release check are validated before any service call.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from clauseguard.api.query_service import QueryOutputError, QueryService
from clauseguard.api.search_service import (
    SearchConfigurationError,
    SearchDependencyError,
    SearchService,
)
from clauseguard.corpus.manifest import load_manifest
from clauseguard.datasets.schema import GoldenEvalRow, RcaRow
from clauseguard.eval.annotations import Annotation, load_annotations
from clauseguard.eval.artifacts import (
    AnswerCitation,
    AnswerOutput,
    ArtifactError,
    Passage,
    RetrievalOutput,
    RowError,
    RowInput,
    RowRecord,
    RunDirectory,
    read_run,
)
from clauseguard.eval.compare import compare_runs, read_comparable
from clauseguard.eval.golden import load_golden, load_rca
from clauseguard.eval.metrics import (
    DEFAULT_POLICY,
    MetricPolicy,
    Reference,
    RetrievalScore,
    RowOutcome,
    TruncatedRankingError,
    aggregate,
    clause_of,
    reference_for,
    score_answer,
    score_ranking,
)
from clauseguard.eval.release import (
    ANNOTATIONS_FILE,
    RCA_FILE,
    SPEC_QA_FILE,
    ReleasePaths,
    check_release,
)
from clauseguard.llm.clients import GenerationUnavailable, MalformedGenerationOutput

DEFAULT_OUTPUT_ROOT = Path("artifacts/eval")
DEFAULT_PATHS = ReleasePaths()
MAX_TOP_K = 50  # the API's top_k ceiling

# Service errors that fail one row; anything else stops the run and leaves it marked `running`.
ROW_ERRORS = (
    SearchDependencyError,
    SearchConfigurationError,
    QueryOutputError,
    GenerationUnavailable,
    MalformedGenerationOutput,
)


class EvalConfigError(ValueError):
    """The run cannot start: bad configuration, a failing release check or an invalid replay."""


@dataclass(frozen=True)
class GoldenItem:
    row: GoldenEvalRow | RcaRow
    annotation: Annotation | None
    reference: Reference

    @property
    def id(self) -> str:
        return self.row.id

    @property
    def split(self) -> str | None:
        return self.annotation.membership if self.annotation else None

    @property
    def families(self) -> list[str]:
        return list(self.annotation.families) if self.annotation else []


@dataclass(frozen=True)
class GoldenSet:
    items: list[GoldenItem]
    release: dict[str, Any]
    split: str | None

    def identities(self, policy: MetricPolicy) -> dict[str, Any]:
        sources = self.release["sources"]
        return {
            "golden_release": self.release["release_id"],
            "golden_files": self.release["files"],
            "corpus": _digest(sources["specs"]),
            "scenario_release": sources["scenario_release"],
            "split": self.split,
            "policy": policy.identity(),
        }


@dataclass(frozen=True)
class RunResult:
    path: Path
    status: str
    metrics: dict[str, Any]


def load_golden_set(paths: ReleasePaths, split: str | None = None) -> GoldenSet:
    """Load the golden rows after the release check passes, optionally for one split."""
    if split not in (None, "dev", "test"):
        raise EvalConfigError(f"split must be 'dev' or 'test', not {split!r}")
    report = check_release(paths)
    if not report.ok or report.manifest is None:
        raise EvalConfigError("golden release check failed: " + "; ".join(report.problems))

    corpus = load_manifest(paths.corpus_manifest)
    rows = [
        *load_golden(paths.golden_dir / SPEC_QA_FILE),
        *load_rca(paths.golden_dir / RCA_FILE),
    ]
    annotations = {a.id: a for a in load_annotations(paths.golden_dir / ANNOTATIONS_FILE, corpus)}
    items = [
        GoldenItem(row, annotations.get(row.id), reference_for(row, annotations.get(row.id)))
        for row in rows
    ]
    if split is not None:
        items = [item for item in items if item.split == split]
    return GoldenSet(items, report.manifest, split)


def score_record(
    record: RowRecord, item: GoldenItem, policy: MetricPolicy
) -> tuple[RowOutcome | None, dict[str, Any] | None]:
    """Score one saved row; excluded rows return `(None, None)` and stay out of aggregates."""
    if record.status == "excluded":
        return None, None
    cohorts = {"split": item.split or "none", "type": item.row.type, "family": tuple(item.families)}
    answerable = item.reference.answerable
    if record.status == "failed":
        error_type = record.error.type if record.error else None
        outcome = RowOutcome(
            item.id, answerable, failed=True, cohorts=cohorts, error_type=error_type
        )
        return outcome, {"failed": True}

    retrieval = None
    if answerable:
        ranking = [
            (p.spec_id, clause) if (clause := clause_of(p.section)) else None
            for p in record.retrieval.passages
        ]
        try:
            retrieval = score_ranking(
                ranking, item.reference, record.retrieval.requested_top_k, policy
            )
        except TruncatedRankingError as exc:
            raise EvalConfigError(f"{item.id}: {exc}") from exc

    answer = None
    if record.answer is not None:
        citations = [
            (c.spec_id, clause) if (clause := clause_of(c.section)) else None
            for c in record.answer.citations
        ]
        answer = score_answer(record.answer.answer, citations, item.reference)

    outcome = RowOutcome(item.id, answerable, retrieval=retrieval, answer=answer, cohorts=cohorts)
    scores = {
        "failed": False,
        "retrieval": _retrieval_scores(retrieval) if retrieval else None,
        "answer": vars(answer) if answer else None,
    }
    return outcome, scores


def replay(
    source: Path | str,
    *,
    paths: ReleasePaths = DEFAULT_PATHS,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    policy: MetricPolicy = DEFAULT_POLICY,
    compare_to: Path | str | None = None,
) -> RunResult:
    """Re-score a finished run offline and write the result as a new run."""
    source = Path(source)
    _check_previous(compare_to)
    try:
        source_run, records = read_run(source)
    except ArtifactError as exc:
        raise EvalConfigError(f"cannot replay: {exc}") from exc

    split = source_run.get("identities", {}).get("split")
    golden = load_golden_set(paths, split)
    source_release = source_run.get("identities", {}).get("golden_release")
    if source_release != golden.release["release_id"]:
        raise EvalConfigError(
            f"cannot replay: run used golden release {source_release!r}, current release is "
            f"{golden.release['release_id']!r}"
        )

    items = {item.id: item for item in golden.items}
    unknown = [r.id for r in records if r.id not in items]
    if unknown:
        raise EvalConfigError(f"cannot replay: rows not in the golden set: {', '.join(unknown)}")
    for record in records:
        if record.input.question != items[record.id].row.question:
            raise EvalConfigError(f"cannot replay: {record.id} input differs from the golden row")

    # Score everything first so a bad saved row fails before a directory is created.
    for record in records:
        score_record(record, items[record.id], policy)

    result = _execute(
        output_root,
        "replay",
        golden,
        policy,
        ((record, items[record.id]) for record in records),
        answers_captured=bool(source_run.get("answers_captured")),
        settings=source_run.get("settings", {}),
        extra={
            "source_run": {
                "run_id": source_run.get("run_id"),
                "path": str(source),
                "mode": source_run.get("mode"),
                "code": source_run.get("code"),
            }
        },
    )
    return _with_comparison(result, compare_to)


def run_live(
    mode: Literal["retrieval", "query"],
    *,
    search_service: SearchService,
    query_service: QueryService | None = None,
    paths: ReleasePaths = DEFAULT_PATHS,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    policy: MetricPolicy = DEFAULT_POLICY,
    split: str | None = None,
    top_k: int | None = None,
    settings: dict[str, Any] | None = None,
    compare_to: Path | str | None = None,
) -> RunResult:
    """Retrieve for every golden row (and answer spec rows in `query` mode), saving each row.

    `query` mode searches once per row and passes those exact hits to `QueryService.answer`.
    RCA rows are excluded from `query` mode with a recorded reason. Rows whose request raises
    a known service error are saved as failed and the run continues.
    """
    top_k = policy.max_k if top_k is None else top_k
    if mode not in ("retrieval", "query"):
        raise EvalConfigError(f"unknown live mode {mode!r}")
    if not policy.max_k <= top_k <= MAX_TOP_K:
        raise EvalConfigError(
            f"top_k must be between k={policy.max_k} and {MAX_TOP_K}, not {top_k}"
        )
    if mode == "query" and query_service is None:
        raise EvalConfigError("query mode needs a query service")
    _check_previous(compare_to)
    golden = load_golden_set(paths, split)
    if not golden.items:
        raise EvalConfigError("no golden rows to evaluate")

    rows = _live_rows(golden, mode, search_service, query_service, top_k)
    result = _execute(
        output_root,
        mode,
        golden,
        policy,
        rows,
        answers_captured=mode == "query",
        settings=settings or {},
    )
    return _with_comparison(result, compare_to)


def _check_previous(previous: Path | str | None) -> None:
    """Reject an unusable comparison run before any service call or new directory."""
    if previous is None:
        return
    try:
        read_comparable(previous)
    except ArtifactError as exc:
        raise EvalConfigError(f"cannot compare: {exc}") from exc


def _with_comparison(result: RunResult, previous: Path | str | None) -> RunResult:
    if previous is not None:
        RunDirectory(result.path).write_comparison(compare_runs(result.path, previous))
    return result


def _live_rows(
    golden: GoldenSet,
    mode: str,
    search_service: SearchService,
    query_service: QueryService | None,
    top_k: int,
) -> Iterator[tuple[RowRecord, GoldenItem]]:
    for item in golden.items:
        base = {
            "id": item.id,
            "type": item.row.type,
            "split": item.split,
            "families": item.families,
            "answerable": item.reference.answerable,
            "input": RowInput(question=item.row.question, top_k=top_k),
        }
        if mode == "query" and item.row.type != "spec_qa":
            reason = "query generation is specification-only"
            yield RowRecord(**base, status="excluded", exclusion_reason=reason), item
            continue

        timings: dict[str, float] = {}
        started = time.perf_counter()
        try:
            hits = search_service.search(item.row.question, top_k)
        except ROW_ERRORS as exc:
            timings["search"] = _elapsed_ms(started)
            yield (
                RowRecord(**base, status="failed", error=_row_error(exc), timings_ms=timings),
                item,
            )
            continue
        timings["search"] = _elapsed_ms(started)
        retrieval = RetrievalOutput(
            requested_top_k=top_k,
            passages=[Passage(rank=rank, **hit.model_dump()) for rank, hit in enumerate(hits, 1)],
        )

        answer = None
        if mode == "query":
            started = time.perf_counter()
            try:
                response = query_service.answer(item.row.question, hits)
            except ROW_ERRORS as exc:
                timings["answer"] = _elapsed_ms(started)
                failed = RowRecord(
                    **base,
                    status="failed",
                    error=_row_error(exc),
                    retrieval=retrieval,
                    timings_ms=timings,
                )
                yield failed, item
                continue
            timings["answer"] = _elapsed_ms(started)
            answer = AnswerOutput(
                answer=response.answer,
                citations=[AnswerCitation(**c.model_dump()) for c in response.citations],
            )
        yield (
            RowRecord(**base, status="ok", retrieval=retrieval, answer=answer, timings_ms=timings),
            item,
        )


def _execute(
    output_root: Path | str,
    mode: str,
    golden: GoldenSet,
    policy: MetricPolicy,
    rows: Iterable[tuple[RowRecord, GoldenItem]],
    *,
    answers_captured: bool,
    settings: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> RunResult:
    """Create the run directory, then score and append each row as it arrives.

    `run.json` says `running` until the end, so a crash leaves partial, recognisable output.
    """
    directory = RunDirectory.create(output_root, mode)
    run = {
        "run_id": directory.path.name,
        "mode": mode,
        "created_at": _now(),
        "identities": golden.identities(policy),
        "policy": policy.to_dict(),
        "code": _code_identity(),
        "settings": settings,
        "answers_captured": answers_captured,
        "golden_release_complete": golden.release["complete"],
        **(extra or {}),
    }
    directory.write_run({**run, "status": "running", "rows": {"planned": len(golden.items)}})

    records: list[RowRecord] = []
    outcomes: list[RowOutcome] = []
    for record, item in rows:
        outcome, scores = score_record(record, item, policy)
        record = record.model_copy(update={"scores": scores})
        directory.append_row(record)
        records.append(record)
        if outcome is not None:
            outcomes.append(outcome)

    passages = sum(len(r.retrieval.passages) for r in records if r.retrieval)
    metrics = aggregate(outcomes, policy, answers_captured, retrieved_passages=passages)
    metrics["policy"] = policy.to_dict()
    directory.write_metrics(metrics)

    run["identities"]["inputs"] = _digest(
        [[r.id, r.input.question, r.input.top_k] for r in records if r.status != "excluded"]
    )
    status = "row_failures" if any(r.status == "failed" for r in records) else "complete"
    run.update(
        status=status,
        finished_at=_now(),
        rows={
            "total": len(records),
            "ok": sum(1 for r in records if r.status == "ok"),
            "failed": sum(1 for r in records if r.status == "failed"),
            "excluded": [
                {"id": r.id, "reason": r.exclusion_reason}
                for r in records
                if r.status == "excluded"
            ],
        },
    )
    directory.write_run(run)
    return RunResult(directory.path, status, metrics)


def _row_error(exc: Exception) -> RowError:
    # Service errors carry fixed messages; the underlying cause is named, never its text,
    # so provider responses or credentials cannot leak into artifacts.
    message = str(exc)
    if exc.__cause__ is not None:
        cause = type(exc.__cause__).__name__
        message = f"{message} (caused by {cause})" if message else f"caused by {cause}"
    return RowError(type=type(exc).__name__, message=message)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _retrieval_scores(score: RetrievalScore) -> dict[str, Any]:
    return {
        "mrr": score.mrr,
        "first_relevant_rank": score.first_relevant_rank,
        "cutoffs": {str(k): vars(cut) for k, cut in score.cutoffs.items()},
    }


def _code_identity() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
