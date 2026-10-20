"""Retrieval and answer metrics for golden evaluation (Week 5).

Pure functions over plain values: a ranking is a list of `(spec_id, clause)` pairs in the order
retrieval returned them, and a `Reference` holds a row's required evidence groups. Nothing here
calls a service or reads a file.

The scoring rules are fixed by `MetricPolicy` and recorded with every run:

- The original ranking is scored as returned; duplicates are kept.
- A passage matches when its spec and clause number equal a reference clause exactly. The clause
  number is the first word of the passage section (`clause_of`).
- The relevant set is the union of every evidence group's alternatives.
- Clause recall@k counts distinct relevant clauses in the top k, over all relevant clauses.
- Group coverage@k counts groups with any alternative in the top k; a complete-reference match
  means every group is covered. Repeated hits never raise recall or coverage.
- Precision@k counts matching slots in the top k (duplicates included) divided by k, even when
  a confirmed short ranking returned fewer than k passages. The duplicate rate is reported
  beside it.
- MRR is 1 / the original rank of the first relevant passage within the requested depth.
- No-answer rows are excluded from retrieval averages. Failed answerable requests score 0 in
  the headline averages; successful-only averages are reported separately.
- Correct abstention needs a successful request, the answer "not found" and no citations.
- Judged precision and semantic complete-evidence scores need human passage judgments, which do
  not exist yet; they are reported as unavailable with their judgment coverage.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field

from clauseguard.datasets.schema import NOT_FOUND_ANSWER, GoldenEvalRow, RcaRow
from clauseguard.eval.annotations import Annotation, section_clause

Clause = tuple[str, str]

DEFAULT_KS = (1, 3, 5, 10)
_UNNUMBERED_SECTIONS = frozenset({"preamble", "unsectioned"})


class TruncatedRankingError(ValueError):
    """A ranking was requested at a shallower depth than a metric cutoff needs."""


def clause_of(section: str) -> str | None:
    """Return the clause number at the start of a passage section, e.g. "5.1.1.3.1".

    Chunk sections are whole heading lines ("5.1.1.3.1 Average DL UE throughput"), sometimes with
    a trailing dot ("5.1. General"). Preamble and unsectioned chunks have no clause.
    """
    words = section.split()
    if not words:
        return None
    clause = words[0].rstrip(".")
    if not clause or clause in _UNNUMBERED_SECTIONS:
        return None
    return clause


@dataclass(frozen=True)
class MetricPolicy:
    """The scoring rules a run used; two runs are only comparable under the same policy."""

    ks: tuple[int, ...] = DEFAULT_KS
    ranking: str = "original_order_with_duplicates"
    clause_match: str = "exact_spec_id_and_clause_number"
    relevant_set: str = "union_of_evidence_alternatives"
    precision_denominator: str = "k"
    failed_answerable_requests: str = "zero_in_headline"
    abstention: str = "successful_not_found_without_citations"
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.ks or any(k < 1 for k in self.ks) or list(self.ks) != sorted(set(self.ks)):
            raise ValueError("ks must be distinct positive integers in increasing order")

    @property
    def max_k(self) -> int:
        return self.ks[-1]

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "ks": list(self.ks)}

    def identity(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


DEFAULT_POLICY = MetricPolicy()


@dataclass(frozen=True)
class Reference:
    """A row's required evidence: each group is satisfied by any one of its clauses.

    No-answer rows have no groups.
    """

    groups: tuple[frozenset[Clause], ...] = ()

    def __post_init__(self) -> None:
        if any(not group for group in self.groups):
            raise ValueError("evidence groups need at least one clause")

    @property
    def answerable(self) -> bool:
        return bool(self.groups)

    @property
    def relevant(self) -> frozenset[Clause]:
        return frozenset().union(*self.groups)


def reference_for(row: GoldenEvalRow | RcaRow, annotation: Annotation | None) -> Reference:
    """Build a row's reference from its annotation's evidence groups.

    Without an annotation, each expected citation is its own required group.
    """
    if not row.is_answerable:
        return Reference()
    if annotation is not None and annotation.evidence:
        return Reference(
            tuple(
                frozenset((loc.spec_id, loc.clause) for loc in group.alternatives)
                for group in annotation.evidence
            )
        )
    return Reference(
        tuple(
            frozenset({(citation.spec_id, section_clause(citation.section))})
            for citation in row.expected_citations
        )
    )


@dataclass(frozen=True)
class CutoffScore:
    clause_recall: float
    group_coverage: float
    complete_reference_match: bool
    precision: float
    matching_slots: int
    returned_slots: int
    duplicate_slots: int


@dataclass(frozen=True)
class RetrievalScore:
    mrr: float
    first_relevant_rank: int | None
    cutoffs: dict[int, CutoffScore]


def score_ranking(
    ranking: Sequence[Clause | None],
    reference: Reference,
    requested_top_k: int | None,
    policy: MetricPolicy = DEFAULT_POLICY,
) -> RetrievalScore:
    """Score one answerable row's ranking at every cutoff in `policy.ks`.

    `requested_top_k` is the depth retrieval was asked for. A ranking shorter than k is a
    confirmed short ranking only when `requested_top_k >= k`; otherwise the missing slots are
    unknown and `TruncatedRankingError` is raised. Entries of `None` are passages without a
    clause number; they occupy a slot but never match.
    """
    if not reference.answerable:
        raise ValueError("no-answer rows have no relevant clauses to score")
    if requested_top_k is None or requested_top_k < policy.max_k:
        raise TruncatedRankingError(
            f"ranking depth {requested_top_k} is unknown or shallower than k={policy.max_k}"
        )
    if len(ranking) > requested_top_k:
        raise ValueError(
            f"ranking has {len(ranking)} passages but only {requested_top_k} were requested"
        )

    relevant = reference.relevant
    first_rank = next(
        (rank for rank, clause in enumerate(ranking, start=1) if clause in relevant), None
    )

    cutoffs = {}
    for k in policy.ks:
        top = list(ranking[:k])
        found = {clause for clause in top if clause in relevant}
        seen: set[Clause | None] = set()
        duplicates = 0
        for clause in top:
            if clause is not None and clause in seen:
                duplicates += 1
            seen.add(clause)
        covered = sum(1 for group in reference.groups if group & found)
        cutoffs[k] = CutoffScore(
            clause_recall=len(found) / len(relevant),
            group_coverage=covered / len(reference.groups),
            complete_reference_match=covered == len(reference.groups),
            precision=sum(1 for clause in top if clause in relevant) / k,
            matching_slots=sum(1 for clause in top if clause in relevant),
            returned_slots=len(top),
            duplicate_slots=duplicates,
        )
    return RetrievalScore(
        mrr=1 / first_rank if first_rank else 0.0,
        first_relevant_rank=first_rank,
        cutoffs=cutoffs,
    )


@dataclass(frozen=True)
class AnswerScore:
    abstained: bool
    correct_abstention: bool | None
    false_abstention: bool | None
    cited_groups_covered: float | None


def score_answer(
    answer: str, citations: Sequence[Clause | None], reference: Reference
) -> AnswerScore:
    """Score one successful answer: abstention for no-answer rows, citation support otherwise."""
    abstained = answer.strip() == NOT_FOUND_ANSWER
    if not reference.answerable:
        return AnswerScore(
            abstained=abstained,
            correct_abstention=abstained and not citations,
            false_abstention=None,
            cited_groups_covered=None,
        )
    cited = set(citations)
    covered = sum(1 for group in reference.groups if group & cited)
    return AnswerScore(
        abstained=abstained,
        correct_abstention=None,
        false_abstention=abstained,
        cited_groups_covered=covered / len(reference.groups),
    )


@dataclass(frozen=True)
class RowOutcome:
    """Everything aggregation needs about one row; `failed` means its request raised."""

    row_id: str
    answerable: bool
    failed: bool = False
    retrieval: RetrievalScore | None = None
    answer: AnswerScore | None = None
    cohorts: dict[str, str | tuple[str, ...]] = field(default_factory=dict)
    error_type: str | None = None

    def __post_init__(self) -> None:
        if self.answerable and not self.failed and self.retrieval is None:
            raise ValueError(f"{self.row_id}: successful answerable rows need a retrieval score")


def aggregate(
    outcomes: Iterable[RowOutcome],
    policy: MetricPolicy = DEFAULT_POLICY,
    answers_captured: bool = False,
    retrieved_passages: int | None = None,
) -> dict[str, object]:
    """Summarise row outcomes, overall and per cohort value (split, type, family, ...)."""
    outcomes = list(outcomes)
    summary = _summary(outcomes, policy, answers_captured)
    total_passages = retrieved_passages or 0
    summary["judged"] = {
        "status": "unavailable",
        "reason": "no human passage judgments",
        "judged_passages": 0,
        "retrieved_passages": total_passages,
        "judgment_coverage": 0.0 if total_passages else None,
    }

    cohorts: dict[str, dict[str, list[RowOutcome]]] = defaultdict(lambda: defaultdict(list))
    for outcome in outcomes:
        for name, value in outcome.cohorts.items():
            for item in (value,) if isinstance(value, str) else value:
                cohorts[name][item].append(outcome)
    summary["cohorts"] = {
        name: {
            value: _summary(rows, policy, answers_captured)
            for value, rows in sorted(groups.items())
        }
        for name, groups in sorted(cohorts.items())
    }
    return summary


def _summary(
    outcomes: list[RowOutcome], policy: MetricPolicy, answers_captured: bool
) -> dict[str, object]:
    answerable = [o for o in outcomes if o.answerable]
    successful = [o for o in answerable if not o.failed]
    no_answer = [o for o in outcomes if not o.answerable]
    return {
        "rows": len(outcomes),
        "failed": sum(1 for o in outcomes if o.failed),
        "errors": dict(
            sorted(Counter(o.error_type or "unknown" for o in outcomes if o.failed).items())
        ),
        "retrieval": {
            "denominators": {
                "answerable": len(answerable),
                "successful": len(successful),
                "failed": len(answerable) - len(successful),
                "no_answer_excluded": len(no_answer),
            },
            "headline": _retrieval_means(answerable, policy),
            "successful_only": _retrieval_means(successful, policy),
            "duplicate_rate": _duplicate_rates(successful, policy),
        },
        "abstention": _abstention(answerable, no_answer, answers_captured),
    }


def _retrieval_means(outcomes: list[RowOutcome], policy: MetricPolicy) -> dict[str, float] | None:
    if not outcomes:
        return None
    scores = [o.retrieval for o in outcomes]
    means = {"mrr": _mean(s.mrr if s else 0.0 for s in scores)}
    for k in policy.ks:
        cuts = [s.cutoffs[k] if s else None for s in scores]
        means[f"clause_recall@{k}"] = _mean(c.clause_recall if c else 0.0 for c in cuts)
        means[f"group_coverage@{k}"] = _mean(c.group_coverage if c else 0.0 for c in cuts)
        means[f"complete_reference_match@{k}"] = _mean(
            1.0 if c and c.complete_reference_match else 0.0 for c in cuts
        )
        means[f"precision@{k}"] = _mean(c.precision if c else 0.0 for c in cuts)
    return means


def _duplicate_rates(outcomes: list[RowOutcome], policy: MetricPolicy) -> dict[str, float | None]:
    rates: dict[str, float | None] = {}
    for k in policy.ks:
        cuts = [o.retrieval.cutoffs[k] for o in outcomes if o.retrieval]
        returned = sum(c.returned_slots for c in cuts)
        rates[f"@{k}"] = sum(c.duplicate_slots for c in cuts) / returned if returned else None
    return rates


def _abstention(
    answerable: list[RowOutcome], no_answer: list[RowOutcome], answers_captured: bool
) -> dict[str, object]:
    if not answers_captured:
        return {"status": "unavailable", "reason": "answers were not captured in this run"}
    correct = sum(1 for o in no_answer if not o.failed and o.answer and o.answer.correct_abstention)
    answered = [o for o in answerable if not o.failed and o.answer]
    false = sum(1 for o in answered if o.answer.false_abstention)
    return {
        "status": "available",
        "correct_abstention": {
            "count": correct,
            "denominator": len(no_answer),
            "rate": correct / len(no_answer) if no_answer else None,
            "failed": sum(1 for o in no_answer if o.failed),
        },
        "false_abstention": {
            "count": false,
            "denominator": len(answered),
            "rate": false / len(answered) if answered else None,
        },
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)
