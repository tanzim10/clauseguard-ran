from __future__ import annotations

import pytest

from clauseguard.corpus.chunker import chunk_document
from clauseguard.datasets.schema import GoldenEvalRow
from clauseguard.eval.annotations import Annotation
from clauseguard.eval.metrics import (
    MetricPolicy,
    Reference,
    RowOutcome,
    TruncatedRankingError,
    aggregate,
    clause_of,
    reference_for,
    score_answer,
    score_ranking,
)

A = ("TS 1", "1.1")
B = ("TS 1", "1.2")
C = ("TS 2", "4.3")  # an alternative that supports the same claim as B
NOISE = [("TS 9", f"9.{n}") for n in range(1, 10)]  # distinct irrelevant clauses


def _group(*clauses):
    return frozenset(clauses)


# Expected values below are worked out by hand from each ranking, not from the code under test.


def test_single_clause_ranking():
    ranking = [NOISE[0], A, *NOISE[1:9]]  # relevant passage at rank 2 of 10

    score = score_ranking(ranking, Reference((_group(A),)), requested_top_k=10)

    assert score.mrr == 0.5
    assert score.first_relevant_rank == 2
    at = score.cutoffs
    assert (at[1].clause_recall, at[1].group_coverage, at[1].precision) == (0.0, 0.0, 0.0)
    assert at[1].complete_reference_match is False
    assert (at[3].clause_recall, at[3].group_coverage) == (1.0, 1.0)
    assert at[3].precision == pytest.approx(1 / 3)
    assert at[5].precision == pytest.approx(1 / 5)
    assert at[10].precision == pytest.approx(1 / 10)
    assert at[10].complete_reference_match is True


def test_multi_clause_confirmed_short_ranking_keeps_denominator_k():
    ranking = [A, NOISE[0], NOISE[1], B]  # only 4 passages exist; 10 were requested

    score = score_ranking(ranking, Reference((_group(A), _group(B))), requested_top_k=10)

    at = score.cutoffs
    assert score.mrr == 1.0
    assert (at[1].clause_recall, at[1].group_coverage, at[1].precision) == (0.5, 0.5, 1.0)
    assert at[3].clause_recall == 0.5
    assert at[3].precision == pytest.approx(1 / 3)
    assert (at[5].clause_recall, at[5].group_coverage) == (1.0, 1.0)
    assert at[5].complete_reference_match is True
    assert at[5].precision == pytest.approx(2 / 5)
    assert at[10].precision == pytest.approx(2 / 10)
    assert at[10].returned_slots == 4


def test_alternative_support_covers_the_group_but_not_full_recall():
    score = score_ranking([C], Reference((_group(B, C),)), requested_top_k=10)

    at = score.cutoffs[1]
    assert at.clause_recall == 0.5  # relevant set is {B, C}; only C was found
    assert at.group_coverage == 1.0
    assert at.complete_reference_match is True
    assert at.precision == 1.0
    assert score.cutoffs[10].precision == pytest.approx(1 / 10)


def test_duplicates_count_as_slots_but_not_as_extra_recall():
    ranking = [A, A, A, NOISE[0]]

    score = score_ranking(ranking, Reference((_group(A), _group(B))), requested_top_k=10)

    at3, at5 = score.cutoffs[3], score.cutoffs[5]
    assert (at3.clause_recall, at3.group_coverage) == (0.5, 0.5)
    assert at3.precision == 1.0  # 3 matching slots / 3
    assert (at3.duplicate_slots, at3.returned_slots) == (2, 3)
    assert at5.precision == pytest.approx(3 / 5)
    assert (at5.duplicate_slots, at5.returned_slots) == (2, 4)


def test_empty_ranking_scores_zero():
    score = score_ranking([], Reference((_group(A),)), requested_top_k=10)

    assert score.mrr == 0.0
    assert score.first_relevant_rank is None
    assert all(cut.precision == 0.0 and cut.clause_recall == 0.0 for cut in score.cutoffs.values())


def test_unnumbered_passages_take_a_slot_but_never_match_or_duplicate():
    score = score_ranking([None, None, A], Reference((_group(A),)), requested_top_k=10)

    assert score.mrr == pytest.approx(1 / 3)
    assert score.cutoffs[3].precision == pytest.approx(1 / 3)
    assert score.cutoffs[3].duplicate_slots == 0


@pytest.mark.parametrize("depth", [None, 5])
def test_unknown_or_shallow_depth_fails(depth):
    with pytest.raises(TruncatedRankingError, match="k=10"):
        score_ranking([A], Reference((_group(A),)), requested_top_k=depth)


def test_ranking_deeper_than_requested_is_rejected():
    with pytest.raises(ValueError, match="only 1 were requested"):
        score_ranking(
            [A, B], Reference((_group(A),)), requested_top_k=1, policy=MetricPolicy(ks=(1,))
        )


def test_no_answer_rows_cannot_be_scored_for_retrieval():
    with pytest.raises(ValueError, match="no-answer"):
        score_ranking([A], Reference(), requested_top_k=10)


def test_correct_abstention_needs_not_found_and_no_citations():
    no_answer = Reference()

    assert score_answer("not found", [], no_answer).correct_abstention is True
    assert score_answer("not found", [A], no_answer).correct_abstention is False
    assert score_answer("Some answer.", [A], no_answer).correct_abstention is False


def test_answerable_rows_report_false_abstention_and_cited_coverage():
    reference = Reference((_group(B, C), _group(A)))

    abstained = score_answer("not found", [], reference)
    answered = score_answer("Answer.", [C, NOISE[0]], reference)

    assert (abstained.false_abstention, abstained.cited_groups_covered) == (True, 0.0)
    assert (answered.false_abstention, answered.cited_groups_covered) == (False, 0.5)


def test_aggregate_keeps_failures_visible_and_excludes_no_answer_rows():
    ranking = [NOISE[0], A, *NOISE[1:9]]
    reference = Reference((_group(A),))
    outcomes = [
        RowOutcome(
            "r1",
            answerable=True,
            retrieval=score_ranking(ranking, reference, 10),
            answer=score_answer("Answer.", [A], reference),
            cohorts={"split": "dev", "family": ("direct",)},
        ),
        RowOutcome(
            "r2", answerable=True, failed=True, cohorts={"split": "test", "family": ("direct",)}
        ),
        RowOutcome(
            "r3",
            answerable=False,
            answer=score_answer("not found", [], Reference()),
            cohorts={"split": "dev", "family": ("no_answer",)},
        ),
        RowOutcome(
            "r4", answerable=False, failed=True, cohorts={"split": "test", "family": ("no_answer",)}
        ),
    ]

    summary = aggregate(outcomes, answers_captured=True, retrieved_passages=10)

    retrieval = summary["retrieval"]
    assert (summary["rows"], summary["failed"]) == (4, 2)
    assert retrieval["denominators"] == {
        "answerable": 2,
        "successful": 1,
        "failed": 1,
        "no_answer_excluded": 2,
    }
    assert retrieval["headline"]["mrr"] == 0.25  # (0.5 + 0 for the failure) / 2
    assert retrieval["successful_only"]["mrr"] == 0.5
    assert retrieval["headline"]["clause_recall@3"] == 0.5
    assert retrieval["successful_only"]["clause_recall@3"] == 1.0
    assert retrieval["duplicate_rate"]["@3"] == 0.0

    abstention = summary["abstention"]
    assert abstention["correct_abstention"] == {
        "count": 1,
        "denominator": 2,
        "rate": 0.5,
        "failed": 1,
    }
    assert abstention["false_abstention"] == {"count": 0, "denominator": 1, "rate": 0.0}

    assert summary["judged"]["status"] == "unavailable"
    assert summary["judged"]["judgment_coverage"] == 0.0

    split = summary["cohorts"]["split"]
    assert split["dev"]["retrieval"]["headline"]["mrr"] == 0.5
    assert split["test"]["retrieval"]["headline"]["mrr"] == 0.0
    assert split["test"]["retrieval"]["successful_only"] is None
    assert summary["cohorts"]["family"]["no_answer"]["retrieval"]["headline"] is None


def test_abstention_is_unavailable_without_answers():
    summary = aggregate([RowOutcome("r3", answerable=False)])

    assert summary["abstention"]["status"] == "unavailable"


def test_pooled_duplicate_rate():
    reference = Reference((_group(A),))
    outcomes = [
        RowOutcome("r1", True, retrieval=score_ranking([A, A, A], reference, 10)),  # 2 dup of 3
        RowOutcome("r2", True, retrieval=score_ranking([A], reference, 10)),  # 0 dup of 1
    ]

    rates = aggregate(outcomes)["retrieval"]["duplicate_rate"]

    assert rates["@3"] == 0.5  # (2 + 0) / (3 + 1)
    assert rates["@1"] == 0.0


def test_clause_of_reads_chunker_section_headings():
    text = (
        "Front matter text.\n\n"
        "5.1.1.3.1 Average DL UE throughput in gNB\nBody.\n\n"
        "5.1.\tGeneral\nBody.\n"
    )
    sections = [
        chunk["metadata"]["section"]
        for chunk in chunk_document(text, spec_id="S", source_file="s.pdf")
    ]

    assert [clause_of(section) for section in sections] == [None, "5.1.1.3.1", "5.1"]
    assert clause_of("unsectioned") is None
    assert clause_of("") is None


def test_reference_for_uses_evidence_groups_or_falls_back_to_citations():
    row = GoldenEvalRow.model_validate(
        {
            "id": "q1",
            "question": "Q?",
            "expected_answer": "A.",
            "expected_citations": [{"spec_id": "TS 1", "section": "1.2 Title"}],
        }
    )
    locator = {"version": "1", "sha256": "a" * 64, "pdf_page": 1}
    annotation = Annotation.model_validate(
        {
            "id": "q1",
            "families": ["direct"],
            "evidence": [
                {
                    "claim": "Claim.",
                    "alternatives": [
                        {"spec_id": "TS 1", "clause": "1.2", **locator},
                        {"spec_id": "TS 2", "clause": "4.3", **locator},
                    ],
                }
            ],
        }
    )

    assert reference_for(row, annotation) == Reference((_group(B, C),))
    assert reference_for(row, None) == Reference((_group(B),))


def test_policy_identity_changes_with_the_rules():
    assert MetricPolicy().identity() == MetricPolicy().identity()
    assert MetricPolicy().identity() != MetricPolicy(ks=(1, 5)).identity()
    with pytest.raises(ValueError, match="increasing"):
        MetricPolicy(ks=(5, 1))
