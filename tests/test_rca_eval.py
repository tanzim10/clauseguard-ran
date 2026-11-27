"""Task 4.8: identification, structure, support, status, resource and error-tag scoring."""

import json

from rca_eval_helpers import CLAIMS, make_record

from clauseguard.eval.rca_eval import (
    CaseReference,
    EvalPolicy,
    EvaluationInputs,
    case_rows,
    evaluate,
    identification,
    judge_metrics,
    match_case,
    ratio,
    resources,
    status_block,
    structure,
)
from clauseguard.eval.rca_judge import ClaimJudgment, JudgePolicy
from clauseguard.eval.rca_runs import view_from_record

POLICY = EvalPolicy()
SEVERE, MILD = "Co-Channel Interference (Severe)", "Co-Channel Interference (Mild)"


def ref(case_id="rca_001", sets=((SEVERE,),), exploratory=False, **extra):
    return CaseReference(case_id=case_id, approved=bool(sets), answerable=True if sets else None,
                         acceptable_sets=tuple(frozenset(s) for s in sets), exploratory=exploratory, **extra)


def view(case_id="rca_001", **kwargs):
    return view_from_record(make_record(case_id, **kwargs))


def judgment(v, claim_id, decision=None, status="judged", model="judge/b"):
    claim = next(c for c in v.claims if c["claim_id"] == claim_id)
    from clauseguard.eval.rca_judge import claim_sha256
    return ClaimJudgment(run_id=v.run_id, case_id=v.case_id, claim_id=claim_id, claim_sha256=claim_sha256(claim),
                         status=status, decision=decision, judge_model=model, prompt_sha256="p", seconds=2.0,
                         prompt_tokens=10, completion_tokens=5)


def test_ratio_with_zero_denominator_is_unavailable():
    assert ratio(0, 0)["value"] is None and ratio(1, 4)["value"] == 0.25


def test_exact_match_uses_any_alternative_set_and_does_not_merge_them():
    sets = [frozenset({SEVERE}), frozenset({MILD, "Antenna Failure"})]
    assert match_case(frozenset({SEVERE}), sets).exact
    assert match_case(frozenset({MILD, "Antenna Failure"}), sets).exact
    assert not match_case(frozenset({SEVERE, MILD}), sets).exact


def test_pooled_counts_use_the_best_overlapping_set_and_multi_cause_partial_credit():
    sets = [frozenset({SEVERE, "Antenna Failure"})]
    match = match_case(frozenset({SEVERE, MILD}), sets)
    assert (match.tp, match.fp, match.fn, match.exact) == (1, 1, 1, False)


def test_failures_and_false_abstentions_are_wrong_and_add_false_negatives():
    views = [view("rca_001"), view("rca_002", outcome="abstained"),
             view("rca_003", status="failed", stage="generation", reason="down")]
    refs = {c: ref(c) for c in ("rca_001", "rca_002", "rca_003")}
    result = identification(views, refs, POLICY, exploratory=False)
    pooled = result["with_failures"]
    assert pooled["exact_cause_set_accuracy"] == {"value": 1 / 3, "numerator": 1, "denominator": 3}
    assert pooled["cause_recall"]["denominator"] == 3 and pooled["cause_precision"]["denominator"] == 1
    assert result["successful_only"]["cases"] == 2
    assert result["successful_only"]["exact_cause_set_accuracy"]["numerator"] == 1


def test_normal_is_an_ordinary_category_and_empty_prediction_is_never_exact():
    normal = view(causes=[{"category": "normal", "claim_ids": ["C1", "C2", "C3"]}])
    assert identification([normal], {"rca_001": ref(sets=(("normal",),))}, POLICY, False)["with_failures"]["exact_cause_set_accuracy"]["numerator"] == 1
    assert not match_case(frozenset(), [frozenset({"normal"})]).exact
    assert not match_case(frozenset({SEVERE}), [frozenset({"normal"})]).exact


def test_declared_category_mappings_apply_and_unmapped_categories_do_not_match():
    policy = EvalPolicy(category_mappings={"CCI Severe": SEVERE})
    mapped = view(causes=[{"category": "CCI Severe", "claim_ids": ["C1", "C2", "C3"]}])
    other = view("rca_002", causes=[{"category": "cci severe", "claim_ids": ["C1", "C2", "C3"]}])
    refs = {"rca_001": ref(), "rca_002": ref("rca_002")}
    out = identification([mapped, other], refs, policy, False)["with_failures"]
    assert out["exact_cause_set_accuracy"]["numerator"] == 1


def test_missing_approved_references_block_scored_identification_but_not_structure_or_support():
    views = [view()]
    refs = {"rca_001": ref(sets=(), exploratory=True)}
    out = identification(views, refs, POLICY, exploratory=False)
    assert out["status"] == "blocked" and "expected_causes is empty" in out["reason"]
    judgments = [judgment(views[0], c["claim_id"], "fully_supported") for c in CLAIMS]
    report = evaluate(EvaluationInputs(views, refs, judgments))
    assert report["identification"]["scored"]["status"] == "blocked"
    assert report["structure"]["successful_only"]["citation_coverage"]["value"] == 1.0
    assert report["support"]["provisional_faithfulness"]["value"] == 1.0
    assert report["human_review_status"] == "pending" and report["evaluation_status"] == "exploratory_provisional"


def test_exploratory_and_scored_cases_are_reported_separately():
    views = [view("rca_001"), view("rca_002")]
    refs = {"rca_001": ref("rca_001", exploratory=False), "rca_002": ref("rca_002", exploratory=True)}
    assert identification(views, refs, POLICY, False)["with_failures"]["cases"] == 1
    assert identification(views, refs, POLICY, True)["with_failures"]["cases"] == 1


def test_claim_free_abstention_adds_nothing_so_coverage_is_unavailable_not_perfect():
    out = structure([view(outcome="abstained")], POLICY)["successful_only"]
    assert out["citation_coverage"]["value"] is None and out["cause_link_coverage"]["value"] is None


def test_invalid_claim_links_lower_citation_and_cause_link_coverage():
    bad = [dict(CLAIMS[0], evidence_ids=["K99"]), CLAIMS[1], dict(CLAIMS[2], evidence_ids=["K1", "K1"])]
    out = structure([view(claims=bad)], POLICY)["successful_only"]
    assert out["citation_coverage"] == {"value": 1 / 3, "numerator": 1, "denominator": 3}
    assert out["cause_link_coverage"]["numerator"] == 0


def test_structure_with_failures_includes_parsed_validation_failures():
    failed = view("rca_002", status="failed", stage="validation", reason="numeric.v2")
    out = structure([view(), failed], POLICY)
    assert out["successful_only"]["runs"] == 1 and out["with_failures"]["runs"] == 2


def test_judge_metrics_count_only_fully_supported_and_keep_failures_visible():
    v = view()
    judgments = [judgment(v, "C1", "fully_supported"), judgment(v, "C2", "cannot_determine"),
                 judgment(v, "C3", status="malformed")]
    out = judge_metrics([v], judgments, POLICY)
    assert out["judgment_coverage"] == {"value": 2 / 3, "numerator": 2, "denominator": 3}
    assert out["provisional_faithfulness"] == {"value": 0.5, "numerator": 1, "denominator": 2}
    assert out["provisional_supported_coverage"]["value"] is None
    assert out["provisional_supported_coverage"]["unavailable_reason"] == "judging is incomplete"
    assert out["status_counts"] == {"judged": 2, "malformed": 1} and out["decision_counts"]["cannot_determine"] == 1


def test_complete_judging_gives_supported_coverage_and_failed_judging_is_not_supported():
    v = view()
    full = [judgment(v, c["claim_id"], "fully_supported") for c in CLAIMS]
    assert judge_metrics([v], full, POLICY)["provisional_supported_coverage"]["value"] == 1.0
    none = [judgment(v, c["claim_id"], status="judge_failed") for c in CLAIMS]
    out = judge_metrics([v], none, POLICY)
    assert out["provisional_faithfulness"]["value"] is None and out["status_counts"] == {"judge_failed": 3}


def test_status_block_separates_answered_abstained_failed_and_stages():
    views = [view("rca_001"), view("rca_002", outcome="abstained"),
             view("rca_003", status="failed", stage="generation", reason="x"),
             view("rca_004", status="failed", stage="validation", reason="y")]
    out = status_block(views, {"rca_001": ref("rca_001"), "rca_002": ref("rca_002")})
    assert (out["answered"], out["abstained"], out["failed"]) == (1, 1, 2)
    assert out["failed_by_stage"] == {"generation": 1, "validation": 1}
    assert out["false_abstention"]["numerator"] == 1 and out["false_abstention"]["denominator"] == 2
    assert out["correct_abstention"]["value"] is None


def test_correct_abstention_is_measured_only_on_declared_unanswerable_fixtures():
    fixture = CaseReference(case_id="fx_1", approved=True, answerable=False, acceptable_sets=(), exploratory=False, fixture=True)
    out = status_block([view("fx_1", outcome="abstained")], {"fx_1": fixture})
    assert out["correct_abstention"] == {"value": 1.0, "numerator": 1, "denominator": 1}
    assert out["false_abstention"]["value"] is None


def test_resources_keep_pipeline_judge_and_memory_scopes_apart():
    v = view(seconds=12.0)
    j = [judgment(v, "C1", "fully_supported")]
    out = resources([v, view("rca_002", generator_kind="replay")], j, {"concurrency": 1})
    assert out["pipeline"]["by_generator_kind"]["live"]["seconds"]["generation"]["mean"] == 12.0
    assert "replay_loading" in out["pipeline"]["by_generator_kind"]["replay"]["seconds"]
    assert out["pipeline"]["hosted_model_memory"].startswith("unavailable") and out["pipeline"]["gpu_memory"].startswith("unavailable")
    assert out["judge"]["seconds"]["total"] == 2.0 and out["judge"]["prompt_tokens"] == 10
    assert out["pipeline"]["completion_tokens"] == 400


def test_error_tags_carry_evidence_and_semantic_tags_are_provisional():
    expected = (("3GPP TS 32.425", "A.9"), ("3GPP TS 38.300", "10.4"))
    reference = ref(expected_clauses=expected, annotated_clauses=frozenset(expected), sets=((MILD,),))
    v = view()
    rows = case_rows([v], {"rca_001": reference}, [judgment(v, "C2", "unsupported"), judgment(v, "C1", "fully_supported")], POLICY)
    tags = {(t["tag"], t["provisional"]): t["evidence"] for t in rows[0]["tags"]}
    assert tags[("retrieval_miss", False)]["clauses_absent_from_packet"] == [["3GPP TS 38.300", "10.4"]]
    assert tags[("citation_mismatch", True)] == {"judge_decisions": {"C2": "unsupported"}}
    assert ("generation_miss", False) not in tags


def test_generation_miss_needs_approved_reference_and_supplied_evidence():
    supplied = (("3GPP TS 32.425", "A.9"),)
    reference = ref(expected_clauses=supplied, annotated_clauses=frozenset(supplied), sets=((MILD,),))
    rows = case_rows([view()], {"rca_001": reference}, [], POLICY)
    assert [t["tag"] for t in rows[0]["tags"]] == ["generation_miss"]
    unapproved = ref(expected_clauses=supplied, annotated_clauses=frozenset(supplied), sets=())
    assert case_rows([view()], {"rca_001": unapproved}, [], POLICY)[0]["tags"] == []


def test_execution_failure_and_undetermined_abstention_tags():
    rows = case_rows([view(status="failed", stage="generation", reason="down"), view("rca_002", outcome="abstained")],
                     {"rca_001": ref(sets=()), "rca_002": ref("rca_002", sets=())}, [], POLICY)
    assert rows[0]["tags"][0]["tag"] == "execution_failure" and rows[0]["tags"][0]["evidence"]["stage"] == "generation"
    assert rows[1]["tags"][0]["tag"] == "undetermined"


def test_evaluation_is_deterministic_and_the_policy_hash_changes_with_the_rules():
    inputs = EvaluationInputs([view()], {"rca_001": ref()}, [], POLICY, JudgePolicy())
    assert json.dumps(evaluate(inputs), sort_keys=True) == json.dumps(evaluate(inputs), sort_keys=True)
    assert EvalPolicy(category_mappings={"a": "b"}).sha256 != POLICY.sha256


def test_references_are_not_part_of_the_generator_packet():
    packet = json.dumps(make_record()["projected_packet"])
    for field in ("expected_causes", "acceptable_conclusions", "expected_citations", "source_labels", "rca_label"):
        assert field not in packet


def test_judge_equal_to_a_generator_model_is_recorded_as_a_blocker():
    v = view(model="same/model")
    out = evaluate(EvaluationInputs([v], {"rca_001": ref()}, [judgment(v, "C1", "fully_supported", model="same/model")]))
    assert any("also a generator" in b for b in out["blockers"])


def test_support_block_counts_splits_and_discloses_prior_test_exposure():
    from clauseguard.eval.rca_eval import support_block

    refs = {"rca_001": ref("rca_001", split="dev", family="f1"), "rca_002": ref("rca_002", split="test", family="f1")}
    out = support_block([view("rca_001"), view("rca_002")], refs)
    assert out["by_split"] == {"dev": 1, "test": 1} and out["recording_families"] == 1
    assert "already run and read" in out["test_exposure"] and "not a frozen test comparison" in out["test_exposure"]
