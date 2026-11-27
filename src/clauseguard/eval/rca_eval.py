"""Scoring of saved base-LLM RCA runs (Task 4.8).

Pure functions over run views (`rca_runs`), reference records and judge records
(`rca_judge`). Nothing here calls a model, reads a generator input or edits a reference.

Four things are measured apart, because they fail apart:

- identification: does the returned cause set match an approved acceptable set;
- structure: do claims and causes carry valid links into the packet the generator saw;
- support: does a separate judge find each claim supported by the evidence it cites (provisional);
- execution: which runs answered, abstained or failed, and where.

Every ratio is reported as `{value, numerator, denominator}`; a zero denominator gives
`value: null` (unavailable), never 0 or 1. All judge-based results are provisional and
`human_review_status` stays `pending`.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from clauseguard.eval.metrics import clause_of
from clauseguard.eval.rca_judge import DECISIONS, ClaimJudgment, JudgePolicy
from clauseguard.eval.rca_runs import FACTUAL_CLAIM_TYPES, RunView

EVAL_POLICY_VERSION = "rca-eval-v1"
FAITHFUL_DECISION = "fully_supported"
SEMANTIC_TAG_DECISIONS = ("partially_supported", "unsupported", "contradicted")

RULES = {
    "cause_matching": "Categories are compared after the declared category_mappings; there is no free-text similarity. "
    "An unmapped category only matches an identical category.",
    "alternative_sets": "A case may list several approved acceptable cause sets. They are never merged. A prediction is "
    "exact when it equals any one set. Pooled counts use the set with the largest overlap (ties: the smaller set, then "
    "the first listed).",
    "pooled_counts": "Precision, recall and F1 pool true positives, false positives and false negatives over eligible cases.",
    "failures": "A failed or incomplete run predicts the empty set. A false abstention also predicts the empty set. Both "
    "count as incorrect in exact accuracy and add the smallest acceptable set size to false negatives.",
    "empty_set": "An empty set is never an acceptable reference; an empty prediction is always incorrect.",
    "normal": "The normal label is an ordinary category. A reference of exactly {normal} is matched only by exactly {normal}.",
    "claim_free_abstention": "An abstention has no claims and no causes. It adds nothing to any citation, cause-link or "
    "support denominator, so it never earns a perfect citation score.",
    "citation_coverage": "A claim has a valid evidence link when it cites at least one evidence ID, cites no ID twice and "
    "every ID exists in the packet the generator received.",
    "cause_link_coverage": "A returned cause has a valid link when all its claim IDs exist and at least one is a "
    "diagnostic_inference claim with a valid evidence link.",
    "faithfulness": "Only a fully_supported decision counts as faithful. Partial, unsupported, contradicted and "
    "cannot_determine claims fail the support check. Malformed replies and judge failures are never counted either way.",
    "supported_coverage": "Fully supported claims over all factual claims, reported only when every claim has a completed "
    "judge decision.",
    "scored_results": "Scored identification needs an approved reference (non-empty expected_causes) and a case that is "
    "not exploratory. Exploratory cases are reported in their own section.",
    "abstention": "False abstention is measured on approved answerable references. Correct abstention is measured only on "
    "references declared unanswerable in a separate abstention file.",
}


TEST_EXPOSURE = (
    "Prior exposure: the golden RCA cases, including any in the test split (see by_split), were already run and read "
    "while the Task 4.7 prompt and checks were developed. These results are development evidence only. They are not a "
    "frozen test comparison, and nothing here may be used to revise prompts or select checkpoints for one."
)


class EvalPolicy(BaseModel):
    """Versioned scoring rules, saved with every evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = EVAL_POLICY_VERSION
    category_mappings: dict[str, str] = Field(default_factory=dict)
    normal_label: str = "normal"
    factual_claim_types: tuple[str, ...] = FACTUAL_CLAIM_TYPES
    faithful_decision: str = FAITHFUL_DECISION
    rules: dict[str, str] = Field(default_factory=lambda: dict(RULES))

    def canonical(self, category: str) -> str:
        return self.category_mappings.get(category, category)

    def canonical_set(self, categories: Iterable[str]) -> frozenset[str]:
        return frozenset(self.canonical(c) for c in categories)

    @property
    def sha256(self) -> str:
        return _digest(self.model_dump(mode="json"))


@dataclass(frozen=True)
class CaseReference:
    """What the evaluator knows about one case. It is never shown to the generator."""

    case_id: str
    approved: bool
    answerable: bool | None
    acceptable_sets: tuple[frozenset[str], ...]
    source_label: str | None = None
    expected_clauses: tuple[tuple[str, str], ...] = ()
    annotated_clauses: frozenset[tuple[str, str]] = frozenset()
    corpus_gap_clauses: tuple[tuple[str, str], ...] = ()
    exploratory: bool = True
    split: str | None = None
    family: str | None = None
    modality: str | None = None
    source: str | None = None
    fixture: bool = False


def load_references(
    rca_path: Path | str,
    annotations_path: Path | str | None = None,
    policy_path: Path | str | None = None,
    overlay_path: Path | str | None = None,
    abstention_path: Path | str | None = None,
    eval_policy: EvalPolicy | None = None,
) -> dict[str, CaseReference]:
    """Load golden RCA references, an optional alternative-set overlay and abstention fixtures.

    A golden case is approved only when its `expected_causes` is non-empty. The overlay maps
    `case_id` to `{"acceptable_sets": [[...], ...], "corpus_gap_clauses": [[spec, clause], ...]}`
    and may add alternatives to an approved case. The abstention file maps `case_id` to
    `{"answerable": false}` and is kept apart from the golden RCA cases.
    """
    from clauseguard.eval.golden import load_rca

    policy = eval_policy or EvalPolicy()
    annotated = _annotated_clauses(annotations_path)
    policy_doc = json.loads(Path(policy_path).read_text()) if policy_path else {}
    mode = policy_doc.get("source_mode", "exploratory")
    overlay = json.loads(Path(overlay_path).read_text()) if overlay_path else {}
    refs: dict[str, CaseReference] = {}
    for row in load_rca(rca_path):
        sets = [policy.canonical_set(c.category for c in row.expected_causes)] if row.expected_causes else []
        extra = overlay.get(row.id, {})
        sets += [policy.canonical_set(s) for s in extra.get("acceptable_sets", [])]
        membership = policy_doc.get("cases", {}).get(row.id, {}).get("membership", {})
        refs[row.id] = CaseReference(
            case_id=row.id,
            approved=bool(sets),
            answerable=True if sets else None,
            acceptable_sets=tuple(dict.fromkeys(sets)),
            source_label=row.source_labels.get("rca_label"),
            expected_clauses=tuple(dict.fromkeys((c.spec_id, clause_of(c.section) or c.section) for c in row.expected_citations)),
            annotated_clauses=annotated.get(row.id, frozenset()),
            corpus_gap_clauses=tuple((s, c) for s, c in extra.get("corpus_gap_clauses", [])),
            exploratory=mode == "exploratory",
            split=membership.get("split"),
            family=membership.get("group_id"),
            modality=row.inputs,
            source=row.origin,
        )
    for case_id, spec in (json.loads(Path(abstention_path).read_text()) if abstention_path else {}).items():
        refs[case_id] = CaseReference(case_id=case_id, approved=True, answerable=bool(spec.get("answerable", False)),
                                      acceptable_sets=(), exploratory=False, fixture=True)
    return refs


def _annotated_clauses(path: Path | str | None) -> dict[str, frozenset[tuple[str, str]]]:
    if not path:
        return {}
    out: dict[str, frozenset[tuple[str, str]]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["id"]] = frozenset(
            (alt["spec_id"], alt["clause"]) for group in row.get("evidence", []) for alt in group["alternatives"]
        )
    return out


def ratio(numerator: int, denominator: int) -> dict[str, Any]:
    """A count ratio with its counts; a zero denominator is unavailable, not 0 or 1."""
    return {"value": numerator / denominator if denominator else None, "numerator": numerator, "denominator": denominator}


# ---------------------------------------------------------------- identification


@dataclass
class _CaseMatch:
    exact: bool
    tp: int
    fp: int
    fn: int
    best: frozenset[str]


def match_case(predicted: frozenset[str], sets: Sequence[frozenset[str]]) -> _CaseMatch:
    """Compare a predicted set with every approved acceptable set (see `RULES`)."""
    exact = bool(predicted) and predicted in sets
    if not predicted:
        best = min(sets, key=len)
        return _CaseMatch(False, 0, 0, len(best), best)
    best = max(sets, key=lambda s: (len(s & predicted), -len(s)))
    return _CaseMatch(exact, len(best & predicted), len(predicted - best), len(best - predicted), best)


def identification(views: Sequence[RunView], refs: Mapping[str, CaseReference], policy: EvalPolicy, exploratory: bool) -> dict[str, Any]:
    """Cause-set accuracy and pooled precision, recall and F1 for one eligibility group."""
    eligible = [
        (v, refs[v.case_id]) for v in views
        if v.case_id in refs and refs[v.case_id].approved and not refs[v.case_id].fixture
        and refs[v.case_id].exploratory == exploratory
    ]
    label = "exploratory" if exploratory else "scored"
    if not eligible:
        reason = ("no case has an approved reference (expected_causes is empty for every golden RCA case)"
                  if not any(r.approved and not r.fixture for r in refs.values())
                  else f"no approved {label} case has a saved run")
        return {"status": "blocked", "reason": reason, "cases": 0}
    return {
        "status": "available",
        "with_failures": _pooled(eligible, policy),
        "successful_only": _pooled([(v, r) for v, r in eligible if v.complete], policy),
        "per_category": _per_category(eligible, policy),
    }


def _pooled(pairs: Sequence[tuple[RunView, CaseReference]], policy: EvalPolicy) -> dict[str, Any]:
    tp = fp = fn = exact = 0
    for view, ref in pairs:
        match = match_case(policy.canonical_set(view.category_set), ref.acceptable_sets)
        tp, fp, fn, exact = tp + match.tp, fp + match.fp, fn + match.fn, exact + match.exact
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    p, r = precision["value"], recall["value"]
    f1 = None if p is None or r is None or p + r == 0 else 2 * p * r / (p + r)
    return {"cases": len(pairs), "exact_cause_set_accuracy": ratio(exact, len(pairs)), "cause_precision": precision,
            "cause_recall": recall, "cause_f1": {"value": f1, "numerator": None, "denominator": None}}


def _per_category(pairs: Sequence[tuple[RunView, CaseReference]], policy: EvalPolicy) -> dict[str, dict[str, int]]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for view, ref in pairs:
        predicted = policy.canonical_set(view.category_set)
        match = match_case(predicted, ref.acceptable_sets)
        for category in match.best:
            out[category]["reference"] += 1
            out[category]["correct"] += category in predicted
        for category in predicted:
            out[category]["predicted"] += 1
    return {k: {n: v[n] for n in ("reference", "predicted", "correct")} for k, v in sorted(out.items())}


def agreement_blocks(views: Sequence[RunView], refs: Mapping[str, CaseReference], policy: EvalPolicy) -> dict[str, Any]:
    """Source-label and detector agreement. The source label is a dataset claim, not an approved reference."""
    labelled = [(v, refs[v.case_id]) for v in views if v.case_id in refs and refs[v.case_id].source_label]
    answered = [(v, r) for v, r in labelled if v.answered]
    final = sum(policy.canonical(r.source_label or "") in policy.canonical_set(v.category_set) for v, r in answered)
    detected = [(v, r) for v, r in labelled if v.detector_label]
    detector = sum(v.detector_label == r.source_label for v, r in detected)
    return {
        "note": "The source label is a dataset claim, not an approved reference. These are not accuracy scores.",
        "final_answer_contains_source_label": ratio(final, len(answered)),
        "detector_prediction_equals_source_label": ratio(detector, len(detected)),
    }


# --------------------------------------------------------------------- structure


def _claim_link_valid(claim: Mapping[str, Any], known: frozenset[str]) -> bool:
    ids = claim.get("evidence_ids", [])
    return bool(ids) and len(set(ids)) == len(ids) and all(i in known for i in ids)


def case_structure(view: RunView, policy: EvalPolicy) -> dict[str, int]:
    """Claim and cause link counts for one run."""
    known = view.packet_evidence_ids
    claims = {c["claim_id"]: c for c in view.claims if c["claim_type"] in policy.factual_claim_types}
    valid = {cid for cid, c in claims.items() if _claim_link_valid(c, known)}
    linked = 0
    for cause in view.causes:
        ids = cause.get("claim_ids", [])
        if ids and all(i in claims for i in ids) and any(
            claims[i]["claim_type"] == "diagnostic_inference" and i in valid for i in ids
        ):
            linked += 1
    return {"claims": len(claims), "claims_with_valid_links": len(valid), "causes": len(view.causes), "causes_with_valid_links": linked}


def structure(views: Sequence[RunView], policy: EvalPolicy) -> dict[str, Any]:
    """Citation and cause-link coverage over complete runs and over every run with a parsed output."""
    def pooled(selected: Sequence[RunView]) -> dict[str, Any]:
        totals: Counter[str] = Counter()
        for view in selected:
            totals.update(case_structure(view, policy))
        return {
            "runs": len(selected),
            "citation_coverage": ratio(totals["claims_with_valid_links"], totals["claims"]),
            "cause_link_coverage": ratio(totals["causes_with_valid_links"], totals["causes"]),
        }

    return {
        "successful_only": pooled([v for v in views if v.complete]),
        "with_failures": pooled([v for v in views if v.complete or v.claims]),
        "note": "with_failures adds runs that failed validation after their output was parsed; runs that failed "
        "earlier return no claims and add nothing.",
    }


# ------------------------------------------------------------------------ judging


def judge_metrics(views: Sequence[RunView], judgments: Sequence[ClaimJudgment], policy: EvalPolicy) -> dict[str, Any]:
    """Provisional faithfulness, supported coverage and judgment coverage."""
    index = {(j.run_id, j.claim_id): j for j in judgments}
    submitted = [(v, c) for v in views if v.complete for c in v.claims if c["claim_type"] in policy.factual_claim_types]
    found = [index.get((v.run_id, c["claim_id"])) for v, c in submitted]
    status = Counter(j.status if j else "not_judged" for j in found)
    decisions = Counter(j.decision for j in found if j and j.status == "judged")
    completed = status["judged"]
    cited = sum(1 for (_, c), j in zip(submitted, found, strict=True) if c.get("evidence_ids") and j and j.status == "judged")
    faithful = decisions[policy.faithful_decision]
    complete_set = bool(submitted) and completed == len(submitted)
    return {
        "provisional": True,
        "human_review_status": "pending",
        "claims_submitted": len(submitted),
        "status_counts": dict(sorted(status.items())),
        "decision_counts": {d: decisions[d] for d in DECISIONS},
        "judgment_coverage": ratio(completed, len(submitted)),
        "provisional_faithfulness": ratio(faithful, cited),
        "provisional_supported_coverage": ratio(faithful, len(submitted)) if complete_set else {
            "value": None, "numerator": faithful, "denominator": len(submitted),
            "unavailable_reason": "judging is incomplete" if submitted else "no claims were submitted"},
    }


# -------------------------------------------------------------- status and failures


def status_block(views: Sequence[RunView], refs: Mapping[str, CaseReference]) -> dict[str, Any]:
    """Answered, abstained, incomplete and failed runs, and the abstention measures."""
    failed = Counter(v.execution_stage or "unknown" for v in views if v.execution_status == "failed")
    answerable = [v for v in views if refs.get(v.case_id) and refs[v.case_id].approved and refs[v.case_id].answerable is True]
    unanswerable = [v for v in views if refs.get(v.case_id) and refs[v.case_id].answerable is False]
    return {
        "runs": len(views),
        "answered": sum(v.answered for v in views),
        "abstained": sum(v.abstained for v in views),
        "incomplete": sum(v.execution_status == "incomplete" for v in views),
        "failed": sum(v.execution_status == "failed" for v in views),
        "failed_by_stage": dict(sorted(failed.items())),
        "false_abstention": ratio(sum(v.abstained for v in answerable), len(answerable)),
        "correct_abstention": ratio(sum(v.abstained for v in unanswerable), len(unanswerable)),
        "abstention_note": "False abstention needs approved answerable references; correct abstention needs references "
        "declared unanswerable in a separate abstention file. Zero denominators are unavailable.",
    }


# ------------------------------------------------------------------------ resources


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "max": None}
    return {"count": len(values), "mean": round(statistics.fmean(values), 3), "median": round(statistics.median(values), 3),
            "max": round(max(values), 3)}


def resources(views: Sequence[RunView], judgments: Sequence[ClaimJudgment], settings: Mapping[str, Any]) -> dict[str, Any]:
    """Pipeline and judge resources, kept apart. Hosted model memory is not reported by the provider."""
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in sorted({v.generator_kind for v in views}):
        subset = [v for v in views if v.generator_kind == kind]
        stage_name = "replay_loading" if kind == "replay" else "generation"
        by_kind[kind] = {
            "runs": len(subset),
            "seconds": {
                "retrieval_at_snapshot_time": _stats([v.timings["original_retrieval"] for v in subset if "original_retrieval" in v.timings]),
                stage_name: _stats([v.timings["generation"] for v in subset if "generation" in v.timings]),
                "validation": _stats([v.timings["validation"] for v in subset if "validation" in v.timings]),
                "total": _stats([v.timings["total"] for v in subset if "total" in v.timings]),
            },
        }
    rss = [v.resources["peak_rss"] for v in views if v.resources.get("peak_rss") is not None]
    judge_seconds = [j.seconds for j in judgments if j.status != "judge_failed"]
    return {
        "pipeline": {
            "by_generator_kind": by_kind,
            "prompt_tokens": sum(v.usage.get("prompt_tokens") or 0 for v in views),
            "completion_tokens": sum(v.usage.get("completion_tokens") or 0 for v in views),
            "client_process_peak_rss_bytes": max(rss) if rss else None,
            "client_memory_scope": "whole client process at run time, not a single run; not hosted model memory",
            "hosted_model_memory": "unavailable: the hosted provider does not report it",
            "gpu_memory": "unavailable: not measured",
        },
        "judge": {
            "judgments": len(judgments),
            "seconds": _stats(judge_seconds) | {"total": round(sum(judge_seconds), 3)},
            "prompt_tokens": sum(j.prompt_tokens or 0 for j in judgments),
            "completion_tokens": sum(j.completion_tokens or 0 for j in judgments),
            "cost": "unavailable: the provider reports tokens, not price",
        },
        "settings": dict(settings),
    }


# ---------------------------------------------------------------------- error tags


def error_tags(view: RunView, ref: CaseReference | None, judgments: Sequence[ClaimJudgment], policy: EvalPolicy) -> list[dict[str, Any]]:
    """Error tags for one run. Each tag records the evidence that justifies it."""
    tags: list[dict[str, Any]] = []
    if not view.complete:
        tags.append({"tag": "execution_failure", "provisional": False,
                     "evidence": {"stage": view.execution_stage, "reason": view.execution_reason}})
    packet_clauses = {
        (r["document_id"], clause_of(r["clause"]) or r["clause"]) for r in view.packet.get("specification_evidence", [])
    }
    if ref and not ref.fixture:
        missing = [list(c) for c in ref.expected_clauses if c in ref.annotated_clauses and c not in packet_clauses]
        if missing:
            tags.append({"tag": "retrieval_miss", "provisional": False, "evidence": {"clauses_absent_from_packet": missing}})
        gap = [list(c) for c in ref.corpus_gap_clauses]
        if gap:
            tags.append({"tag": "corpus_gap", "provisional": False, "evidence": {"clauses_missing_from_corpus": gap}})
        supplied = all(c in packet_clauses for c in ref.expected_clauses)
        if ref.approved and view.complete and supplied and not (
            view.answered and match_case(policy.canonical_set(view.category_set), ref.acceptable_sets).exact
        ):
            tags.append({"tag": "generation_miss", "provisional": False,
                         "evidence": {"expected_evidence_supplied": True, "outcome": view.result_status}})
    known = view.packet_evidence_ids
    broken = [c["claim_id"] for c in view.claims if not _claim_link_valid(c, known)]
    if broken:
        tags.append({"tag": "citation_mismatch", "provisional": False, "evidence": {"claims_with_invalid_links": broken}})
    weak = {j.claim_id: j.decision for j in judgments if j.run_id == view.run_id and j.status == "judged"
            and j.decision in SEMANTIC_TAG_DECISIONS}
    if weak:
        tags.append({"tag": "citation_mismatch", "provisional": True, "evidence": {"judge_decisions": dict(sorted(weak.items()))}})
    if view.abstained and not (ref and ref.approved):
        tags.append({"tag": "undetermined", "provisional": False,
                     "evidence": {"reason": "the abstention cannot be checked without an approved reference"}})
    return tags


def case_rows(views: Sequence[RunView], refs: Mapping[str, CaseReference], judgments: Sequence[ClaimJudgment], policy: EvalPolicy) -> list[dict[str, Any]]:
    """One row per case with its outcome, link counts, judge decisions and error tags."""
    rows = []
    for view in views:
        ref = refs.get(view.case_id)
        mine = [j for j in judgments if j.run_id == view.run_id]
        rows.append({
            "case_id": view.case_id, "run_id": view.run_id, "superseded_run_ids": list(view.superseded),
            "execution_status": view.execution_status, "execution_stage": view.execution_stage,
            "result_status": view.result_status, "abstention_reason": view.abstention_reason,
            "categories": sorted(view.category_set), "reference_approved": bool(ref and ref.approved),
            "case_status": "fixture" if ref and ref.fixture else "exploratory" if (ref is None or ref.exploratory) else "scored",
            "structure": case_structure(view, policy),
            "judgments": dict(sorted(Counter(j.decision or j.status for j in mine).items())),
            "tags": error_tags(view, ref, mine, policy),
        })
    return rows


def tag_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for tag in row["tags"]:
            counts[tag["tag"]]["provisional" if tag["provisional"] else "deterministic"] += 1
    return {k: dict(v) for k, v in sorted(counts.items())}


# ------------------------------------------------------------------------- support


def support_block(views: Sequence[RunView], refs: Mapping[str, CaseReference]) -> dict[str, Any]:
    """Source, modality, cause-count and recording-family support, and test exposure."""
    present = [refs[v.case_id] for v in views if v.case_id in refs and not refs[v.case_id].fixture]
    families = {r.family for r in present if r.family}
    return {
        "cases": len(present),
        "by_source": dict(Counter(r.source or "unknown" for r in present)),
        "by_modality": dict(Counter(r.modality or "unknown" for r in present)),
        "by_cause_count": dict(Counter(str(len(r.acceptable_sets[0])) if r.acceptable_sets else "unknown" for r in present)),
        "by_split": dict(Counter(r.split or "unknown" for r in present)),
        "recording_families": len(families),
        "note": "Cases that share a recording family or overlapping windows are not independent.",
        "test_exposure": TEST_EXPOSURE,
    }


# -------------------------------------------------------------------------- report


@dataclass
class EvaluationInputs:
    views: Sequence[RunView]
    refs: Mapping[str, CaseReference]
    judgments: Sequence[ClaimJudgment] = field(default_factory=list)
    policy: EvalPolicy = field(default_factory=EvalPolicy)
    judge_policy: JudgePolicy | None = None
    retrieval_run: Path | str | None = None


def retrieval_section(run_dir: Path | str | None) -> dict[str, Any]:
    """Spec-QA Recall@k and MRR from an existing `run_eval` directory; never recomputed here."""
    if not run_dir:
        return {"status": "not_run", "reason": "no spec_qa evaluation directory was given"}
    path = Path(run_dir) / "metrics.json"
    if not path.is_file():
        return {"status": "not_run", "reason": f"{path} does not exist"}
    return {"status": "included", "source": str(run_dir), "metrics": json.loads(path.read_text())}


def evaluate(inputs: EvaluationInputs) -> dict[str, Any]:
    """Assemble the full, deterministic metrics report."""
    views, refs, policy = inputs.views, inputs.refs, inputs.policy
    scored = identification(views, refs, policy, exploratory=False)
    exploratory = identification(views, refs, policy, exploratory=True)
    rows = case_rows(views, refs, inputs.judgments, policy)
    judge_models = {j.judge_model for j in inputs.judgments}
    generators = {v.served_model or v.requested_model for v in views}
    blockers = []
    if scored["status"] == "blocked":
        blockers.append("Scored identification is blocked: " + scored["reason"] + ".")
    if exploratory["status"] == "blocked":
        blockers.append("Exploratory identification is blocked: " + exploratory["reason"] + ".")
    if judge_models & {g for g in generators if g}:
        blockers.append("The judge model is also a generator model for these runs.")
    return {
        "evaluation_status": "exploratory_provisional",
        "human_review_status": "pending",
        "policy_version": policy.version,
        "policy_sha256": policy.sha256,
        "identification": {"scored": scored, "exploratory": exploratory,
                           "agreement_with_source_labels": agreement_blocks(views, refs, policy)},
        "structure": structure(views, policy),
        "support": judge_metrics(views, inputs.judgments, policy),
        "status": status_block(views, refs),
        "retrieval": retrieval_section(inputs.retrieval_run),
        "case_support": support_block(views, refs),
        "error_tags": tag_summary(rows),
        "blockers": blockers,
        "cases": rows,
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
