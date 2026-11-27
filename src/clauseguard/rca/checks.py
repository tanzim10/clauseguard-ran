"""Runtime checks on generator output against its exact evidence packet (Task 4.7).

Each check has a versioned ID and reports every failure, not only the first. The checks enforce
the approved ``rca-v1`` output contract plus the #63 rules (separate support per cause, declared
taxonomy, competing causes abstain, normal kept distinct) without changing the schema. Passing
them shows structure and citation types only; it cannot show that a diagnosis is correct.
"""

from __future__ import annotations

import json
import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from clauseguard.api.schemas import RcaClaim, RcaModelOutput
from clauseguard.datasets.contracts import CATEGORY_LABELS
from clauseguard.rca.contracts import (
    EvidencePacket,
    KpiStatisticEvidence,
    SpecificationEvidence,
    TextEvidence,
    validate_output_for_packet,
)
from clauseguard.rca.evidence import relative_change
from clauseguard.rca.prompts import ABSTENTION_REASONS

CHECKS_VERSION = "rca-checks-v1"
NORMAL = "normal"


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    check_id: str
    status: Literal["passed", "failed"]
    subject: str | None = None
    detail: str | None = None


class Taxonomy(BaseModel):
    """The categories a run may return; ``normal`` is allowed only when declared."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    taxonomy_id: str = Field(min_length=1)
    labels: tuple[str, ...] = Field(min_length=1)
    allows_normal: bool

    @model_validator(mode="after")
    def check_labels(self) -> Taxonomy:
        if len(set(self.labels)) != len(self.labels) or NORMAL in self.labels:
            raise ValueError("labels must be unique and must not list normal; use allows_normal")
        return self

    def categories(self) -> tuple[str, ...]:
        return (*self.labels, NORMAL) if self.allows_normal else self.labels


TELECOMTS_TAXONOMY = Taxonomy(taxonomy_id="telecomts-v1", labels=CATEGORY_LABELS, allows_normal=True)
# Source taxonomies for training runs (Task 4.9). Labels are never mapped across sources
# (data/candidates/source-fields.md), so each source is run with its own list. The TelecomTS
# candidate taxonomy adds the real-capture Jamming cause to the 10 synthetic categories.
RUN_TAXONOMIES: dict[str, Taxonomy] = {
    "telecomts-v1": TELECOMTS_TAXONOMY,
    "telecomts-v1-jamming": Taxonomy(taxonomy_id="telecomts-v1-jamming", labels=(*CATEGORY_LABELS, "Jamming"),
                                     allows_normal=True),
    "bubbleran-pi-v1": Taxonomy(taxonomy_id="bubbleran-pi-v1", labels=("persistent_interference",),
                                allows_normal=True),
}
# Candidate taxonomy (Task 4.6 reference) -> run taxonomy used to draft its training target.
CANDIDATE_RUN_TAXONOMY = {"telecomts-v1": "telecomts-v1-jamming", "bubbleran-pi-v1": "bubbleran-pi-v1"}


class OutputParseError(ValueError):
    """Raw output is not one strict ``RcaModelOutput`` JSON object."""


def parse_model_output(raw: str) -> RcaModelOutput:
    """Strict parse: no code-fence stripping, no repair, unknown fields rejected."""
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise OutputParseError("output is not valid JSON") from exc
    if not isinstance(value, dict):
        raise OutputParseError("output is not a JSON object")
    try:
        return RcaModelOutput.model_validate(value)
    except ValidationError as exc:
        errors = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'output'}: {e['msg']}" for e in exc.errors())
        raise OutputParseError(errors) from exc


def _incident(item: object) -> bool:
    return isinstance(item, (KpiStatisticEvidence, TextEvidence))


def structural_checks(output: RcaModelOutput, packet: EvidencePacket, taxonomy: Taxonomy) -> list[CheckResult]:
    """Every check ID appears once as passed, or once per failing claim or cause."""
    evidence = {item.evidence_id: item for item in
                (*packet.kpi_evidence, *packet.text_evidence, *packet.specification_evidence)}
    claims = {claim.claim_id: claim for claim in output.claims}
    failures: dict[str, list[CheckResult]] = {}

    def fail(check_id: str, subject: str, detail: str) -> None:
        failures.setdefault(check_id, []).append(
            CheckResult(check_id=check_id, status="failed", subject=subject, detail=detail))

    def cited(claim: RcaClaim) -> list:
        return [evidence[eid] for eid in claim.evidence_ids if eid in evidence]

    def spec_ids(claim: RcaClaim) -> set[str]:
        return {eid for eid in claim.evidence_ids if isinstance(evidence.get(eid), SpecificationEvidence)}

    def incident_ids(claim: RcaClaim) -> set[str]:
        return {eid for eid in claim.evidence_ids if _incident(evidence.get(eid))}

    for claim in output.claims:
        unknown = sorted(set(claim.evidence_ids) - evidence.keys())
        if unknown:
            fail("evidence_ids_known.v1", claim.claim_id, f"unknown evidence IDs {unknown}")
        if len(set(claim.evidence_ids)) != len(claim.evidence_ids):
            fail("evidence_ids_unique_in_claim.v1", claim.claim_id, "an evidence ID is repeated")
        items = cited(claim)
        if claim.claim_type == "observation" and not all(_incident(item) for item in items):
            fail("observation_cites_incident.v1", claim.claim_id, "observation cites non-incident evidence")
        if claim.claim_type == "mechanism" and not all(isinstance(i, SpecificationEvidence) for i in items):
            fail("mechanism_cites_specification.v1", claim.claim_id, "mechanism cites non-specification evidence")
        if claim.claim_type == "diagnostic_inference" and not (incident_ids(claim) and spec_ids(claim)):
            fail("diagnostic_cites_incident_and_specification.v1", claim.claim_id,
                 "diagnostic claim needs incident and specification evidence")

    if output.status == "abstained" and output.abstention_reason not in ABSTENTION_REASONS:
        fail("abstention_reason_known.v1", "abstention_reason",
             f"{output.abstention_reason!r} is not one of {list(ABSTENTION_REASONS)}")

    causes = output.final_fault_categories
    diagnostics = {cause.category: [claims[cid] for cid in cause.claim_ids
                                    if cid in claims and claims[cid].claim_type == "diagnostic_inference"]
                   for cause in causes}
    for cause in causes:
        if cause.category not in taxonomy.categories():
            fail("category_in_taxonomy.v1", cause.category, f"not in taxonomy {taxonomy.taxonomy_id}")
        mechanisms = [claims[cid] for cid in cause.claim_ids if cid in claims and claims[cid].claim_type == "mechanism"]
        if not any(spec_ids(d) & spec_ids(m) for d in diagnostics[cause.category] for m in mechanisms):
            fail("diagnostic_links_mechanism.v1", cause.category,
                 "no linked mechanism claim cites a specification passage that a linked diagnostic claim cites")
        others = [other for other in causes if other is not cause]
        shared = {cid for other in others for cid in other.claim_ids}
        if not any(d.claim_id not in shared for d in diagnostics[cause.category]):
            fail("cause_separate_support.v1", cause.category, "every linked diagnostic claim is shared with another cause")
        if others:
            own = {eid for d in diagnostics[cause.category] for eid in incident_ids(d)}
            rest = {eid for other in others for d in diagnostics[other.category] for eid in incident_ids(d)}
            if not own - rest:
                fail("competing_causes_abstain.v1", cause.category,
                     "explains the same incident evidence as another cause; competing causes must abstain "
                     "with conflicting_evidence")
    if NORMAL in {cause.category for cause in causes} and len(causes) > 1:
        fail("normal_exclusive.v1", NORMAL, "normal cannot be returned with a fault category")

    if not failures:
        # Guard against drift from the approved packet rules in ``rca.contracts``.
        try:
            validate_output_for_packet(output, packet)
        except ValueError as exc:
            fail("packet_contract.v1", "output", str(exc))

    results = []
    for check_id in STRUCTURAL_CHECK_IDS:
        results += failures.get(check_id) or [CheckResult(check_id=check_id, status="passed")]
    return results


STRUCTURAL_CHECK_IDS = (
    "evidence_ids_known.v1",
    "evidence_ids_unique_in_claim.v1",
    "observation_cites_incident.v1",
    "mechanism_cites_specification.v1",
    "diagnostic_cites_incident_and_specification.v1",
    "diagnostic_links_mechanism.v1",
    "category_in_taxonomy.v1",
    "cause_separate_support.v1",
    "competing_causes_abstain.v1",
    "normal_exclusive.v1",
    "abstention_reason_known.v1",
    "packet_contract.v1",
)


NUMERIC_CHECK_ID = "numeric.v2"
NUMBER = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?(?![A-Za-z0-9_])")
KPI_NUMBER_FIELDS = ("value", "valid_count", "sample_count", "missing_fraction", "sample_start", "sample_end")


class NumberMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    token: str
    status: Literal["matched", "unmatched", "ignored"]
    evidence_id: str | None = None
    field: str | None = None


class NumericCheckResult(BaseModel):
    """Numbers quoted in one claim against the records that claim cites."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    check_id: Literal["numeric.v2"] = NUMERIC_CHECK_ID
    claim_id: str
    status: Literal["passed", "failed", "no_numbers"]
    numbers: list[NumberMatch] = Field(default_factory=list)


def _shown_equal(token: str, value: float) -> bool:
    """``value`` written at the precision ``token`` shows (decimal places, or digits in e-notation)."""
    shown = float(token)
    mantissa = token.lower().split("e")[0].lstrip("+-")
    if "e" in token.lower():
        digits = max(1, len(mantissa.replace(".", "").lstrip("0")))
        return math.isclose(float(f"{value:.{digits - 1}e}"), shown, rel_tol=1e-12)
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return math.isclose(round(value, decimals), shown, rel_tol=0, abs_tol=10 ** -(decimals + 6))


def _tokens(text: str) -> list[str]:
    return NUMBER.findall(text)


def numeric_checks(output: RcaModelOutput, packet: EvidencePacket) -> list[NumericCheckResult]:
    """Check each number in observation and diagnostic claims against the cited records.

    Numbers that also appear in a cited specification passage (clause or document numbers) are
    ignored. A percentage matches only a missing fraction or a verified baseline change. Since
    ``numeric.v2`` a number written in the packet question is an accepted source (field
    ``question``), because the question is supplied input. Invented, truncated or mis-rounded
    numbers still fail. This shows that a number was copied correctly, not that it was used correctly.
    """
    question_tokens = set(_tokens(packet.question))
    evidence = {item.evidence_id: item for item in
                (*packet.kpi_evidence, *packet.text_evidence, *packet.specification_evidence)}
    results = []
    for claim in output.claims:
        if claim.claim_type == "mechanism":
            continue
        cited = [evidence[eid] for eid in claim.evidence_ids if eid in evidence]
        spec_tokens = {token for item in cited if isinstance(item, SpecificationEvidence)
                       for field in (item.text, item.clause, item.version, item.document_id)
                       for token in _tokens(field)}
        numbers = []
        for token in _tokens(claim.text):
            if token in spec_tokens:
                numbers.append(NumberMatch(token=token, status="ignored", field="specification_reference"))
                continue
            match = _match(token, cited, evidence)
            if match is None and token in question_tokens:
                match = NumberMatch(token=token, status="matched", field="question")
            numbers.append(match or NumberMatch(token=token, status="unmatched"))
        status = ("no_numbers" if not any(n.status != "ignored" for n in numbers)
                  else "failed" if any(n.status == "unmatched" for n in numbers) else "passed")
        results.append(NumericCheckResult(claim_id=claim.claim_id, status=status, numbers=numbers))
    return results


def _match(token: str, cited: list, evidence: dict) -> NumberMatch | None:
    percent = token.endswith("%")
    number = token.rstrip("%")
    for item in cited:
        if isinstance(item, KpiStatisticEvidence):
            if percent:
                candidates = {"missing_fraction": item.missing_fraction * 100}
                baseline = evidence.get(item.baseline_evidence_id) if item.baseline_evidence_id else None
                if isinstance(baseline, KpiStatisticEvidence):
                    change = relative_change(baseline.value, item.value)
                    if change is not None:
                        candidates["baseline_change"] = change * 100
            else:
                candidates = {name: getattr(item, name) for name in KPI_NUMBER_FIELDS
                              if getattr(item, name) is not None}
            for name, value in candidates.items():
                if _shown_equal(number, float(value)):
                    return NumberMatch(token=token, status="matched", evidence_id=item.evidence_id, field=name)
        elif isinstance(item, TextEvidence) and token in _tokens(item.text):
            return NumberMatch(token=token, status="matched", evidence_id=item.evidence_id, field="text")
    return None
