"""Label and KPI-support checks run on teacher drafts before judging (Task 4.9).

The label check compares a draft with the source label kept outside model input. The KPI-support
check applies "evidence beyond the label" to every source: a named fault must rest on measured
incident evidence, not on the anomaly label or on specification text alone.
"""

from __future__ import annotations

from clauseguard.api.schemas import RcaModelOutput
from clauseguard.rca.candidates import CandidateReference
from clauseguard.rca.checks import NORMAL, CheckResult
from clauseguard.rca.contracts import EvidencePacket

LABEL_CHECK_ID = "label.v1"
KPI_SUPPORT_CHECK_ID = "kpi_support.v1"


def label_check(output: RcaModelOutput | None, reference: CandidateReference) -> CheckResult:
    """Compare the returned categories with the source label (``label.v1``)."""
    def result(passed: bool, detail: str | None = None) -> CheckResult:
        return CheckResult(check_id=LABEL_CHECK_ID, status="passed" if passed else "failed", detail=detail)

    if output is None:
        return result(False, "no parsed output")
    if reference.label_status not in ("labelled_fault", "labelled_normal"):
        return result(False, f"reference is {reference.label_status}, not a usable label")
    kind = "fault" if reference.label_status == "labelled_fault" else "normal"
    if output.status == "abstained":
        return result(False, f"abstained on a labelled {kind}")
    returned = {cause.category for cause in output.final_fault_categories}
    if kind == "normal":
        return result(returned == {NORMAL}, None if returned == {NORMAL} else f"returned {sorted(returned)}, label is normal")
    if returned & set(reference.causes):
        return result(True)
    return result(False, f"returned {sorted(returned)}, none matches the labelled cause")


def kpi_support_check(output: RcaModelOutput | None, packet: EvidencePacket) -> CheckResult:
    """Every diagnostic claim of a named fault cites packet KPI evidence (``kpi_support.v1``)."""
    problems: list[str] = []
    if output is not None and output.status == "answered":
        kpi_ids = {item.evidence_id for item in packet.kpi_evidence}
        claims = {claim.claim_id: claim for claim in output.claims}
        for cause in output.final_fault_categories:
            if cause.category == NORMAL:
                continue
            diagnostics = [claims[cid] for cid in cause.claim_ids
                           if cid in claims and claims[cid].claim_type == "diagnostic_inference"]
            if not diagnostics:
                problems.append(f"{cause.category}: no diagnostic claim")
            problems += [f"{claim.claim_id}: cites no KPI evidence" for claim in diagnostics
                         if not kpi_ids & set(claim.evidence_ids)]
    if problems:
        return CheckResult(check_id=KPI_SUPPORT_CHECK_ID, status="failed", detail="; ".join(problems))
    return CheckResult(check_id=KPI_SUPPORT_CHECK_ID, status="passed")
