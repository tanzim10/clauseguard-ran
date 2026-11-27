"""Evaluation reservations, source inventory and source candidate builders (Task 4.6).

Reservations are fixed before any candidate is admitted. Builders reuse the Task 4.4 adapters
for observations and read source labels only into separate ``CandidateReference`` rows.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import Field

from clauseguard.baselines.contracts import Task
from clauseguard.datasets.schema import NonEmptyStr, ScenarioRecord
from clauseguard.rca.candidates import (
    CandidateIdentity,
    CandidateRecord,
    CandidateReference,
    CandidateVersions,
    EligibilityStatus,
    Grouping,
    ModalityStates,
    Observations,
    PredictionRef,
    Sha256,
    SourceInventory,
    StrictModel,
    Taxonomy,
)
from clauseguard.rca.contracts import DetectorPrediction, EvidencePacket
from clauseguard.rca.cross_fit import reserved_families
from clauseguard.rca.detector import CaseRecord, dataset_absence, read_cases
from clauseguard.rca.evidence import check_audits
from clauseguard.rca.sources import (
    SourceEvidence,
    _read_rows,
    bubbleran_evidence,
    spotlight_evidence,
    telecomts_evidence,
)
from clauseguard.rca.summarizer import CALCULATION_VERSION

RESERVATION_VERSION = "rca-reservations-v1"
SOURCE_REVIEW = "Task 4.6 source review: licence files, dataset cards and repository APIs"
SOURCE_REVIEW_DATE = "2026-11-13"
# Recorded terms per source. Local files and hashes are filled in from disk by ``build_inventory``.
SOURCE_TERMS: dict[str, dict[str, Any]] = {
    "TelecomTS": {
        "identifier": "https://huggingface.co/datasets/AliMaatouk/TelecomTS", "licence": "MIT",
        "access": "public", "redistribution": "allowed", "training_use": "cleared",
        "adapter": "rca.sources.telecomts_evidence", "blockers": [],
    },
    "BubbleRAN": {
        "identifier": "https://github.com/bubbleran/open-telco-datasets/tree/main/datasets/"
                      "anomaly-detection-persistent-interference",
        "licence": "CC BY-SA 4.0 (dataset metadata.yaml)", "access": "public", "redistribution": "share_alike",
        "training_use": "pending", "adapter": "rca.sources.bubbleran_evidence",
        "blockers": ["CC BY-SA 4.0 share-alike terms are not yet cleared for model training"],
    },
    "SpotLight": {
        "identifier": "https://github.com/netsys-edinburgh/SpotLight", "licence": None,
        "access": "public", "redistribution": "unknown", "training_use": "blocked",
        "adapter": "rca.sources.spotlight_evidence",
        "blockers": ["no licence is published for the repository or its data",
                     "only the one-experiment sample is local and its anomaly type is not stated"],
    },
    "TeleLogs": {
        "identifier": "https://huggingface.co/datasets/netop/TeleLogs", "licence": "MIT (dataset card)",
        "access": "gated", "redistribution": "forbidden_by_request", "training_use": "blocked",
        "adapter": None,
        "blockers": ["gated access has not been accepted; source files are not available locally",
                     "the dataset card asks users not to share, re-upload or distribute the benchmark",
                     "revision, fields, families and multi-cause rows are unverified",
                     "settings evidence has no rca-v1 contract type"],
    },
    "5G core faults": {
        "identifier": "https://huggingface.co/datasets/greenwich157/telco-5G-core-faults "
                      "(identified by name only; unconfirmed)",
        "licence": "Apache-2.0 (dataset card)", "access": "public", "redistribution": "allowed",
        "training_use": "blocked", "adapter": None,
        "blockers": ["dataset identity is unconfirmed by the project",
                     "790 of 1,323 inputs state the cause; outputs are remediation steps, not RCA",
                     "source files are not available locally"],
    },
}
TEXT_NORMALIZATION = "NFKC, lowercase, whitespace collapsed to single spaces, trimmed"
GOLDEN_FILES = ("rca.jsonl", "spec_qa.jsonl", "rca_policy.json", "rca_questions.candidate.json")
# Official upstream test material that must stay out of training once its source is available.
EXTERNAL_RESERVATIONS = (
    {"dataset": "TeleLogs", "scope": "official test split (test.json)",
     "status": "declared; source files are not available locally"},
    {"dataset": "TeleLogs", "scope": "100 GSMA ot-lite TeleLogs items (test_telelogs.json)",
     "status": "declared; source files are not available locally"},
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def shingles(text: str, size: int) -> set[str]:
    normal = normalize_text(text)
    return {normal[i:i + size] for i in range(max(len(normal) - size + 1, 1))}


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


class ReservedText(StrictModel):
    text_id: NonEmptyStr
    source: NonEmptyStr
    sha256: Sha256
    text: NonEmptyStr  # normalized; golden files are tracked, so no restricted content is copied


class Reservations(StrictModel):
    version: str = RESERVATION_VERSION
    telecomts_release_id: NonEmptyStr
    golden_release_id: NonEmptyStr
    golden_files: dict[NonEmptyStr, Sha256]
    families: list[NonEmptyStr]
    scenario_ids: list[NonEmptyStr]
    texts: list[ReservedText]
    external: list[dict[str, str]] = Field(default_factory=list)
    text_normalization: str = TEXT_NORMALIZATION

    def reasons(self, *, family: str | None, record_id: str) -> list[str]:
        """Why a candidate overlaps evaluation data; empty when it does not."""
        found = []
        if family is not None and family in set(self.families):
            found.append(f"reserved evaluation family {family}")
        if record_id in set(self.scenario_ids):
            found.append(f"reserved evaluation case {record_id}")
        return found

    def text_matches(self, text: str, *, threshold: float, size: int) -> list[str]:
        """IDs of reserved texts that ``text`` duplicates exactly or nearly (character shingles)."""
        digest, grams = hashlib.sha256(normalize_text(text).encode()).hexdigest(), shingles(text, size)
        return [r.text_id for r in self.texts
                if r.sha256 == digest or jaccard(grams, shingles(r.text, size)) >= threshold]


def _reserved_text(text_id: str, source: str, text: str) -> ReservedText:
    normal = normalize_text(text)
    return ReservedText(text_id=text_id, source=source, text=normal,
                        sha256=hashlib.sha256(normal.encode()).hexdigest())


def build_reservations(dataset, golden_dir: Path) -> Reservations:
    """Golden cases, their families, and every golden question, paraphrase and answer."""
    golden_dir = Path(golden_dir)
    manifest = json.loads((golden_dir / "manifest.json").read_text(encoding="utf-8"))
    files = {name: hashlib.sha256((golden_dir / name).read_bytes()).hexdigest() for name in GOLDEN_FILES}
    for name, entry in manifest["files"].items():
        if name in files and files[name] != entry["sha256"]:
            raise ValueError(f"golden file differs from its release manifest: {name}")
    policy = json.loads((golden_dir / "rca_policy.json").read_text(encoding="utf-8"))
    rca = [json.loads(line) for line in (golden_dir / "rca.jsonl").read_text(encoding="utf-8").splitlines()
           if line.strip()]
    spec = [json.loads(line) for line in (golden_dir / "spec_qa.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    paraphrases = json.loads((golden_dir / "rca_questions.candidate.json").read_text(encoding="utf-8"))
    scenario_ids = {row["scenario_id"] for row in rca} | {c["scenario_id"] for c in policy["cases"].values()}
    if policy["source_release_id"] != dataset.release_id:
        raise ValueError("golden policy belongs to a different TelecomTS release")
    texts = []
    for row in rca:
        texts.append(_reserved_text(f"{row['id']}:question", "rca.jsonl", row["question"]))
        texts.append(_reserved_text(f"{row['id']}:expected_conclusion", "rca.jsonl", row["expected_conclusion"]))
        texts.extend(_reserved_text(f"{row['id']}:acceptable_conclusion:{i}", "rca.jsonl", text)
                     for i, text in enumerate(row.get("acceptable_conclusions", [])))
    for row in spec:
        texts.append(_reserved_text(f"{row['id']}:question", "spec_qa.jsonl", row["question"]))
        texts.append(_reserved_text(f"{row['id']}:expected_answer", "spec_qa.jsonl", row["expected_answer"]))
    for row in paraphrases:
        texts.extend(_reserved_text(f"{row['id']}:{key}", "rca_questions.candidate.json", row[key])
                     for key in ("qualitative_question", "numeric_question") if row.get(key))
    return Reservations(
        telecomts_release_id=dataset.release_id, golden_release_id=manifest["release_id"], golden_files=files,
        families=sorted(reserved_families(dataset, scenario_ids)), scenario_ids=sorted(scenario_ids),
        texts=sorted(texts, key=lambda t: t.text_id), external=[dict(e) for e in EXTERNAL_RESERVATIONS])


def write_json(path: Path, value: Any) -> str:
    """Canonical, sorted JSON; returns the SHA-256 of the written bytes."""
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    Path(path).write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode()).hexdigest()


def load_reservations(path: Path, expected_sha256: str | None = None) -> Reservations:
    content = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("reservations changed since they were recorded")
    return Reservations.model_validate_json(content)



def _local_hashes(directory: Path | None) -> dict[str, str]:
    if directory is None or not Path(directory).is_dir():
        return {}
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(directory).iterdir()) if path.is_file()}


def build_inventory(*, telecomts_acquisition: Path | None = None, bubbleran_dir: Path | None = None,
                    spotlight_dir: Path | None = None) -> list[SourceInventory]:
    """All five sources, available or not; TelecomTS hashes come from its acquisition record."""
    local: dict[str, tuple[str | None, dict[str, str], str]] = {
        name: (None, {}, SOURCE_REVIEW_DATE) for name in SOURCE_TERMS}
    if telecomts_acquisition is not None and Path(telecomts_acquisition).is_file():
        record = json.loads(Path(telecomts_acquisition).read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            raise ValueError("TelecomTS acquisition is incomplete")
        local["TelecomTS"] = (record["revision"], {f["source_file"]: f["sha256"] for f in record["files"]},
                              record["acquired_at"][:10])
    for name, directory in (("BubbleRAN", bubbleran_dir), ("SpotLight", spotlight_dir)):
        files = _local_hashes(directory)
        revision = f"local-copy-sha256:{canonical_sha256(files)[:16]}" if files else None
        local[name] = (revision, files, SOURCE_REVIEW_DATE)
    inventory = []
    for name, terms in SOURCE_TERMS.items():
        revision, files, verified_on = local[name]
        verified_by = ("TelecomTS acquisition record (declared licence and file hashes)"
                       if name == "TelecomTS" and files else SOURCE_REVIEW)
        blockers = list(terms["blockers"])
        if not files and terms["adapter"] and "source files are not available locally" not in blockers:
            blockers.append("source files are not available locally")
        inventory.append(SourceInventory(
            dataset=name, revision=revision, local_status="present" if files else "absent", local_files=files,
            verified_by=verified_by, verified_on=verified_on, **{**terms, "blockers": blockers}))
    return inventory


# --- Configuration and TelecomTS candidates ----------------------------------------------------

MINIMAL_QUESTION = "Explain the observed KPI changes using the supplied specification evidence."
ADAPTER_VERSION = "rca-evidence-v1"
GROUPING_VERSION = "candidate-grouping-v1"


class CandidateConfig(StrictModel):
    """Every pool decision that is not fixed by a contract; saved beside the outputs."""

    packet_head: Task = "diagnosis"  # matches the Task 4.4 packets (build_evidence.py)
    question_template_version: NonEmptyStr = "candidate-question-v1"
    # Fixed questions per modality, without numbers; never copied from golden questions.
    question_templates: dict[str, NonEmptyStr] = Field(default_factory=lambda: {
        "kpi": MINIMAL_QUESTION,
        "text": "Explain the reported observations using the supplied specification evidence.",
        "kpi+text": "Explain the observed KPI changes and reported observations using the supplied "
                    "specification evidence.",
    })
    bubbleran_window_rows: int = Field(default=60, ge=2)
    bubbleran_stride: int = Field(default=60, ge=1)
    near_duplicate_threshold: float = Field(default=0.8, gt=0, le=1)
    shingle_size: int = Field(default=5, ge=1)
    pending_terms_status: Literal["ineligible", "exploratory"] = "ineligible"
    validation_fraction: float = Field(default=0.2, ge=0, lt=1)
    split_seed: int = 42
    batch_size: int = Field(default=30, ge=0, le=30)
    batch_seed: int = 42
    batch_statuses: list[EligibilityStatus] = Field(default_factory=lambda: ["eligible_pilot", "exploratory"])
    batch_per_category: int = Field(default=2, ge=0)
    batch_jamming: int = Field(default=2, ge=0)
    batch_normal: int = Field(default=3, ge=0)
    batch_bubbleran: int = Field(default=3, ge=0)
    # Share of each batch stratum drawn from detector-correction cases; the rest agree with the label.
    batch_correction_share: float = Field(default=0.5, ge=0, le=1)
    spotlight_window_rows: int = Field(default=100, ge=2)
    spotlight_stride: int = Field(default=100, ge=1)

    def question(self, kpi: str, text: str) -> str:
        key = "+".join(name for name, state in (("kpi", kpi), ("text", text)) if state == "included")
        if key not in self.question_templates:
            raise ValueError("a candidate needs at least one included modality")
        return self.question_templates[key]


class BuiltCandidate(NamedTuple):
    candidate: CandidateRecord
    reference: CandidateReference
    texts: tuple[str, ...] = ()  # approved text the packet carries, for duplicate checks only


def packet_digest(packet: EvidencePacket) -> str:
    return canonical_sha256(packet.model_dump(mode="json"))


def observation_digest(packet: EvidencePacket) -> str:
    """Identity-free fingerprint of what the model would observe, for exact-duplicate checks."""
    return canonical_sha256({
        "kpi": sorted([k.channel, k.statistic, k.value, k.valid_count, k.sample_count] for k in packet.kpi_evidence),
        "text": sorted(normalize_text(t.text) for t in packet.text_evidence)})


def _rejected(built: SourceEvidence) -> list[str]:
    return sorted({r.channel or "(unnamed index column)" for r in built.rejections})


def load_predictions(path: Path, folds_path: Path, *, release_id: str) -> tuple[dict[str, CaseRecord], str]:
    """Task 4.2 out-of-fold records, checked against the hash and release saved in folds.json."""
    content = Path(path).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    folds = json.loads(Path(folds_path).read_text(encoding="utf-8"))
    if digest != folds["predictions_sha256"] or folds["release_id"] != release_id:
        raise ValueError("detector predictions differ from their cross-fit record or release")
    records = {r.scenario_id: r for r in read_cases(path)}
    if any(r.purpose != "slm_training_candidate" for r in records.values()):
        raise ValueError("training candidates need out-of-fold slm_training_candidate predictions")
    return records, digest


def _prediction_ref(case: CaseRecord | None, *, head: str, file_sha256: str | None) -> PredictionRef:
    if case is None:
        return PredictionRef(packet_head=head, status="absent",
                             absence_reason="no out-of-fold prediction for this window")
    return PredictionRef(packet_head=head, status="available", file_sha256=file_sha256,
                         selection_sha256=case.selection_sha256, record_id=case.scenario_id, fold=case.fold,
                         heads={task: getattr(case, task).prediction for task in ("detection", "category", "diagnosis")})


def telecomts_packet(record: ScenarioRecord, member: Mapping[str, Any], prediction: DetectorPrediction | None,
                     config: CandidateConfig) -> tuple[EvidencePacket, SourceEvidence]:
    """The Task 4.4 packet for one window with the fixed candidate question; retrieval not run."""
    built = telecomts_evidence(record, family_id=member["group_id"], family_status=member["evidence_status"])
    packet = built.registry.packet(source=built.source, question=config.question(built.kpi_state, built.text_state),
                                   kpi_state=built.kpi_state, text_state=built.text_state,
                                   retrieval_status="not_run", detector_prediction=prediction)
    check_audits(packet, built.registry.audits())
    return packet, built


def telecomts_reference(record: ScenarioRecord, candidate_id: str) -> CandidateReference:
    """Labels, annotation and reference-text locations; the reference text itself is not copied."""
    labels = {"rca_label": [record.rca_label], "fault_label": [record.fault_label]}
    if record.anomaly.source_type:
        labels["anomaly.source_type"] = [record.anomaly.source_type]
    fault = record.fault_label == "abnormal"
    return CandidateReference(
        candidate_id=candidate_id, taxonomy_id="telecomts-v1", source_labels=labels,
        causes=[record.rca_label] if fault else [], label_status="labelled_fault" if fault else "labelled_normal",
        annotation={"origin": record.origin, "anomaly_start_index": record.anomaly.start_index,
                    "anomaly_end_index": record.anomaly.end_index,
                    "affected_channels": list(record.anomaly.affected_channels)},
        source_text_refs=[f"reference.{key}" for key in sorted(record.reference)])


def telecomts_candidates(dataset, predictions: Mapping[str, CaseRecord], *, predictions_sha256: str | None,
                         config: CandidateConfig, split: str = "train") -> Iterator[BuiltCandidate]:
    """One candidate per window of ``split``; development and test windows are never built."""
    if split != "train":
        raise ValueError("only training-split windows become SLM candidates")
    membership = dataset.splits["membership"]
    for sid in sorted(s for s, row in membership.items() if row["split"] == split):
        record, member = dataset.load_scenario(sid), membership[sid]
        case = predictions.get(sid)
        prediction = getattr(case, config.packet_head).prediction if case else None
        packet, built = telecomts_packet(record, member, prediction, config)
        candidate_id = f"TelecomTS:{sid}"
        reference = telecomts_reference(record, candidate_id)
        provenance = record.provenance
        candidate = CandidateRecord(
            identity=CandidateIdentity(candidate_id=candidate_id, source_dataset="TelecomTS",
                                       source_revision=provenance.revision, source_record_id=sid,
                                       source_location={"file": provenance.source_file, "row": provenance.source_row},
                                       source_sha256=provenance.source_sha256),
            modalities=ModalityStates(kpi=built.kpi_state, text=built.text_state, settings="missing",
                                      text_reason=built.text_reason),
            observations=Observations(packet_status="built", packet_sha256=packet_digest(packet),
                                      observation_sha256=observation_digest(packet), question_template=packet.question,
                                      kpi_records=len(packet.kpi_evidence), text_records=len(packet.text_evidence),
                                      rejected_channels=_rejected(built)),
            prediction_ref=_prediction_ref(case, head=config.packet_head, file_sha256=predictions_sha256),
            grouping=Grouping(group_id=member["group_id"], group_status=member["evidence_status"],
                              grouping_version=GROUPING_VERSION),
            taxonomy=Taxonomy(taxonomy_id=reference.taxonomy_id, cause_count=reference.cause_count,
                              label_status=reference.label_status),
            versions=CandidateVersions(adapter=ADAPTER_VERSION, calculation=CALCULATION_VERSION,
                                       question_template=config.question_template_version))
        yield BuiltCandidate(candidate, reference, tuple(t.text for t in packet.text_evidence))


# --- BubbleRAN candidates ----------------------------------------------------------------------

BUBBLERAN_CAUSE = "persistent_interference"
BUBBLERAN_GRID_RULE = ("fixed grid of {rows} consecutive 1 s rows at stride {stride} inside one capture "
                       "segment; a segment ends at a time gap, a timestamp reset or a UE/gNB change; "
                       "label values never move a window boundary")


def bubbleran_windows(path: Path, *, rows: int, stride: int) -> list[tuple[int, int]]:
    """Label-blind grid windows (first and last zero-based data rows) that never cross a segment."""
    _, data = _read_rows(path)
    segments, start = [], 0
    for i in range(1, len(data) + 1):
        if i == len(data) or int(data[i]["timestamp"]) - int(data[i - 1]["timestamp"]) != 1 or (
                (data[i]["ran_ue_id"], data[i]["e2node_nb_id"]) != (data[i - 1]["ran_ue_id"],
                                                                    data[i - 1]["e2node_nb_id"])):
            segments.append((start, i - 1))
            start = i
    return [(first, first + rows - 1) for begin, end in segments
            for first in range(begin, end - rows + 2, stride)]


def _table_packet(built: SourceEvidence, config: CandidateConfig) -> tuple[EvidencePacket, SourceEvidence]:
    """Packet for a source without a compatible detector: its prediction is the recorded absence."""
    prediction = dataset_absence(built.source.dataset)[config.packet_head]
    packet = built.registry.packet(source=built.source, question=config.question(built.kpi_state, built.text_state),
                                   kpi_state=built.kpi_state, text_state=built.text_state,
                                   retrieval_status="not_run", detector_prediction=prediction)
    check_audits(packet, built.registry.audits())
    return packet, built


def _table_candidate(packet: EvidencePacket, built: SourceEvidence, reference: CandidateReference, *,
                     location: dict[str, str | int], family_id: str, event_id: str,
                     config: CandidateConfig) -> BuiltCandidate:
    source = built.source
    candidate = CandidateRecord(
        identity=CandidateIdentity(candidate_id=reference.candidate_id, source_dataset=source.dataset,
                                   source_revision=source.revision, source_record_id=source.case_id,
                                   source_location=location, source_sha256=source.artifact_sha256),
        modalities=ModalityStates(kpi=built.kpi_state, text=built.text_state, settings="missing",
                                  text_reason=built.text_reason),
        observations=Observations(packet_status="built", packet_sha256=packet_digest(packet),
                                  observation_sha256=observation_digest(packet), question_template=packet.question,
                                  kpi_records=len(packet.kpi_evidence), rejected_channels=_rejected(built)),
        prediction_ref=PredictionRef(packet_head=config.packet_head, status="absent",
                                     absence_reason=packet.detector_prediction.absence_reason),
        grouping=Grouping(group_id=family_id, group_status="unresolved", grouping_version=GROUPING_VERSION,
                          event_id=event_id),
        taxonomy=Taxonomy(taxonomy_id=reference.taxonomy_id, cause_count=reference.cause_count,
                          label_status=reference.label_status),
        versions=CandidateVersions(adapter=ADAPTER_VERSION, calculation=CALCULATION_VERSION,
                                   question_template=config.question_template_version))
    return BuiltCandidate(candidate, reference)


def bubbleran_packet(path: Path, start: int, end: int, config: CandidateConfig) -> tuple[EvidencePacket, SourceEvidence]:
    return _table_packet(bubbleran_evidence(path, start_row=start, end_row=end), config)


def _episodes(flags: list[bool]) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    return runs


def bubbleran_candidates(paths: list[Path], *, config: CandidateConfig, family_id: str) -> list[BuiltCandidate]:
    """Grid windows from each capture file; the label column is read only into references."""
    out = []
    for path in paths:
        _, data = _read_rows(path)
        flags = [row["persistent_anomaly"] == "True" for row in data]
        if {row["persistent_anomaly"] for row in data} - {"True", "False"}:
            raise ValueError(f"{Path(path).name}: unexpected persistent_anomaly values")
        episodes = _episodes(flags)
        for start, end in bubbleran_windows(path, rows=config.bubbleran_window_rows, stride=config.bubbleran_stride):
            packet, built = bubbleran_packet(path, start, end, config)
            window = flags[start:end + 1]
            hits = [i for i, flag in enumerate(window) if flag]
            episode = next((e for e in episodes if e[0] <= start + hits[0] <= e[1]), None) if hits else None
            event_rows = episode or (start, end)
            candidate_id = f"BubbleRAN:{built.source.case_id}"
            reference = CandidateReference(
                candidate_id=candidate_id, taxonomy_id="bubbleran-pi-v1",
                source_labels={"persistent_anomaly": sorted({str(flag) for flag in window})},
                causes=[BUBBLERAN_CAUSE] if hits else [], label_status="labelled_fault" if hits else "labelled_normal",
                annotation={"origin": "bubbleran", "window_rows": len(window), "anomalous_rows": len(hits),
                            "first_anomalous_offset": hits[0] if hits else None,
                            "last_anomalous_offset": hits[-1] if hits else None,
                            "anomaly_episode_rows": list(episode) if episode else None})
            out.append(_table_candidate(
                packet, built, reference, location={"file": Path(path).name, "first_row": start, "last_row": end},
                family_id=family_id, event_id="event-" + canonical_sha256([Path(path).name, *event_rows])[:16],
                config=config))
    return out


# --- SpotLight candidates ----------------------------------------------------------------------

SPOTLIGHT_GRID_RULE = ("fixed grid of {rows} consecutive rows at stride {stride} inside one capture of one "
                       "stream (radio or platform); a repeated header starts a new capture; streams are never "
                       "joined; the label file and anomaly column never move a window boundary")
SPOTLIGHT_LABEL_TIME = ("label file holds Unix nanosecond intervals; CSV timestamps are naive, so overlap is "
                        "computed assuming UTC (unverified)")


def spotlight_windows(path: Path, *, rows: int, stride: int) -> list[tuple[int, int, int]]:
    """Label-blind (capture, first row, last row) windows; row numbers index the CSV data rows."""
    _, data = _read_rows(path)
    captures: list[list[int]] = [[]]
    for index, row in enumerate(data):
        if row["timestamp"] == "timestamp":
            captures.append([])
        else:
            captures[-1].append(index)
    return [(number, indices[k], indices[k + rows - 1]) for number, indices in enumerate(captures)
            for k in range(0, len(indices) - rows + 1, stride)]


def spotlight_packet(path: Path, start_time: str, end_time: str,
                     config: CandidateConfig) -> tuple[EvidencePacket, SourceEvidence]:
    return _table_packet(spotlight_evidence(path, start_time=start_time, end_time=end_time), config)


def read_label_intervals(path: Path) -> list[tuple[int, int]]:
    intervals = []
    for line in Path(path).read_text(encoding="utf-8").split():
        start, end = (int(value) for value in line.split(","))
        if end < start:
            raise ValueError("label interval ends before it starts")
        intervals.append((start, end))
    return intervals


def _naive_ns(stamp: str) -> int:
    """Nanoseconds since the epoch for a naive source timestamp read as UTC (an assumption)."""
    whole, _, fraction = stamp.partition(".")
    seconds = datetime.fromisoformat(whole).replace(tzinfo=UTC).timestamp()
    return int(seconds) * 10**9 + int((fraction + "000000000")[:9])


def spotlight_candidates(paths: list[Path], *, label_path: Path | None, config: CandidateConfig,
                         family_id: str) -> list[BuiltCandidate]:
    """Grid windows per stream and capture; anomaly column and label intervals go to references only.

    The anomaly type is not stated and label-time alignment is unverified, so every window's label
    status is ``ambiguous`` and no cause is recorded.
    """
    intervals = read_label_intervals(label_path) if label_path and Path(label_path).is_file() else []
    out = []
    for path in paths:
        _, data = _read_rows(path)
        for capture, first, last in spotlight_windows(path, rows=config.spotlight_window_rows,
                                                      stride=config.spotlight_stride):
            start_time, end_time = data[first]["timestamp"], data[last]["timestamp"]
            packet, built = spotlight_packet(path, start_time, end_time, config)
            begin, finish = _naive_ns(start_time), _naive_ns(end_time)
            overlapping = sum(1 for a, b in intervals if a <= finish and b >= begin)
            labels = {"anomaly_column": sorted({row["anomaly"] for row in data[first:last + 1]})}
            if intervals:
                labels["label_file_overlap"] = ["yes" if overlapping else "no"]
            reference = CandidateReference(
                candidate_id=f"SpotLight:{built.source.case_id}", taxonomy_id="spotlight-v1",
                source_labels=labels, label_status="ambiguous",
                annotation={"origin": "spotlight", "stream": Path(path).stem, "capture": capture,
                            "overlapping_label_intervals": overlapping, "label_time_rule": SPOTLIGHT_LABEL_TIME,
                            "anomaly_type": "not stated by the source"})
            location = {"file": Path(path).name, "capture": capture, "first_row": first, "last_row": last,
                        "start_time": start_time, "end_time": end_time}
            out.append(_table_candidate(
                packet, built, reference, location=location, family_id=family_id,
                event_id="event-" + canonical_sha256([Path(path).name, capture, first, last])[:16], config=config))
    return out
