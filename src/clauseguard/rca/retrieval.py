"""Frozen retrieval and per-case specification snapshots for RCA (Task 4.5).

Queries come from the evidence packet only, which by contract carries no reference label,
expected conclusion, golden citation or review decision. Retrieval runs once through the shared
``SearchService``; the saved snapshot is replayed so the base LLM and the adapted SLM receive the
same passages in the same order.
"""

import hashlib
import json
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from clauseguard.api.search_service import SearchConfigurationError, SearchDependencyError
from clauseguard.corpus.chunker import chunk_document
from clauseguard.corpus.manifest import load_manifest
from clauseguard.rca.contracts import EvidencePacket, SpecificationEvidence
from clauseguard.rca.evidence import EvidenceRegistry, _digest

QUERY_TEMPLATE_VERSION = "rca-query-v1"
MANIFEST_VERSION = "rca-retrieval-manifest-v1"
SNAPSHOT_VERSION = "rca-retrieval-snapshot-v1"
CONFIGURATIONS = ("base_llm_full", "adapted_slm_full", "text_rag", "kpi_summary_rag")
QUERY_RULES = {
    "question": "the packet question, verbatim",
    "kpi": "one line per channel with a mean record, in channel order: name, mean, minimum and maximum "
           "rounded to 4 significant digits",
    "text": "approved text records, verbatim, in evidence-ID order",
    "detector": "the detector prediction label when its status is ok; otherwise omitted and recorded",
    "excluded": "reference labels, expected conclusions, golden citations and review decisions have no "
                "field in the packet and cannot enter the query",
}
PROCESSING = {
    "ordering": "Qdrant rank order (descending score); ranks start at 1",
    "duplicates": "a repeated chunk ID keeps its first rank; later copies are recorded as filtered",
    "excerpt": "the whole indexed chunk text is supplied; no shortening",
    "scores": "ranking values only, never diagnosis probabilities",
}
SHARED_RETRIEVAL_NOTE = ("all four configurations reuse the same saved passages; because the query used "
                         "the full packet, the passages can retain information from modalities that a "
                         "configuration omits. Modality-specific queries are a separate experiment.")
State = Literal["completed", "empty", "not_run", "failed"]


class SnapshotError(ValueError):
    """A snapshot or manifest is missing, altered or inconsistent."""


def _seal(value: dict, key: str) -> dict:
    body = {k: v for k, v in value.items() if k != key}
    return {**body, key: _digest(body)}


def _significant(value: float) -> str:
    return f"{value:.4g}"


def build_query(packet: EvidencePacket) -> tuple[str, dict]:
    """Deterministic query text plus the inputs that contributed to it."""
    stats: dict[str, dict[str, Any]] = {}
    for item in packet.kpi_evidence:
        if item.statistic in ("mean", "minimum", "maximum"):
            stats.setdefault(item.channel, {})[item.statistic] = item
    lines, kpi_ids = [packet.question.strip()], []
    kpi_lines = []
    for channel in sorted(stats):
        group = stats[channel]
        if "mean" not in group:
            continue
        parts = [f"mean {_significant(group['mean'].value)}"]
        parts += [f"{name} {_significant(group[name].value)}" for name in ("minimum", "maximum") if name in group]
        kpi_lines.append(f"{channel}: " + ", ".join(parts))
        kpi_ids += [group[name].evidence_id for name in ("mean", "minimum", "maximum") if name in group]
    if kpi_lines:
        lines.append("Observed KPI statistics: " + "; ".join(kpi_lines))
    text_ids = [item.evidence_id for item in packet.text_evidence]
    lines += [item.text.strip() for item in packet.text_evidence]
    prediction = packet.detector_prediction
    used = prediction is not None and prediction.status == "ok" and bool(prediction.predicted_label)
    if used:
        lines.append(f"Detector prediction ({prediction.task}): {prediction.predicted_label}")
    query = "\n".join(line for line in lines if line)
    return query, {
        "template_version": QUERY_TEMPLATE_VERSION, "kpi_evidence_ids": kpi_ids, "text_evidence_ids": text_ids,
        "kpi_state": packet.kpi_state, "text_state": packet.text_state,
        "detector_prediction": {"used": used,
                                "status": prediction.status if prediction else "not_supplied",
                                "model_run_id": prediction.model_run_id if prediction else None},
    }


def chunk_inventory(manifest_path: Path, parsed_dir: Path) -> list[dict]:
    """The chunks ``index_corpus`` writes, rebuilt deterministically from parsed text."""
    chunks = []
    for row in load_manifest(manifest_path):
        if not row.get("acquired_at"):
            continue
        text = (Path(parsed_dir) / f"{Path(row['filename']).stem}.txt").read_text(encoding="utf-8")
        chunks.extend(chunk_document(text, spec_id=row["spec_id"], source_file=row["filename"]))
    return chunks


def freeze_manifest(*, settings, manifest_path: Path, parsed_dir: Path, top_k: int, output: Path,
                    frozen_on: str, index_status: Mapping[str, Any] | None = None) -> dict:
    """Seal corpus, chunk, index, embedding, query and search settings before any scored run."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("retrieval manifest exists; a frozen manifest is never overwritten")
    if not 1 <= top_k <= 50:
        raise SnapshotError("top_k must be between 1 and 50")
    rows = [row for row in load_manifest(manifest_path) if row.get("acquired_at")]
    documents = {}
    for row in rows:
        if row["spec_id"] in documents and documents[row["spec_id"]]["version"] != row["version"]:
            raise SnapshotError(f"conflicting versions for {row['spec_id']}")
        documents[row["spec_id"]] = {"version": row["version"], "sha256": row["sha256"],
                                     "filename": row["filename"]}
    chunks = chunk_inventory(manifest_path, parsed_dir)
    inventory = {chunk["id"]: hashlib.sha256(chunk["text"].encode()).hexdigest() for chunk in chunks}
    if len(inventory) != len(chunks):
        raise SnapshotError("chunk inventory contains duplicate chunk IDs")
    manifest = _seal({
        "version": MANIFEST_VERSION, "frozen_on": frozen_on,
        "corpus": {"manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
                   "documents": documents},
        "chunks": {"chunker": "clauseguard.corpus.chunker.chunk_document",
                   "chunk_size_tokens": settings.chunk_size_tokens, "count": len(inventory),
                   "inventory_sha256": _digest(inventory), "chunk_sha256": inventory},
        "index": {"collection": settings.qdrant_collection, "distance": settings.qdrant_distance,
                  "content": "the chunk inventory above, embedded as passages",
                  "point_id": "uuid5(NAMESPACE_URL, chunk_id)",
                  "rebuild": "uv run python -m clauseguard.cli.index_corpus",
                  "status": dict(index_status or {"checked": False})},
        "embeddings": {"model": settings.nvidia_embedding_model, "dimensions": settings.embedding_dimensions,
                       "query_input_type": "query", "revision": None,
                       "revision_note": "the provider does not expose a model revision; recorded as a "
                                        "reproducibility limit"},
        "query_builder": {"version": QUERY_TEMPLATE_VERSION, "rules": QUERY_RULES},
        "search": {"top_k": top_k, "filters": None, "score_threshold": None,
                   "parameters": "Qdrant query_points defaults (approximate HNSW)"},
        "processing": PROCESSING,
        "limits": ["a fresh approximate search can rank differently; the snapshot preserves what was supplied",
                   SHARED_RETRIEVAL_NOTE],
    }, "manifest_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def load_manifest_record(path: Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("version") != MANIFEST_VERSION or _seal(manifest, "manifest_sha256") != manifest:
        raise SnapshotError("retrieval manifest is not a sealed rca-retrieval-manifest-v1 record")
    return manifest


class RetrievedPassage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    score: float = Field(allow_inf_nan=False)
    evidence_id: str
    point_id: str
    chunk_id: str
    document_id: str
    version: str
    clause: str
    source_file: str
    text: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RetrievalSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["rca-retrieval-snapshot-v1"] = SNAPSHOT_VERSION
    manifest_sha256: str
    case: dict[str, Any]
    query: str | None
    query_inputs: dict[str, Any]
    state: State
    stage: str | None = None
    reason: str | None = None
    passages: list[RetrievedPassage] = Field(default_factory=list)
    filtered: list[dict[str, Any]] = Field(default_factory=list)
    timing: dict[str, Any]
    snapshot_id: str = ""


def _resolve(hits, manifest: Mapping) -> tuple[list[RetrievedPassage], list[dict]]:
    """Check every hit against the frozen corpus and chunk inventory; drop repeated chunks."""
    documents, chunks = manifest["corpus"]["documents"], manifest["chunks"]["chunk_sha256"]
    passages, filtered, seen = [], [], set()
    for rank, hit in enumerate(hits, start=1):
        document = documents.get(hit.spec_id)
        if document is None:
            raise SnapshotError(f"unresolved document {hit.spec_id!r} for chunk {hit.chunk_id}")
        if hit.source_file != document["filename"]:
            raise SnapshotError(f"chunk {hit.chunk_id} source file conflicts with the corpus manifest")
        if hit.id != str(uuid.uuid5(uuid.NAMESPACE_URL, hit.chunk_id)):
            raise SnapshotError(f"point ID conflicts with chunk {hit.chunk_id}")
        content = hashlib.sha256(hit.text.encode()).hexdigest()
        if chunks.get(hit.chunk_id) != content:
            raise SnapshotError(f"chunk {hit.chunk_id} text differs from the frozen inventory")
        if hit.chunk_id in seen:
            filtered.append({"rank": rank, "chunk_id": hit.chunk_id, "reason": "duplicate_chunk"})
            continue
        seen.add(hit.chunk_id)
        passages.append(RetrievedPassage(
            rank=rank, score=hit.score, evidence_id=f"S{len(passages) + 1}", point_id=hit.id,
            chunk_id=hit.chunk_id, document_id=hit.spec_id, version=document["version"], clause=hit.section,
            source_file=hit.source_file, text=hit.text, content_sha256=content))
    return passages, filtered


def snapshot_case(packet: EvidencePacket, manifest: Mapping, search_service, *, case_id: str) -> RetrievalSnapshot:
    """Run retrieval once for one packet and record exactly what happened."""
    query, inputs = build_query(packet)
    case = {"case_id": case_id, "scenario_id": packet.source.case_id, "dataset": packet.source.dataset,
            "packet_sha256": _digest(packet.model_dump(mode="json"))}
    common = {"manifest_sha256": manifest["manifest_sha256"], "case": case, "query_inputs": inputs}
    if not query.strip():
        return _sealed(RetrievalSnapshot(query=None, state="not_run", stage="query", reason="empty query",
                                         timing={"mode": "live", "retrieval_seconds": None}, **common))
    started = time.perf_counter()
    try:
        hits = search_service.search(query, manifest["search"]["top_k"])
        elapsed = time.perf_counter() - started
        passages, filtered = _resolve(hits, manifest)
    except SearchConfigurationError as exc:
        return _failed(common, query, "configuration", exc, started)
    except SearchDependencyError as exc:
        return _failed(common, query, "dependency", exc, started)
    except SnapshotError as exc:
        return _failed(common, query, "source_identity", exc, started)
    return _sealed(RetrievalSnapshot(
        query=query, state="completed" if passages else "empty", passages=passages, filtered=filtered,
        timing={"mode": "live", "retrieval_seconds": elapsed}, **common))


def _failed(common, query, stage, exc, started) -> RetrievalSnapshot:
    reason = f"{type(exc).__name__}: {exc}" + (f" ({exc.__cause__!r})" if exc.__cause__ else "")
    return _sealed(RetrievalSnapshot(query=query, state="failed", stage=stage, reason=reason,
                                     timing={"mode": "live", "retrieval_seconds": time.perf_counter() - started},
                                     **common))


def _sealed(snapshot: RetrievalSnapshot) -> RetrievalSnapshot:
    body = snapshot.model_dump(mode="json", exclude={"snapshot_id"})
    return snapshot.model_copy(update={"snapshot_id": _digest(body)})


def save_snapshot(snapshot: RetrievalSnapshot, directory: Path) -> Path:
    """Snapshots are content-addressed; an earlier snapshot is never overwritten."""
    path = Path(directory) / f"{snapshot.snapshot_id}.json"
    content = json.dumps(snapshot.model_dump(mode="json"), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise SnapshotError(f"refusing to overwrite snapshot {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def load_snapshot(path: Path, manifest: Mapping) -> RetrievalSnapshot:
    """Verify the snapshot hash, its manifest and every passage hash before replay."""
    snapshot = RetrievalSnapshot.model_validate_json(Path(path).read_text(encoding="utf-8"))
    if _sealed(snapshot).snapshot_id != snapshot.snapshot_id:
        raise SnapshotError(f"snapshot {Path(path).name} was altered")
    if snapshot.manifest_sha256 != manifest["manifest_sha256"]:
        raise SnapshotError("snapshot belongs to a different retrieval manifest")
    chunks = manifest["chunks"]["chunk_sha256"]
    for passage in snapshot.passages:
        if (hashlib.sha256(passage.text.encode()).hexdigest() != passage.content_sha256
                or chunks.get(passage.chunk_id) != passage.content_sha256):
            raise SnapshotError(f"passage {passage.evidence_id} is missing or inconsistent with the manifest")
    return snapshot


def replay(packet: EvidencePacket, snapshot: RetrievalSnapshot) -> tuple[EvidencePacket, dict]:
    """The packet with the snapshot's passages, without any live retrieval call."""
    started = time.perf_counter()
    if snapshot.case["scenario_id"] != packet.source.case_id:
        raise SnapshotError("snapshot belongs to a different case")
    if snapshot.case["packet_sha256"] != _digest(packet.model_dump(mode="json")):
        raise SnapshotError("evidence packet differs from the one the snapshot was built from")
    registry = EvidenceRegistry()
    specs: list[SpecificationEvidence] = [
        registry.add_specification(chunk_id=p.chunk_id, document_id=p.document_id, version=p.version,
                                   clause=p.clause, chunk_text=p.text, evidence_id=p.evidence_id)
        for p in snapshot.passages]
    status = {"completed": "completed", "empty": "completed", "not_run": "not_run", "failed": "failed"}
    replayed = packet.model_copy(update={"specification_evidence": specs,
                                         "retrieval_status": status[snapshot.state]})
    return replayed, {"mode": "replay", "replay_seconds": time.perf_counter() - started,
                      "original_retrieval_seconds": snapshot.timing.get("retrieval_seconds")}


def specification_for_configurations(packet: EvidencePacket, snapshot: RetrievalSnapshot,
                                     configurations: Sequence[str] = CONFIGURATIONS) -> dict[str, list]:
    """Every configuration receives the same passages in the same order."""
    unknown = set(configurations) - set(CONFIGURATIONS)
    if unknown:
        raise SnapshotError(f"unknown configurations: {sorted(unknown)}")
    replayed, _ = replay(packet, snapshot)
    return {name: list(replayed.specification_evidence) for name in configurations}
