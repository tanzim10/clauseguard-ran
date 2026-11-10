"""Task 4.5: query construction, frozen retrieval settings, snapshots, replay and failures."""

import json
import uuid
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from clauseguard.api.search_service import SearchService
from clauseguard.cli import retrieval_snapshots as cli
from clauseguard.config import Settings
from clauseguard.rca import retrieval
from clauseguard.rca.contracts import DetectorPrediction, EvidencePacket, SourceIdentity
from clauseguard.rca.evidence import EvidenceRegistry
from clauseguard.rca.summarizer import summarize_channel

SOURCE = SourceIdentity(dataset="TelecomTS", revision="r", case_id="telecomts-1", artifact_sha256="a" * 64,
                        recording_family_status="unresolved")
PREDICTION = DetectorPrediction(task="diagnosis", status="ok", predicted_label_id=3,
                                predicted_label="Co-Channel Interference (Severe)", label_mapping_version="1",
                                model_run_id="pipeline:run", source_modalities=["kpi"])
TEXTS = {"TS 38.300": "1 Scope\nThe scheduler assigns resources.\n2 Interference\nUplink SNR falls under interference.",
         "TS 28.552": "1 Scope\nPerformance measurements are defined here."}


def _packet(prediction=PREDICTION, question="Why did uplink SNR fall?"):
    registry = EvidenceRegistry()
    for channel, values in (("UL_SNR", [21.4, 6.4]), ("RSRP", [-87.8, -98.1])):
        for stat in summarize_channel(channel, values, semantics="reported_value")[0]:
            registry.add_kpi(stat, source=SOURCE, location={"row": 1})
    return registry.packet(source=SOURCE, question=question, kpi_state="included", text_state="missing",
                           retrieval_status="not_run", detector_prediction=prediction)


@pytest.fixture
def corpus(tmp_path):
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    rows = ["filename,spec_id,version,sha256,acquired_at"]
    for spec, text in TEXTS.items():
        name = spec.replace(" ", "_")
        (parsed / f"{name}.txt").write_text(text, encoding="utf-8")
        rows.append(f"{name}.pdf,{spec},18.0.0,{'b' * 64},2026-01-01")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    settings = Settings(_env_file=None, qdrant_collection="test", embedding_dimensions=4, nvidia_api_key="x")
    frozen = retrieval.freeze_manifest(settings=settings, manifest_path=manifest, parsed_dir=parsed, top_k=3,
                                       output=tmp_path / "retrieval-manifest.json", frozen_on="2026-11-09")
    chunks = retrieval.chunk_inventory(manifest, parsed)
    return SimpleNamespace(manifest=frozen, chunks=chunks, settings=settings, path=tmp_path,
                           manifest_csv=manifest, parsed=parsed)


def _point(chunk, score):
    return {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["id"])), "score": score,
            "payload": {"text": chunk["text"], **chunk["metadata"]}}


def _service(corpus, points, calls=None):
    def search(collection, vector, top_k, *, settings):
        if calls is not None:
            calls.append((collection, top_k))
        return points

    return SearchService(settings=corpus.settings, embedder=lambda texts, **kw: [[0.0] * 4 for _ in texts],
                         qdrant_search=search)


def test_query_uses_packet_inputs_only_and_records_contributions():
    query, inputs = retrieval.build_query(_packet())
    assert query.splitlines() == [
        "Why did uplink SNR fall?",
        ("Observed KPI statistics: RSRP: mean -92.95, minimum -98.1, maximum -87.8; "
         "UL_SNR: mean 13.9, minimum 6.4, maximum 21.4"),
        "Detector prediction (diagnosis): Co-Channel Interference (Severe)"]
    assert inputs["kpi_evidence_ids"] == ["K4", "K5", "K6", "K1", "K2", "K3"]
    assert inputs["detector_prediction"] == {"used": True, "status": "ok", "model_run_id": "pipeline:run"}
    assert retrieval.build_query(_packet()) == (query, inputs)
    absent = DetectorPrediction(task="diagnosis", status="absent", absence_reason="no detector")
    query, inputs = retrieval.build_query(_packet(prediction=absent))
    assert "Detector prediction" not in query and inputs["detector_prediction"]["used"] is False
    with pytest.raises(ValidationError):
        EvidencePacket.model_validate({**_packet().model_dump(), "expected_conclusion": "interference"})


def test_manifest_is_sealed_and_never_overwritten(corpus):
    manifest = retrieval.load_manifest_record(corpus.path / "retrieval-manifest.json")
    assert manifest["chunks"]["count"] == len(corpus.chunks) and manifest["search"]["top_k"] == 3
    assert manifest["embeddings"]["revision"] is None and manifest["index"]["collection"] == "test"
    assert set(manifest["corpus"]["documents"]) == set(TEXTS)
    with pytest.raises(FileExistsError):
        retrieval.freeze_manifest(settings=corpus.settings, manifest_path=corpus.manifest_csv,
                                  parsed_dir=corpus.parsed, top_k=3, output=corpus.path / "retrieval-manifest.json",
                                  frozen_on="x")
    tampered = corpus.path / "tampered.json"
    tampered.write_text(json.dumps({**manifest, "search": {**manifest["search"], "top_k": 9}}))
    with pytest.raises(retrieval.SnapshotError, match="sealed"):
        retrieval.load_manifest_record(tampered)
    with pytest.raises(retrieval.SnapshotError, match="top_k"):
        retrieval.freeze_manifest(settings=corpus.settings, manifest_path=corpus.manifest_csv,
                                  parsed_dir=corpus.parsed, top_k=0, output=corpus.path / "m2.json", frozen_on="x")
    conflict = corpus.path / "conflict.csv"
    conflict.write_text(corpus.manifest_csv.read_text() + "x.pdf,TS 38.300,17.0.0,c,2026-01-01\n")
    (corpus.parsed / "x.txt").write_text("1 A\nB")
    with pytest.raises(retrieval.SnapshotError, match="conflicting versions"):
        retrieval.freeze_manifest(settings=corpus.settings, manifest_path=conflict, parsed_dir=corpus.parsed,
                                  top_k=3, output=corpus.path / "m3.json", frozen_on="x")


def test_completed_snapshot_resolves_sources_and_filters_duplicates(corpus):
    calls = []
    first, second = corpus.chunks[0], corpus.chunks[2]
    service = _service(corpus, [_point(second, 0.9), _point(first, 0.8), _point(second, 0.7)], calls)
    snapshot = retrieval.snapshot_case(_packet(), corpus.manifest, service, case_id="rca_001")
    assert snapshot.state == "completed" and calls == [("test", 3)]
    assert [(p.rank, p.evidence_id, p.chunk_id) for p in snapshot.passages] == [
        (1, "S1", second["id"]), (2, "S2", first["id"])]
    assert snapshot.filtered == [{"rank": 3, "chunk_id": second["id"], "reason": "duplicate_chunk"}]
    assert snapshot.passages[0].version == "18.0.0" and snapshot.passages[0].clause == second["metadata"]["section"]
    assert snapshot.timing["mode"] == "live" and snapshot.timing["retrieval_seconds"] >= 0
    assert snapshot.case["packet_sha256"] and len(snapshot.snapshot_id) == 64


def test_empty_not_run_and_failed_searches_stay_distinct(corpus, monkeypatch):
    empty = retrieval.snapshot_case(_packet(), corpus.manifest, _service(corpus, []), case_id="c")
    assert (empty.state, empty.passages, empty.stage) == ("empty", [], None)
    no_key = SearchService(settings=Settings(_env_file=None, embedding_dimensions=4, nvidia_api_key=""))
    failed = retrieval.snapshot_case(_packet(), corpus.manifest, no_key, case_id="c")
    assert (failed.state, failed.stage) == ("failed", "configuration") and "NVIDIA_API_KEY" in failed.reason

    def broken(*args, **kwargs):
        raise ConnectionError("qdrant down")

    down = SearchService(settings=corpus.settings, embedder=lambda texts, **kw: [[0.0] * 4], qdrant_search=broken)
    failed = retrieval.snapshot_case(_packet(), corpus.manifest, down, case_id="c")
    assert (failed.state, failed.stage) == ("failed", "dependency") and "qdrant down" in failed.reason
    chunk = corpus.chunks[0]
    for point, message in (
            ({**_point(chunk, 0.5), "payload": {**_point(chunk, 0.5)["payload"], "spec_id": "TS 99.999"}}, "unresolved"),
            ({**_point(chunk, 0.5), "payload": {**_point(chunk, 0.5)["payload"], "text": "edited"}}, "differs"),
            ({**_point(chunk, 0.5), "id": str(uuid.uuid4())}, "point ID")):
        failed = retrieval.snapshot_case(_packet(), corpus.manifest, _service(corpus, [point]), case_id="c")
        assert (failed.state, failed.stage) == ("failed", "source_identity") and message in failed.reason
    calls = []
    monkeypatch.setattr(retrieval, "build_query", lambda packet: ("  ", {"template_version": "x"}))
    skipped = retrieval.snapshot_case(_packet(), corpus.manifest, _service(corpus, [], calls), case_id="c")
    assert (skipped.state, skipped.stage, skipped.query, calls) == ("not_run", "query", None, [])


def test_snapshots_replay_identically_without_live_calls(corpus, tmp_path):
    packet = _packet()
    snapshot = retrieval.snapshot_case(packet, corpus.manifest, _service(
        corpus, [_point(corpus.chunks[1], 0.9), _point(corpus.chunks[0], 0.4)]), case_id="rca_001")
    path = retrieval.save_snapshot(snapshot, tmp_path / "snaps")
    assert retrieval.save_snapshot(snapshot, tmp_path / "snaps") == path  # identical content is idempotent
    loaded = retrieval.load_snapshot(path, corpus.manifest)
    filled, timing = retrieval.replay(packet, loaded)
    assert [s.chunk_id for s in filled.specification_evidence] == [corpus.chunks[1]["id"], corpus.chunks[0]["id"]]
    assert [s.evidence_id for s in filled.specification_evidence] == ["S1", "S2"]
    assert filled.retrieval_status == "completed" and filled.kpi_evidence == packet.kpi_evidence
    assert timing["mode"] == "replay" and timing["original_retrieval_seconds"] == snapshot.timing["retrieval_seconds"]
    shared = retrieval.specification_for_configurations(packet, loaded)
    assert list(shared) == list(retrieval.CONFIGURATIONS)
    assert all(value == filled.specification_evidence for value in shared.values())
    with pytest.raises(retrieval.SnapshotError, match="unknown"):
        retrieval.specification_for_configurations(packet, loaded, ["other"])
    with pytest.raises(retrieval.SnapshotError, match="differs"):
        retrieval.replay(_packet(question="Another question?"), loaded)
    data = json.loads(path.read_text())
    data["passages"][0]["score"] = 0.1
    path.write_text(json.dumps(data))
    with pytest.raises(retrieval.SnapshotError, match="altered"):
        retrieval.load_snapshot(path, corpus.manifest)
    changed = snapshot.model_copy(update={"query": "changed"})
    with pytest.raises(retrieval.SnapshotError, match="overwrite"):
        retrieval.save_snapshot(changed.model_copy(update={"snapshot_id": snapshot.snapshot_id}), tmp_path / "snaps")
    other = {**corpus.manifest, "manifest_sha256": "0" * 64}
    resealed = retrieval._sealed(snapshot)
    with pytest.raises(retrieval.SnapshotError, match="different retrieval manifest"):
        retrieval.load_snapshot(retrieval.save_snapshot(resealed, tmp_path / "s2"), other)


def test_cli_freeze_run_and_replay(corpus, tmp_path, monkeypatch, capsys):
    packets = tmp_path / "packets.jsonl"
    packets.write_text(_packet().model_dump_json() + "\n")
    monkeypatch.setattr("clauseguard.config.get_settings", lambda: corpus.settings)
    points = [_point(corpus.chunks[0], 0.9)]
    monkeypatch.setattr("clauseguard.api.search_service.SearchService",
                        lambda settings: _service(corpus, points))
    manifest = tmp_path / "frozen.json"
    assert cli.main(["freeze", "--corpus-manifest", str(corpus.manifest_csv), "--parsed-dir", str(corpus.parsed),
                     "--top-k", "3", "--frozen-on", "2026-11-09", "--output", str(manifest)]) == 0
    capsys.readouterr()
    run = tmp_path / "run"
    common = ["--manifest", str(manifest), "--packets", str(packets), "--run-dir", str(run)]
    assert cli.main(["run", *common]) == 0
    assert json.loads(capsys.readouterr().out)["states"]["completed"] == 1
    assert cli.main(["replay", *common, "--output", str(tmp_path / "replayed.jsonl")]) == 0
    assert json.loads(capsys.readouterr().out)["identical_across_configurations"] is True
    replayed = EvidencePacket.model_validate_json((tmp_path / "replayed.jsonl").read_text())
    assert replayed.specification_evidence[0].chunk_id == corpus.chunks[0]["id"]
    assert cli.main(["run", *common]) == 1 and "exists" in capsys.readouterr().err
    monkeypatch.setattr("clauseguard.config.get_settings",
                        lambda: corpus.settings.model_copy(update={"qdrant_collection": "other"}))
    assert cli.main(["run", "--manifest", str(manifest), "--packets", str(packets),
                     "--run-dir", str(tmp_path / "run2")]) == 1
    assert "settings differ" in capsys.readouterr().err


def test_published_example_snapshots_keep_their_sealed_ids():
    from pathlib import Path

    example = json.loads(Path("data/golden/rca-retrieval.example.json").read_text())
    for key in ("real_snapshot", "fixture_snapshot"):
        snapshot = retrieval.RetrievalSnapshot.model_validate(example[key])
        assert retrieval._sealed(snapshot).snapshot_id == snapshot.snapshot_id
    assert example["real_snapshot"]["state"] == "failed" and example["real_snapshot"]["stage"] == "configuration"
    fixture = example["fixture_snapshot"]
    assert [s["chunk_id"] for s in example["fixture_replay"]["specification_evidence"]] == [
        p["chunk_id"] for p in fixture["passages"]]
    assert "chunk_sha256" not in example["frozen_manifest_summary"]["chunks"]
