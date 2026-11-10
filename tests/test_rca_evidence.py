"""Task 4.4: deterministic KPI evidence, traceable IDs, baselines, rejections and source adapters."""

import csv
import json
import math
from datetime import datetime, timedelta

import pytest

from clauseguard.cli import build_evidence as cli
from clauseguard.datasets.release import prepare_release
from clauseguard.datasets.scenarios import ScenarioDataset
from clauseguard.rca import sources
from clauseguard.rca.contracts import SourceIdentity
from clauseguard.rca.evidence import (
    BaselineVerification,
    EvidenceError,
    EvidenceRegistry,
    check_audits,
    relative_change,
)
from clauseguard.rca.summarizer import (
    TimeIntegrityError,
    check_timestamps,
    counter_resets,
    summarize_channel,
    uniform_times,
)

SOURCE = SourceIdentity(dataset="Fixture", revision="r1", case_id="case-1", artifact_sha256="a" * 64,
                        recording_family_status="unresolved")


def test_statistics_use_observed_values_and_record_extremum_positions():
    stats, rejected = summarize_channel("UL_SNR", [3.0, None, 0.0, 7.5, 7.5, -1.0], semantics="reported_value",
                                        start=10)
    by_name = {s.statistic: s for s in stats}
    assert by_name["mean"].value == pytest.approx((3.0 + 0.0 + 7.5 + 7.5 - 1.0) / 5)
    assert (by_name["minimum"].value, by_name["minimum"].extremum_index) == (-1.0, 15)
    assert (by_name["maximum"].value, by_name["maximum"].extremum_index) == (7.5, 13)  # first occurrence
    assert by_name["mean"].extremum_index is None
    assert all((s.valid_count, s.sample_count, s.sample_start, s.sample_end) == (5, 6, 10, 15) for s in stats)
    assert by_name["mean"].missing_fraction == pytest.approx(1 / 6) and not rejected


def test_zero_is_valid_missing_is_not_and_non_finite_values_are_excluded():
    zero = {s.statistic: s for s in summarize_channel("a", [0.0, 0.0], semantics="reported_value")[0]}
    assert zero["mean"].value == 0.0 and zero["mean"].missing_fraction == 0.0
    stats, rejected = summarize_channel("b", [1.0, math.inf, None, math.nan, 3.0], semantics="reported_value")
    assert {s.valid_count for s in stats} == {2} and {s.non_finite_count for s in stats} == {2}
    assert stats[0].missing_fraction == pytest.approx(1 / 5)
    assert [(r.scope, r.reason, r.detail) for r in rejected] == [
        ("values", "non_finite_values_excluded", {"indices": [1, 3]})]
    empty, rejected = summarize_channel("c", [None, None], semantics="reported_value")
    assert not empty and rejected[0].reason == "no_valid_values" and rejected[0].detail["missing"] == 2


@pytest.mark.parametrize("values, semantics, reason", [
    (["TCP", "UDP"], "categorical", "categorical_channel_no_numeric_statistics"),
    ([1, 2], "identifier", "identifier_field_not_a_measurement"),
    ([1.0, "x"], "reported_value", "non_numeric_value"),
    ([], "reported_value", "empty_interval"),
])
def test_unsupported_channels_are_rejected_with_a_reason(values, semantics, reason):
    stats, rejected = summarize_channel("ch", values, semantics=semantics)
    assert not stats and rejected[0].reason == reason and rejected[0].scope == "channel"


def test_cumulative_counters_are_not_differenced_and_resets_are_recorded():
    assert counter_resets([5, 7, None, 2, 9, 1], start=100) == [103, 105]
    stats, rejected = summarize_channel("mac_dl_aggr_prb", [5, 7, 2], semantics="cumulative_counter", start=4)
    assert not stats and rejected[0].reason == "counter_semantics_unverified"
    assert rejected[0].detail["resets"] == [6]


def test_time_rules_reject_duplicates_and_disorder_and_report_gaps():
    t0 = datetime.fromisoformat("2026-01-01 00:00:00")
    stamps = [t0 + timedelta(seconds=s) for s in (0, 1, 2, 5, 6)]
    assert check_timestamps(stamps, nominal_period=1.0) == [{"after_sample": 2, "seconds": 3.0}]
    with pytest.raises(TimeIntegrityError, match="duplicate"):
        check_timestamps([t0, t0])
    with pytest.raises(TimeIntegrityError, match="order"):
        check_timestamps([t0, t0 - timedelta(seconds=1)])
    assert uniform_times("2026-09-02 00:08:44.800", 10.0, 3) == [
        "2026-09-02 00:08:44.800", "2026-09-02 00:08:44.900", "2026-09-02 00:08:45.000"]
    assert uniform_times("2025-01-02 00:00:03.200000", 10.0, 2)[1] == "2025-01-02 00:00:03.300000"


def _registry_with(values, start=0):
    registry = EvidenceRegistry()
    stats, _ = summarize_channel("RSRP", values, semantics="reported_value", start=start)
    return registry, [registry.add_kpi(s, source=SOURCE, location={"row": 1}, times=None) for s in stats]


def test_registry_assigns_unique_ids_and_rejects_conflicting_mappings():
    registry, records = _registry_with([1.0, 2.0])
    assert [r.evidence_id for r in records] == ["K1", "K2", "K3"]
    again = summarize_channel("RSRP", [1.0, 2.0], semantics="reported_value")[0][0]
    assert registry.add_kpi(again, source=SOURCE, location={"row": 1}).evidence_id == "K1"  # same fact, same ID
    changed = again.model_copy(update={"value": 99.0})
    with pytest.raises(EvidenceError, match="conflicting"):
        registry.add_kpi(changed, source=SOURCE, location={"row": 1})
    spec = registry.add_specification(chunk_id="c1", document_id="TS 38.300", version="18", clause="5.1",
                                      chunk_text="abc", evidence_id="S9")
    with pytest.raises(EvidenceError, match="duplicate evidence ID"):
        registry.add_specification(chunk_id="c2", document_id="d", version="1", clause="x", chunk_text="z",
                                   evidence_id=spec.evidence_id)
    packet = registry.packet(source=SOURCE, question="q", kpi_state="included", text_state="missing",
                             retrieval_status="completed")
    check_audits(packet, registry.audits())
    audit = registry.audits()[0]
    assert audit.persistent_id == "Fixture:case-1:RSRP:mean:0-1:kpi-stats-v1"
    assert audit.calculation["imputation"] == "none" and audit.source["location"] == {"row": 1}
    tampered = packet.model_copy(update={"kpi_evidence": [packet.kpi_evidence[0].model_copy(update={"value": 5.0}),
                                                          *packet.kpi_evidence[1:]]})
    with pytest.raises(EvidenceError, match="no matching audit"):
        check_audits(tampered, registry.audits())


def test_scalar_evidence_has_no_invented_range_or_time():
    registry = EvidenceRegistry()
    scalar = registry.add_scalar(channel="cell_load", value=0.0, source=SOURCE, location={"field": "cell_load"})
    assert (scalar.statistic, scalar.value, scalar.sample_start, scalar.time_start) == ("scalar", 0.0, None, None)
    for bad in (math.nan, True, "1"):
        with pytest.raises(EvidenceError, match="finite"):
            registry.add_scalar(channel="x", value=bad, source=SOURCE, location={})


def test_eligible_text_becomes_a_text_record(dataset, monkeypatch):
    from clauseguard.rca.detector import TextEligibility

    record = dataset.load_scenario(next(iter(dataset.splits["membership"])))
    record = record.model_copy(update={"nl_context": "Users reported stalls."})
    monkeypatch.setattr(sources, "rca_text_eligibility",
                        lambda r: TextEligibility(state="eligible", origin="independent_observation"))
    built = sources.telecomts_evidence(record, family_id=None)
    [text] = built.registry.records("text")
    assert (built.text_state, text.evidence_id, text.text) == ("included", "T1", "Users reported stalls.")


def test_specification_excerpts_keep_their_link_to_the_chunk():
    registry = EvidenceRegistry()
    excerpt = registry.add_specification(chunk_id="c1", document_id="TS 38.300", version="18.0.0",
                                         clause="5.1", chunk_text="0123456789", span=(2, 5))
    assert excerpt.text == "234"
    source = registry.audits()[0].source
    assert source["shortened"] and source["span"] == [2, 5] and source["chunk_sha256"] != excerpt.content_sha256
    with pytest.raises(EvidenceError, match="outside"):
        registry.add_specification(chunk_id="c1", document_id="d", version="1", clause="x",
                                   chunk_text="abc", span=(1, 9))


def test_baselines_need_verification_and_compatibility_and_zero_baseline_is_undefined():
    assert relative_change(0.0, 3.0) is None and relative_change(-2.0, -1.0) == pytest.approx(0.5)
    registry = EvidenceRegistry()
    first = summarize_channel("RSRP", [1.0, 1.0], semantics="reported_value", start=0)[0][0]
    second = summarize_channel("RSRP", [3.0, 5.0], semantics="reported_value", start=2)[0][0]
    base = registry.add_kpi(first, source=SOURCE, location={})
    incident = registry.add_kpi(second, source=SOURCE, location={})
    earlier = BaselineVerification(kind="earlier_interval", interval_rule="window_first_half", status="verified")
    with pytest.raises(EvidenceError, match="unverified"):
        registry.attach_baseline(incident.evidence_id, base.evidence_id,
                                 earlier.model_copy(update={"status": "unverified"}))
    with pytest.raises(EvidenceError, match="fixed rule"):
        registry.attach_baseline(incident.evidence_id, base.evidence_id,
                                 earlier.model_copy(update={"interval_rule": "annotated anomaly start"}))
    with pytest.raises(EvidenceError, match="end before"):
        registry.attach_baseline(base.evidence_id, incident.evidence_id, earlier)
    with pytest.raises(EvidenceError, match="reviewer"):
        registry.attach_baseline(incident.evidence_id, base.evidence_id,
                                 BaselineVerification(kind="verified_normal", interval_rule="clean", status="verified"))
    other = registry.add_kpi(summarize_channel("UL_SNR", [1.0], semantics="reported_value")[0][0],
                             source=SOURCE, location={})
    with pytest.raises(EvidenceError, match="channel"):
        registry.attach_baseline(incident.evidence_id, other.evidence_id, earlier)
    linked = registry.attach_baseline(incident.evidence_id, base.evidence_id, earlier)
    assert linked.baseline_evidence_id == base.evidence_id
    packet = registry.packet(source=SOURCE, question="q", kpi_state="included", text_state="missing",
                             retrieval_status="not_run")
    check_audits(packet, registry.audits())
    assert registry.audits()[1].calculation["baseline"]["kind"] == "earlier_interval"


@pytest.fixture
def dataset(release_sources, recording_evidence):
    source = release_sources
    prepare_release(source["scenarios"], source["output"], source["output"], trials=128)
    return ScenarioDataset(source["output"])


def test_telecomts_window_evidence_reads_observations_only(dataset):
    sid = next(iter(dataset.splits["membership"]))
    record = dataset.load_scenario(sid)
    built = sources.telecomts_evidence(record, family_id="fam", family_status="verified")
    kpi = built.registry.records("kpi")
    numeric = [n for n, c in record.kpi_window.channels.items() if c.kind == "numeric"]
    assert len(kpi) == 3 * len(numeric)
    assert {r.reason for r in built.rejections} == {"categorical_channel_no_numeric_statistics"}
    assert all(r.unit is None and r.sample_start == 0 and r.sample_end == 127 for r in kpi)
    assert kpi[0].time_start == record.kpi_window.start_time and kpi[0].time_end == record.kpi_window.end_time
    first = next(r for r in kpi if r.channel == numeric[0] and r.statistic == "mean")
    values = record.kpi_window.channels[numeric[0]].values
    assert first.value == pytest.approx(sum(values) / len(values))
    assert built.text_state == "missing" and built.text_reason == "missing: no_approved_text"
    packet = built.registry.packet(source=built.source, question="What happened?", kpi_state=built.kpi_state,
                                   text_state=built.text_state, retrieval_status="not_run")
    dumped = packet.model_dump_json()
    for leaked in ("Revealing source text", "There is a jammer", '"anomalies"', '"QnA"', "fault_label"):
        assert leaked not in dumped
    assert packet.source.artifact_sha256 == record.provenance.source_sha256
    broken = record.model_copy(update={"kpi_window": record.kpi_window.model_copy(
        update={"end_time": "2030-01-01 00:00:00.000"})})
    with pytest.raises(TimeIntegrityError, match="cadence"):
        sources.telecomts_evidence(broken, family_id=None)


def _write_csv(path, header, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_bubbleran_rejects_counters_and_identifiers_and_checks_time(tmp_path):
    header = ["", "timestamp", "ran_ue_id", "e2node_nb_id", "mac_dl_aggr_prb", "mac_pusch_snr", "mac_dl_bler",
              "persistent_anomaly"]
    rows = [[0, 100, 2, 50, 10, 18.5, 0.0, 1], [1, 101, 2, 50, 20, "", 0.1, 1], [2, 104, 2, 50, 5, 19.0, 0.0, 1]]
    path = _write_csv(tmp_path / "anomalous_data.csv", header, rows)
    built = sources.bubbleran_evidence(path, start_row=0, end_row=2)
    reasons = {r.channel: r.reason for r in built.rejections}
    assert reasons["persistent_anomaly"] == "reference_label_excluded"
    assert all(r.channel != "persistent_anomaly" for r in built.registry.records("kpi"))
    assert reasons["mac_dl_aggr_prb"] == "counter_semantics_unverified"
    assert reasons["ran_ue_id"] == reasons["timestamp"] == "identifier_field_not_a_measurement"
    resets = next(r for r in built.rejections if r.channel == "mac_dl_aggr_prb").detail["resets"]
    assert resets == [2]
    snr = {r.statistic: r for r in built.registry.records("kpi") if r.channel == "mac_pusch_snr"}
    assert snr["mean"].value == pytest.approx(18.75) and snr["mean"].missing_fraction == pytest.approx(1 / 3)
    assert built.time["gaps"] == [{"after_sample": 1, "seconds": 3.0}]
    assert all(r.unit is None for r in built.registry.records("kpi"))
    _write_csv(tmp_path / "dup.csv", header, [rows[0], [1, 100, 2, 50, 11, 1.0, 0.0]])
    with pytest.raises(TimeIntegrityError, match="duplicate"):
        sources.bubbleran_evidence(tmp_path / "dup.csv", start_row=0, end_row=1)
    _write_csv(tmp_path / "two_ue.csv", header, [rows[0], [1, 101, 3, 50, 11, 1.0, 0.0]])
    with pytest.raises(ValueError, match="one UE"):
        sources.bubbleran_evidence(tmp_path / "two_ue.csv", start_row=0, end_row=1)


def test_spotlight_reads_one_stream_and_keeps_empty_cells_missing(tmp_path):
    header = ["timestamp", "BHTX_In_max", "rx_power_mean", "anomaly"]
    rows = [["2024-02-15 05:28:33.419523584", "", "-80.5", "LABEL"], ["2024-02-15 05:28:33.569523584", "", "-81.5", "LABEL"],
            header, ["2024-02-15 05:28:34.000000000", "", "-79.0", "LABEL"]]
    path = _write_csv(tmp_path / "radio.csv", header, rows)
    built = sources.spotlight_evidence(path, start_time="2024-02-15 05:28:33.400",
                                       end_time="2024-02-15 05:28:33.600")
    assert built.source.case_id == "radio.csv:rows-0-1"
    reasons = {r.channel: r.reason for r in built.rejections}
    assert reasons["BHTX_In_max"] == "no_valid_values" and reasons["anomaly"] == "reference_label_excluded"
    mean = next(r for r in built.registry.records("kpi") if r.statistic == "mean")
    assert mean.value == pytest.approx(-81.0) and mean.time_end == "2024-02-15 05:28:33.569523584"
    assert "LABEL" not in built.registry.packet(source=built.source, question="q", kpi_state="included",
                                                text_state="missing", retrieval_status="not_run").model_dump_json()
    with pytest.raises(TimeIntegrityError, match="capture boundary"):
        sources.spotlight_evidence(path, start_time="2024-02-15 05:28:33", end_time="2024-02-15 05:28:35")
    with pytest.raises(ValueError, match="no rows"):
        sources.spotlight_evidence(path, start_time="2030-01-01 00:00:00", end_time="2030-01-02 00:00:00")


def test_unavailable_sources_are_recorded():
    assert sources.unavailable_source("TeleLogs")["status"] == "unavailable"
    with pytest.raises(ValueError):
        sources.unavailable_source("TelecomTS")


def test_cli_builds_packets_and_matches_raw_metrics(dataset, release_sources, tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    ids = list(dataset.splits["membership"])[:2]
    rows = [SimpleNamespace(id=f"rca_{i}", scenario_id=sid, question="Which cause fits these KPIs?")
            for i, sid in enumerate(ids)]
    monkeypatch.setattr("clauseguard.eval.golden.load_rca", lambda path: rows)
    output = tmp_path / "evidence"
    release = ["--release-dir", str(dataset.release_dir)]
    assert cli.main(["telecomts", *release, "--output-dir", str(output)]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["cases"] == 2 and manifest["detector_predictions"] == "absent"
    packets = [json.loads(line) for line in (output / "packets.jsonl").read_text().splitlines()]
    assert {p["retrieval_status"] for p in packets} == {"not_run"} and packets[0]["question"] == rows[0].question
    raw_dir = release_sources["local"]
    assert cli.main(["verify-raw", *release, "--output-dir", str(output), "--raw-dir", str(raw_dir)]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["statistics_checked"] == manifest["kpi_records"] and checked["max_abs_difference"] <= 1e-9
    assert cli.main(["telecomts", *release, "--output-dir", str(output)]) == 1
    assert "output exists" in capsys.readouterr().err


def test_published_example_validates_against_the_contract():
    from pathlib import Path

    from clauseguard.rca.contracts import EvidencePacket
    from clauseguard.rca.evidence import EvidenceAudit, _digest, check_packet_ids

    example = json.loads(Path("data/golden/rca-evidence.example.json").read_text())
    packet = EvidencePacket.model_validate(example["packet"])
    check_packet_ids(packet)
    records = {item.evidence_id: item for item in packet.kpi_evidence}
    for raw in example["audits"]:
        audit = EvidenceAudit.model_validate({k: v for k, v in raw.items() if k != "case_id"})
        assert audit.content_sha256 == _digest(records[audit.evidence_id].model_dump(mode="json"))
    assert packet.retrieval_status == "not_run" and packet.text_state == "missing"
    assert all(item.unit is None and item.baseline_evidence_id is None for item in packet.kpi_evidence)
    assert {r["reason"] for r in example["rejections"]} == {"categorical_channel_no_numeric_statistics"}
