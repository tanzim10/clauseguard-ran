import hashlib
import json

from clauseguard.cli.ingest_telecomts import main


def offline_args(tmp_path):
    source = tmp_path / "local"
    source.mkdir()
    row = {"KPIs": {"RX_Bytes": [1, 2.5]}, "sampling_rate": 10,
           "start_time": "2025-01-01 00:00:00", "end_time": "2025-01-01 00:00:00.100",
           "anomalies": {"exists": False}, "description": "Unreviewed symptoms."}
    payload = (json.dumps(row) + "\n").encode()
    (source / "file.jsonl").write_bytes(payload)
    checksums = tmp_path / "checksums.json"
    checksums.write_text(json.dumps({"file.jsonl": hashlib.sha256(payload).hexdigest()}))
    return ["--revision", "a" * 40, "--input-dir", str(source), "--checksums", str(checksums),
            "--acquisition-dir", str(tmp_path / "acquired"), "--output-dir", str(tmp_path / "output")]


def test_offline_command_outputs(tmp_path, capsys):
    assert main(offline_args(tmp_path)) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["accepted"] == 1
    record = json.loads((tmp_path / "output/scenarios.jsonl").read_text())
    assert record["nl_context"] == ""
    assert json.loads((tmp_path / "acquired/acquisition.json").read_text())["status"] == "complete"


def test_invalid_revision_exits_nonzero_and_reports(tmp_path, capsys):
    args = offline_args(tmp_path)
    args[1] = "main"
    assert main(args) == 1
    assert "revision" in capsys.readouterr().err
    assert "revision" in json.loads((tmp_path / "output/ingest_error.json").read_text())["error"]


def test_rejections_exit_nonzero(tmp_path):
    args = offline_args(tmp_path)
    path = tmp_path / "local/file.jsonl"
    path.write_text("not JSON\n")
    (tmp_path / "checksums.json").write_text(json.dumps(
        {"file.jsonl": hashlib.sha256(path.read_bytes()).hexdigest()},
    ))
    assert main(args) == 2
    assert json.loads((tmp_path / "output/rejected.jsonl").read_text())["source_row"] == 1
