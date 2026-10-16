import hashlib
import json

import httpx
import pytest

from clauseguard.datasets.acquisition import (
    acquire_telecomts,
    check_revision,
    git_blob_sha1,
    source_path,
    verify_source,
)


@pytest.mark.parametrize("revision", ["main", "a" * 39, "A" * 40, "../main"])
def test_requires_immutable_revision(revision):
    with pytest.raises(ValueError, match="revision"):
        check_revision(revision)


@pytest.mark.parametrize("name", ["../file.jsonl", "/file.jsonl", "file.txt", "a\\b.jsonl"])
def test_path_must_stay_under_root(tmp_path, name):
    with pytest.raises(ValueError):
        source_path(tmp_path, name)


def test_checks_git_blob_and_sha256(tmp_path):
    path = tmp_path / "file.jsonl"
    path.write_bytes(b'{"a": 1}\n')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify_source(path, {"sha256": sha})["verification"] == "sha256"
    assert verify_source(path, {"blobId": git_blob_sha1(path)})["sha256"] == sha
    with pytest.raises(ValueError, match="checksum"):
        verify_source(path, {"sha256": "0" * 64})
    with pytest.raises(ValueError, match="size"):
        verify_source(path, {"sha256": sha, "size": 999})


def test_existing_symlink_cannot_escape_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "outside").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        source_path(root, "outside/file.jsonl")


def test_offline_acquisition_and_failure_report(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    (local / "file.jsonl").write_bytes(b'{"source": true}\n')
    sha = hashlib.sha256((local / "file.jsonl").read_bytes()).hexdigest()
    output = tmp_path / "out"
    result = acquire_telecomts(revision="a" * 40, output_dir=output, local_dir=local,
                              checksums={"file.jsonl": sha}, declared_license="mit")
    assert result["status"] == "complete"
    assert result["declared_license"] == "mit"
    assert result["files"][0]["sha256"] == sha
    with pytest.raises(ValueError, match="checksum"):
        acquire_telecomts(revision="a" * 40, output_dir=tmp_path / "bad", local_dir=local,
                         checksums={"file.jsonl": "0" * 64})
    assert json.loads((tmp_path / "bad/acquisition.json").read_text())["status"] == "failed"


def test_remote_metadata_and_download_are_mocked(tmp_path):
    payload = b'{"source": true}\n'
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if "/api/datasets/" in request.url.path:
            return httpx.Response(200, json={"sha": "a" * 40, "cardData": {"license": "mit"},
                "siblings": [{"rfilename": "normal/processed/chunked.jsonl",
                              "size": len(payload), "lfs": {
                                  "sha256": hashlib.sha256(payload).hexdigest()}}]})
        return httpx.Response(200, content=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = acquire_telecomts(revision="a" * 40, output_dir=tmp_path, client=client)
        acquire_telecomts(revision="a" * 40, output_dir=tmp_path, client=client)
    assert result["files"][0]["verification"] == "sha256"
    assert len(calls) == 3  # Second acquisition verifies the cached file instead of downloading.


def test_remote_revision_mismatch_retains_manifest(tmp_path):
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"sha": "b" * 40}),
    )) as client, pytest.raises(ValueError, match="different revision"):
        acquire_telecomts(revision="a" * 40, output_dir=tmp_path, client=client)
    assert "different revision" in json.loads((tmp_path / "acquisition.json").read_text())["error"]
