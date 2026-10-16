"""Integrity checks for pinned TelecomTS files; no dataset-library cache is required."""

import hashlib
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import httpx

from clauseguard.datasets.telecomts import (
    ADAPTER_VERSION,
    REPOSITORY,
    SCHEMA_VERSION,
    file_sha256,
)


def check_revision(revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full 40-character lowercase commit hash")


def source_path(root: Path, name: str, *, supplemental: bool = False) -> Path:
    """Source names must remain relative and cannot escape through existing symlinks."""
    parts = PurePosixPath(name).parts
    if (not name or str(PurePosixPath(name)) != name
            or "\\" in name or PurePosixPath(name).is_absolute()
            or ".." in parts or not (name.endswith(".jsonl") or supplemental and
                name.endswith(("/raw/metrics.csv", "/raw/description.txt")))):
        raise ValueError(f"invalid JSONL source path: {name}")
    result = root / name
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"source path escapes root: {name}")
    return result


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(path: Path, metadata: dict) -> dict:
    """Check LFS SHA256 or Git blob identity and retain SHA256 for every acquired file."""
    sha = file_sha256(path)
    expected = metadata.get("sha256") or (metadata.get("lfs") or {}).get("sha256")
    blob = metadata.get("blobId")
    if expected:
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or sha != expected:
            raise ValueError(f"SHA256 checksum mismatch: {path.name}")
        method = "sha256"
    elif blob:
        if git_blob_sha1(path) != blob:
            raise ValueError(f"Git blob checksum mismatch: {path.name}")
        method = "git_blob_sha1"
    else:
        raise ValueError("source metadata must supply a SHA256 or Git blob checksum")
    size = metadata.get("size")
    if size is not None and path.stat().st_size != size:
        raise ValueError(f"file size mismatch: {path.name}")
    return {"sha256": sha, "size_bytes": path.stat().st_size,
            "verification": method, "expected_checksum": expected or blob}


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def acquire_telecomts(
    *, revision: str, output_dir: Path, files: list[str] | None = None,
    local_dir: Path | None = None, checksums: dict[str, str] | None = None,
    declared_license: str = "unknown", client: httpx.Client | None = None,
    supplemental: bool = False,
) -> dict:
    """Acquire explicit local files or all/selected processed files from an immutable revision."""
    check_revision(revision)
    manifest_path = output_dir / ("supplements.json" if supplemental else "acquisition.json")
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["revision"] != revision:
            raise FileExistsError("acquisition directory belongs to another revision")
    manifest = {
        "repository": REPOSITORY, "revision": revision,
        "source_url": f"https://huggingface.co/datasets/{REPOSITORY}/tree/{revision}",
        "declared_license": declared_license, "adapter_version": ADAPTER_VERSION,
        "schema_version": SCHEMA_VERSION, "code_version": "0.1.0",
        "acquired_at": datetime.now(UTC).isoformat(), "files": [], "status": "failed",
        "settings": {"mode": "local" if local_dir else "remote", "selected_files": files},
    }
    owned_client = local_dir is None and client is None
    try:
        if local_dir is not None:
            if not checksums:
                raise ValueError("offline acquisition requires supplied SHA256 checksums")
            selected = sorted(files if files is not None else (
                name for name in checksums if name.endswith(".jsonl")
            ))
            metadata = {name: {"sha256": checksums[name]} for name in selected}
        else:
            client = client or httpx.Client(follow_redirects=True, timeout=30)
            response = client.get(
                f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/{revision}",
                params={"blobs": "true"},
            )
            response.raise_for_status()
            snapshot = response.json()
            if snapshot["sha"] != revision:
                raise ValueError("upstream metadata returned a different revision")
            metadata = {item["rfilename"]: item for item in snapshot["siblings"]}
            selected = sorted(files if files is not None else (
                name for name in metadata if name.endswith("/processed/chunked.jsonl")
            ))
            manifest["declared_license"] = (snapshot.get("cardData") or {}).get("license", "unknown")
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("select at least one file without duplicate source paths")
        manifest["settings"]["selected_files"] = selected
        for name in selected:
            target = source_path(output_dir / "raw", name, supplemental=supplemental)
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = metadata[name]
            if local_dir is not None:
                source = source_path(local_dir, name, supplemental=supplemental)
                verified = verify_source(source, expected)
                if source.resolve() != target.resolve():
                    shutil.copyfile(source, target)
            elif target.exists():
                verified = verify_source(target, expected)
            else:
                temporary = target.with_suffix(target.suffix + ".part")
                url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{quote(name)}"
                with client.stream("GET", url) as download:
                    download.raise_for_status()
                    with temporary.open("wb") as stream:
                        for block in download.iter_bytes():
                            stream.write(block)
                verified = verify_source(temporary, expected)
                temporary.replace(target)
            if file_sha256(target) != verified["sha256"]:
                raise ValueError(f"copied source checksum mismatch: {name}")
            manifest["files"].append({"source_file": name, **verified})
        manifest["status"] = "complete"
    except (OSError, ValueError, KeyError, httpx.HTTPError) as exc:
        manifest["error"] = str(exc)
        raise
    finally:
        write_manifest(manifest_path, manifest)
        if owned_client and client is not None:
            client.close()
    return manifest


def acquire_supplements(*, sources: list[str], **kwargs) -> dict:
    """Acquire the two pinned raw assets per processed trace in a separate manifest."""
    names = []
    for name in sources:
        if not name.endswith("/processed/chunked.jsonl"):
            raise ValueError(f"supplemental mapping requires a processed trace: {name}")
        family = name.removesuffix("/processed/chunked.jsonl")
        names.extend(f"{family}/raw/{asset}" for asset in ("metrics.csv", "description.txt"))
    return acquire_telecomts(files=names, supplemental=True, **kwargs)
