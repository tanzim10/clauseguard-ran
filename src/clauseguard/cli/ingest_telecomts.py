"""Prepare acquisition and validated scenarios; split assignment belongs to issue #22."""

import argparse
import json
import sys
from pathlib import Path

import httpx

from clauseguard.datasets.acquisition import source_path, write_manifest
from clauseguard.datasets.telecomts import build_scenarios, download_telecomts, file_sha256


def _json_object(path: Path | None) -> dict:
    if path is None:
        return {}
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise TypeError(f"configuration must be a JSON object: {path.name}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Full upstream commit hash")
    parser.add_argument("--file", action="append", help="Relative source path; repeat to select files")
    parser.add_argument("--input-dir", type=Path, help="Already acquired source tree (offline mode)")
    parser.add_argument("--checksums", type=Path, help="Offline source-path to SHA256 JSON object")
    parser.add_argument("--license", default="unknown", help="Declared license for offline sources")
    parser.add_argument("--acquisition-dir", type=Path, default=Path("data/telecomts"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/scenarios"))
    parser.add_argument("--reviews", type=Path, help="Human review JSON keyed by scenario ID")
    parser.add_argument("--lineage", type=Path, help="Recording evidence JSON keyed by scenario ID")
    parser.add_argument("--units", type=Path, help="Verified channel-unit JSON object")
    args = parser.parse_args(argv)
    try:
        # Avoid downloading again when the requested conversion outputs already exist.
        if any((args.output_dir / name).exists() for name in
               ("scenarios.jsonl", "rejected.jsonl", "conversion.json")):
            raise FileExistsError("conversion outputs already exist; choose a new output directory")
        checksums = _json_object(args.checksums)
        reviews, lineage, units = (_json_object(path) for path in
                                   (args.reviews, args.lineage, args.units))
        manifest = download_telecomts(
            revision=args.revision, output_dir=args.acquisition_dir, files=args.file,
            local_dir=args.input_dir, checksums=checksums, declared_license=args.license,
        )
        manifest["settings"].update({name: file_sha256(path) if path else None for name, path in
                                    (("reviews_sha256", args.reviews), ("lineage_sha256", args.lineage),
                                     ("units_sha256", args.units))})
        write_manifest(args.acquisition_dir / "acquisition.json", manifest)
        files = manifest["files"]
        report = build_scenarios(
            {item["source_file"]: source_path(args.acquisition_dir / "raw", item["source_file"])
             for item in files}, revision=args.revision,
            checksums={item["source_file"]: item["sha256"] for item in files},
            output_dir=args.output_dir, reviews=reviews, lineage=lineage, units=units,
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if report["counts"]["accepted"] and not report["counts"]["rejected"] else 2
    except (OSError, ValueError, TypeError, KeyError, httpx.HTTPError) as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(args.output_dir / "ingest_error.json", {"error": str(exc)})
        print(f"TelecomTS preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
