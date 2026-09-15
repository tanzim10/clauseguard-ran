from __future__ import annotations

import hashlib

from clauseguard.corpus.manifest import load_manifest, validate_manifest_row


def test_load_manifest_reads_multiple_rows(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "filename,source_url,spec_id,doc_type,working_group,version,sha256,acquired_at,"
        "license_note,has_text_layer\n"
        "a.pdf,https://example.com/a,O-RAN.WG2.A1AP,oran,WG2,1.0,pending,,blocked,true\n"
        "b.pdf,https://example.com/b,3GPP TS 28.552,3gpp,SA5,18.7,pending,,blocked,false\n",
        encoding="utf-8",
    )

    rows = load_manifest(manifest_path)

    assert [row["filename"] for row in rows] == ["a.pdf", "b.pdf"]
    assert rows[0]["has_text_layer"] == "true"


def test_validate_manifest_row_accepts_matching_file_hash(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pdf_path = raw_dir / "spec.pdf"
    pdf_path.write_bytes(b"fixture pdf bytes")

    row = {
        "filename": "spec.pdf",
        "source_url": "https://example.com/spec.pdf",
        "spec_id": "O-RAN.WG2.A1AP",
        "doc_type": "oran",
        "working_group": "WG2",
        "version": "04.03",
        "sha256": hashlib.sha256(b"fixture pdf bytes").hexdigest(),
        "acquired_at": "2026-09-07T05:02:37Z",
        "license_note": "fixture license note",
        "has_text_layer": "true",
    }

    assert validate_manifest_row(row, corpus_dir=raw_dir)


def test_validate_manifest_row_rejects_hash_mismatch(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "spec.pdf").write_bytes(b"actual bytes")

    row = {
        "filename": "spec.pdf",
        "source_url": "https://example.com/spec.pdf",
        "spec_id": "O-RAN.WG2.A1AP",
        "doc_type": "oran",
        "working_group": "WG2",
        "version": "04.03",
        "sha256": hashlib.sha256(b"different bytes").hexdigest(),
        "acquired_at": "2026-09-07T05:02:37Z",
        "license_note": "fixture license note",
        "has_text_layer": "true",
    }

    assert not validate_manifest_row(row, corpus_dir=raw_dir)


def test_validate_manifest_row_rejects_missing_required_field():
    row = {
        "filename": "",
        "source_url": "https://example.com/spec.pdf",
        "spec_id": "O-RAN.WG2.A1AP",
        "doc_type": "oran",
        "working_group": "WG2",
        "version": "04.03",
        "sha256": "pending",
        "acquired_at": "",
        "license_note": "documented blocker",
        "has_text_layer": "",
    }

    assert not validate_manifest_row(row)


def test_validate_manifest_row_rejects_invalid_doc_type():
    row = {
        "filename": "spec.pdf",
        "source_url": "https://example.com/spec.pdf",
        "spec_id": "O-RAN.WG2.A1AP",
        "doc_type": "ietf",
        "working_group": "WG2",
        "version": "04.03",
        "sha256": "pending",
        "acquired_at": "",
        "license_note": "documented blocker",
        "has_text_layer": "",
    }

    assert not validate_manifest_row(row)


def test_validate_manifest_row_accepts_documented_blocker():
    row = {
        "filename": "blocked.pdf",
        "source_url": "https://example.com/blocked.pdf",
        "spec_id": "O-RAN.WG2.A1AP",
        "doc_type": "oran",
        "working_group": "WG2",
        "version": "04.03",
        "sha256": "pending",
        "acquired_at": "",
        "license_note": "Unavailable publicly; waiting on accessible source.",
        "has_text_layer": "",
    }

    assert validate_manifest_row(row)


def test_validate_manifest_row_rejects_blocker_with_acquired_at():
    row = {
        "filename": "blocked.pdf",
        "source_url": "https://example.com/blocked.pdf",
        "spec_id": "O-RAN.WG2.A1AP",
        "doc_type": "oran",
        "working_group": "WG2",
        "version": "04.03",
        "sha256": "pending",
        "acquired_at": "2026-09-07T05:02:37Z",
        "license_note": "Unavailable publicly; waiting on accessible source.",
        "has_text_layer": "",
    }

    assert not validate_manifest_row(row)


def test_validate_manifest_row_rejects_invalid_has_text_layer():
    row = {
        "filename": "spec.pdf",
        "source_url": "https://example.com/spec.pdf",
        "spec_id": "O-RAN.WG2.A1AP",
        "doc_type": "oran",
        "working_group": "WG2",
        "version": "04.03",
        "sha256": "pending",
        "acquired_at": "",
        "license_note": "documented blocker",
        "has_text_layer": "yes",
    }

    assert not validate_manifest_row(row)
