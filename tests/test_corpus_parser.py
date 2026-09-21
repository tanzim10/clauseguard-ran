from __future__ import annotations

import hashlib
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from clauseguard.corpus import parser


def test_parse_pdf_writes_manifest_aligned_text(monkeypatch, tmp_path) -> None:
    source = tmp_path / "spec.pdf"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(
        parser,
        "PdfReader",
        lambda path: SimpleNamespace(
            pages=[
                SimpleNamespace(extract_text=lambda: "first page"),
                SimpleNamespace(extract_text=lambda: "second page"),
            ]
        ),
    )

    output = parser.parse_pdf(source, tmp_path / "parsed")

    assert output.name == "spec.txt"
    assert output.read_text(encoding="utf-8") == "first page\n\nsecond page\n"


def test_parse_docx_writes_paragraph_text(tmp_path) -> None:
    source = tmp_path / "spec.docx"
    with ZipFile(source, "w") as archive:
        archive.writestr(
            "word/document.xml",
            """<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">
            <w:body><w:p><w:r><w:t>First paragraph</w:t></w:r></w:p>
            <w:p><w:r><w:t>Second paragraph</w:t></w:r></w:p></w:body></w:document>""",
        )

    output = parser.parse_docx(source, tmp_path / "parsed")

    assert output.name == "spec.txt"
    assert output.read_text(encoding="utf-8") == "First paragraph\n\nSecond paragraph\n"


def test_parse_document_rejects_unsupported_format(tmp_path) -> None:
    source = tmp_path / "spec.txt"
    source.write_text("not a source document", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported corpus format"):
        parser.parse_document(source, tmp_path / "parsed")


def test_parse_corpus_validates_sources_before_parsing(monkeypatch, tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source = raw_dir / "spec.pdf"
    source.write_bytes(b"verified corpus source")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "filename,source_url,spec_id,doc_type,working_group,version,sha256,acquired_at,"
        "license_note,has_text_layer\n"
        f"spec.pdf,https://example.com/spec,O-RAN.WG2.A1AP,oran,WG2,1.0,"
        f"{hashlib.sha256(source.read_bytes()).hexdigest()},2026-09-07T00:00:00Z,fixture,true\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "parsed"
    parsed_sources: list[tuple[object, object]] = []

    def fake_parse_document(source_path, destination):
        parsed_sources.append((source_path, destination))
        output = output_dir / "spec.txt"
        output.parent.mkdir(exist_ok=True)
        output.write_text("parsed text\n", encoding="utf-8")
        return output

    monkeypatch.setattr(parser, "parse_document", fake_parse_document)

    summary = parser.parse_corpus(
        manifest_path=manifest,
        raw_dir=raw_dir,
        output_dir=output_dir,
    )

    assert summary.documents == 1
    assert summary.outputs == (output_dir / "spec.txt",)
    assert parsed_sources == [(source, output_dir)]


def test_parse_corpus_rejects_hash_mismatch_before_parsing(monkeypatch, tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "spec.pdf").write_bytes(b"actual source")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "filename,source_url,spec_id,doc_type,working_group,version,sha256,acquired_at,"
        "license_note,has_text_layer\n"
        "spec.pdf,https://example.com/spec,O-RAN.WG2.A1AP,oran,WG2,1.0,"
        "not-the-real-hash,2026-09-07T00:00:00Z,fixture,true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        parser,
        "parse_document",
        lambda *args: pytest.fail("parser must not run for invalid sources"),
    )

    with pytest.raises(ValueError, match="manifest validation failed"):
        parser.parse_corpus(manifest_path=manifest, raw_dir=raw_dir)
