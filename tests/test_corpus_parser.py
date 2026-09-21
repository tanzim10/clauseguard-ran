from __future__ import annotations

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
