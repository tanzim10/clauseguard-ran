from __future__ import annotations

import pytest

from clauseguard.cli import parse_corpus
from clauseguard.corpus.parser import ParseSummary


def test_main_runs_parser_and_prints_summary(monkeypatch, capsys, tmp_path) -> None:
    output = tmp_path / "spec.txt"
    calls: list[dict[str, object]] = []

    def fake_parse_corpus(**kwargs: object) -> ParseSummary:
        calls.append(kwargs)
        return ParseSummary(documents=12, outputs=(output,) * 12)

    monkeypatch.setattr(parse_corpus, "parse_corpus", fake_parse_corpus)

    assert parse_corpus.main(
        ["--manifest", "/tmp/manifest.csv", "--raw-dir", "/tmp/raw", "--output-dir", "/tmp/parsed"]
    ) == 0
    assert calls == [
        {
            "manifest_path": "/tmp/manifest.csv",
            "raw_dir": "/tmp/raw",
            "output_dir": "/tmp/parsed",
        }
    ]
    assert capsys.readouterr().out == "Parsed 12 documents into 12 text files.\n"


def test_main_reports_parser_failures(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        parse_corpus,
        "parse_corpus",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("invalid source")),
    )

    assert parse_corpus.main([]) == 1
    assert "parsing failed: invalid source" in capsys.readouterr().err


def test_help_is_available_without_external_services() -> None:
    with pytest.raises(SystemExit) as error:
        parse_corpus.main(["--help"])

    assert error.value.code == 0
