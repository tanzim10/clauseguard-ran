import pytest

from clauseguard.cli import index_corpus
from clauseguard.retrieval.indexer import IndexSummary


def test_main_runs_indexer_and_prints_summary(monkeypatch, capsys) -> None:
    calls: list[dict[str, object]] = []

    def fake_index_corpus(**kwargs: object) -> IndexSummary:
        calls.append(kwargs)
        return IndexSummary(documents=12, chunks=34)

    monkeypatch.setattr(index_corpus, "index_corpus", fake_index_corpus)

    assert index_corpus.main(
        ["--manifest", "/tmp/manifest.csv", "--parsed-dir", "/tmp/parsed"]
    ) == 0
    assert calls == [
        {
            "manifest_path": "/tmp/manifest.csv",
            "parsed_dir": "/tmp/parsed",
            "collection": None,
        }
    ]
    assert capsys.readouterr().out == "Indexed 12 documents into 34 chunks.\n"


def test_main_returns_nonzero_and_reports_failure(monkeypatch, capsys) -> None:
    def fake_index_corpus(**kwargs: object) -> IndexSummary:
        raise FileNotFoundError("parsed corpus files are missing: first.txt")

    monkeypatch.setattr(index_corpus, "index_corpus", fake_index_corpus)

    assert index_corpus.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "parsed corpus files are missing: first.txt" in captured.err


def test_help_is_available_without_external_services() -> None:
    with pytest.raises(SystemExit) as error:
        index_corpus.main(["--help"])

    assert error.value.code == 0
