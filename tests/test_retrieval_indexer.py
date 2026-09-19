from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from clauseguard.config import Settings
from clauseguard.retrieval.indexer import index_corpus


@dataclass
class FakeEmbedding:
    index: int
    embedding: list[float]


class FakeEmbeddingClient:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        texts = kwargs["input"]
        return SimpleNamespace(
            data=[
                FakeEmbedding(index=index, embedding=[float(index)] * self.dimensions)
                for index, _ in enumerate(texts)
            ]
        )


class FakeEmbeddings:
    def __init__(self, client: FakeEmbeddingClient) -> None:
        self.client = client

    def create(self, **kwargs: object) -> SimpleNamespace:
        return self.client.create(**kwargs)


class FakeQdrantClient:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.upserts: list[dict[str, object]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return bool(self.created)

    def create_collection(self, **kwargs: object) -> None:
        self.created.append(kwargs)

    def get_collection(self, collection_name: str) -> object:
        raise AssertionError("compatible collection should not need inspection in fixture")

    def upsert(self, **kwargs: object) -> None:
        self.upserts.append(kwargs)


def _write_manifest(path: Path) -> None:
    path.write_text(
        "filename,source_url,spec_id,doc_type,working_group,version,sha256,acquired_at,license_note\n"
        "first.pdf,https://example.com/first,O-RAN.first,oran,WG1,1.0,hash,2026-09-01,,\n"
        "second.pdf,https://example.com/second,3GPP.second,3gpp,RAN2,18.1,hash,2026-09-01,,\n",
        encoding="utf-8",
    )


def test_index_corpus_chunks_embeds_and_upserts_fixture_corpus(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    _write_manifest(manifest)
    (parsed / "first.txt").write_text("1.1 First section\n\nA first passage.", encoding="utf-8")
    (parsed / "second.txt").write_text(
        "2.1 Second section\n\nA second passage.", encoding="utf-8"
    )
    settings = Settings(
        _env_file=None,
        openai_embedding_dimensions=2,
        openai_embedding_batch_size=8,
    )
    embedding_client = FakeEmbeddingClient(dimensions=2)
    qdrant_client = FakeQdrantClient()

    summary = index_corpus(
        manifest_path=manifest,
        parsed_dir=parsed,
        collection="fixture_specs",
        settings=settings,
        embedding_client=SimpleNamespace(embeddings=FakeEmbeddings(embedding_client)),
        qdrant_client=qdrant_client,
    )

    assert summary.documents == 2
    assert summary.chunks == 2
    assert len(embedding_client.calls) == 1
    assert len(qdrant_client.upserts) == 1
    points = qdrant_client.upserts[0]["points"]
    assert [point.payload["spec_id"] for point in points] == [
        "O-RAN.first",
        "3GPP.second",
    ]
    assert [point.payload["text"] for point in points] == [
        "1.1 First section\n\nA first passage.",
        "2.1 Second section\n\nA second passage.",
    ]


def test_index_corpus_fails_before_embedding_when_parsed_file_is_missing(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    _write_manifest(manifest)
    settings = Settings(_env_file=None, openai_api_key="unused")
    embedding_client = FakeEmbeddingClient(dimensions=3072)
    qdrant_client = FakeQdrantClient()

    with pytest.raises(FileNotFoundError, match="first.txt"):
        index_corpus(
            manifest_path=manifest,
            parsed_dir=parsed,
            settings=settings,
            embedding_client=SimpleNamespace(embeddings=FakeEmbeddings(embedding_client)),
            qdrant_client=qdrant_client,
        )

    assert embedding_client.calls == []
    assert qdrant_client.upserts == []
