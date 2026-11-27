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
        embedding_dimensions=2,
        embedding_batch_size=8,
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
    assert embedding_client.calls[0]["extra_body"] == {"input_type": "passage"}
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


def _api_error(status: int) -> Exception:
    import httpx
    from openai import APIStatusError, InternalServerError, UnprocessableEntityError

    response = httpx.Response(status, request=httpx.Request("POST", "http://embed"))
    cls = {422: UnprocessableEntityError, 502: InternalServerError}.get(status, APIStatusError)
    return cls("error", response=response, body=None)


def test_embed_screened_drops_only_rejected_chunks_and_keeps_order() -> None:
    from clauseguard.retrieval.indexer import embed_screened

    chunks = [{"id": f"c{i}", "text": f"t{i}"} for i in range(5)]

    def embed(texts: list[str]) -> list[list[float]]:
        if "t3" in texts:
            raise _api_error(422)
        return [[float(t[1:])] for t in texts]

    kept, vectors, rejected = embed_screened(chunks, embed, batch_size=4, sleep=lambda _: None)

    assert [c["id"] for c in kept] == ["c0", "c1", "c2", "c4"]
    assert vectors == [[0.0], [1.0], [2.0], [4.0]]
    assert rejected == ["c3"]


def test_embed_screened_retries_transient_errors_then_raises_others() -> None:
    from clauseguard.retrieval.indexer import embed_screened

    calls = {"n": 0}

    def flaky(texts: list[str]) -> list[list[float]]:
        calls["n"] += 1
        if calls["n"] < 3:
            raise _api_error(502)
        return [[1.0] for _ in texts]

    kept, _, rejected = embed_screened([{"id": "a", "text": "x"}], flaky, batch_size=8, sleep=lambda _: None)
    assert [c["id"] for c in kept] == ["a"] and rejected == [] and calls["n"] == 3

    def forbidden(texts: list[str]) -> list[list[float]]:
        raise _api_error(401)

    from openai import APIStatusError

    with pytest.raises(APIStatusError):
        embed_screened([{"id": "a", "text": "x"}], forbidden, batch_size=8, sleep=lambda _: None)
