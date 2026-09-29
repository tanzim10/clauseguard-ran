from types import SimpleNamespace

import pytest
from qdrant_client import models

from clauseguard.config import Settings
from clauseguard.retrieval.qdrant_store import (
    CollectionCompatibilityError,
    ensure_collection,
    search,
    upsert_chunks,
)


class FakeQdrantClient:
    def __init__(self, *, exists: bool, vectors: object | None = None) -> None:
        self.exists = exists
        self.vectors = vectors
        self.create_calls: list[dict[str, object]] = []
        self.upsert_calls: list[dict[str, object]] = []
        self.query_calls: list[dict[str, object]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def create_collection(self, **kwargs: object) -> None:
        self.create_calls.append(kwargs)
        self.exists = True

    def get_collection(self, collection_name: str) -> object:
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=self.vectors))
        )

    def upsert(self, **kwargs: object) -> None:
        self.upsert_calls.append(kwargs)

    def query_points(self, **kwargs: object) -> object:
        self.query_calls.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="point-1",
                    score=0.9,
                    payload={"text": "passage", "spec_id": "spec"},
                )
            ]
        )


def test_ensure_collection_creates_missing_collection() -> None:
    client = FakeQdrantClient(exists=False)
    settings = Settings(_env_file=None, embedding_dimensions=4)

    ensure_collection("specs", client=client, settings=settings)

    assert len(client.create_calls) == 1
    assert client.create_calls[0]["collection_name"] == "specs"
    vector_config = client.create_calls[0]["vectors_config"]
    assert isinstance(vector_config, models.VectorParams)
    assert vector_config.size == 4
    assert vector_config.distance == models.Distance.COSINE


def test_ensure_collection_reuses_compatible_collection() -> None:
    client = FakeQdrantClient(
        exists=True,
        vectors=models.VectorParams(size=4, distance=models.Distance.COSINE),
    )
    settings = Settings(_env_file=None, embedding_dimensions=4)

    ensure_collection("specs", client=client, settings=settings)

    assert client.create_calls == []


def test_ensure_collection_rejects_incompatible_collection() -> None:
    client = FakeQdrantClient(
        exists=True,
        vectors=models.VectorParams(size=3, distance=models.Distance.COSINE),
    )
    settings = Settings(_env_file=None, embedding_dimensions=4)

    with pytest.raises(CollectionCompatibilityError, match="dimension 3"):
        ensure_collection("specs", client=client, settings=settings)

    assert client.create_calls == []


def test_ensure_collection_rejects_incompatible_distance() -> None:
    client = FakeQdrantClient(
        exists=True,
        vectors=models.VectorParams(size=4, distance=models.Distance.DOT),
    )
    settings = Settings(_env_file=None, embedding_dimensions=4)

    with pytest.raises(CollectionCompatibilityError, match="distance"):
        ensure_collection("specs", client=client, settings=settings)


def test_upsert_chunks_preserves_text_metadata_and_uses_deterministic_uuid() -> None:
    client = FakeQdrantClient(exists=True)
    settings = Settings(_env_file=None, embedding_dimensions=2)
    chunks = [
        {
            "id": "spec::1.1::0",
            "text": "A citation passage.",
            "metadata": {
                "spec_id": "spec",
                "section": "1.1",
                "source_file": "spec.txt",
                "chunk_id": "spec::1.1::0",
            },
        }
    ]

    count = upsert_chunks(
        "specs",
        chunks,
        embeddings=[[0.1, 0.2]],
        client=client,
        settings=settings,
    )

    assert count == 1
    point = client.upsert_calls[0]["points"][0]
    assert isinstance(point, models.PointStruct)
    assert isinstance(point.id, str)
    assert point.id == "0bcfa5b7-0cd4-5aa5-8b45-8625c09ec784"
    assert point.payload == {
        "text": "A citation passage.",
        "spec_id": "spec",
        "section": "1.1",
        "source_file": "spec.txt",
        "chunk_id": "spec::1.1::0",
    }


def test_upsert_chunks_batches_and_validates_before_writing() -> None:
    client = FakeQdrantClient(exists=True)
    settings = Settings(
        _env_file=None,
        embedding_dimensions=1,
        qdrant_upsert_batch_size=1,
    )
    chunks = [
        {
            "id": f"spec::{index}",
            "text": f"text {index}",
            "metadata": {
                "spec_id": "spec",
                "section": "1",
                "source_file": "spec.txt",
                "chunk_id": f"spec::{index}",
            },
        }
        for index in range(2)
    ]

    assert upsert_chunks(
        "specs",
        chunks,
        embeddings=[[0.1], [0.2]],
        client=client,
        settings=settings,
    ) == 2
    assert len(client.upsert_calls) == 2

    with pytest.raises(ValueError, match="same number"):
        upsert_chunks(
            "specs",
            chunks,
            embeddings=[[0.1]],
            client=client,
            settings=settings,
        )
    assert len(client.upsert_calls) == 2


def test_search_maps_query_points_response() -> None:
    client = FakeQdrantClient(exists=True)
    settings = Settings(_env_file=None, embedding_dimensions=2)

    results = search("specs", [0.1, 0.2], client=client, settings=settings)

    assert results == [
        {
            "id": "point-1",
            "score": 0.9,
            "payload": {"text": "passage", "spec_id": "spec"},
        }
    ]
    assert client.query_calls[0]["query"] == [0.1, 0.2]
