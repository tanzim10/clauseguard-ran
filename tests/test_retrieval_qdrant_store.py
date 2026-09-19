from types import SimpleNamespace

import pytest
from qdrant_client import models

from clauseguard.config import Settings
from clauseguard.retrieval.qdrant_store import (
    CollectionCompatibilityError,
    ensure_collection,
)


class FakeQdrantClient:
    def __init__(self, *, exists: bool, vectors: object | None = None) -> None:
        self.exists = exists
        self.vectors = vectors
        self.create_calls: list[dict[str, object]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def create_collection(self, **kwargs: object) -> None:
        self.create_calls.append(kwargs)
        self.exists = True

    def get_collection(self, collection_name: str) -> object:
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=self.vectors))
        )


def test_ensure_collection_creates_missing_collection() -> None:
    client = FakeQdrantClient(exists=False)
    settings = Settings(_env_file=None, openai_embedding_dimensions=4)

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
    settings = Settings(_env_file=None, openai_embedding_dimensions=4)

    ensure_collection("specs", client=client, settings=settings)

    assert client.create_calls == []


def test_ensure_collection_rejects_incompatible_collection() -> None:
    client = FakeQdrantClient(
        exists=True,
        vectors=models.VectorParams(size=3, distance=models.Distance.COSINE),
    )
    settings = Settings(_env_file=None, openai_embedding_dimensions=4)

    with pytest.raises(CollectionCompatibilityError, match="dimension 3"):
        ensure_collection("specs", client=client, settings=settings)

    assert client.create_calls == []


def test_ensure_collection_rejects_incompatible_distance() -> None:
    client = FakeQdrantClient(
        exists=True,
        vectors=models.VectorParams(size=4, distance=models.Distance.DOT),
    )
    settings = Settings(_env_file=None, openai_embedding_dimensions=4)

    with pytest.raises(CollectionCompatibilityError, match="distance"):
        ensure_collection("specs", client=client, settings=settings)
