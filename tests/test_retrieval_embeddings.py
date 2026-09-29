from dataclasses import dataclass

import pytest

from clauseguard.config import Settings
from clauseguard.retrieval.embeddings import embed_texts


@dataclass
class FakeEmbedding:
    index: int
    embedding: list[float]


@dataclass
class FakeResponse:
    data: list[FakeEmbedding]


class FakeEmbeddings:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        return next(self.responses)


class FakeClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.embeddings = FakeEmbeddings(responses)


def test_embed_texts_returns_empty_without_calling_client() -> None:
    client = FakeClient([])

    assert embed_texts([], client=client) == []
    assert client.embeddings.calls == []


def test_embed_texts_batches_requests_and_preserves_input_order() -> None:
    client = FakeClient(
        [
            FakeResponse(
                data=[
                    FakeEmbedding(index=1, embedding=[2.0]),
                    FakeEmbedding(index=0, embedding=[1.0]),
                ]
            ),
            FakeResponse(data=[FakeEmbedding(index=0, embedding=[3.0])]),
        ]
    )
    settings = Settings(
        _env_file=None,
        embedding_batch_size=2,
        embedding_dimensions=1,
    )

    result = embed_texts(["a", "b", "c"], client=client, settings=settings, input_type="passage")

    assert result == [[1.0], [2.0], [3.0]]
    assert [call["input"] for call in client.embeddings.calls] == [["a", "b"], ["c"]]
    assert all(call["model"] == "nvidia/llama-nemotron-embed-vl-1b-v2" for call in client.embeddings.calls)
    assert all(call["extra_body"] == {"input_type": "passage"} for call in client.embeddings.calls)
    assert all(call["encoding_format"] == "float" for call in client.embeddings.calls)


def test_embed_texts_rejects_empty_text() -> None:
    client = FakeClient([])

    with pytest.raises(ValueError, match="empty text"):
        embed_texts(["valid", "  "], client=client)

    assert client.embeddings.calls == []


def test_embed_texts_rejects_wrong_response_count() -> None:
    client = FakeClient([FakeResponse(data=[FakeEmbedding(index=0, embedding=[1.0])])])

    with pytest.raises(ValueError, match="expected 2 embeddings"):
        embed_texts(["a", "b"], client=client)


def test_embed_texts_rejects_wrong_embedding_dimensions() -> None:
    client = FakeClient([FakeResponse(data=[FakeEmbedding(index=0, embedding=[1.0])])])
    settings = Settings(_env_file=None, embedding_dimensions=2)

    with pytest.raises(ValueError, match="expected dimension 2"):
        embed_texts(["a"], client=client, settings=settings)
