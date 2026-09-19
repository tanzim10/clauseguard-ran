from clauseguard.config import Settings


def test_retrieval_defaults_define_one_vector_contract() -> None:
    settings = Settings(_env_file=None)

    assert settings.openai_embedding_model == "text-embedding-3-large"
    assert settings.openai_embedding_dimensions == 3072
    assert settings.openai_embedding_batch_size == 128
    assert settings.qdrant_distance == "Cosine"
    assert settings.qdrant_upsert_batch_size == 128
    assert settings.parsed_corpus_dir == "./data/corpus/parsed"


def test_retrieval_settings_can_be_overridden_without_network_calls() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        openai_embedding_dimensions=1024,
        openai_embedding_batch_size=16,
        qdrant_upsert_batch_size=32,
        parsed_corpus_dir="/tmp/parsed",
    )

    assert settings.openai_api_key == "test-key"
    assert settings.openai_embedding_dimensions == 1024
    assert settings.openai_embedding_batch_size == 16
    assert settings.qdrant_upsert_batch_size == 32
    assert settings.parsed_corpus_dir == "/tmp/parsed"
