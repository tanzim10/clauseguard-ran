"""Application settings (scaffold). Values load from env / .env; no side effects."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration contract for later weeks."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"

    data_dir: str = "./data"
    artifacts_dir: str = "./artifacts"
    models_dir: str = "./models"

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "oran_specs"
    qdrant_distance: str = "Cosine"
    qdrant_upsert_batch_size: int = 128

    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_dimensions: int = 3072
    openai_embedding_batch_size: int = 128
    openai_chat_model: str = "gpt-4o-mini"

    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_cooldown_seconds: int = 30
    nvidia_timeout_seconds: float = 30.0

    chunk_size_tokens: int = 512
    retrieval_top_k: int = 8
    raw_corpus_dir: str = "./data/corpus/raw"
    parsed_corpus_dir: str = "./data/corpus/parsed"

    local_llm_base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_adapter_path: str = "./models/lora-rca-v1"


def get_settings() -> Settings:
    """Return a Settings instance."""
    return Settings()
