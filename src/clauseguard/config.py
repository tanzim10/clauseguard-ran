"""Application settings (scaffold). Values load from env / .env; no side effects."""

from pydantic import Field
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
    qdrant_collection: str = "oran_specs_nemotron_vl_2048"
    qdrant_distance: str = "Cosine"
    qdrant_upsert_batch_size: int = 128

    nvidia_embedding_model: str = "nvidia/llama-nemotron-embed-vl-1b-v2"
    embedding_dimensions: int = 2048
    embedding_batch_size: int = 128
    openai_chat_model: str = "gpt-4o-mini"

    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_cooldown_seconds: int = Field(default=30, ge=0)
    nvidia_timeout_seconds: float = Field(default=30.0, gt=0)

    chunk_size_tokens: int = 512
    retrieval_top_k: int = 8
    raw_corpus_dir: str = "./data/corpus/raw"
    parsed_corpus_dir: str = "./data/corpus/parsed"

    local_llm_base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_adapter_path: str = "./models/lora-rca-v1"


def get_settings() -> Settings:
    """Return a Settings instance."""
    return Settings()
