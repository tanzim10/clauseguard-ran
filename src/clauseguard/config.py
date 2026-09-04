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

    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-large"
    openai_chat_model: str = "gpt-4o-mini"

    chunk_size_tokens: int = 512
    retrieval_top_k: int = 8

    local_llm_base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_adapter_path: str = "./models/lora-rca-v1"


def get_settings() -> Settings:
    """Return a Settings instance."""
    return Settings()
