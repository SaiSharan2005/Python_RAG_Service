from __future__ import annotations

import logging
import os
import sys

from pydantic_settings import BaseSettings, SettingsConfigDict

# Pick .env file: set ENV=prod to load .env.prod, otherwise .env
_env_mode = os.getenv("ENV", "").strip().lower()
_env_file = f".env.{_env_mode}" if _env_mode else ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_env_file,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Qdrant
    qdrant_url: str = ""
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""
    qdrant_collection: str = "rag_chunks"

    # Embedding
    embedding_model: str = "BAAI/bge-small-en"
    embedding_dimension: int = 384
    embedding_batch_size: int = 10

    # Ingestion — path can be a single .json file OR a directory of .json files
    ingestion_json_path: str = "../lucene-service/chunk-exports"
    upsert_batch_size: int = 10

    # LLM
    llm_api_key: str = ""
    llm_api_url: str = "https://api.anthropic.com/v1/messages"
    llm_model: str = "claude-sonnet-4-5-20250929"
    llm_max_tokens: int = 1024

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


settings = Settings()


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("rag")
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.flush = lambda: sys.stdout.flush()
    logger.addHandler(handler)
    return logger


log = setup_logging()
