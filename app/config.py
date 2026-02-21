from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine which .env file to use
_env_mode = os.getenv("ENV", "").strip().lower()
_env_file = f".env.{_env_mode}" if _env_mode else ".env"

# Get the directory of this file
_config_dir = Path(__file__).parent.parent

# Explicitly load the .env file using python-dotenv
_env_path = _config_dir / _env_file
print(f"[CONFIG] ENV mode: {_env_mode if _env_mode else 'default (development)'}")
print(f"[CONFIG] Loading: {_env_path}")
print(f"[CONFIG] File exists: {_env_path.exists()}")

if _env_path.exists():
    load_dotenv(_env_path, override=True)
    print(f"[CONFIG] Successfully loaded {_env_file}")
else:
    print(f"[CONFIG] WARNING: {_env_file} not found, using defaults")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_env_path),
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
    port: int = 8081
    log_level: str = "info"


settings = Settings()

# Log the loaded settings
print(f"[CONFIG] Qdrant URL: {settings.qdrant_url if settings.qdrant_url else 'NOT SET (using localhost)'}")
print(f"[CONFIG] Qdrant Host: {settings.qdrant_host}")
print(f"[CONFIG] Qdrant Port: {settings.qdrant_port}")
print(f"[CONFIG] Qdrant Collection: {settings.qdrant_collection}")


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
