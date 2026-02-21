from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import List

import torch
from sentence_transformers import SentenceTransformer

from app.config import log, settings

_model: SentenceTransformer | None = None
_query_embedding_cache: dict[str, List[float]] = {}  # ⚡ Cache query embeddings

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Cache models locally to avoid re-downloading
MODELS_CACHE_DIR = Path(__file__).parent.parent / ".model_cache"


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("Loading embedding model: %s", settings.embedding_model)

        # Create cache directory if it doesn't exist
        MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Set HuggingFace cache to local directory
        os.environ["HF_HOME"] = str(MODELS_CACHE_DIR)
        os.environ["TRANSFORMERS_CACHE"] = str(MODELS_CACHE_DIR)
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODELS_CACHE_DIR)

        log.info("Model cache directory: %s", MODELS_CACHE_DIR)

        # ⚡ Auto-detect device: GPU if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            log.info("🚀 GPU detected! Using CUDA for embeddings (10-20x faster)")
        else:
            log.warning("⚠️  No GPU detected. Using CPU (slow: ~1000ms per query)")

        _model = SentenceTransformer(
            settings.embedding_model,
            device=device,
            cache_folder=str(MODELS_CACHE_DIR),
        )
        _model.eval()
        log.info("Embedding model loaded on device: %s", device)
    return _model


def embed_texts(texts: List[str], is_query: bool = False) -> List[List[float]]:
    """Embed a list of texts. For queries, prepend the BGE search prefix."""
    model = get_model()

    if is_query:
        texts = [f"{QUERY_PREFIX}{t}" for t in texts]

    with torch.no_grad():
        embeddings = model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    result = embeddings.tolist()

    del embeddings
    gc.collect()

    return result


def embed_query(query: str) -> List[float]:
    """Embed a single query string with caching."""
    # ⚡ Check cache first (huge speedup for repeated queries)
    if query in _query_embedding_cache:
        return _query_embedding_cache[query]

    # Not in cache, embed it
    embedding = embed_texts([query], is_query=True)[0]

    # Store in cache for next time
    _query_embedding_cache[query] = embedding

    return embedding
