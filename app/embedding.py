from __future__ import annotations

import gc
from typing import List

import torch
from sentence_transformers import SentenceTransformer

from app.config import log, settings

_model: SentenceTransformer | None = None

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("Loading embedding model: %s", settings.embedding_model)
        _model = SentenceTransformer(
            settings.embedding_model,
            device="cpu",
        )
        _model.eval()
        log.info("Embedding model loaded.")
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
    """Embed a single query string."""
    return embed_texts([query], is_query=True)[0]
