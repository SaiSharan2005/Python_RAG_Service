from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.config import log
from app.embedding import embed_query
from app.qdrant_store import search_by_ids


@dataclass
class RetrievedChunk:
    id: str
    content: str
    source: str
    title: str
    page_number: Optional[int]
    chunk_index: Optional[int]
    document_id: str
    score: float


def retrieve(
    query: str,
    candidate_ids: List[str],
    top_k: int = 10,
) -> List[RetrievedChunk]:
    """Embed query, search Qdrant within candidate_ids, return top_k chunks.

    Only the query is embedded at runtime.  Candidate chunks are never
    re-embedded — their vectors already exist in Qdrant from ingestion.
    Search is strictly filtered to the supplied candidate_ids via
    HasIdCondition.
    """
    if not candidate_ids:
        log.warning("Empty candidate_ids received.")
        return []

    log.info(
        "Retrieving: query=%r, candidates=%d, top_k=%d",
        query[:80],
        len(candidate_ids),
        top_k,
    )

    query_vector = embed_query(query)

    results = search_by_ids(
        query_vector=query_vector,
        candidate_ids=candidate_ids,
        top_k=top_k,
    )

    chunks: List[RetrievedChunk] = []
    for hit in results:
        payload = hit.payload or {}
        chunks.append(
            RetrievedChunk(
                id=payload.get("chunk_id", str(hit.id)),
                content=payload.get("content", ""),
                source=payload.get("source", ""),
                title=payload.get("title", ""),
                page_number=payload.get("page_number"),
                chunk_index=payload.get("chunk_index"),
                document_id=payload.get("document_id", ""),
                score=hit.score,
            )
        )

    log.info("Retrieved %d chunks.", len(chunks))
    return chunks
