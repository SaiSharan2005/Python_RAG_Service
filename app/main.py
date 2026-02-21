from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.config import log, settings
from app.embedding import get_model
from app.qdrant_store import ensure_collection, collection_count
from app.retriever import retrieve, retrieve_with_bm25_scores
from app.generator import generate_answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting RAG service...")
    get_model()
    ensure_collection()
    count = collection_count()
    log.info("Qdrant collection '%s' has %d points.", settings.qdrant_collection, count)
    log.info("RAG service ready.")
    yield
    log.info("Shutting down RAG service.")


app = FastAPI(
    title="RAG Service",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    candidate_ids: List[str] = Field(..., min_length=1, max_length=1000)


class SourceInfo(BaseModel):
    id: str
    source: str
    title: str
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    document_id: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]


class HybridChunkResult(BaseModel):
    """Result from hybrid search with all scores"""
    chunkId: str
    content: str
    source: str
    title: str
    author: Optional[str] = None
    pageNumber: Optional[int] = None
    chunkIndex: Optional[int] = None
    bm25Score: float
    bm25Normalized: float
    cosineScore: float
    combinedScore: float
    explanation: str

    class Config:
        populate_by_name = True  # Allow both snake_case and camelCase


class SemanticSearchRequest(BaseModel):
    """Request from Lucene service for semantic search"""
    query: str = Field(..., min_length=1, max_length=500)  # Original search query for embedding
    bm25Scores: Dict[str, float] = Field(..., min_length=1, max_length=1000)  # IDs extracted from keys
    topK: int = Field(default=10, ge=1, le=100)
    bm25Weight: float = Field(default=0.3, ge=0.0, le=1.0)
    semanticWeight: float = Field(default=0.7, ge=0.0, le=1.0)

    class Config:
        populate_by_name = True  # Allow both snake_case and camelCase


class SemanticSearchResponse(BaseModel):
    """Response with top K results and timing info"""
    requestedIds: int
    topK: int
    results: List[HybridChunkResult]
    searchTimeMs: float

    class Config:
        populate_by_name = True  # Allow both snake_case and camelCase


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    log.info(
        "POST /ask — query=%r, candidates=%d",
        request.query[:80],
        len(request.candidate_ids),
    )

    chunks = retrieve(
        query=request.query,
        candidate_ids=request.candidate_ids,
        top_k=10,
    )

    result = await generate_answer(
        query=request.query,
        chunks=chunks,
    )

    return AskResponse(**result)


@app.post("/api/v1/semantic/search-by-ids", response_model=SemanticSearchResponse)
async def semantic_search_by_ids(request: SemanticSearchRequest) -> SemanticSearchResponse:
    """
    Semantic search endpoint for hybrid search from Lucene service.

    Receives 1000 chunk IDs from BM25 search along with their scores.
    Returns top 10 chunks with combined BM25 + semantic scores.

    Workflow:
    1. Receive 1000 chunk IDs with BM25 scores from Lucene
    2. Normalize BM25 scores from [0,20] → [0,1]
    3. Query Qdrant with those IDs to get cosine similarity scores
    4. Combine: (0.3×BM25_norm) + (0.7×Cosine) for each chunk
    5. Sort by combined score, return top 10
    """
    import time
    request_start = time.time()

    # Extract chunk IDs from bm25Scores keys
    chunk_ids = list(request.bm25Scores.keys())

    log.info(
        "\n📥 REQUEST RECEIVED: /api/v1/semantic/search-by-ids\n"
        "   Input: %d chunk IDs from Lucene with BM25 scores",
        len(chunk_ids),
    )

    # Verify weights sum to 1.0
    weight_sum = request.bm25Weight + request.semanticWeight
    if abs(weight_sum - 1.0) > 0.01:
        log.warning("Weights don't sum to 1.0: %.2f", weight_sum)

    # Call retriever with detailed timing
    retrieve_start = time.time()
    results = retrieve_with_bm25_scores(
        query=request.query,
        chunk_ids=chunk_ids,
        bm25_scores=request.bm25Scores,
        top_k=request.topK,
        bm25_weight=request.bm25Weight,
        semantic_weight=request.semanticWeight,
    )
    retrieve_time = (time.time() - retrieve_start) * 1000

    # Format response
    format_start = time.time()
    hybrid_results = [
        HybridChunkResult(
            chunkId=r["chunk_id"],
            content=r["content"],
            source=r["source"],
            title=r["title"],
            author=r.get("author"),
            pageNumber=r.get("page_number"),
            chunkIndex=r.get("chunk_index"),
            bm25Score=r["bm25_score"],
            bm25Normalized=r["bm25_normalized"],
            cosineScore=r["cosine_score"],
            combinedScore=r["combined_score"],
            explanation=r["explanation"],
        )
        for r in results
    ]
    format_time = (time.time() - format_start) * 1000

    total_time_ms = (time.time() - request_start) * 1000

    log.info(
        "\n📤 RESPONSE READY\n"
        "   ├─ Hybrid search logic: %.2f ms\n"
        "   ├─ Format results: %.2f ms\n"
        "   └─ TOTAL endpoint time: %.2f ms\n"
        "   Returning %d results (top %d from %d evaluated)",
        retrieve_time, format_time, total_time_ms, len(hybrid_results), request.topK, len(chunk_ids)
    )

    return SemanticSearchResponse(
        requestedIds=len(chunk_ids),
        topK=request.topK,
        results=hybrid_results,
        searchTimeMs=total_time_ms,
    )


@app.get("/health")
async def health() -> Dict[str, Any]:
    try:
        count = collection_count()
    except Exception:
        count = -1
    return {"status": "ok", "collection_points": count}
