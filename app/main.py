from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.config import log, settings
from app.embedding import get_model
from app.qdrant_store import ensure_collection, collection_count
from app.retriever import retrieve
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


@app.get("/health")
async def health() -> Dict[str, Any]:
    try:
        count = collection_count()
    except Exception:
        count = -1
    return {"status": "ok", "collection_points": count}
