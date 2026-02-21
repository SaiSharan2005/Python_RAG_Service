"""
Qdrant vector store interface.

HNSW disabled intentionally due to 1GB RAM constraint. System relies on
Lucene pre-filtering and brute-force reranking over 1000 candidates.

Design invariants:
  - HnswConfigDiff(m=0) disables the HNSW graph entirely.
  - indexing_threshold=0 prevents the optimizer from building any index.
  - SearchParams(exact=True) forces brute-force scoring on every query.
  - No quantization, no payload indexing, no ANN, no hybrid search.
  - Payloads are stored on disk (on_disk_payload=True) to minimize RAM.
"""
from __future__ import annotations

import uuid
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    Filter,
    HasIdCondition,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    SearchParams,
    VectorParams,
)

from app.config import log, settings

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        if settings.qdrant_url:
            log.info("Using Qdrant Cloud: %s", settings.qdrant_url)
            _client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
                timeout=120,
            )
            log.info("Successfully connected to Qdrant Cloud!")
        else:
            log.warning("WARNING: QDRANT_URL not set! Falling back to localhost (this will fail if Qdrant is not running locally)")
            log.info("To use cloud Qdrant, set ENV=prod before running")
            _client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=120,
            )
            log.info(
                "Connected to Qdrant at %s:%s",
                settings.qdrant_host,
                settings.qdrant_port,
            )
    return _client


def to_qdrant_id(chunk_id: str) -> str:
    """Convert an arbitrary chunk ID string to a valid Qdrant UUID.

    Qdrant accepts only unsigned integers or valid UUIDs as point IDs.
    Lucene exports IDs like '5c4a9c97-..._p1_c0_088c5634' which are not
    valid UUIDs.  uuid5 produces a deterministic UUID from any string,
    so the same chunk_id always maps to the same Qdrant point ID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def ensure_collection() -> None:
    """Create the collection if it does not exist.

    HNSW is fully disabled (m=0). No ANN index is built.
    Optimizer indexing threshold is 0 — no automatic index construction.
    Payloads stored on disk to save RAM.
    No payload indexes are created.
    No quantization is applied.
    """
    client = get_client()
    name = settings.qdrant_collection

    collections = [c.name for c in client.get_collections().collections]
    if name in collections:
        log.info("Collection '%s' already exists.", name)
        return

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=settings.embedding_dimension,
            distance=Distance.COSINE,
        ),
        hnsw_config=HnswConfigDiff(m=0),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=0,
        ),
        on_disk_payload=True,
    )
    log.info("Created collection '%s' (HNSW disabled, on-disk payload).", name)


def upsert_points(points: List[PointStruct]) -> None:
    client = get_client()
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
        wait=True,
    )


def collection_count() -> int:
    client = get_client()
    info = client.get_collection(settings.qdrant_collection)
    return info.points_count or 0


def search_by_ids(
    query_vector: List[float],
    candidate_ids: List[str],
    top_k: int = 10,
) -> list:
    """Brute-force similarity search filtered to candidate_ids only.

    - exact=True forces exhaustive (non-ANN) scoring.
    - HasIdCondition restricts the search space to Lucene's pre-filtered IDs.
    - Vectors are NOT returned (with_vectors=False) to save bandwidth.
    - Candidate chunks are never re-embedded; only the query is embedded.
    """
    client = get_client()
    qdrant_ids = [to_qdrant_id(cid) for cid in candidate_ids]

    results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                HasIdCondition(has_id=qdrant_ids),
            ]
        ),
        search_params=SearchParams(
            exact=True,
        ),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    return results
