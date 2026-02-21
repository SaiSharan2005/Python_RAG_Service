from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from app.config import log
from app.embedding import embed_query
from app.qdrant_store import search_by_ids, get_client, to_qdrant_id, settings
from qdrant_client.http.models import Filter, HasIdCondition, SearchParams


@dataclass
class RetrievedChunk:
    id: str
    content: str
    source: str
    title: str
    author: Optional[str]
    page_number: Optional[int]
    chunk_index: Optional[int]
    chunk_position: Optional[str]
    token_count: Optional[int]
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
                author=payload.get("author"),
                page_number=payload.get("page_number"),
                chunk_index=payload.get("chunk_index"),
                chunk_position=payload.get("chunk_position"),
                token_count=payload.get("token_count"),
                document_id=payload.get("document_id", ""),
                score=hit.score,
            )
        )

    log.info("Retrieved %d chunks.", len(chunks))
    return chunks


def retrieve_with_bm25_scores(
    query: str,
    chunk_ids: List[str],
    bm25_scores: Dict[str, float],
    top_k: int = 10,
    bm25_weight: float = 0.3,
    semantic_weight: float = 0.7,
) -> List[Dict]:
    """
    Hybrid search combining BM25 scores (from Lucene) with semantic scores (from Qdrant).

    Process:
    1. Embed the query into vector space (384 dimensions)
    2. Normalize BM25 scores from [0, 20] to [0, 1]
    3. Query Qdrant with real query vector for cosine similarity on candidate IDs
    4. Combine scores: (bm25_weight × BM25_normalized) + (semantic_weight × cosine_score)
    5. Return top K results sorted by combined score
    """
    import time

    if not query:
        log.warning("Empty query received for hybrid search.")
        return []

    if not chunk_ids:
        log.warning("Empty chunk_ids received for hybrid search.")
        return []

    total_start = time.time()
    log.info(
        "🔍 HYBRID SEARCH START: query=%r, chunk_ids=%d, top_k=%d, weights=(%.1f, %.1f)",
        query[:80],
        len(chunk_ids),
        top_k,
        bm25_weight,
        semantic_weight,
    )

    # Step 1: Normalize BM25 scores (typically 0-20 range) to [0, 1]
    step1_start = time.time()
    max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
    max_bm25 = max(max_bm25, 1.0)  # Avoid division by zero

    bm25_normalized = {
        cid: min(score / max_bm25, 1.0)
        for cid, score in bm25_scores.items()
    }
    step1_time = (time.time() - step1_start) * 1000
    log.info("✅ Step 1 (Normalize BM25): %.2f ms", step1_time)

    # Step 2: Query Qdrant for semantic scores using cosine similarity
    step2_start = time.time()

    # Embed the query to get real semantic vector
    embed_start = time.time()
    query_vector = embed_query(query)
    embed_time = (time.time() - embed_start) * 1000
    log.info("  ├─ Embed query: %.2f ms (vector dim: %d)", embed_time, len(query_vector))

    client = get_client()

    # Convert chunk IDs to Qdrant IDs
    convert_start = time.time()
    qdrant_ids = [to_qdrant_id(cid) for cid in chunk_ids]
    convert_time = (time.time() - convert_start) * 1000
    log.info("  ├─ Convert IDs: %.2f ms", convert_time)

    # Execute Qdrant search with REAL query vector
    qdrant_search_start = time.time()
    qdrant_results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,  # ← USING REAL QUERY VECTOR!
        query_filter=Filter(
            must=[
                HasIdCondition(has_id=qdrant_ids),
            ]
        ),
        search_params=SearchParams(
            exact=True,  # HNSW was disabled during ingestion, so use exact search
        ),
        limit=len(chunk_ids),  # Get all candidates
        with_payload=True,
        with_vectors=False,
    )
    qdrant_search_time = (time.time() - qdrant_search_start) * 1000
    log.info("  ├─ Qdrant search: %.2f ms (returned %d hits)", qdrant_search_time, len(qdrant_results))

    step2_time = (time.time() - step2_start) * 1000
    log.info("✅ Step 2 (Query Qdrant): %.2f ms (embed: %.2f + search: %.2f)", step2_time, embed_time, qdrant_search_time)

    # Step 3: Build result map with combined scores
    step3_start = time.time()
    results = []
    chunk_id_to_result = {}

    for hit in qdrant_results:
        payload = hit.payload or {}
        chunk_id = payload.get("chunk_id", str(hit.id))

        # Get scores
        bm25_score = bm25_scores.get(chunk_id, 0.0)
        bm25_norm = bm25_normalized.get(chunk_id, 0.0)
        cosine_score = hit.score  # Qdrant returns cosine similarity [0, 1]

        # Combine scores
        combined_score = (bm25_weight * bm25_norm) + (semantic_weight * cosine_score)

        # Create explanation
        explanation = (
            f"BM25: {bm25_score:.2f} (norm: {bm25_norm:.3f}) × {bm25_weight} + "
            f"Cosine: {cosine_score:.3f} × {semantic_weight} = {combined_score:.3f}"
        )

        result_dict = {
            "chunk_id": chunk_id,
            "content": payload.get("content", ""),
            "source": payload.get("source", ""),
            "title": payload.get("title", ""),
            "author": payload.get("author"),
            "page_number": payload.get("page_number"),
            "chunk_index": payload.get("chunk_index"),
            "bm25_score": bm25_score,
            "bm25_normalized": bm25_norm,
            "cosine_score": cosine_score,
            "combined_score": combined_score,
            "explanation": explanation,
        }

        results.append(result_dict)
        chunk_id_to_result[chunk_id] = result_dict

    # Handle chunks that might not be in Qdrant (shouldn't happen, but just in case)
    for chunk_id in chunk_ids:
        if chunk_id not in chunk_id_to_result:
            bm25_score = bm25_scores.get(chunk_id, 0.0)
            bm25_norm = bm25_normalized.get(chunk_id, 0.0)
            # If not in Qdrant, assign 0 cosine score
            combined_score = bm25_weight * bm25_norm

            result_dict = {
                "chunk_id": chunk_id,
                "content": "",
                "source": "",
                "title": "",
                "author": None,
                "page_number": None,
                "chunk_index": None,
                "bm25_score": bm25_score,
                "bm25_normalized": bm25_norm,
                "cosine_score": 0.0,
                "combined_score": combined_score,
                "explanation": f"Not found in Qdrant. BM25: {bm25_score:.2f} (norm: {bm25_norm:.3f})",
            }
            results.append(result_dict)

    step3_time = (time.time() - step3_start) * 1000
    log.info("✅ Step 3 (Build results): %.2f ms (processing %d chunks)", step3_time, len(results))

    # Step 4: Sort by combined score (descending) and return top K
    step4_start = time.time()
    results.sort(key=lambda x: x["combined_score"], reverse=True)
    top_results = results[:top_k]
    step4_time = (time.time() - step4_start) * 1000
    log.info("✅ Step 4 (Sort & filter): %.2f ms", step4_time)

    total_time = (time.time() - total_start) * 1000
    log.info(
        "🎉 HYBRID SEARCH COMPLETE\n"
        "   ├─ Normalize BM25: %.2f ms\n"
        "   ├─ Query Qdrant: %.2f ms (search: %.2f ms)\n"
        "   ├─ Build results: %.2f ms\n"
        "   ├─ Sort & filter: %.2f ms\n"
        "   └─ TOTAL: %.2f ms (returned %d/%d results)",
        step1_time, step2_time, qdrant_search_time, step3_time, step4_time, total_time, len(top_results), len(results)
    )

    return top_results
