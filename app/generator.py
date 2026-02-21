from __future__ import annotations

from typing import Any, Dict, List

import httpx

from app.config import log, settings
from app.retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a precise document-answering assistant. "
    "Answer the user's question using ONLY the provided context passages. "
    "If the answer cannot be found in the context, respond exactly with: "
    '"Not found in provided documents." '
    "Always cite the source filename for each piece of information you use."
)

NO_CONTEXT_ANSWER = "Not found in provided documents."


def build_context(chunks: List[RetrievedChunk]) -> str:
    parts: List[str] = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] Source: {c.source} | Title: {c.title} | Page: {c.page_number}"
        if c.author:
            header += f" | Author: {c.author}"
        parts.append(f"{header}\n{c.content}")
    return "\n\n---\n\n".join(parts)


def build_sources(chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    seen: set = set()
    for c in chunks:
        key = (c.source, c.page_number)
        if key in seen:
            continue
        seen.add(key)
        source_entry: Dict[str, Any] = {
            "id": c.id,
            "source": c.source,
            "title": c.title,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
            "document_id": c.document_id,
            "score": round(c.score, 4),
        }
        # Add optional metadata fields if available
        if c.author:
            source_entry["author"] = c.author
        if c.chunk_position:
            source_entry["chunk_position"] = c.chunk_position
        # token_count is critical for context window management in LLM calls
        if c.token_count:
            source_entry["token_count"] = c.token_count
        sources.append(source_entry)
    return sources


async def generate_answer(
    query: str,
    chunks: List[RetrievedChunk],
) -> Dict[str, Any]:
    """Build prompt from chunks and call LLM API."""
    if not chunks:
        return {"answer": NO_CONTEXT_ANSWER, "sources": []}

    context = build_context(chunks)
    user_message = (
        f"Context:\n{context}\n\n---\n\nQuestion: {query}\n\n"
        "Answer using only the context above. Cite sources."
    )

    log.info("Calling LLM: model=%s", settings.llm_model)

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                settings.llm_api_url,
                headers={
                    "x-api-key": settings.llm_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": settings.llm_model,
                    "max_tokens": settings.llm_max_tokens,
                    "system": SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": user_message},
                    ],
                },
            )
            response.raise_for_status()
            data = response.json()

        answer = data["content"][0]["text"]
    except httpx.HTTPStatusError as e:
        log.error("LLM API error: %s — %s", e.response.status_code, e.response.text)
        answer = NO_CONTEXT_ANSWER
    except Exception as e:
        log.error("LLM call failed: %s", e)
        answer = NO_CONTEXT_ANSWER

    return {
        "answer": answer,
        "sources": build_sources(chunks),
    }
