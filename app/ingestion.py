from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Generator, List, Dict, Any

from qdrant_client.http.models import PointStruct

from app.config import log, settings
from app.embedding import embed_texts
from app.qdrant_store import ensure_collection, upsert_points, collection_count, to_qdrant_id


def resolve_json_sources(path: str) -> List[Path]:
    """Resolve the ingestion path to a list of JSON files.

    Accepts either:
      - A single .json file path
      - A directory containing one or more .json files (Lucene chunk-exports)

    Files are sorted by name so ingestion order is deterministic.
    """
    p = Path(path)

    if p.is_file() and p.suffix == ".json":
        return [p]

    if p.is_dir():
        files = sorted(p.glob("*.json"))
        if not files:
            log.error("No .json files found in directory: %s", path)
            sys.exit(1)
        return files

    log.error("Path is neither a .json file nor a directory: %s", path)
    sys.exit(1)


def stream_json_records(path: Path) -> Generator[Dict[str, Any], None, None]:
    """Stream JSON records one at a time from a single JSON file.

    Uses a streaming approach: reads tokens incrementally so the full
    file is never held in memory.  Falls back to newline-delimited JSON
    parsing for robustness.
    """
    if not path.exists():
        log.error("JSON file not found: %s", path)
        sys.exit(1)

    file_size_mb = path.stat().st_size / (1024 * 1024)
    log.info("Opening JSON file: %s (%.1f MB)", path.name, file_size_mb)

    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1).strip()
        f.seek(0)

        if first_char == "[":
            yield from _stream_json_array(f)
        else:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    log.warning("Skipping malformed line %d: %s", line_no, e)


def _stream_json_array(f) -> Generator[Dict[str, Any], None, None]:
    """Parse a JSON array without loading it entirely into memory.

    Uses json.JSONDecoder.raw_decode to pull one object at a time from
    a buffered read of the file.  Buffer is kept to a minimum — only
    the current incomplete object tail is retained between reads.
    """
    decoder = json.JSONDecoder()
    buffer = ""
    started = False

    for chunk in _read_chunks(f, chunk_size=64 * 1024):
        buffer += chunk

        while buffer:
            buffer = buffer.lstrip(" \t\n\r,")

            if not buffer:
                break

            if not started:
                if buffer[0] == "[":
                    buffer = buffer[1:]
                    started = True
                    continue
                else:
                    break

            if buffer[0] == "]":
                return

            try:
                obj, end_idx = decoder.raw_decode(buffer)
                yield obj
                buffer = buffer[end_idx:]
            except json.JSONDecodeError:
                break


def _read_chunks(f, chunk_size: int = 64 * 1024) -> Generator[str, None, None]:
    while True:
        data = f.read(chunk_size)
        if not data:
            break
        yield data


def _build_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata payload to store alongside the vector.

    Stores all essential metadata fields for response construction and context:
    - Document identification (chunk_id, document_id)
    - Content (content)
    - Document metadata (source, title, author)
    - Positional metadata (page_number, chunk_index, chunk_position)
    - Statistics (token_count)
    """
    metadata = record.get("metadata", {})
    return {
        "chunk_id": record.get("id", ""),
        "content": record.get("content", ""),
        "source": metadata.get("source", ""),
        "title": metadata.get("title", ""),
        "author": metadata.get("author", ""),
        "page_number": metadata.get("page_number"),
        "chunk_index": metadata.get("chunk_index"),
        "chunk_position": metadata.get("chunk_position"),
        "token_count": metadata.get("token_count"),
        "document_id": record.get("document_id", ""),
    }


def run_ingestion() -> None:
    log.info("=== Starting ingestion pipeline ===")
    ensure_collection()

    existing = collection_count()
    log.info("Existing points in collection: %d", existing)

    json_files = resolve_json_sources(settings.ingestion_json_path)
    log.info("Found %d JSON file(s) to ingest.", len(json_files))

    embed_batch_size = settings.embedding_batch_size
    upsert_size = settings.upsert_batch_size

    record_buffer: List[Dict[str, Any]] = []
    total_ingested = 0
    total_skipped = 0

    for file_idx, json_file in enumerate(json_files, 1):
        log.info("Processing file %d/%d: %s", file_idx, len(json_files), json_file.name)

        for record in stream_json_records(json_file):
            record_id = record.get("id")
            if record_id is None:
                total_skipped += 1
                continue

            raw_content = record.get("content")
            if not isinstance(raw_content, str) or not raw_content.strip():
                total_skipped += 1
                continue

            record_buffer.append(record)

            if len(record_buffer) >= upsert_size:
                count = _process_batch(record_buffer, embed_batch_size)
                total_ingested += count
                record_buffer.clear()
                gc.collect()
                log.info("Ingested %d points so far.", total_ingested)

    if record_buffer:
        count = _process_batch(record_buffer, embed_batch_size)
        total_ingested += count
        record_buffer.clear()
        gc.collect()

    log.info(
        "=== Ingestion complete: %d ingested, %d skipped, %d files processed ===",
        total_ingested,
        total_skipped,
        len(json_files),
    )


def _process_batch(
    records: List[Dict[str, Any]],
    embed_batch_size: int,
) -> int:
    """Embed and upsert a batch of records.

    Embeddings are computed in sub-batches (size = embedding_batch_size)
    to cap peak memory.  gc.collect() is called after each sub-batch and
    after the full upsert to release memory promptly.
    """
    texts = [str(r["content"]) for r in records]
    raw_ids = [r["id"] for r in records]
    ids = [to_qdrant_id(rid) for rid in raw_ids]
    payloads = [_build_payload(r) for r in records]

    all_vectors: List[List[float]] = []
    total_sub = (len(texts) + embed_batch_size - 1) // embed_batch_size
    for i in range(0, len(texts), embed_batch_size):
        sub = texts[i : i + embed_batch_size]
        sub_idx = i // embed_batch_size + 1
        log.info("  Embedding sub-batch %d/%d (%d texts)...", sub_idx, total_sub, len(sub))
        vecs = embed_texts(sub, is_query=False)
        all_vectors.extend(vecs)
        del vecs
        gc.collect()

    points = [
        PointStruct(id=ids[j], vector=all_vectors[j], payload=payloads[j])
        for j in range(len(ids))
    ]

    log.info("  Upserting %d points to Qdrant...", len(points))
    upsert_points(points)

    del all_vectors
    del points
    gc.collect()

    return len(ids)


if __name__ == "__main__":
    run_ingestion()
