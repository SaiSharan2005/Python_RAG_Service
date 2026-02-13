"""
Production RAG — Cloud Ingestion Pipeline
==========================================

Embeds chunked documents exported by the Lucene service and stores
the resulting vectors in Qdrant Cloud.  Designed to run on Google
Colab with GPU acceleration, but works on any CUDA or CPU machine.

Architecture Context
--------------------
This is the offline indexing step for a two-service RAG system:

    ┌────────────────┐      ┌──────────────────┐
    │  Lucene Service │─────▶│  Qdrant (vectors) │
    │  (BM25 search)  │      │  (cosine rerank)  │
    └────────────────┘      └──────────────────┘
           │                         │
           └──── candidate IDs ──────┘
                        │
                   RAG Service
                   (embed query → rerank → LLM)

The Lucene service exports chunked documents as JSON files.
This script reads those files, embeds the text content using
BAAI/bge-small-en (384 dimensions), and upserts the vectors
into a Qdrant Cloud collection with HNSW fully disabled.

Design Decisions
----------------
- HNSW disabled (m=0): The deployed server has only 1 GB RAM.
  Qdrant uses brute-force scoring (exact=True) over ~1000
  candidates pre-filtered by Lucene.
- Deterministic UUID5 IDs: Lucene exports non-UUID chunk IDs.
  uuid5(NAMESPACE_URL, chunk_id) maps them deterministically
  so ingestion and query-time lookups stay in sync.
- On-disk payloads: Minimizes RAM at the cost of slightly
  slower payload retrieval (acceptable for 1 GB constraint).
- No quantization: Brute-force over small candidate sets
  doesn't benefit from quantized vectors.

Usage
-----
Run the script directly — it auto-detects the data source::

    python run_ingestion_cloud.py

On Google Colab, it shows an interactive menu to upload files.
If ./chunk-exports/ already has JSON files, it uses them directly.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import sys
import time
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import torch
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    SearchParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

# ────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("ingestion")


# ────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IngestionConfig:
    """All tunable parameters for the ingestion pipeline.

    Frozen dataclass ensures configuration is immutable once created,
    preventing accidental mid-run changes.

    Attributes:
        qdrant_url:       Full HTTPS URL of the Qdrant Cloud cluster.
        qdrant_api_key:   JWT token for Qdrant Cloud authentication.
        collection_name:  Name of the Qdrant collection to populate.
        embedding_model:  HuggingFace model ID for sentence embeddings.
        embedding_dim:    Output dimension of the embedding model.
        embed_batch_size: Number of texts per model.encode() call.
                          GPU can handle 256+; reduce for CPU.
        upsert_batch_size: Number of points per Qdrant upsert call.
                           Higher = fewer network round-trips.
        json_dir:         Default directory for JSON chunk exports.
    """

    # Qdrant Cloud
    qdrant_url: str = (
        "https://b210317b-feb7-4514-89c0-44668fffeba0"
        ".eu-central-1-0.aws.cloud.qdrant.io:6333"
    )
    qdrant_api_key: str = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJhY2Nlc3MiOiJtIn0"
        ".MC3P9BZdG63yqfXKnG3udz5XAyS-wbqctc52fEcmYGk"
    )
    collection_name: str = "rag_chunks"

    # Embedding
    embedding_model: str = "BAAI/bge-small-en"
    embedding_dim: int = 384
    embed_batch_size: int = 256

    # Ingestion
    upsert_batch_size: int = 1000
    json_dir: str = "./chunk-exports"


# ────────────────────────────────────────────────────────────
# Latency Tracker
# ────────────────────────────────────────────────────────────

class LatencyTracker:
    """Accumulates wall-clock time per named pipeline stage.

    Supports both explicit start/stop and context-manager usage::

        tracker = LatencyTracker()

        with tracker.measure("embedding"):
            vectors = model.encode(texts)

        # Or explicit:
        tracker.start("upsert")
        client.upsert(...)
        tracker.stop()

    Call ``tracker.report()`` at the end to print a formatted
    breakdown of where time was spent.
    """

    def __init__(self) -> None:
        self._stages: Dict[str, float] = {}
        self._active_stage: Optional[str] = None
        self._stage_start: float = 0.0

    @contextmanager
    def measure(self, stage: str) -> Generator[None, None, None]:
        """Context manager that times a named stage."""
        self.start(stage)
        try:
            yield
        finally:
            self.stop()

    def start(self, stage: str) -> None:
        """Begin timing a named stage."""
        self._active_stage = stage
        self._stage_start = time.perf_counter()

    def stop(self) -> float:
        """Stop the current stage timer and accumulate elapsed time.

        Returns:
            Elapsed seconds for this interval.

        Raises:
            RuntimeError: If no stage is currently being timed.
        """
        if self._active_stage is None:
            raise RuntimeError("stop() called without a matching start()")

        elapsed = time.perf_counter() - self._stage_start
        self._stages[self._active_stage] = (
            self._stages.get(self._active_stage, 0.0) + elapsed
        )
        self._active_stage = None
        return elapsed

    def report(self) -> str:
        """Print and return a formatted latency breakdown."""
        total = sum(self._stages.values())
        lines = [
            "",
            "=" * 60,
            "LATENCY BREAKDOWN",
            "=" * 60,
        ]
        for stage, secs in self._stages.items():
            pct = (secs / total * 100) if total > 0 else 0.0
            lines.append(f"  {stage:<30s} {secs:>8.2f}s  ({pct:>5.1f}%)")
        lines.append("-" * 60)
        lines.append(f"  {'TOTAL':<30s} {total:>8.2f}s")
        lines.append(f"  {'TOTAL (minutes)':<30s} {total / 60:>8.2f}m")
        lines.append("=" * 60)

        report_text = "\n".join(lines)
        print(report_text)
        return report_text


# ────────────────────────────────────────────────────────────
# Text Processing
# ────────────────────────────────────────────────────────────

def sanitize_text(text: str) -> str:
    """Clean text content for the Rust tokenizer.

    The HuggingFace tokenizers library (Rust-backed) rejects inputs
    containing NUL bytes or certain control characters with a
    ``TypeError: TextEncodeInput must be Union[...]`` error.

    This function:
    1. Removes NUL bytes (\\x00) which can appear in PDF extractions.
    2. Replaces ASCII control characters (0x01–0x1F) with spaces,
       except for whitespace characters (\\n, \\r, \\t) which are kept.
    3. Strips leading/trailing whitespace.

    Args:
        text: Raw text content from the JSON chunk export.

    Returns:
        Cleaned text safe for tokenization.
    """
    # Fast path: most texts are clean
    text = text.replace("\x00", "")

    cleaned: list[str] = []
    for ch in text:
        cp = ord(ch)
        if cp < 32 and ch not in ("\n", "\r", "\t"):
            cleaned.append(" ")
        else:
            cleaned.append(ch)

    return "".join(cleaned).strip()


def to_qdrant_id(chunk_id: str) -> str:
    """Convert an arbitrary chunk ID string to a deterministic UUID.

    Qdrant accepts only unsigned integers or valid UUIDs as point IDs.
    Lucene exports IDs like ``5c4a9c97-..._p1_c0_088c5634`` which are
    not valid UUIDs.

    ``uuid5(NAMESPACE_URL, chunk_id)`` produces a deterministic UUID
    from any string, so the same chunk_id always maps to the same
    Qdrant point ID.

    IMPORTANT: The deployed rag-service (``qdrant_store.py``) uses the
    identical function so ingestion IDs and query-time lookups match.

    Args:
        chunk_id: The original Lucene chunk ID string.

    Returns:
        A valid UUID string for Qdrant.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


# ────────────────────────────────────────────────────────────
# JSON Loading & Validation
# ────────────────────────────────────────────────────────────

def resolve_json_sources(path: str) -> List[Path]:
    """Resolve a path to a list of JSON files to ingest.

    Accepts either:
    - A single ``.json`` file path
    - A directory containing one or more ``.json`` files

    Args:
        path: File or directory path.

    Returns:
        Sorted list of Path objects for each JSON file.

    Raises:
        FileNotFoundError: If the path doesn't exist or contains no
            JSON files.
    """
    p = Path(path)
    if p.is_file() and p.suffix == ".json":
        return [p]
    if p.is_dir():
        files = sorted(p.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No .json files found in: {path}")
        return files
    raise FileNotFoundError(
        f"Path is neither a .json file nor a directory: {path}"
    )


@dataclass
class LoadResult:
    """Result of loading and validating a single JSON file.

    Attributes:
        records:  Valid records ready for embedding.
        skipped:  Number of records filtered out during validation.
        errors:   List of error messages for records that failed.
    """

    records: List[Dict[str, Any]] = field(default_factory=list)
    skipped: int = 0
    errors: List[str] = field(default_factory=list)


def load_json_records(path: Path) -> LoadResult:
    """Load a JSON file and return validated records.

    Validation rules:
    - Record must have a non-null ``id`` field.
    - Record must have a ``content`` field that is a non-empty string.
    - Records failing validation are counted as skipped with a reason.

    The entire file is loaded with ``json.load()`` instead of streaming.
    This is safe on Colab (12+ GB RAM) and avoids subtle text corruption
    that was observed with chunked streaming parsers.

    Args:
        path: Path to a JSON file containing a list of chunk records.

    Returns:
        LoadResult with validated records and skip counts.

    Raises:
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the top-level structure is not a list.
    """
    result = LoadResult()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array at top level, got {type(data).__name__} "
            f"in {path.name}"
        )

    for idx, rec in enumerate(data):
        # Validate: must be a dict
        if not isinstance(rec, dict):
            result.skipped += 1
            result.errors.append(f"Record {idx}: not a dict ({type(rec).__name__})")
            continue

        # Validate: must have an id
        rid = rec.get("id")
        if rid is None:
            result.skipped += 1
            result.errors.append(f"Record {idx}: missing 'id' field")
            continue

        # Validate: content must be a non-empty string
        content = rec.get("content")
        if not isinstance(content, str):
            result.skipped += 1
            result.errors.append(
                f"Record {idx} (id={rid}): content is {type(content).__name__}, "
                f"expected str"
            )
            continue

        if not content.strip():
            result.skipped += 1
            result.errors.append(f"Record {idx} (id={rid}): content is empty/whitespace")
            continue

        result.records.append(rec)

    return result


def build_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the payload to store alongside each vector in Qdrant.

    Only fields needed by the RAG service at query time are kept.
    The original ``chunk_id`` is stored so the API returns it instead
    of the Qdrant-internal UUID.

    Args:
        record: A validated chunk record from the JSON export.

    Returns:
        Dict with the fields the RAG service expects in each payload.
    """
    metadata = record.get("metadata") or {}
    return {
        "chunk_id": record.get("id", ""),
        "content": record.get("content", ""),
        "source": metadata.get("source", ""),
        "title": metadata.get("title", ""),
        "page_number": metadata.get("page_number"),
        "chunk_index": metadata.get("chunk_index"),
        "document_id": record.get("document_id", ""),
    }


# ────────────────────────────────────────────────────────────
# Model & Qdrant Setup
# ────────────────────────────────────────────────────────────

def load_model(
    model_name: str,
    device: str,
    tracker: LatencyTracker,
) -> SentenceTransformer:
    """Load the sentence-transformer embedding model.

    The model is set to eval mode to disable dropout, ensuring
    deterministic embeddings for the same input.

    Args:
        model_name: HuggingFace model identifier.
        device: ``"cuda"`` or ``"cpu"``.
        tracker: Latency tracker for timing.

    Returns:
        Loaded SentenceTransformer model on the specified device.
    """
    with tracker.measure("model_loading"):
        log.info("Loading embedding model: %s on %s ...", model_name, device)
        model = SentenceTransformer(model_name, device=device)
        model.eval()

    log.info(
        "Model loaded (max_seq_length=%d, device=%s)",
        model.max_seq_length,
        device,
    )
    return model


def connect_qdrant(
    url: str,
    api_key: str,
    tracker: LatencyTracker,
) -> QdrantClient:
    """Connect to Qdrant Cloud and verify the connection.

    Args:
        url: Full HTTPS URL of the Qdrant Cloud cluster.
        api_key: JWT authentication token.
        tracker: Latency tracker for timing.

    Returns:
        Connected QdrantClient instance.

    Raises:
        ConnectionError: If the Qdrant cluster is unreachable.
    """
    with tracker.measure("qdrant_connect"):
        log.info("Connecting to Qdrant Cloud ...")
        client = QdrantClient(url=url, api_key=api_key, timeout=120)
        # Verify the connection by listing collections
        collections = [c.name for c in client.get_collections().collections]

    log.info("Connected. Existing collections: %s", collections)
    return client


def ensure_collection(
    client: QdrantClient,
    name: str,
    dim: int,
    tracker: LatencyTracker,
) -> None:
    """Create the vector collection if it does not already exist.

    Collection configuration is optimized for the 1 GB RAM constraint
    of the deployed Qdrant instance:

    - ``HnswConfigDiff(m=0)``: Disables the HNSW graph entirely.
      No ANN index is built, saving significant RAM.
    - ``indexing_threshold=0``: Prevents the optimizer from
      automatically building any index.
    - ``on_disk_payload=True``: Payloads stored on disk to
      minimize RAM usage.
    - ``Distance.COSINE``: Cosine similarity scoring for
      normalized BGE embeddings.

    Args:
        client: Connected QdrantClient.
        name: Collection name.
        dim: Vector dimensionality (384 for bge-small-en).
        tracker: Latency tracker for timing.
    """
    with tracker.measure("collection_setup"):
        existing = [c.name for c in client.get_collections().collections]
        if name in existing:
            info = client.get_collection(name)
            log.info(
                "Collection '%s' already exists (%d points)",
                name,
                info.points_count or 0,
            )
            return

        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=0),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
            on_disk_payload=True,
        )

    log.info("Created collection '%s' (HNSW disabled, on-disk payload)", name)


# ────────────────────────────────────────────────────────────
# Embedding
# ────────────────────────────────────────────────────────────

def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 256,
) -> List[List[float]]:
    """Embed a batch of texts into normalized float vectors.

    Uses ``torch.no_grad()`` to disable gradient computation and
    save GPU memory. Embeddings are L2-normalized for cosine
    similarity. GPU cache is cleared after encoding to prevent
    OOM on large ingestion runs.

    Args:
        model: Loaded SentenceTransformer.
        texts: List of text strings to embed.
        batch_size: Internal encode batch size.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    with torch.no_grad():
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    vectors = embeddings.tolist()

    # Free GPU memory
    del embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vectors


# ────────────────────────────────────────────────────────────
# Ingestion Pipeline
# ────────────────────────────────────────────────────────────

@dataclass
class IngestionStats:
    """Aggregated statistics from the ingestion run.

    Attributes:
        total_ingested: Number of chunks successfully embedded and upserted.
        total_skipped:  Number of chunks skipped due to validation failures.
        files_processed: Number of JSON files processed.
        total_size_mb:   Total size of all JSON files in megabytes.
        elapsed_seconds: Wall-clock time for the full pipeline.
        validation_errors: Aggregated error messages from all files.
    """

    total_ingested: int = 0
    total_skipped: int = 0
    files_processed: int = 0
    total_size_mb: float = 0.0
    elapsed_seconds: float = 0.0
    validation_errors: List[str] = field(default_factory=list)

    @property
    def elapsed_minutes(self) -> float:
        return self.elapsed_seconds / 60

    @property
    def throughput(self) -> float:
        """Chunks per second."""
        if self.elapsed_seconds > 0:
            return self.total_ingested / self.elapsed_seconds
        return 0.0

    def report(self) -> str:
        """Format and print ingestion statistics."""
        lines = [
            "",
            "=" * 60,
            "INGESTION SUMMARY",
            "=" * 60,
            f"  Files processed      : {self.files_processed}",
            f"  Total data size      : {self.total_size_mb:.1f} MB",
            f"  Chunks ingested      : {self.total_ingested}",
            f"  Chunks skipped       : {self.total_skipped}",
            f"  Elapsed time         : {self.elapsed_seconds:.1f}s "
            f"({self.elapsed_minutes:.1f} min)",
            f"  Throughput           : {self.throughput:.1f} chunks/sec",
        ]
        if self.validation_errors:
            lines.append(f"  Validation warnings  : {len(self.validation_errors)}")
            # Show first 10 errors at most
            for err in self.validation_errors[:10]:
                lines.append(f"    - {err}")
            if len(self.validation_errors) > 10:
                lines.append(
                    f"    ... and {len(self.validation_errors) - 10} more"
                )
        lines.append("=" * 60)

        report_text = "\n".join(lines)
        print(report_text)
        return report_text


def _process_and_upsert(
    records: List[Dict[str, Any]],
    model: SentenceTransformer,
    client: QdrantClient,
    collection: str,
    embed_batch_size: int,
    tracker: LatencyTracker,
) -> int:
    """Embed a batch of records and upsert them to Qdrant.

    Pipeline for each batch:
    1. Sanitize text content (remove control chars for tokenizer).
    2. Convert chunk IDs to deterministic UUIDs.
    3. Build payloads with metadata.
    4. Embed texts in sub-batches on GPU.
    5. Construct PointStruct objects and upsert to Qdrant.

    Args:
        records: Validated chunk records to process.
        model: Loaded embedding model.
        client: Connected QdrantClient.
        collection: Target collection name.
        embed_batch_size: Sub-batch size for embedding.
        tracker: Latency tracker.

    Returns:
        Number of points successfully upserted.
    """
    # Prepare data
    texts = [sanitize_text(str(r["content"])) for r in records]
    raw_ids = [r["id"] for r in records]
    qdrant_ids = [to_qdrant_id(rid) for rid in raw_ids]
    payloads = [build_payload(r) for r in records]

    # Embed in sub-batches to control GPU memory
    with tracker.measure("embedding"):
        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), embed_batch_size):
            sub_texts = texts[i : i + embed_batch_size]
            sub_vectors = embed_texts(model, sub_texts, batch_size=embed_batch_size)
            all_vectors.extend(sub_vectors)
            del sub_vectors

    # Build points and upsert to Qdrant
    with tracker.measure("qdrant_upsert"):
        points = [
            PointStruct(
                id=qdrant_ids[j],
                vector=all_vectors[j],
                payload=payloads[j],
            )
            for j in range(len(qdrant_ids))
        ]
        client.upsert(collection_name=collection, points=points, wait=True)

    # Free memory
    del all_vectors, points, texts, payloads
    gc.collect()

    return len(qdrant_ids)


def run_ingestion(
    config: IngestionConfig,
    model: SentenceTransformer,
    client: QdrantClient,
    tracker: LatencyTracker,
    json_path: Optional[str] = None,
) -> IngestionStats:
    """Run the full ingestion pipeline across all JSON files.

    For each file:
    1. Load and validate all records with ``json.load()``.
    2. Split into upsert-sized batches.
    3. Embed + upsert each batch.
    4. Track progress and accumulate statistics.

    Args:
        config: Pipeline configuration.
        model: Loaded embedding model.
        client: Connected QdrantClient.
        tracker: Latency tracker.
        json_path: Override path for JSON source (file or directory).
                   Defaults to ``config.json_dir``.

    Returns:
        IngestionStats with totals, timings, and any validation errors.
    """
    source = json_path or config.json_dir
    json_files = resolve_json_sources(source)

    stats = IngestionStats()
    stats.files_processed = len(json_files)
    stats.total_size_mb = sum(
        f.stat().st_size for f in json_files
    ) / (1024 * 1024)

    log.info(
        "Starting ingestion: %d file(s), %.1f MB total",
        len(json_files),
        stats.total_size_mb,
    )
    for f in json_files:
        log.info("  %s (%.1f MB)", f.name, f.stat().st_size / (1024 * 1024))

    pipeline_start = time.perf_counter()

    for file_idx, json_file in enumerate(json_files, 1):
        file_size_mb = json_file.stat().st_size / (1024 * 1024)
        log.info(
            "[%d/%d] Processing: %s (%.1f MB)",
            file_idx,
            len(json_files),
            json_file.name,
            file_size_mb,
        )

        # Load and validate
        with tracker.measure("json_loading"):
            load_result = load_json_records(json_file)

        records = load_result.records
        stats.total_skipped += load_result.skipped
        stats.validation_errors.extend(load_result.errors)

        log.info(
            "  Loaded %d valid records (skipped %d)",
            len(records),
            load_result.skipped,
        )

        if load_result.errors:
            log.warning(
                "  %d validation warning(s) — first: %s",
                len(load_result.errors),
                load_result.errors[0],
            )

        # Process in upsert-sized batches
        total_batches = (
            (len(records) + config.upsert_batch_size - 1)
            // config.upsert_batch_size
        )
        for batch_idx, i in enumerate(
            range(0, len(records), config.upsert_batch_size), 1
        ):
            batch = records[i : i + config.upsert_batch_size]
            log.info(
                "  Batch %d/%d: embedding + upserting %d chunks ...",
                batch_idx,
                total_batches,
                len(batch),
            )

            count = _process_and_upsert(
                batch, model, client, config.collection_name,
                config.embed_batch_size, tracker,
            )
            stats.total_ingested += count

        # Free file-level memory
        del records, load_result
        gc.collect()

    stats.elapsed_seconds = time.perf_counter() - pipeline_start
    return stats


# ────────────────────────────────────────────────────────────
# Verification & Testing
# ────────────────────────────────────────────────────────────

def verify_collection(client: QdrantClient, collection: str) -> int:
    """Print collection metadata and a sample point.

    Used after ingestion to confirm vectors were stored correctly.

    Args:
        client: Connected QdrantClient.
        collection: Collection name to inspect.

    Returns:
        Total number of points in the collection.
    """
    info = client.get_collection(collection)

    print("\n" + "=" * 60)
    print("COLLECTION VERIFICATION")
    print("=" * 60)
    print(f"  Collection       : {collection}")
    print(f"  Total points     : {info.points_count}")
    print(f"  Vector dimension : {info.config.params.vectors.size}")
    print(f"  Distance metric  : {info.config.params.vectors.distance}")
    print(f"  HNSW m           : {info.config.hnsw_config.m}")
    print(f"  On-disk payload  : {info.config.params.on_disk_payload}")

    # Fetch one sample point to verify payload structure
    sample = client.scroll(
        collection_name=collection,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if sample[0]:
        point = sample[0][0]
        print(f"\n  Sample point ID  : {point.id}")
        print("  Sample payload   :")
        for key, val in point.payload.items():
            display_val = str(val)
            if len(display_val) > 80:
                display_val = display_val[:80] + "..."
            print(f"    {key}: {display_val}")

    print("=" * 60)
    return info.points_count or 0


def test_search(
    model: SentenceTransformer,
    client: QdrantClient,
    collection: str,
    tracker: LatencyTracker,
    query: str = "What is retrieval augmented generation?",
    top_k: int = 3,
) -> None:
    """Run a test similarity search to verify embeddings.

    Uses the BGE query prefix for proper asymmetric retrieval.
    Searches with ``exact=True`` (brute-force) to match the
    deployed service behavior.

    Args:
        model: Loaded embedding model.
        client: Connected QdrantClient.
        collection: Collection to search.
        tracker: Latency tracker.
        query: Test query string.
        top_k: Number of results to return.
    """
    bge_query_prefix = (
        "Represent this sentence for searching relevant passages: "
    )

    with tracker.measure("test_search"):
        # Embed the query
        with torch.no_grad():
            vector = model.encode(
                [bge_query_prefix + query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0].tolist()

        # Search with brute-force scoring
        results = client.search(
            collection_name=collection,
            query_vector=vector,
            search_params=SearchParams(exact=True),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

    print(f"\n{'=' * 60}")
    print(f"TEST SEARCH")
    print(f"{'=' * 60}")
    print(f"  Query: \"{query}\"")
    print(f"  Results: {len(results)}\n")

    for i, hit in enumerate(results, 1):
        payload = hit.payload
        content_preview = (
            payload.get("content", "")[:120].replace("\n", " ")
        )
        print(f"  [{i}] Score: {hit.score:.4f}")
        print(
            f"      Source: {payload.get('source', 'N/A')} | "
            f"Page: {payload.get('page_number', 'N/A')}"
        )
        print(f"      {content_preview}...")
        print()

    print("=" * 60)


# ────────────────────────────────────────────────────────────
# Data Source Handlers (Colab / Drive / URL / Local)
# ────────────────────────────────────────────────────────────

def _is_colab() -> bool:
    """Detect if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def upload_via_colab(dest_dir: str = "./chunk-exports") -> str:
    """Upload JSON files using the Colab file picker.

    Opens an interactive file picker dialog. All uploaded files
    are saved to ``dest_dir``. ZIP files are auto-extracted.

    Args:
        dest_dir: Directory to save uploaded files into.

    Returns:
        Path to the directory containing JSON files.
    """
    from google.colab import files  # type: ignore[import]

    os.makedirs(dest_dir, exist_ok=True)
    print("Select your JSON files (or a .zip containing them) ...")
    uploaded = files.upload()

    for name, data in uploaded.items():
        save_path = os.path.join(dest_dir, name)
        with open(save_path, "wb") as f:
            f.write(data)

        if name.endswith(".zip"):
            log.info("Extracting %s ...", name)
            with zipfile.ZipFile(save_path, "r") as zf:
                zf.extractall(dest_dir)
            os.remove(save_path)

    json_count = len(list(Path(dest_dir).glob("*.json")))
    log.info("Upload complete: %d JSON file(s) in %s", json_count, dest_dir)
    return dest_dir


def upload_via_drive(
    drive_path: str = "/content/drive/MyDrive/chunk-exports",
) -> str:
    """Mount Google Drive and use JSON files from there.

    The JSON files should already be uploaded to Google Drive
    before running this function.

    Args:
        drive_path: Path to the chunk-exports directory on Drive.

    Returns:
        Path to the directory containing JSON files.

    Raises:
        FileNotFoundError: If no JSON files exist at the Drive path.
    """
    from google.colab import drive  # type: ignore[import]

    drive.mount("/content/drive")
    json_count = len(list(Path(drive_path).glob("*.json")))

    if json_count == 0:
        raise FileNotFoundError(
            f"No JSON files found at {drive_path}. "
            f"Upload them to Google Drive first."
        )

    log.info("Google Drive mounted: %d JSON file(s) at %s", json_count, drive_path)
    return drive_path


def download_via_url(
    url: str,
    dest_dir: str = "./chunk-exports",
) -> str:
    """Download JSON files from a direct URL.

    Supports:
    - Direct ``.json`` file URLs
    - ``.zip`` file URLs (auto-extracted)
    - Google Drive shareable links (via ``gdown``)

    Args:
        url: URL pointing to a JSON or ZIP file.
        dest_dir: Directory to save downloaded files into.

    Returns:
        Path to the directory containing JSON files.
    """
    os.makedirs(dest_dir, exist_ok=True)

    if "drive.google.com" in url:
        # Google Drive links need gdown
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "gdown"],
            check=True,
        )
        import gdown  # type: ignore[import]

        log.info("Downloading from Google Drive ...")
        output = gdown.download(url, fuzzy=True, quiet=False)

        if output and output.endswith(".zip"):
            log.info("Extracting %s ...", output)
            with zipfile.ZipFile(output, "r") as zf:
                zf.extractall(dest_dir)
            os.remove(output)
        elif output:
            os.rename(output, os.path.join(dest_dir, os.path.basename(output)))
    else:
        # Direct URL — use wget
        filename = url.split("/")[-1].split("?")[0] or "download"
        save_path = os.path.join(dest_dir, filename)
        log.info("Downloading from %s ...", url)
        subprocess.run(["wget", "-q", "-O", save_path, url], check=True)

        if save_path.endswith(".zip"):
            log.info("Extracting %s ...", filename)
            with zipfile.ZipFile(save_path, "r") as zf:
                zf.extractall(dest_dir)
            os.remove(save_path)

    json_count = len(list(Path(dest_dir).glob("*.json")))
    log.info("Download complete: %d JSON file(s) in %s", json_count, dest_dir)
    return dest_dir


def resolve_data_source(default_dir: str) -> str:
    """Determine where to get JSON files from.

    Resolution order:
    1. If ``default_dir`` already contains JSON files, use them.
    2. Otherwise, show an interactive menu with upload options.

    Args:
        default_dir: Default directory to check for JSON files.

    Returns:
        Path to a directory (or file) containing JSON data.
    """
    # Check if files already exist locally
    default_path = Path(default_dir)
    if default_path.exists() and list(default_path.glob("*.json")):
        json_count = len(list(default_path.glob("*.json")))
        log.info("Found %d JSON file(s) in %s", json_count, default_dir)
        return default_dir

    # Show interactive menu
    is_colab = _is_colab()

    print("\n" + "=" * 60)
    print("DATA SOURCE — How to load the JSON chunk exports?")
    print("=" * 60)
    if is_colab:
        print("  [1] Upload files (Colab file picker)")
        print("  [2] Google Drive (mount and read)")
    print("  [3] Download from URL / Google Drive link")
    print("  [4] Enter a local path")
    print("=" * 60)

    choice = input("Enter choice: ").strip()

    if choice == "1" and is_colab:
        return upload_via_colab()
    elif choice == "2" and is_colab:
        drive_path = input(
            "Drive path [/content/drive/MyDrive/chunk-exports]: "
        ).strip()
        return upload_via_drive(
            drive_path or "/content/drive/MyDrive/chunk-exports"
        )
    elif choice == "3":
        url = input("Paste URL (zip or json): ").strip()
        return download_via_url(url)
    elif choice == "4":
        path = input("Enter local path: ").strip()
        if not Path(path).exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        return path
    else:
        log.warning("Invalid choice '%s'. Using default: %s", choice, default_dir)
        return default_dir


# ────────────────────────────────────────────────────────────
# Main Entry Point
# ────────────────────────────────────────────────────────────

def main() -> None:
    """Run the complete ingestion pipeline.

    Steps:
    1. Detect compute device (CUDA / CPU).
    2. Resolve the JSON data source (local, upload, or download).
    3. Load the embedding model.
    4. Connect to Qdrant Cloud.
    5. Create the collection (if needed).
    6. Ingest all JSON files (load → sanitize → embed → upsert).
    7. Verify the collection.
    8. Run a test search.
    9. Print latency breakdown.
    """
    config = IngestionConfig()
    tracker = LatencyTracker()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Banner ──────────────────────────────────────────────
    print("=" * 60)
    print("Production RAG — Cloud Ingestion Pipeline")
    print("=" * 60)
    print(f"  Device          : {device}")
    print(f"  PyTorch         : {torch.__version__}")
    if device == "cuda":
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
    print(f"  Embedding model : {config.embedding_model}")
    print(f"  Embed batch     : {config.embed_batch_size}")
    print(f"  Upsert batch    : {config.upsert_batch_size}")
    print(f"  Collection      : {config.collection_name}")
    print("=" * 60)

    # ── Step 1: Resolve data source ─────────────────────────
    json_path = resolve_data_source(config.json_dir)

    # ── Step 2: Load embedding model ────────────────────────
    model = load_model(config.embedding_model, device, tracker)

    # ── Step 3: Connect to Qdrant Cloud ─────────────────────
    client = connect_qdrant(config.qdrant_url, config.qdrant_api_key, tracker)

    # ── Step 4: Create collection ───────────────────────────
    ensure_collection(
        client, config.collection_name, config.embedding_dim, tracker,
    )

    # ── Step 5: Run ingestion ───────────────────────────────
    stats = run_ingestion(config, model, client, tracker, json_path)

    # ── Step 6: Print results ───────────────────────────────
    stats.report()

    # ── Step 7: Verify collection ───────────────────────────
    verify_collection(client, config.collection_name)

    # ── Step 8: Test search ─────────────────────────────────
    test_search(model, client, config.collection_name, tracker)

    # ── Step 9: Latency breakdown ───────────────────────────
    tracker.report()

    log.info("Done. Vectors are stored in Qdrant Cloud.")


if __name__ == "__main__":
    main()
