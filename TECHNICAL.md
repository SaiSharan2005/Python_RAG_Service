# Technical Reference

Deep-dive technical documentation for the Production RAG system. For the high-level overview, see [README.md](README.md).

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup Guide](#setup-guide)
- [API Reference — Java Service](#api-reference--java-service)
- [API Reference — Python Service](#api-reference--python-service)
- [End-to-End Query Flow](#end-to-end-query-flow)
- [Chunk JSON Format](#chunk-json-format)
- [Qdrant Collection Internals](#qdrant-collection-internals)
- [Embedding Pipeline](#embedding-pipeline)
- [Ingestion Pipeline](#ingestion-pipeline)
- [Prompt Construction](#prompt-construction)
- [Configuration Reference](#configuration-reference)
- [Deployment Guide](#deployment-guide)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Dependency | Version | Required For |
|------------|---------|-------------|
| Java | 17+ | Lucene service |
| Maven | 3.8+ | Building the Java service |
| Python | 3.10+ | RAG service |
| Docker | Any recent | Running Qdrant |
| Anthropic API key | — | Claude answer generation |

---

## Setup Guide

### Step 1: Start Qdrant

```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
```

Verify it's running:

```bash
curl http://localhost:6333/collections
# Expected: {"result":{"collections":[]},"status":"ok","time":0.00001}
```

### Step 2: Start the Java Service

```bash
cd lucene-service
mvn spring-boot:run
```

The service starts on `http://localhost:8080`.

Verify:

```bash
curl http://localhost:8080/api/v1/ingest/health
# Expected: {"status":"UP"}
```

### Step 3: Upload and Process PDFs

```bash
# Single file
curl -X POST http://localhost:8080/api/v1/ingest/pdf \
  -F "file=@research-paper.pdf"

# Multiple files
curl -X POST http://localhost:8080/api/v1/ingest/pdf \
  -F "file=@paper1.pdf" \
  -F "file=@paper2.pdf" \
  -F "file=@paper3.pdf"
```

Response:

```json
{
  "jobId": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "PROCESSING",
  "message": "3 file(s) submitted for processing",
  "filesSubmitted": 3,
  "skippedFiles": null
}
```

Check progress:

```bash
curl http://localhost:8080/api/v1/ingest/status/f47ac10b-58cc-4372-a567-0e02b2c3d479
```

Once status is `COMPLETED`, the chunks are:
- Indexed in Lucene (searchable via BM25)
- Exported as JSON to `lucene-service/chunk-exports/`

### Step 4: Install Python Dependencies

```bash
cd rag-service

# IMPORTANT: Install CPU-only PyTorch first
pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Then install everything else
pip install -r requirements.txt
```

If you skip the first command, pip may pull a CUDA-enabled PyTorch build (~2 GB instead of ~115 MB).

### Step 5: Configure

```bash
cp .env.example .env
```

At minimum, set:

```
LLM_API_KEY=sk-ant-your-key-here
```

All other defaults work for local development. See [Configuration Reference](#configuration-reference) for every option.

### Step 6: Run Ingestion

```bash
python run_ingestion.py
```

Expected output:

```
2026-02-12 21:00:00 | INFO     | rag | JSON source: ../lucene-service/chunk-exports
2026-02-12 21:00:00 | INFO     | rag | Loading embedding model: BAAI/bge-small-en
2026-02-12 21:00:05 | INFO     | rag | Embedding model loaded.
2026-02-12 21:00:05 | INFO     | rag | Created collection 'rag_chunks' (HNSW disabled, on-disk payload).
2026-02-12 21:00:05 | INFO     | rag | Existing points in collection: 0
2026-02-12 21:00:05 | INFO     | rag | Found 11 JSON file(s) to ingest.
2026-02-12 21:00:05 | INFO     | rag | Processing file 1/11: 2026-02-12T20-47-19-454.json
2026-02-12 21:00:05 | INFO     | rag | Opening JSON file: 2026-02-12T20-47-19-454.json (8.9 MB)
2026-02-12 21:01:30 | INFO     | rag | Ingested 1000 points so far.
2026-02-12 21:03:00 | INFO     | rag | Ingested 2000 points so far.
...
2026-02-12 21:45:00 | INFO     | rag | === Ingestion complete: 50000 ingested, 0 skipped, 11 files processed ===
```

First run downloads the `BAAI/bge-small-en` model (~130 MB). Subsequent runs use the cached model.

Override the JSON path:

```bash
# Point to a specific directory
python run_ingestion.py --json-path ../lucene-service/chunk-exports

# Point to a single file
python run_ingestion.py --json-path ../lucene-service/chunk-exports/2026-02-12T20-47-19-454.json
```

### Step 7: Start the Query Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Expected startup output:

```
2026-02-12 22:00:00 | INFO     | rag | Starting RAG service...
2026-02-12 22:00:00 | INFO     | rag | Loading embedding model: BAAI/bge-small-en
2026-02-12 22:00:03 | INFO     | rag | Embedding model loaded.
2026-02-12 22:00:03 | INFO     | rag | Collection 'rag_chunks' already exists.
2026-02-12 22:00:03 | INFO     | rag | Qdrant collection 'rag_chunks' has 50000 points.
2026-02-12 22:00:03 | INFO     | rag | RAG service ready.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## API Reference — Java Service

Base URL: `http://localhost:8080`

### POST /api/v1/ingest/pdf

Upload PDF files for processing.

**Request:**

```bash
curl -X POST http://localhost:8080/api/v1/ingest/pdf \
  -F "file=@document.pdf"
```

**Response (202 Accepted):**

```json
{
  "jobId": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "PROCESSING",
  "message": "1 file(s) submitted for processing",
  "filesSubmitted": 1,
  "skippedFiles": null
}
```

**Duplicate handling:** If a file was already processed, it's skipped:

```json
{
  "status": "SUCCESS",
  "message": "All 2 file(s) already ingested",
  "filesSubmitted": 0,
  "skippedFiles": ["paper1.pdf", "paper2.pdf"]
}
```

### GET /api/v1/ingest/status/{jobId}

Check ingestion job progress.

**Response:**

```json
{
  "jobId": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "COMPLETED",
  "totalFiles": 3,
  "processedFiles": 3,
  "failedFiles": 0
}
```

### DELETE /api/v1/ingest/document/{documentId}

Delete a document and all its chunks from the index.

```bash
curl -X DELETE http://localhost:8080/api/v1/ingest/document/a51f219d-4cdb-4006-b42a-80f094ee55cb
```

### GET /api/v1/ingest/stats

```json
{
  "indexedChunks": 50000,
  "indexedPdfs": 95,
  "status": "healthy"
}
```

### GET /api/v1/search

Keyword search using BM25.

**Request:**

```bash
# GET with query params
curl "http://localhost:8080/api/v1/search?q=transformer+attention+mechanism&topK=10"

# Filter by document
curl "http://localhost:8080/api/v1/search?q=attention&topK=5&documentId=a51f219d-4cdb-4006-b42a-80f094ee55cb"
```

**Response:**

```json
{
  "query": "transformer attention mechanism",
  "totalHits": 342,
  "searchTimeMs": 12,
  "results": [
    {
      "chunkId": "a51f219d-4cdb-4006-b42a-80f094ee55cb_p5_c8_ea21961b",
      "documentId": "a51f219d-4cdb-4006-b42a-80f094ee55cb",
      "content": "The attention mechanism allows the model to...",
      "pageNumber": 5,
      "chunkIndex": 8,
      "score": 12.456
    }
  ]
}
```

### POST /api/v1/search

Same as GET, but with JSON body.

```bash
curl -X POST http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "attention mechanism", "topK": 1000}'
```

### GET /api/v1/search/chunk-stats

Returns statistics about the indexed chunks.

---

## API Reference — Python Service

Base URL: `http://localhost:8000`

### POST /ask

The primary endpoint. Takes a user question and candidate chunk IDs (from Lucene), returns a grounded answer with source citations.

**Request:**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the limitations of transformer models?",
    "candidate_ids": [
      "a51f219d-4cdb-4006-b42a-80f094ee55cb_p5_c8_ea21961b",
      "a51f219d-4cdb-4006-b42a-80f094ee55cb_p6_c9_bb33ff12",
      "b72e331a-1234-5678-9abc-def012345678_p12_c3_ff12ab09"
    ]
  }'
```

**Request fields:**

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `query` | string | 1–2000 chars | The user's question |
| `candidate_ids` | string[] | 1–1000 items | Chunk IDs from Lucene search |

**Response (200 OK):**

```json
{
  "answer": "According to the provided documents, transformer models have several limitations. The primary constraint is their quadratic attention complexity with respect to sequence length, making them computationally expensive for long documents (2601.16344v1.pdf, page 5). Additionally, they require large amounts of training data and are prone to hallucination when the training data is insufficient (2601.15953v1.pdf, page 12).",
  "sources": [
    {
      "id": "a51f219d-4cdb-4006-b42a-80f094ee55cb_p5_c8_ea21961b",
      "source": "2601.16344v1.pdf",
      "title": "DSGym: A Holistic Framework for Evaluating and Training Data Science Agents",
      "page_number": 5,
      "chunk_index": 8,
      "document_id": "a51f219d-4cdb-4006-b42a-80f094ee55cb",
      "score": 0.8734
    },
    {
      "id": "b72e331a-1234-5678-9abc-def012345678_p12_c3_ff12ab09",
      "source": "2601.15953v1.pdf",
      "title": "Scaling Laws for Neural Architectures",
      "page_number": 12,
      "chunk_index": 3,
      "document_id": "b72e331a-1234-5678-9abc-def012345678",
      "score": 0.8156
    }
  ]
}
```

**Response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | LLM-generated answer citing sources, or `"Not found in provided documents."` |
| `sources` | object[] | Deduplicated list of source chunks used |
| `sources[].id` | string | Chunk ID |
| `sources[].source` | string | PDF filename |
| `sources[].title` | string | Document title |
| `sources[].page_number` | int/null | Page number in the PDF |
| `sources[].chunk_index` | int/null | Chunk position within the document |
| `sources[].document_id` | string | Parent document UUID |
| `sources[].score` | float | Cosine similarity score (0–1) |

**When no answer is found:**

```json
{
  "answer": "Not found in provided documents.",
  "sources": []
}
```

### GET /health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "collection_points": 50000
}
```

Returns `-1` for `collection_points` if Qdrant is unreachable.

---

## End-to-End Query Flow

This is how a client application calls both services together:

```bash
# Step 1: Search Lucene for top 1000 keyword-relevant candidates
SEARCH_RESULT=$(curl -s -X POST http://localhost:8080/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer attention limitations", "topK": 1000}')

# Step 2: Extract chunk IDs
CANDIDATE_IDS=$(echo $SEARCH_RESULT | jq '[.results[].chunkId]')

# Step 3: Send to Python service for semantic reranking + answer
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"What are the limitations of transformer models?\",
    \"candidate_ids\": $CANDIDATE_IDS
  }"
```

What happens internally at Step 3:

1. **Query embedding** — The question is prefixed with `"Represent this sentence for searching relevant passages: "` and encoded into a 384-dim vector using `BAAI/bge-small-en`. Only the query is embedded; candidate chunks are not re-embedded.

2. **Filtered search** — Qdrant receives the query vector and a `HasIdCondition` filter containing the 1,000 candidate IDs. It computes brute-force cosine similarity (`exact=True`) against only those 1,000 vectors and returns the top 10.

3. **Prompt construction** — The top 10 chunks are formatted into a prompt with numbered headers showing source, title, and page number. The LLM system prompt instructs Claude to answer only from the provided context and cite sources.

4. **Answer generation** — The prompt is sent to the Claude API. The response is returned along with a deduplicated source list.

---

## Chunk JSON Format

The Java service exports chunks as JSON arrays. Each file contains one batch of chunks.

```json
[
  {
    "id": "a51f219d-4cdb-4006-b42a-80f094ee55cb_p1_c0_ea21961b",
    "content": "The full text content of this chunk...",
    "metadata": {
      "source": "2601.16344v1.pdf",
      "title": "DSGym: A Holistic Framework",
      "author": "Fan Nie; Junlin Wang; Harper Hua",
      "page_number": 1,
      "total_pages": 37,
      "chunk_index": 0,
      "chunk_position": "start",
      "token_count": 398,
      "created_at": "2026-02-12T15:30:30.121266600Z"
    },
    "document_id": "a51f219d-4cdb-4006-b42a-80f094ee55cb"
  }
]
```

**Field descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique chunk ID: `{documentId}_p{page}_c{index}_{hash}` |
| `content` | string | The chunk text |
| `metadata.source` | string | PDF filename |
| `metadata.title` | string | Document title (from PDF metadata) |
| `metadata.author` | string | Document author(s) |
| `metadata.page_number` | int | Primary page number (1-indexed) |
| `metadata.total_pages` | int | Total pages in the PDF |
| `metadata.chunk_index` | int | Chunk position within the document (0-indexed) |
| `metadata.chunk_position` | string | `"start"`, `"middle"`, or `"end"` |
| `metadata.token_count` | int | Approximate token count (whitespace split) |
| `metadata.created_at` | string | ISO 8601 timestamp |
| `document_id` | string | UUID of the parent document |

**What gets stored in Qdrant (minimal payload):**

Only 6 fields are stored to save disk:

| Stored Field | Source |
|-------------|--------|
| `content` | `record.content` |
| `source` | `metadata.source` |
| `title` | `metadata.title` |
| `page_number` | `metadata.page_number` |
| `chunk_index` | `metadata.chunk_index` |
| `document_id` | `record.document_id` |

Fields like `author`, `total_pages`, `chunk_position`, `token_count`, and `created_at` are intentionally dropped to minimize disk usage.

---

## Qdrant Collection Internals

The collection is created with these exact parameters:

```python
client.create_collection(
    collection_name="rag_chunks",
    vectors_config=VectorParams(
        size=384,                          # BGE-small-en dimension
        distance=Distance.COSINE,          # Cosine similarity
    ),
    hnsw_config=HnswConfigDiff(m=0),       # HNSW completely disabled
    optimizers_config=OptimizersConfigDiff(
        indexing_threshold=0,              # No automatic index building
    ),
    on_disk_payload=True,                  # Payloads stored on disk, not RAM
)
```

**Why `m=0`?**
The `m` parameter controls the number of edges per node in the HNSW graph. Setting it to 0 means no graph is built at all. No graph = no RAM consumed by the index = more room for vectors and the embedding model.

**Why `indexing_threshold=0`?**
This prevents Qdrant's optimizer from automatically building any index when the collection reaches a certain size. We want zero indexing overhead.

**Why `on_disk_payload=True`?**
Payloads (the chunk text, source, title, etc.) are stored on disk instead of in RAM. This trades slightly slower payload retrieval for significant RAM savings. Since we only retrieve payloads for the top 10 results, the disk read is negligible.

**Search is always brute-force:**

```python
results = client.search(
    collection_name="rag_chunks",
    query_vector=query_vector,
    query_filter=Filter(
        must=[HasIdCondition(has_id=candidate_ids)]
    ),
    search_params=SearchParams(exact=True),  # Forces exhaustive search
    limit=10,
    with_payload=True,
    with_vectors=False,                       # Don't return vectors (saves bandwidth)
)
```

`exact=True` forces Qdrant to compute cosine similarity against every matching vector instead of using any approximate index. Combined with the `HasIdCondition` filter (which limits the search to 1,000 IDs), this means we're doing brute-force scoring over exactly 1,000 vectors per query.

**What is NOT used:**
- No HNSW graph (`m=0`)
- No ANN (approximate nearest neighbor)
- No quantization (scalar, binary, or product)
- No payload indexes (no `create_payload_index` calls)
- No hybrid search
- No snapshots

---

## Embedding Pipeline

### Model

`BAAI/bge-small-en` — a 33M parameter model producing 384-dimensional normalized vectors.

- Download size: ~130 MB
- RAM when loaded: ~200 MB
- Device: CPU only (`device="cpu"`)
- Normalization: enabled (`normalize_embeddings=True`)
- Output: unit-length vectors (L2 norm = 1), suitable for cosine similarity

### Query Prefix

BGE models require a prefix for queries (but not for document chunks):

```
Represent this sentence for searching relevant passages: {query}
```

This prefix is added automatically in `embedding.py`. Document chunks are embedded without any prefix.

### Batching

- **Embedding batch size:** 128 (configurable via `EMBEDDING_BATCH_SIZE`)
- **During ingestion:** chunks are embedded in sub-batches of 128 within each upsert batch of 1,000
- **During query:** only 1 text is embedded (the question), so batching doesn't apply

---

## Ingestion Pipeline

The ingestion pipeline in `ingestion.py` processes exported JSON files and stores vectors in Qdrant.

### Memory Management

The pipeline is designed to never hold more than ~1,000 records in memory:

1. **JSON streaming** — Files are read in 64 KB chunks using `json.JSONDecoder.raw_decode`. The full file is never loaded into memory. Both JSON array (`[{...}, {...}]`) and newline-delimited JSON formats are supported.

2. **Record buffer** — Records accumulate in a buffer. When the buffer reaches `UPSERT_BATCH_SIZE` (default 1,000), the batch is processed and the buffer is cleared.

3. **Embedding sub-batches** — Within each batch of 1,000 records, embeddings are computed in sub-batches of `EMBEDDING_BATCH_SIZE` (default 128). Each sub-batch's vectors are extended into the result list and the intermediate tensors are deleted.

4. **Garbage collection** — `gc.collect()` is called after every embedding sub-batch and after every upsert. This ensures PyTorch and Python release memory promptly.

### Multi-File Support

The Lucene service exports one JSON file per ingestion batch. The ingestion pipeline auto-detects whether the path is a file or a directory:

- **Single file:** processes that file
- **Directory:** finds all `*.json` files, sorts by name, processes each sequentially

### Idempotency

Qdrant upserts are idempotent — re-inserting a point with the same ID overwrites the existing one. This makes ingestion safe to re-run after interruption.

### Pipeline Flow

```
resolve_json_sources(path)
  → List[Path]  (1 file or N files from directory)

For each file:
  stream_json_records(file)
    → yields records one at a time (64 KB streaming)

  For each record:
    validate (has id? has content?)
    append to record_buffer

    When buffer reaches 1,000:
      _process_batch(records, embed_batch_size)
        → embed in sub-batches of 128
        → build PointStruct list
        → upsert to Qdrant
        → gc.collect()
      clear buffer

Flush remaining buffer
Log totals
```

---

## Prompt Construction

The prompt sent to Claude is built from the top 10 retrieved chunks:

**System prompt:**

```
You are a precise document-answering assistant. Answer the user's question
using ONLY the provided context passages. If the answer cannot be found in
the context, respond exactly with: "Not found in provided documents." Always
cite the source filename for each piece of information you use.
```

**User message format:**

```
Context:
[1] Source: 2601.16344v1.pdf | Title: DSGym | Page: 5
The attention mechanism allows the model to focus on...

---

[2] Source: 2601.15953v1.pdf | Title: Scaling Laws | Page: 12
Transformer architectures face several challenges...

---

...up to 10 chunks...

---

Question: What are the limitations of transformer models?

Answer using only the context above. Cite sources.
```

**Rules enforced by the system prompt:**
- Answer ONLY from the provided chunks
- If the answer isn't in the context → return `"Not found in provided documents."`
- Always cite source filenames
- No hallucination

---

## Configuration Reference

### Python Service (.env)

All configuration is in a single `rag-service/.env` file.

| Variable | Default | Description |
|----------|---------|-------------|
| **Qdrant** | | |
| `QDRANT_HOST` | `localhost` | Qdrant server hostname or IP |
| `QDRANT_PORT` | `6333` | Qdrant HTTP API port |
| `QDRANT_COLLECTION` | `rag_chunks` | Qdrant collection name |
| **Embedding** | | |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en` | HuggingFace model ID |
| `EMBEDDING_DIMENSION` | `384` | Vector size (must match model output) |
| `EMBEDDING_BATCH_SIZE` | `128` | Texts per embedding batch |
| **Ingestion** | | |
| `INGESTION_JSON_PATH` | `../lucene-service/chunk-exports` | File or directory of JSON exports |
| `UPSERT_BATCH_SIZE` | `1000` | Points per Qdrant upsert call |
| **LLM** | | |
| `LLM_API_KEY` | *(none)* | Anthropic API key (required for `/ask`) |
| `LLM_API_URL` | `https://api.anthropic.com/v1/messages` | Claude API endpoint |
| `LLM_MODEL` | `claude-sonnet-4-5-20250929` | Model ID |
| `LLM_MAX_TOKENS` | `1024` | Max tokens in LLM response |
| **Server** | | |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Listen port |
| `LOG_LEVEL` | `info` | `debug`, `info`, `warning`, `error` |

### Java Service (application.yml)

| Setting | Default | Description |
|---------|---------|-------------|
| `chunking.chunk-size-tokens` | `400` | Target tokens per chunk |
| `chunking.chunk-overlap-tokens` | `50` | Overlap between consecutive chunks |
| `chunking.min-chunk-length-tokens` | `100` | Minimum chunk size (avoids tiny fragments) |
| `lucene.index-path` | `./lucene-index` | Lucene index directory |
| `lucene.bm25.k1` | `1.2` | BM25 term frequency saturation |
| `lucene.bm25.b` | `0.75` | BM25 document length normalization |
| `lucene.batch-commit-size` | `100` | Chunks per Lucene commit |
| `rag.export.enabled` | `true` | Whether to export chunks as JSON |
| `rag.export.path` | `./chunk-exports` | Export directory |

---

## Deployment Guide

### Deployment Model

```
LOCAL MACHINE                           DEPLOYED SERVER
─────────────                           ───────────────
Java service ─┐                         Python/FastAPI (query only)
              ├─► Qdrant (remote) ◄──── Qdrant (same instance)
Ingestion ────┘
```

Ingestion is offline-only. The deployed server runs only the query path.

### Phase 1: Offline — Populate Qdrant

Run everything locally. If Qdrant is on the server, point to it:

```bash
# In rag-service/.env
QDRANT_HOST=your-server-ip-or-hostname
QDRANT_PORT=6333
```

Then run ingestion from your local machine:

```bash
cd rag-service
python run_ingestion.py
```

This pushes vectors directly to the remote Qdrant. No file transfers needed.

### Phase 2: Deploy — Query Server Only

Deploy the `rag-service/` directory. The only process needed:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

**Environment variables to set on the platform:**

```
QDRANT_HOST=localhost          # or wherever Qdrant runs
QDRANT_PORT=6333
QDRANT_COLLECTION=rag_chunks
EMBEDDING_MODEL=BAAI/bge-small-en
EMBEDDING_DIMENSION=384
LLM_API_KEY=sk-ant-...
LLM_MODEL=claude-sonnet-4-5-20250929
LLM_MAX_TOKENS=1024
LOG_LEVEL=info
```

### Runtime Resource Usage

| Component | RAM Usage |
|-----------|----------|
| Embedding model (`bge-small-en`) | ~200 MB |
| FastAPI + uvicorn | ~30 MB |
| Per-query overhead | ~5 MB (single embedding + HTTP) |
| **Total** | **~235 MB** |

This leaves ~765 MB headroom on a 1 GB server. Qdrant runs as a separate container with its own memory allocation — vectors are stored on disk with `on_disk_payload=True`.

### What Does NOT Run on the Server

- Java/Lucene service (unless you also need runtime keyword search on the server)
- `run_ingestion.py`
- Bulk embedding
- PDF processing

---

## Troubleshooting

### Ingestion runs out of memory

**Symptom:** Process killed or `MemoryError` during `run_ingestion.py`.

**Fix:** Reduce batch sizes in `.env`:

```
EMBEDDING_BATCH_SIZE=64    # down from 128
UPSERT_BATCH_SIZE=500      # down from 1000
```

These control peak memory. Smaller batches = less RAM per batch = slower but safer.

### Qdrant connection refused

**Symptom:** `ConnectionRefusedError` when starting the server or running ingestion.

**Fix:**

```bash
# Is Docker running?
docker ps

# Is Qdrant healthy?
curl http://localhost:6333/collections

# If Qdrant container stopped:
docker start qdrant
```

Also verify `QDRANT_HOST` and `QDRANT_PORT` in `.env`.

### Embedding model fails to download

**Symptom:** Timeout or HTTP error during first run when downloading `BAAI/bge-small-en`.

**Fix:** Pre-download the model:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en')"
```

If behind a corporate proxy:

```bash
export HTTPS_PROXY=http://your-proxy:port
export HF_HUB_OFFLINE=0
```

### LLM returns "Not found in provided documents"

**Possible causes:**

1. **`LLM_API_KEY` not set** — Check `.env`. The key should start with `sk-ant-`.
2. **API error** — Check logs for `LLM API error: 401` (bad key) or `429` (rate limit).
3. **Candidate IDs don't exist in Qdrant** — Qdrant returns empty results if the IDs haven't been ingested. Verify with:
   ```bash
   curl http://localhost:6333/collections/rag_chunks
   ```
4. **Chunks genuinely don't answer the question** — The system is designed to say "not found" rather than hallucinate.

### torch installation pulls a 2 GB CUDA build

**Fix:** Always install PyTorch first with the CPU index:

```bash
pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

Then run `pip install -r requirements.txt`. The requirements file pins `torch==2.5.1+cpu` but pip may ignore the `+cpu` suffix without the explicit index URL.

### Ingestion is slow

Embedding on CPU is inherently slower than GPU. Typical throughput: ~50–100 chunks/second on a modern CPU. For 50,000 chunks, expect 8–15 minutes.

To speed up:
- Increase `EMBEDDING_BATCH_SIZE` to 256 (if you have enough RAM)
- Make sure no other heavy processes are running
- CPU with AVX2/AVX-512 support will be faster

### Server starts but /ask returns 422

**Symptom:** Validation error.

**Cause:** Request body doesn't match the schema. Check:
- `query` must be a non-empty string (1–2000 chars)
- `candidate_ids` must be a non-empty array of strings (1–1000 items)
- Content-Type header must be `application/json`

```bash
# Correct
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "candidate_ids": ["id1"]}'
```
