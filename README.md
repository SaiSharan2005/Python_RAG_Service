# Production RAG

Ask questions across **1,000,000+ document chunks**.
On a **1 GB RAM** server.
With **no GPU**.

And still get accurate, source-cited answers.

---

Most RAG systems assume you have GPUs and memory to burn.

This one assumes you don't.

It's a retrieval-augmented generation pipeline built to run under real-world infrastructure constraints — the kind where every megabyte matters and there's no room for bloated vector indexes.

```
User Question
      │
      ▼
  Lucene       1M+ chunks  →  1,000 candidates     (keyword, fast)
      │
      ▼
  Qdrant       1,000 candidates  →  10 best         (semantic, accurate)
      │
      ▼
  Claude       10 chunks  →  1 cited answer          (grounded, no hallucination)
      │
      ▼
  { answer + sources }
```

> Full technical reference — APIs, data formats, Qdrant internals, deployment — in **[TECHNICAL.md](TECHNICAL.md)**

---

## Who Is This For?

This project is for you if:

- You need RAG on **cheap infrastructure** — not a $200/month GPU server
- You care about **citations and grounding** — every answer traces back to a source page
- You want to understand how **large-scale retrieval works under constraints**
- You're building search over **research papers, legal documents, manuals, or reports**

If you want maximum throughput on big hardware — this isn't that.

If you want smart engineering under real limits — keep reading.

---

## Example

**Question:**

> *"What are the limitations of transformer models?"*

**Response:**

```json
{
  "answer": "According to the provided documents, transformer models face several
    key limitations. Their attention mechanism has quadratic complexity with respect
    to sequence length, making them computationally expensive for long documents
    (2601.16344v1.pdf, page 5). They also require large amounts of training data
    and struggle with hallucination when data is insufficient
    (2601.15953v1.pdf, page 12).",
  "sources": [
    {
      "source": "2601.16344v1.pdf",
      "title": "DSGym: A Holistic Framework",
      "page_number": 5,
      "score": 0.8734
    },
    {
      "source": "2601.15953v1.pdf",
      "title": "Scaling Laws for Neural Architectures",
      "page_number": 12,
      "score": 0.8156
    }
  ]
}
```

Every claim is backed by a filename and page number. If the answer isn't in the documents, the system says so — it doesn't guess.

---

## How It Works

The core idea:

> **Cheap filter first. Expensive reasoning later.**
>
> Never do expensive work on a huge dataset if you can shrink it first.

The system is two services working together:

### Stage 1 — The Document Engine (Java + Lucene)

You upload PDFs. The service extracts text, splits it into overlapping chunks (~400 tokens each, aligned to sentence boundaries), and indexes everything with Apache Lucene.

When a question comes in, Lucene uses BM25 keyword scoring to scan the **entire** collection and pull out the **top 1,000 most relevant chunk IDs**. This takes milliseconds — even over a million chunks — because that's exactly what inverted indexes are built for.

Think of it like a librarian who instantly pulls 1,000 possibly-relevant books from the shelves. Fast, but imprecise — she's matching words, not understanding meaning.

### Stage 2 — The Intelligence Layer (Python + Qdrant)

The Python service receives those 1,000 candidate IDs and a question. It:

1. **Embeds the question** into a 384-dimensional vector using `BAAI/bge-small-en`
2. **Searches Qdrant** — but only within those 1,000 candidates, using brute-force cosine similarity
3. **Picks the top 10** chunks that are most semantically relevant
4. **Builds a prompt** from those 10 chunks — with source filenames, titles, and page numbers
5. **Calls Claude** to generate a grounded, cited answer

Think of it like actually reading those 1,000 books and finding the 10 that truly answer your question. Slower, but it understands that "shortcomings of deep learning architectures" means the same as "neural network limitations" — even though the words are completely different.

### Stage 3 — The Answer

Claude reads the 10 best chunks and writes a clear answer. It's instructed to:
- Answer **only** from the provided context
- **Cite sources** with filenames and page numbers
- Say "Not found in provided documents" rather than hallucinate

The magic is in the combination. Neither service alone works under our constraints. Together, they turn 1M+ chunks into 10 precise answers on a server with 1 GB of RAM.

---

## Performance Snapshot

| Metric | Value |
|--------|-------|
| Chunks indexed | 1,000,000+ |
| Lucene keyword search | ~15 ms |
| Qdrant semantic rerank (1,000 vectors) | ~20 ms |
| Query embedding (CPU) | ~8 ms |
| Total latency (excluding LLM) | < 50 ms |
| RAM at query time | ~235 MB |
| Embedding model size | ~130 MB on disk |
| PyTorch CPU build | ~115 MB (vs ~2 GB for CUDA) |

These numbers are from real runs on commodity hardware. The LLM call (Claude API) adds 1–3 seconds depending on response length — that's the bottleneck, not the retrieval.

---

## Architecture

```
                         OFFLINE (your local machine)
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │   PDFs ──► Java/Lucene Service ──► chunk-exports/*.json          │
  │                                          │                       │
  │                                          ▼                       │
  │                                   run_ingestion.py               │
  │                                   (embed + store vectors)        │
  │                                          │                       │
  │                                          ▼                       │
  │                                   Qdrant (populated)             │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘

                        RUNTIME (deployed server, 1 GB RAM)
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │   User Question                                                  │
  │        │                                                         │
  │        ▼                                                         │
  │   Java/Lucene ──► top 1000 candidate IDs                         │
  │                          │                                       │
  │                          ▼                                       │
  │                   Python/FastAPI                                  │
  │                     ├── embed query                               │
  │                     ├── search Qdrant (only those 1000)           │
  │                     ├── pick top 10                               │
  │                     ├── build prompt with citations               │
  │                     ├── call Claude                               │
  │                     └── return answer + sources                   │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
```

**Why offline ingestion?** Embedding hundreds of thousands of chunks is CPU-heavy. The 1 GB server can't handle it. So we do it locally and push vectors to Qdrant ahead of time. The deployed server only ever embeds one query at a time — trivial.

---

## Design Philosophy

This project optimizes for:

- **Simplicity** over novelty
- **Deterministic behavior** over magic
- **Explainability** over black-box pipelines
- **Cheap infrastructure** over impressive demos
- **Correctness** over speed

Every design choice follows from the constraints. We didn't disable HNSW because we wanted to — we disabled it because 1 GB of RAM demanded it. And then we found that with Lucene pre-filtering, we didn't need it anyway.

---

## Why These Decisions

**Why Lucene + Qdrant instead of just one?**

Lucene alone misses meaning. "Neural network limitations" won't match "shortcomings of deep learning architectures." Same idea, different words.

Qdrant alone can't scan a million vectors on 1 GB RAM. The HNSW index graph would eat most of the memory budget. Brute-force over the full collection is too slow.

Together: Lucene handles scale (1M → 1K), Qdrant handles understanding (1K → 10). Each does what it's best at.

**Why is HNSW disabled?**

HNSW builds a multi-layer graph in memory to speed up approximate nearest neighbor search. For 1M vectors at 384 dimensions, that graph consumes hundreds of megabytes — most of our RAM budget.

But we're never searching the full collection. We're searching 1,000 pre-filtered candidates. Brute-force cosine similarity over 1,000 vectors takes ~20 ms. No index needed.

**Why two separate services?**

Java/Lucene is the gold standard for full-text search. Nothing in Python matches its BM25 implementation or indexing performance.

Python has the best ML ecosystem — sentence-transformers, PyTorch, and clean LLM API integration.

Keeping them separate means you can deploy, scale, and update them independently.

---

## What This Project Does NOT Do

No GPU required.
No HNSW index.
No approximate nearest neighbor.
No hybrid search.
No reranker models.
No FAISS. No Elasticsearch. No Redis. No Pinecone.
No re-chunking in Python.
No 10 GB RAM requirement.

Every omission is intentional. The constraints demanded it, and the two-stage architecture makes it work without them.

---

## Tech Stack

| Component | Technology | Why This |
|-----------|-----------|----------|
| Document engine | Java 17, Spring Boot, Lucene 9.11 | Best-in-class BM25, battle-tested inverted index |
| Vector store | Qdrant (Docker) | Lightweight, brute-force mode, on-disk payloads |
| Semantic search | Python 3.10+, FastAPI | Best ML ecosystem, async, clean API layer |
| Embeddings | `BAAI/bge-small-en` (384 dim) | Small, fast on CPU, solid quality for its size |
| Answers | Claude (Anthropic API) | Strong instruction-following, good at citing sources |
| ML runtime | PyTorch (CPU build) | 115 MB vs 2 GB for CUDA — no contest on a tiny server |

---

## Project Structure

```
production-rag/
│
├── lucene-service/                      # Java — PDF processing + keyword search
│   ├── src/main/java/.../
│   │   ├── controller/
│   │   │   ├── PdfIngestionController   # Upload PDFs, track jobs, delete docs
│   │   │   └── SearchController         # BM25 search with document filtering
│   │   ├── service/
│   │   │   ├── PdfIngestionService      # Text extraction via PDFBox
│   │   │   ├── ChunkingService          # Overlap chunking + sentence alignment
│   │   │   └── IngestionJobService      # Background job processing
│   │   └── lucene/
│   │       ├── LuceneIndexService       # Index management
│   │       ├── LuceneSearchService      # BM25 scoring + filtering
│   │       └── LuceneConfig             # Analyzer, stopwords, BM25 params
│   ├── chunk-exports/                   # JSON output (auto-generated)
│   └── pom.xml
│
├── rag-service/                         # Python — semantic reranking + answers
│   ├── app/
│   │   ├── config.py                    # All settings from one .env file
│   │   ├── embedding.py                 # BGE model, query prefix, batched encoding
│   │   ├── qdrant_store.py              # Collection setup, brute-force search
│   │   ├── ingestion.py                 # Streaming JSON → embed → upsert
│   │   ├── retriever.py                 # Query embed → filtered Qdrant search
│   │   ├── generator.py                 # Prompt builder + Claude API
│   │   └── main.py                      # FastAPI — /ask and /health
│   ├── run_ingestion.py                 # Offline CLI for populating Qdrant
│   ├── requirements.txt
│   ├── .env / .env.example
│   ├── README.md                        # ← You are here
│   └── TECHNICAL.md                     # API reference, internals, deployment
│
├── qdrant_storage/                      # Qdrant persistent data
└── Research-Paper-Downloder/            # arXiv paper downloader utility
```

---

## Quick Start

```bash
# 1. Start the vector database
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

# 2. Start the document engine (separate terminal)
cd lucene-service && mvn spring-boot:run

# 3. Feed it some PDFs
curl -X POST http://localhost:8080/api/v1/ingest/pdf -F "file=@paper.pdf"

# 4. Install Python dependencies (CPU-only torch first!)
cd rag-service
pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 5. Set your API key
cp .env.example .env
# Edit .env → set LLM_API_KEY=sk-ant-your-key-here

# 6. Embed all chunks and store in Qdrant (offline, one-time)
python run_ingestion.py

# 7. Start the query server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 8. Ask something
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is attention mechanism?", "candidate_ids": ["id1", "id2"]}'
```

Detailed steps with expected output in [TECHNICAL.md](TECHNICAL.md).

---

## Hardware Reality

| Constraint | What We Did |
|------------|-------------|
| **1 GB RAM** | Disabled HNSW (`m=0`), payloads on disk, streaming JSON ingestion, GC after every batch, model uses ~200 MB leaving room for everything else |
| **4 GB disk** | No graph index, no quantization tables, no snapshots, only 6 fields stored per chunk |
| **No GPU** | PyTorch CPU build (115 MB), BGE-small is fast on CPU, one query embedding is instant |
| **1M+ chunks** | Lucene shrinks the search space from 1M to 1K before Qdrant ever touches it |

---

## What's Next

For the full technical deep-dive, see **[TECHNICAL.md](TECHNICAL.md)**:

- Complete API reference — every endpoint, every field, every status code
- Chunk JSON format — exact schema from the Java export
- Qdrant internals — why `m=0`, why `indexing_threshold=0`, why `exact=True`
- Embedding pipeline — query prefix, normalization, batch mechanics
- Ingestion pipeline — streaming parser, memory management, multi-file support
- Prompt construction — the exact format Claude receives
- Full configuration reference — every environment variable
- Step-by-step deployment guide — with RAM breakdown
- Troubleshooting — the 7 most common issues and how to fix them
