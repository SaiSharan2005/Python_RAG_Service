# 🚀 Python RAG Service

Retrieval-Augmented Generation for large document collections on **1 GB RAM** with **no GPU**.

Combines **Lucene** (keyword search) + **Qdrant** (semantic search) + **Claude** (answer generation) for accurate, cited answers.

---

## 📋 Quick Overview

```
User Question
     │
     ▼
Lucene Service (Java)     → Top 1,000 keyword matches (fast)
     │
     ▼
Qdrant (Vector DB)        → Top 10 semantic matches (accurate)
     │
     ▼
Python RAG Service        → Build prompt + call Claude
     │
     ▼
Answer with Sources       → "According to doc_xyz.pdf, page 5..."
```

---

## 📁 Directory Structure

```
Python_RAG_Service/
├── ingestion-rag-pipeline.ipynb     ⭐ Embed chunks & store in Qdrant
├── run_ingestion.py                 Command-line ingestion runner
├── run_ingestion_cloud.py           Google Colab optimized version
├── requirements.txt                 Python dependencies
├── app/                             FastAPI service code
│   ├── main.py                      Query API endpoints
│   ├── retriever.py                 Embed queries & search Qdrant
│   ├── generator.py                 Claude answer generation
│   ├── qdrant_store.py              Vector DB operations
│   ├── embedding.py                 BGE embedding model
│   └── config.py                    Configuration from .env
├── .env                             Your API keys (git-ignored)
├── .env.example                     Template for .env
├── .env.prod                        Production config
├── README.md                        (this file)
└── TECHNICAL.md                     Deep technical reference
```

---

## 🎯 What This Service Does

**Three-stage RAG pipeline**:

1. **Lucene** (Java) - Fast keyword search
   - Scans 1M+ chunks instantly
   - Returns top 1,000 candidates via BM25 scoring
   - Takes ~15 ms

2. **Qdrant** (Vector DB) - Semantic reranking
   - Embeds query using BAAI/bge-small-en
   - Searches ONLY the 1,000 candidates (brute-force)
   - Returns top 10 most relevant chunks
   - Takes ~20 ms

3. **Claude** (LLM) - Answer generation
   - Reads the 10 best chunks
   - Generates grounded, cited answer
   - References source PDF + page number
   - Takes 1-3 seconds

**Total latency** (excluding LLM): < 50 ms
**Total latency** (with Claude): 1-3 seconds

---

## 🔧 Setup & Installation

### **Step 1: Install Dependencies**

```bash
cd Python_RAG_Service

# Install CPU-only PyTorch (important!)
pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

**Why CPU-only PyTorch?**
- CPU build: 115 MB
- CUDA build: 2+ GB
- For a 1 GB server, CPU build is mandatory

### **Step 2: Configure Environment**

```bash
# Copy template
cp .env.example .env

# Edit .env
nano .env  # or your editor
```

**Required settings**:
```
LLM_API_KEY=sk-ant-xxxxxxx          # Your Claude API key
QDRANT_URL=http://localhost:6333    # Local Qdrant or cloud
QDRANT_API_KEY=                     # Empty for local, required for cloud
LUCENE_URL=http://localhost:8080    # Java service URL
```

### **Step 3: Start Qdrant** (if local)

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant:latest
```

### **Step 4: Start Lucene Service** (separate terminal)

```bash
cd lucene-service
mvn spring-boot:run
```

---

## 📥 Ingestion Pipeline

Two options to populate Qdrant with embeddings:

### **Option A: Google Colab (Recommended for Large Datasets)**

Use the Jupyter notebook with free T4 GPU:

```bash
# Upload to Colab
# Open: ingestion-rag-pipeline.ipynb

# Or use direct Python runner
python run_ingestion_cloud.py
```

**What it does**:
1. Loads JSON chunks from `lucene-service/chunk-exports/`
2. Embeds in batches of 256 on GPU
3. Upserts to Qdrant Cloud in batches of 1000
4. **Speed**: ~99 MB in 2-5 minutes

**Key features**:
- ✅ Text sanitization (removes control characters)
- ✅ Error handling (pinpoints problematic chunks)
- ✅ Memory cleanup (GC after each batch)
- ✅ Progress bar (live status)

### **Option B: Command Line (Local)**

```bash
# One-time ingestion of all chunks
python run_ingestion.py

# Or from the notebook
python -m jupyter nbconvert --to script ingestion-rag-pipeline.ipynb
python ingestion_rag_pipeline.py
```

---

## 🚀 Running the Service

### **Start the Query Server**

```bash
# Development (auto-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### **Test the API**

```bash
# Health check
curl http://localhost:8000/health

# Example query (requires candidate IDs from Lucene)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are transformers?",
    "candidate_ids": ["chunk_001", "chunk_002", "chunk_003"]
  }'

# Response:
{
  "answer": "According to the documents, transformers are...",
  "sources": [
    {
      "source": "2601.16344v1.pdf",
      "title": "DSGym Framework",
      "page_number": 5,
      "score": 0.873
    }
  ],
  "query_embedding_time_ms": 8,
  "search_time_ms": 18,
  "total_time_ms": 42
}
```

---

## 📊 Performance Specs

| Metric | Value |
|--------|-------|
| **Chunks indexed** | 1M+ |
| **Query embedding (CPU)** | ~8 ms |
| **Qdrant semantic search** | ~20 ms |
| **Total retrieval** | < 50 ms |
| **Claude answer generation** | 1-3 sec |
| **RAM usage at query time** | ~235 MB |
| **Model size on disk** | ~130 MB |
| **PyTorch build** | 115 MB (CPU) |

---

## 📚 Ingestion Pipeline Breakdown

### **ingestion-rag-pipeline.ipynb**

The main notebook for embedding and storing chunks:

**Step-by-step**:
1. **Install dependencies** - torch, sentence-transformers, qdrant-client
2. **Configuration** - Set Qdrant URL, API key, model name
3. **Upload JSON files** - From Lucene service exports
4. **Load embedding model** - BAAI/bge-small-en (384 dim)
5. **Create Qdrant collection** - HNSW disabled, brute-force only
6. **Sanitize text** - Remove control characters that break tokenizers
7. **Embed in batches** - 256 texts at a time on GPU
8. **Upsert to Qdrant** - 1000 points per batch
9. **Verify** - Check point count and sample payload
10. **Test search** - Run a quick similarity search

**Key features**:
- ✅ Works on Colab T4 GPU (~2-5 min for 44K chunks)
- ✅ Works on CPU (~30-60 min for 44K chunks)
- ✅ Handles large files (streaming, batch processing)
- ✅ Sanitizes text (removes NUL bytes, control chars)
- ✅ Detailed error messages (tells you exactly which text failed)
- ✅ Memory cleanup (GC + CUDA cache clearing)

**Configuration** (in notebook):
```python
QDRANT_URL = "https://your-cloud-instance"
QDRANT_API_KEY = "your-api-key"
EMBEDDING_MODEL = "BAAI/bge-small-en"
EMBEDDING_DIM = 384
EMBED_BATCH_SIZE = 256
UPSERT_BATCH_SIZE = 1000
JSON_DIR = "./chunk-exports"
```

---

## ⚙️ Configuration

All settings in `.env`:

```bash
# Claude API
LLM_API_KEY=sk-ant-xxxxxxx
LLM_MODEL=claude-3-5-sonnet-20241022

# Qdrant Vector DB
QDRANT_URL=https://your-instance.eu-central-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your-api-key
QDRANT_COLLECTION=rag_chunks

# Lucene Service
LUCENE_URL=http://localhost:8080

# Search parameters
TOP_K_SEMANTIC=10          # Final chunks for Claude
SIMILARITY_THRESHOLD=0.5   # Min similarity score

# Ingestion
EMBED_BATCH_SIZE=256
UPSERT_BATCH_SIZE=1000
MAX_CHUNK_TOKENS=512
```

For detailed reference, see **[TECHNICAL.md](TECHNICAL.md)**.

---

## 🔍 Workflow: End-to-End

### **Setup Phase** (One-time)

```bash
# 1. Start services
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
cd lucene-service && mvn spring-boot:run &

# 2. Upload PDFs to Lucene
curl -X POST http://localhost:8080/api/v1/ingest/pdf \
  -F "file=@paper1.pdf" \
  -F "file=@paper2.pdf"

# 3. Wait for Lucene to process & export JSON
# Check: lucene-service/chunk-exports/*.json

# 4. Embed chunks in Qdrant (on Colab or local)
python run_ingestion.py
# or upload ingestion-rag-pipeline.ipynb to Colab
```

### **Query Phase** (Runtime)

```bash
# 1. Start Python RAG service
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2. User asks a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is attention mechanism?"}'

# 3. Service:
#    a. Queries Lucene → gets 1000 candidate IDs
#    b. Embeds query using BGE-small-en
#    c. Searches Qdrant (only those 1000) → gets top 10
#    d. Builds prompt from top 10 chunks
#    e. Calls Claude
#    f. Returns answer + sources
```

---

## 💾 Data Flow

```
PDFs (lucene-service/*)
  │
  ▼
Lucene Service (Java)
  ├─ Extract text (PDFBox)
  ├─ Split into chunks (overlapping, ~400 tokens)
  ├─ Index with BM25
  └─ Export JSON → chunk-exports/
     │
     ▼
ingestion-rag-pipeline.ipynb (Colab)
  ├─ Load JSON files
  ├─ Sanitize text
  ├─ Embed with BGE-small-en
  ├─ Upsert to Qdrant
  └─ Done!
     │
     ▼
Qdrant Cloud (44,000+ vectors)
  │
  ▼
Python RAG Service (FastAPI)
  ├─ Receive query
  ├─ Query Lucene (1M → 1K)
  ├─ Embed query
  ├─ Search Qdrant (1K → 10)
  ├─ Build prompt
  ├─ Call Claude
  └─ Return answer + sources
```

---

## 🐛 Troubleshooting

### **Embedding fails with "invalid UTF-8"**

The notebook sanitizes text automatically:
```python
def sanitize_text(text: str) -> str:
    text = text.replace("\x00", "")  # Remove NUL bytes
    # Remove other control chars
    cleaned = [
        " " if ord(ch) < 32 and ch not in "\n\r\t" else ch
        for ch in text
    ]
    return "".join(cleaned).strip()
```

If still failing, the notebook tells you exactly which text broke:
```
BROKEN TEXT FOUND
Sub-index: 45
Length: 2341
Preview: "threshold 𝜕 = 10−2..."
Error: ...tokenizer error...
```

### **Qdrant connection refused**

Check if Qdrant is running:
```bash
curl http://localhost:6333/health
# Should return: {"status":"ok"}
```

If using cloud:
- Verify `QDRANT_URL` in `.env`
- Check `QDRANT_API_KEY` is set correctly
- Ensure IP is whitelisted (if cloud instance requires it)

### **Claude API errors**

Check your API key:
```bash
echo $LLM_API_KEY
# Should start with sk-ant-
```

### **High latency on queries**

- **Lucene slow?** Check if Java service has enough memory
- **Qdrant slow?** Verify brute-force search is enabled (`exact=True`)
- **Embedding slow?** Expected on CPU (~8 ms), normal

---

## 📖 Files Guide

| File | Purpose |
|------|---------|
| **ingestion-rag-pipeline.ipynb** | Main notebook: embed chunks, store in Qdrant |
| **run_ingestion.py** | CLI runner for ingestion (local) |
| **run_ingestion_cloud.py** | Optimized for Google Colab |
| **app/main.py** | FastAPI server - `/ask` endpoint |
| **app/retriever.py** | Query embedding + Qdrant search |
| **app/generator.py** | Claude answer generation |
| **app/qdrant_store.py** | Vector DB operations |
| **app/embedding.py** | BGE model loading + encoding |
| **requirements.txt** | Python dependencies |
| **README.md** | (you are here) |
| **TECHNICAL.md** | Full API reference & internals |

---

## 🎓 Key Concepts

### **Why Lucene + Qdrant?**

| Stage | Why |
|-------|-----|
| **Lucene** | Keyword search is instant at scale. BM25 is unbeatable for full-text search. Shrinks 1M → 1K in ~15ms. |
| **Qdrant** | Semantic search understands meaning. "Architecture limitations" matches "transformer shortcomings". But can't scale to 1M on 1 GB. |
| **Together** | Lucene filters, Qdrant understands. Best of both worlds. |

### **Why HNSW Disabled?**

HNSW is a graph index for fast approximate nearest neighbor search. It uses ~100-200 MB per million vectors.

On a 1 GB server, that's most of our memory budget.

But we're never searching the full collection. We're searching 1,000 pre-filtered candidates. Brute-force cosine similarity over 1K vectors takes only ~20 ms. No index needed.

### **Why BGE-small-en?**

- **Size**: 33M parameters, 384 dimensions
- **Speed**: Fast on CPU (~8 ms per query)
- **Quality**: Solid performance for its size
- **Model**: Specifically trained for semantic search (not generation)

---

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Install deps | `pip install -r requirements.txt` |
| Start Qdrant | `docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest` |
| Start Lucene | `cd lucene-service && mvn spring-boot:run` |
| Run ingestion | `python run_ingestion.py` or notebook |
| Start API | `uvicorn app.main:app --reload --port 8000` |
| Test API | `curl http://localhost:8000/health` |
| View logs | Check .env + FastAPI console output |
| Debug embedding | Run notebook cell-by-cell, watch for sanitization errors |

---

## 📚 More Info

**For technical details**, see **[TECHNICAL.md](TECHNICAL.md)**:
- Complete API reference
- Chunk JSON schema
- Qdrant configuration internals
- Embedding pipeline details
- Prompt construction
- Full deployment guide
- Troubleshooting (common issues + fixes)

---

## ✅ Status

- ✓ Ingestion pipeline (notebook + CLI)
- ✓ Query API (FastAPI)
- ✓ Qdrant integration (brute-force search)
- ✓ Claude integration (grounded answers)
- ✓ Citation tracking (sources per answer)
- ✓ CPU-only deployment (no GPU needed)
- ✓ Low memory footprint (1 GB server)
- ✓ Production-ready

**Ready to deploy!** 🚀
