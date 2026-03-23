# Advanced RAG Pipeline (Hybrid Retrieval + Reranking)

A production-oriented Retrieval-Augmented Generation (RAG) system for educational document QA, with both Web UI and API interfaces.

This project combines:
- Dense semantic retrieval (Sentence Transformers + Qdrant)
- Lexical retrieval (BM25)
- Score fusion for hybrid ranking
- Cross-encoder reranking
- Multi-source ingestion (PDF and pre-chunked JSON)
- Pluggable generation layer (Gemini API or vLLM via proxy with fallback)

---

## 1. Overview

This system answers user questions using course materials stored in a vector database and then generates grounded responses with an LLM.

### What makes this RAG system advanced

Compared to a basic vector-only RAG setup, this implementation includes:

- Hybrid retrieval:
	- Dense search in Qdrant
	- BM25 lexical search over in-memory corpus
	- Weighted fusion (`alpha`, `beta`) after score normalization
- Cross-encoder reranking:
	- Second-stage reranker to improve final context quality
- Two ingestion modes:
	- Raw PDFs (extract + chunk)
	- Pre-processed chunk JSON files (preserve metadata, upsert by chunk ID)
- Runtime resilience:
	- LLM proxy with health-check, retry, and fallback to Gemini
	- Embedding/reranker fallback handling for model-load edge cases
- Operational optimizations:
	- Embedding model caching
	- LLM instance caching
	- Processor cache by `(subject, model)`
	- Streaming responses
- Conversation-aware prompting:
	- Includes short history window to improve continuity

---

## 2. Architecture

### High-level components

- Client layer
	- Streamlit app (`src/app_v2.py`)
	- FastAPI RAG API (`src/api_service.py`)
- RAG core
	- `RAGProcessor` (`src/processor.py`)
- LLM abstraction
	- `LLMManager` and adapters (`src/llm_manager.py`)
- LLM proxy (optional but recommended for vLLM)
	- FastAPI proxy (`src/llm_proxy.py`)
- Storage
	- Qdrant (vector store + payload metadata)
	- SQL DB for users/conversations/messages (outside `src`, used by UI)

### Query data flow

```text
User Query
	 |
	 v
RAGProcessor._retrieve_relevant_chunks()
	 |
	 +--> Dense retrieval (Qdrant cosine search)
	 |
	 +--> BM25 retrieval (in-memory lexical index built from Qdrant payloads)
	 |
	 +--> Hybrid score fusion (min-max normalized, weighted alpha/beta)
	 |
	 +--> Cross-encoder reranking
	 |
	 v
Top-N context chunks
	 |
	 v
Prompt construction (+ optional conversation history)
	 |
	 v
LLM generation (Gemini API or ProxyLLM)
	 |
	 v
Response (+ formatted source list)
```

### Indexing/ingestion flow

```text
reload_data()
	 |
	 +--> source_docs/<subject>/*.pdf
	 |      -> text extraction (PyMuPDF)
	 |      -> chunking (RecursiveCharacterTextSplitter)
	 |      -> embeddings
	 |      -> Qdrant upsert
	 |
	 +--> processed_chunks/<subject>/*.json
					-> normalize text/metadata/chunk_id
					-> embeddings
					-> Qdrant upsert (supports metadata updates)
```

---

## 3. Project Structure (`/src`)

```text
src/
├── __init__.py
├── api_service.py
├── app_v2.py
├── config.py
├── llm_manager.py
├── llm_proxy.py
└── processor.py
```

### Module responsibilities

- `src/processor.py`
	- Core RAG pipeline: ingestion, chunking, embedding, indexing, retrieval, rerank, prompting, generation
- `src/llm_manager.py`
	- LLM abstraction layer with multiple backends:
		- Gemini API
		- ProxyLLM (OpenAI-compatible endpoint)
	- Caches LLM instances
- `src/llm_proxy.py`
	- Lightweight proxy for vLLM serving:
		- Health check
		- Retry loop
		- Gemini fallback
		- Streaming passthrough
- `src/api_service.py`
	- FastAPI app exposing health/models/subjects/reload/query/stream endpoints
	- Caches `RAGProcessor` instances by `(subject, model)`
- `src/app_v2.py`
	- Streamlit UI with authentication, conversation list/history, model selector, subject selector, streaming chat
- `src/config.py`
	- Dataclass-based configuration loading from environment

---

## 4. Key Components

### Retriever

- Vector DB: Qdrant (`Distance.COSINE`)
- Embedding model: Sentence Transformers (default env value points to `BAAI/bge-m3` in processor)
- Dense retrieval:
	- Query embedding
	- Optional subject filter
	- Optional score threshold
- Lexical retrieval:
	- BM25 index built from Qdrant payload text
- Hybrid fusion:
	- Min-max normalize dense and BM25 scores
	- Weighted fusion using `RAG_HYBRID_ALPHA` and `RAG_HYBRID_BETA`

### Reranker

- CrossEncoder reranker (default `BAAI/bge-reranker-v2-m3`)
- Applied to hybrid candidates
- Returns final top-N context chunks

### LLM / Generator

- Through `LLMManager`:
	- `GeminiLLM` via `langchain_google_genai`
	- `ProxyLLM` via HTTP endpoint (OpenAI-style `/v1/chat/completions`)
- Supports:
	- Non-streaming and streaming generation
	- Model instance cache
- Proxy layer adds:
	- vLLM health checks
	- retries
	- Gemini fallback

### Chunking and Indexing Pipeline

- PDF ingestion
	- Extract text page-by-page
	- Chunk using `RecursiveCharacterTextSplitter`
	- Deterministic chunk IDs (`md5(subject/file/chunk/text_prefix)`)
- JSON ingestion
	- Supports list or wrapped payloads (`chunks`, `data`, `items`, `documents`)
	- Flexible text keys (`text`, `chunk`, `content`, `page_content`)
	- Preserves metadata and chunk IDs (`id`, `id_`, `metadata.chunk_id`)
- Upserts embeddings + payload to Qdrant

### Caching and Optimizations

- Embedding model is loaded once and reused
- Reranker is lazy-loaded
- LLM instances cached by model key
- API caches processor instances by `(subject, model)`
- BM25 index built in memory from Qdrant for fast lexical retrieval

---

## 5. Setup & Installation

### Prerequisites

- Python 3.10+ (3.11 recommended)
- Docker + Docker Compose (recommended for full stack)
- Qdrant available (local/container)
- Optional: PostgreSQL for persistent conversations
- Optional: Google API key for Gemini/fallback

### Installation

```bash
git clone <your-repo-url>
cd rag-pipeline

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

Create `.env` in project root:

```env
# Core
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=documents
LLM_MODEL=gemini
LLM_PROXY_URL=http://localhost:5000
GOOGLE_API_KEY=your_google_api_key

# DB (optional, Streamlit defaults to sqlite if unset in database layer)
DATABASE_URL=sqlite:///./chatbot.db
# or:
# DATABASE_URL=postgresql://chatbot_user:chatbot_pass_2024@localhost:5432/chatbot_db

# LLM params
LLM_TEMPERATURE=0.7
LLM_MAX_OUTPUT_TOKENS=2048

# Retrieval / embedding tuning (processor.py)
RAG_EMBEDDING_MODEL=BAAI/bge-m3
RAG_EMBEDDING_LOCAL_PATH=./models/bge-m3
RAG_HYBRID_TOP_K=20
RAG_DENSE_TOP_K=20
RAG_BM25_TOP_K=20
RAG_HYBRID_ALPHA=0.7
RAG_HYBRID_BETA=0.3
RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RAG_RERANK_TOP_N=5
RAG_RERANK_BATCH_SIZE=8
RAG_DENSE_SCORE_THRESHOLD=0.3
```

---

## 6. Usage

### Option A: Docker Compose (recommended)

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

Expected services:
- Streamlit app
- RAG API
- LLM proxy
- Qdrant
- PostgreSQL

### Option B: Run services manually

#### 1) Start Qdrant
Use Docker or your own Qdrant deployment.

#### 2) Start LLM proxy (optional for vLLM route)
```bash
python3 src/llm_proxy.py
```

#### 3) Start FastAPI RAG API
```bash
uvicorn src.api_service:app --host 0.0.0.0 --port 9100 --reload
```

#### 4) Start Streamlit UI
```bash
streamlit run src/app_v2.py
```

### Reload/index data

```bash
curl -X POST http://localhost:9100/reload \
	-H "Content-Type: application/json" \
	-d '{"clean": false}'
```

- `clean: true` recreates collection and reindexes all subjects.

### Query API

```bash
curl -X POST http://localhost:9100/query \
	-H "Content-Type: application/json" \
	-d '{
		"question": "Explain historical materialism",
		"subject": "Môn Triết học Mác-Lênin",
		"model_key": "gemini",
		"use_history": true
	}'
```

### Streaming API

```bash
curl -N -X POST http://localhost:9100/query/stream \
	-H "Content-Type: application/json" \
	-d '{
		"question": "Summarize chapter 3",
		"subject": null,
		"model_key": "qwen2-7b",
		"use_history": true
	}'
```

---

## 7. Configuration

Primary configuration is in `src/config.py`.

### Config classes

- `EmbeddingConfig`
	- model location, batch size, cache folder
- `QdrantConfig`
	- host, port, collection name
- `ChunkingConfig`
	- chunk size, overlap, separators
- `LLMConfig`
	- active model, generation params, proxy URL
- `RetrievalConfig`
	- top-k, score threshold, max context length
- `AppConfig`
	- data and log folders, UI title/icon

### Data layout conventions

The processor expects:

```text
data/
├── source_docs/
│   └── <subject>/*.pdf
└── processed_chunks/
		└── <subject>/*.json
```

JSON chunks may include rich metadata. The pipeline keeps metadata and stores normalized fields (subject/page/chunk_id/source markers) in Qdrant payload.

---

## 8. Advanced Features and Why They Help

- Hybrid dense + BM25 retrieval
	- Dense captures semantic similarity
	- BM25 captures exact keyword overlap
	- Fusion improves robustness across query types

- Cross-encoder reranking
	- Re-scores `(query, chunk)` pairs jointly
	- Usually improves precision for final context shown to LLM

- Model and processor caching
	- Reduces repeated initialization cost
	- Improves latency under repeated requests

- Fallback-capable proxy
	- Keeps service available when vLLM is down
	- Automatic retries reduce transient failure impact

- Incremental + full reindex modes
	- `clean=false` for updates
	- `clean=true` for full rebuilds after major model/schema changes

- Conversation-aware prompting
	- Better continuity in multi-turn QA without a heavy long-term memory layer

---

## Notes and Assumptions

- This codebase currently uses single-vector dense embeddings + BM25, not multi-vector retrieval.
- No explicit evaluation pipeline (e.g., benchmark scripts/metrics) is present in `src`.
- Main UI and API coexist; choose one or both depending on deployment needs.

---

## Quick Start Checklist

1. Configure `.env`
2. Start Qdrant (and optional PostgreSQL)
3. Start `src/llm_proxy.py` if using proxy model
4. Start `src/api_service.py` and/or `src/app_v2.py`
5. Call `/reload` once to ingest documents
6. Query via UI or API
