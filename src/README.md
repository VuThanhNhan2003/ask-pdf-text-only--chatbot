# Advanced RAG Pipeline (Hybrid Retrieval, Reranking, Proxy LLM)

Production-oriented Retrieval-Augmented Generation (RAG) stack for educational QA, with both a Streamlit UI and a FastAPI service.

This codebase includes:

- Dense retrieval (Sentence Transformers + Qdrant)
- Lexical retrieval (BM25)
- Hybrid fusion with configurable mode (`score` default, optional `rrf`)
- Cross-encoder reranking
- Multi-source ingestion (PDF + pre-chunked JSON)
- LLM abstraction layer (Gemini API or proxy-backed vLLM)
- Proxy retries + fallback to Gemini
- Conversation-aware prompting with short history window
- Debug retrieval mode with per-chunk score breakdown

---

## 1. Overview

The system answers user questions grounded in educational documents. It retrieves relevant chunks from indexed materials, reranks them for precision, builds a context-aware prompt, and generates responses through a pluggable LLM backend.

### Why this is an advanced RAG implementation

- Hybrid retrieval balances semantic and exact-match behavior:
	- Dense search handles conceptual queries.
	- BM25 catches exact terms (codes, parameter names, rare keywords).
- Configurable fusion strategy:
	- Default: weighted score fusion after normalization.
	- Optional: Reciprocal Rank Fusion (RRF).
- Second-stage cross-encoder reranking improves final context quality.
- Ingestion supports both:
	- Raw PDFs (`data/source_docs/<subject>/*.pdf`)
	- Structured JSON chunks (`data/processed_chunks/<subject>/*.json`)
- Robust serving path:
	- vLLM through lightweight proxy
	- Retry and fallback to Gemini when vLLM is unavailable
- Efficient runtime behavior:
	- Cached embedding model
	- Cached LLM instances
	- Cached processor instances in API service

---

## 2. Architecture

### 2.1 High-level components

- UI layer:
	- Streamlit app: `src/app_v2.py`
	- Frontend rendering package: `frontend/`
- API layer:
	- FastAPI service: `src/api_service.py`
- RAG core:
	- `RAGProcessor`: `src/processor.py`
- LLM abstraction:
	- `LLMManager`, `GeminiLLM`, `ProxyLLM`: `src/llm_manager.py`
- LLM proxy:
	- FastAPI proxy: `src/llm_proxy.py`
- Storage:
	- Qdrant for vectors and chunk payload metadata
	- SQL database for users/conversations/messages (used by UI services)

### 2.2 Query data flow

```text
User question
		|
		v
RAGProcessor._retrieve_relevant_chunks()
		|
		+--> Dense retrieval (Qdrant cosine + optional threshold + subject filter)
		|
		+--> BM25 retrieval (in-memory index from Qdrant payload text)
		|
		+--> Hybrid fusion
		|      - score mode: normalized weighted sum (default)
		|      - rrf mode: reciprocal rank fusion (optional)
		|
		+--> Candidate pool sizing
		|      - RAG_RERANK_CANDIDATE_MULTIPLIER
		|      - RAG_RERANK_CANDIDATE_CAP
		|
		+--> Cross-encoder rerank
		|
		v
Top-N chunks
		|
		v
Prompt build (+ optional short conversation history)
		|
		v
LLM invoke/stream (GeminiLLM or ProxyLLM)
		|
		v
Answer + source list
```

### 2.3 Ingestion/indexing flow

```text
reload_data(clean=False)
		|
		+--> data/source_docs/<subject>/*.pdf
		|      -> validate PDF
		|      -> page text extraction (PyMuPDF)
		|      -> recursive chunking
		|      -> deterministic chunk_id generation
		|      -> embedding + Qdrant upsert
		|
		+--> data/processed_chunks/<subject>/*.json
					 -> flexible payload extraction (chunks/data/items/documents)
					 -> normalize text + metadata + ids
					 -> embedding + Qdrant upsert (metadata updates allowed)
```

### 2.4 Hybrid fusion strategy (current behavior)

The retriever merges dense and lexical candidates before reranking.

- Default runtime mode is `score`:
	- `HYBRID_MODE = os.getenv("RAG_HYBRID_MODE", "score")`
	- Dense/BM25 scores are min-max normalized, then fused by weighted sum.
- Optional mode is `rrf`:
	- Enabled when `RAG_HYBRID_MODE=rrf`
	- Uses rank-based reciprocal rank fusion with `RAG_RRF_K`.

Score fusion formula:

$$
s_{hybrid}(d)=\alpha\cdot\hat{s}_{dense}(d)+\beta\cdot\hat{s}_{bm25}(d),\quad \alpha+\beta=1
$$

Missing-branch score behavior is configurable via:

- `RAG_HYBRID_MISSING_STRATEGY` in `{zero,min,epsilon}`
- `RAG_HYBRID_MISSING_EPSILON`

---

## 3. Module Responsibilities

| Module | Responsibility | Key behavior |
|---|---|---|
| `src/processor.py` | Core RAG pipeline | Ingestion, indexing, dense retrieval, BM25, hybrid fusion, rerank, prompt build, response/debug/stream |
| `src/llm_manager.py` | LLM abstraction and cache | Selects Gemini vs proxy model, handles streaming/non-streaming calls, LLM instance reuse |
| `src/llm_proxy.py` | Proxy + resilience layer | vLLM health checks, retries, streaming passthrough, Gemini fallback |
| `src/api_service.py` | Public API service | `/query`, `/query/debug`, `/query/stream`, `/reload`, `/models`, `/subjects`, processor cache |
| `src/app_v2.py` | Streamlit app entrypoint | Auth gate, sidebar selection, chat UX, DB-backed conversation persistence |
| `src/config.py` | Configuration management | Dataclass config loading + validation from environment |

Related non-`src` modules used by UI/runtime:

- `frontend/`: UI rendering components used by `src/app_v2.py`
- `database/`, `services/`, `auth/`: persistence + authentication services
- `widget/`: static assets mounted by FastAPI when directory exists (`/widget`)

---

## 4. Retrieval, Reranking, and Generation Details

### 4.1 Retrieval pipeline

- Dense retrieval (`retrieve_dense`):
	- Embeds query with sentence-transformers
	- Qdrant cosine search
	- Optional score threshold via `RAG_DENSE_SCORE_THRESHOLD`
	- Optional subject filtering
- BM25 retrieval (`retrieve_bm25`):
	- In-memory BM25 index built from Qdrant payload text
	- Unicode-friendly tokenization
	- Optional subject filtering
- Hybrid retrieval (`hybrid_retrieve`):
	- `score` mode or `rrf` mode
	- Merges branch metadata and scores per chunk id

### 4.2 Reranking pipeline

- Cross-encoder reranker (default: `BAAI/bge-reranker-v2-m3`)
- Lazy-loaded on first rerank call
- Candidate-pool controls:
	- `RAG_RERANK_TOP_N`
	- `RAG_RERANK_CANDIDATE_MULTIPLIER`
	- `RAG_RERANK_CANDIDATE_CAP`
- Fallback behavior:
	- If reranker fails, returns hybrid-ranked candidates

### 4.3 LLM generation

- `LLMManager` supports model keys defined in `config.llm.AVAILABLE_MODELS`
- Current model map in code:
	- `gemini` -> API (`gemini-2.5-flash`)
	- `qwen3-8b` -> proxy (`Qwen/Qwen3-8B-AWQ`)
- `ProxyLLM` features:
	- OpenAI-compatible `/v1/chat/completions` call
	- Non-streaming and streaming modes
	- `presence_penalty` env tuning (`LLM_PRESENCE_PENALTY`, default `1.5`)
	- `chat_template_kwargs.enable_thinking=False`

### 4.4 Conversation-aware prompting

- Processor keeps a short history window (`max_history_messages = 3` pairs)
- History can be set/cleared by UI/API usage
- Prompt template integrates context + optional prior turns
- Sources are appended after generation

### 4.5 Debug mode

- `get_response_with_debug()` returns:
	- `answer`
	- `debug_scores` (dense/bm25 normalized/raw scores, ranks, rerank score)
	- `retrieval_meta` (hybrid mode, candidate counts)
- API endpoint: `POST /query/debug`

---

## 5. Configuration

Primary configuration lives in `src/config.py` plus retrieval/reranker env values read in `src/processor.py`.

### 5.1 Core environment variables

| Variable | Default | Used in | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | empty | config, proxy | Required for Gemini API and fallback |
| `LLM_MODEL` | `gemini` | config | Active model key (`gemini`, `qwen3-8b`) |
| `LLM_PROXY_URL` | `http://localhost:5000` | config, llm_manager | Proxy endpoint for proxy models |
| `LLM_TEMPERATURE` | `0.7` | config | Generation temperature |
| `LLM_MAX_OUTPUT_TOKENS` | `2048` | config | Max output tokens |
| `LLM_PRESENCE_PENALTY` | `1.5` | llm_manager | Proxy request tuning (anti-repetition) |
| `QDRANT_URL` | unset | config | Overrides Qdrant host/port parsing; if unset, config defaults to host=`qdrant`, port=`6333` |
| `QDRANT_COLLECTION` | `documents` | config | Qdrant collection name |
| `VLLM_MODEL_NAME` | `Qwen/Qwen3-8B-AWQ` | llm_proxy | Declared in proxy service config; request routing still uses incoming `request.model` |

### 5.2 Retrieval and reranking environment variables

| Variable | Default | Description |
|---|---|---|
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model id |
| `RAG_EMBEDDING_LOCAL_PATH` | `<cache>/bge-m3` | Local embedding model path |
| `RAG_HYBRID_TOP_K` | `20` | Hybrid candidate count baseline |
| `RAG_DENSE_TOP_K` | `RAG_HYBRID_TOP_K` | Dense candidate count |
| `RAG_BM25_TOP_K` | `RAG_HYBRID_TOP_K` | BM25 candidate count |
| `RAG_HYBRID_MODE` | `score` | Fusion mode (`score` or `rrf`) |
| `RAG_HYBRID_ALPHA` | `0.7` | Dense weight in score fusion |
| `RAG_HYBRID_BETA` | `0.3` | BM25 weight in score fusion |
| `RAG_RRF_K` | `60` | RRF constant when `rrf` mode is enabled |
| `RAG_HYBRID_MISSING_STRATEGY` | `zero` | Missing branch handling (`zero`, `min`, `epsilon`) |
| `RAG_HYBRID_MISSING_EPSILON` | `0.01` | Epsilon for missing strategy |
| `RAG_DENSE_SCORE_THRESHOLD` | `config.retrieval.score_threshold` | Dense threshold (`<=0` disables) |
| `RAG_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker model |
| `RAG_RERANK_TOP_N` | `5` | Final selected chunks |
| `RAG_RERANK_BATCH_SIZE` | `8` | Reranker batch size |
| `RAG_RERANK_CANDIDATE_MULTIPLIER` | `1` | Candidate pool multiplier |
| `RAG_RERANK_CANDIDATE_CAP` | `30` | Max candidate pool cap |

### 5.3 Data layout conventions

```text
data/
├── source_docs/
│   └── <subject>/*.pdf
└── processed_chunks/
		└── <subject>/*.json
```

Both roots are scanned. Subjects are discovered by folder names under these roots.

---

## 6. API Reference and Usage

By default in Docker Compose, the API is exposed at `http://localhost:9100`.

### 6.1 Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health |
| `GET` | `/models` | Available LLM model map |
| `GET` | `/subjects` | Discovered subject list |
| `POST` | `/query` | Normal QA response |
| `POST` | `/query/debug` | QA response + retrieval debug table |
| `POST` | `/query/stream` | SSE streaming response |
| `POST` | `/reload` | Reindex documents |

### 6.2 Example: normal query

Request:

```bash
curl -X POST http://localhost:9100/query \
	-H "Content-Type: application/json" \
	-d '{
		"question": "What is gradient descent learning rate?",
		"subject": null,
		"model_key": "qwen3-8b",
		"use_history": true,
		"debug": false
	}'
```

Response shape:

```json
{
	"answer": "...model answer...\n\n📚 **Nguồn tham khảo:**\n- Subject / file.pdf / Trang 12",
	"debug_scores": null,
	"retrieval_meta": null
}
```

### 6.3 Example: debug query

Request:

```bash
curl -X POST http://localhost:9100/query/debug \
	-H "Content-Type: application/json" \
	-d '{
		"question": "Explain logistic regression",
		"subject": null,
		"model_key": "gemini",
		"use_history": true
	}'
```

Response shape (truncated):

```json
{
	"answer": "...",
	"debug_scores": [
		{
			"rank": 1,
			"hybrid_rank": 2,
			"id": "chunk_id",
			"dense_score": 0.81,
			"bm25_score": 12.4,
			"dense_norm": 0.93,
			"bm25_norm": 0.74,
			"hybrid_score": 0.87,
			"rerank_score": 7.12,
			"dense_rank": null,
			"bm25_rank": null,
			"rrf_score": null,
			"text_preview": "..."
		}
	],
	"retrieval_meta": {
		"hybrid_mode": "score",
		"hybrid_top_k": 20,
		"rerank_top_n": 5,
		"total_candidates": 20,
		"selected_candidates": 5
	}
}
```

### 6.4 Example: stream query (SSE)

```bash
curl -N -X POST http://localhost:9100/query/stream \
	-H "Content-Type: application/json" \
	-d '{
		"question": "Summarize chapter 3",
		"subject": null,
		"model_key": "qwen3-8b",
		"use_history": true
	}'
```

Stream event format:

```text
data: {"chunk":"partial text"}

data: {"chunk":"more text"}

data: {"done":true}
```

### 6.5 Example: reload index

```bash
curl -X POST http://localhost:9100/reload \
	-H "Content-Type: application/json" \
	-d '{"subject": null, "model_key": "gemini", "clean": false}'
```

- `clean=false`: incremental update
- `clean=true`: delete/recreate collection then full reindex

---

## 7. Running the System

### 7.1 Docker Compose (recommended)

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

Expected services (from compose):

- `postgres` (5432)
- `qdrant` (6333)
- `llm_proxy` (5000)
- `app` Streamlit (8501)
- `chatbot_api` FastAPI (9100 host -> 8000 container)

Check status:

```bash
docker compose -f docker/docker-compose.yml ps
```

### 7.2 Manual run

1. Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Start Qdrant and database (docker or external deployment).

3. Start proxy (for proxy model path):

```bash
python3 src/llm_proxy.py
```

4. Start API:

```bash
uvicorn src.api_service:app --host 0.0.0.0 --port 9100 --reload
```

5. Start UI:

```bash
streamlit run src/app_v2.py
```

---

## 8. Advanced Features and Why They Matter

- Hybrid retrieval (`score` / `rrf`):
	- Better robustness across conceptual and keyword-heavy questions.
- Configurable missing-branch behavior:
	- Helps control bias when only dense or BM25 retrieves a chunk.
- Candidate pool controls before reranking:
	- Improves recall without always reranking too many noisy chunks.
- Debug retrieval endpoint:
	- Provides transparent score/rank traces for analysis and tuning.
- Streaming support across layers:
	- Better UX in both UI and API consumers.
- Multi-layer caching:
	- Embedding model cache, LLM cache, and processor cache reduce latency.
- Proxy fallback strategy:
	- Maintains service continuity when vLLM backend is unhealthy.

---

## 9. Notes and Assumptions

- Current retrieval is single-vector dense + BM25 hybrid (no multi-vector retrieval path implemented in `src`).
- The Streamlit UI in `src/app_v2.py` depends on external package directory `frontend/`.
- FastAPI mounts `/widget` only if `widget/` exists.
- In code, `LLM_MODEL` default is `gemini`; in Docker Compose for `app`, default is `qwen3-8b`.
- Gemini fallback in proxy is enabled only when `GOOGLE_API_KEY` is set.

---

## 10. Quick Start Checklist

1. Create and fill `.env` (`GOOGLE_API_KEY`, `QDRANT_URL`, `LLM_MODEL`, `LLM_PROXY_URL`).
2. Start infrastructure (Qdrant, DB, and optionally full Docker Compose stack).
3. Ensure data is placed in:
	 - `data/source_docs/<subject>/*.pdf`
	 - `data/processed_chunks/<subject>/*.json`
4. Start proxy if using proxy models.
5. Start API and/or Streamlit UI.
6. Trigger `/reload` once.
7. Run `/query` or `/query/debug` to validate retrieval and generation.


Dựa trên project RAG của bạn, đây là những diagram tôi gợi ý cho thesis/report:

## 📊 Các diagram khuyến nghị

### **1. System Architecture Overview** (bắt buộc)
```
┌─────────────────────────────────────────────────────────┐
│                    User Interface Layer                 │
│        Streamlit UI (app_v2.py)  │  FastAPI (port 9100)│
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐      ┌────────▼──────────┐
│  RAGProcessor  │      │    LLMManager      │
│ • PDF ingest   │      │  • GeminiLLM       │
│ • Chunking     │      │  • ProxyLLM        │
│ • Embeddings   │      │  • Model cache     │
└────────┬───────┘      └────────┬───────────┘
         │                       │
    ┌────┴───────────┬───────────┴────┐
    │                │                │
┌───▼──────┐  ┌─────▼──────┐  ┌──────▼──────┐
│  Qdrant  │  │ PostgreSQL │  │LLM Proxy    │
│ • Dense  │  │ • Users    │  │ • vLLM      │
│ • BM25   │  │ • Conv     │  │ • Gemini    │
│  Index   │  │ • Messages │  │  fallback   │
└──────────┘  └────────────┘  └─────────────┘
```

### **2. Query Processing Pipeline** (rất quan trọng)
```
User Question
    ↓
[Embedding] → sentence-transformers
    ↓
┌─────────────────────────────────────┐
│        Hybrid Retrieval             │
├─────────────────────────────────────┤
│ Dense Retrieval (Qdrant cosine)    │ → Top-k dense results
│                                    │
│ BM25 Retrieval (in-memory index)   │ → Top-k BM25 results
└─────────────────────────────────────┘
    ↓
[Hybrid Fusion: score or RRF mode]
    ↓
[Candidate Pool Sizing]
    ↓
[Cross-Encoder Reranking]
    ↓
Top-N final chunks
    ↓
[Prompt Building + History]
    ↓
[LLM Generation]
    ↓
Response + Sources
```

### **3. Hybrid Score Fusion Comparison** (lý thuyết + thực tế)
Vẽ hai cột so sánh:
- **Cột trái: Score Fusion (mặc định)**
  - Min-max normalize
  - Weighted sum: α·s_dense + β·s_bm25
  - Công thức toán

- **Cột phải: RRF (tùy chọn)**
  - Rank-based fusion
  - Công thức: 1/(k + rank)
  - Ưu/nhược điểm

### **4. Data Ingestion Pipeline**
```
data/
├── source_docs/<subject>/*.pdf
│   ├─→ [Validate] ─→ [Extract Text] ─→ [Chunk]
│   ├─→ [Tokenize for BM25] ─→ [Embed] ─→ [Qdrant Upsert]
│
└── processed_chunks/<subject>/*.json
    └─→ [Parse] ─→ [Normalize] ─→ [Embed] ─→ [Qdrant Upsert]
                                                  ↓
                                        [Build in-memory BM25]
```

### **5. Reranking Process Flow**
```
Hybrid Candidates (20 chunks)
    ↓
[Candidate Pool Control]
  - multiplier × top_n
  - cap at max_cap
    ↓
[Cross-Encoder Reranker]
  - Score (query, chunk) pairs
  - Batch inference
    ↓
[Sort by rerank score]
    ↓
Final Context (5 chunks)
```

### **6. LLM Abstraction & Proxy Strategy**
```
                    LLMManager
                        ↓
            ┌───────────┴───────────┐
            ↓                       ↓
        GeminiLLM              ProxyLLM
        (API direct)           (HTTP proxy)
                                   ↓
                      ┌────────────┴────────────┐
                      ↓                         ↓
                  [vLLM Server]         [Health Check → ✅/❌]
                                               ↓
                                        [Retry 3x + Backoff]
                                               ↓
                                        [Fallback → Gemini]
```

### **7. Database Schema** (ER Diagram)
Vẽ mối quan hệ:
```
Users ──1:N─→ Conversations ──1:N─→ Messages
  ↓                ↓
UserSettings   SharedConversations
```

### **8. Candidate Pool Tuning Effect** (chart)
Vẽ 3 scenarios:
- Multiplier=1 (no expansion)
- Multiplier=2 (moderate expansion → better recall)
- Multiplier=3 (aggressive → more reranking cost)

### **9. Embedding + Chunking Strategy**
```
PDF Content
    ↓
[RecursiveCharacterTextSplitter]
  chunk_size=1000
  overlap=200
    ↓
Overlapped Chunks
    ↓
[BAAI/bge-m3 Embedding]
    ↓
Dense Vectors (384-dim) + Payload
```

### **10. Docker Deployment Stack** (Compose visualization)
```
┌──────────────────────────────────────────────┐
│         Docker Compose Services               │
├──────────────────────────────────────────────┤
│  postgres:5432     qdrant:6333               │
│   (users/conv)      (vectors)                │
│       ↑                ↑                      │
│       └───────┬────────┘                     │
│               ↓                              │
│        llm_proxy:5000 ──→ vLLM(external)    │
│               ↑                              │
│    ┌──────────┼──────────┐                  │
│    ↓          ↓          ↓                   │
│   app:8501  api:9100  widget:/static        │
│  (Streamlit) (FastAPI)                      │
└──────────────────────────────────────────────┘
```
