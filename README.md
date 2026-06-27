# 📚 Advanced RAG Chatbot for Educational Q&A

<p align="center">
  <strong>A production-oriented Retrieval-Augmented Generation (RAG) system for student Q&A on MOOC course materials.<br>
  Part of a three-pipeline multimodal ingestion architecture.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.49-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/FastAPI-0.110-05998b?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangChain-0.2-green" alt="LangChain">
  <img src="https://img.shields.io/badge/Qdrant-Vector_DB-DC382D" alt="Qdrant">
  <img src="https://img.shields.io/badge/PostgreSQL-15-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker">
</p>

---

## Overview

This repository is the **RAG pipeline** — the final stage of a three-pipeline knowledge system for MOOC platforms:

| Pipeline | Repository | Output |
|---|---|---|
| **ASR** | [whisper-project](https://github.com/VuThanhNhan2003/whisper-project) | Lecture video → Faster-Whisper → TextNode JSON |
| **Vision / VLM** | [`extracting_pdf_pipeline/`](./extracting_pdf_pipeline/) | Course documents → Qwen3-VL-8B → TextNode JSON |
| **RAG Chatbot** ← *this repo* | — | TextNode JSON → Qdrant → answers |

Both upstream pipelines produce a shared **TextNode JSON** format (chunk + structured metadata), which this chatbot ingests directly. There are no raw PDF files involved at the retrieval stage.

> **To use this chatbot you must first generate TextNode JSON files using the two pipelines above.** The `data/` folder and ground-truth evaluation files are excluded from this repository (see `.gitignore`).

---

## Key Technical Features

**Retrieval**
- Hybrid search combining dense (BAAI/bge-m3) and sparse (BM25) retrieval with configurable fusion (`RAG_HYBRID_ALPHA`)
- Weighted score fusion and Reciprocal Rank Fusion (RRF) are both supported
- Cross-encoder reranking with BAAI/bge-reranker-v2-m3 (OpenVINO / ONNX / PyTorch backend, auto-fallback)
- Metadata-aware boosting: keyword and topic fields from TextNode metadata are used to boost BM25 scoring and reranking
- MD5-based deduplication on ingest: nodes already in Qdrant are skipped, making re-indexing fast

**LLM serving**
- Three backends: Google Gemini (cloud), Qwen3-8B-AWQ via vLLM on a Slurm GPU cluster (local), Groq API (cloud)
- `llm_proxy` provides health-check, retry, and automatic Gemini fallback when vLLM is unavailable
- Model selection via a single environment variable; switchable without rebuilding

**Interfaces**
- Streamlit web app with authentication, conversation history, pinning, archiving, and sharing
- FastAPI service (`chatbot_api`) with streaming (SSE), debug, and reload endpoints
- Embeddable JavaScript widget (`widget/`) for drop-in LMS integration

---

## Architecture

```
  TextNode JSON          TextNode JSON
  (ASR pipeline)         (Vision pipeline)
         │                      │
         └──────────┬───────────┘
                    │
              data/*.json
                    │
         ┌──────────▼──────────────────┐
         │     Ingest on startup        │
         │  BGE-M3 embed + BM25 index  │
         │  Qdrant upsert (MD5 dedup)  │
         └──────────┬──────────────────┘
                    │
         ┌──────────▼──────────────────────────────┐
         │              Query flow                  │
         │                                          │
         │  Question → BGE-M3 embed                 │
         │    → Dense search  (Qdrant)              │
         │    → Sparse search (BM25 + kw boost)     │
         │    → Score fusion (alpha/RRF)            │
         │    → BGE Reranker  (+ metadata boost)    │
         │    → Top-N chunks → LLM → Answer         │
         └──────────┬──────────────────────────────┘
                    │
       ┌────────────┴───────────────┐
       ▼                            ▼
  Streamlit :8501           FastAPI API :9100
  (web UI)                  (REST + streaming + widget)
```

---

## Project Structure

```
.
├── src/
│   ├── app_v2.py               # Streamlit UI
│   ├── api_service.py          # FastAPI RAG service (port 9100)
│   ├── config.py               # All configuration dataclasses
│   ├── processor.py            # Ingest, embed, hybrid search, rerank, generate
│   ├── llm_manager.py          # LLM abstraction (Gemini / vLLM / Groq)
│   └── llm_proxy.py            # FastAPI proxy with health-check & fallback (port 5000)
├── auth/
│   └── authentication.py       # bcrypt-based sign-up / login
├── components/ & frontend/     # Streamlit UI components and styles
├── database/
│   ├── database.py             # SQLAlchemy session
│   └── models.py               # ORM: User, Conversation, Message, Settings, SharedConversation
├── services/
│   └── conversation_service.py # Conversation & message business logic
├── extracting_pdf_pipeline/    # Vision pipeline (Qwen3-VL-8B + Google Drive crawler)
│   ├── main.py                 # PDF → TextNode JSON via Qwen3-VL
│   └── CrawlFileFromDrive.ipynb# Google Colab notebook: Drive → PDF normalisation
├── widget/                     # Embeddable JS chat widget
│   ├── chatbot-widget.js
│   ├── loader.js
│   └── README.md
├── scripts/                    # Evaluation & benchmarking scripts
│   ├── evaluate_rag_english.py # End-to-end RAG evaluation (nDCG, MRR, RAGAS)
│   └── benchmark_*.py
├── docker/
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── Dockerfile.proxy
├── .env.example
└── requirements.txt
```

*`data/`, `models/`, `vectordb/`, `evaluation/`, and `vpn_config/` are excluded from version control.*

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose v2
- A Google API key (Gemini — used as LLM fallback)
- TextNode JSON files generated by the [ASR pipeline](https://github.com/VuThanhNhan2003/whisper-project) or the [Vision pipeline](./extracting_pdf_pipeline/)

### 1. Clone

```bash
git clone https://github.com/VuThanhNhan2003/ask-pdf-text-only--chatbot.git
cd ask-pdf-text-only--chatbot
```

### 2. Configure environment

```bash
cp .env.example .env
```

Key variables to set:

```env
# Required
GOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key          # only needed for groq-qwen3-32b

# LLM backend: gemini | qwen3-8b | groq-qwen3-32b
LLM_MODEL=gemini
LLM_PROXY_URL=http://llm_proxy:5000
LLM_TEMPERATURE=0.7
LLM_PRESENCE_PENALTY=1.5

# Hybrid retrieval
RAG_EMBEDDING_MODEL=BAAI/bge-m3
RAG_HYBRID_ALPHA=0.7                # weight for dense vs sparse
RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RAG_RERANK_BACKEND=openvino         # openvino | onnx | torch
RAG_RERANK_TOP_N=8

# Database
POSTGRES_PASSWORD=chatbot_pass_2024
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=documents
```

### 3. Add TextNode JSON files

Place the `.json` files produced by the ASR or Vision pipelines into `data/`:

```bash
mkdir -p data

# From the ASR pipeline
cp /path/to/whisper-project/file_textnodes/*.textnodes.json data/

# From the Vision pipeline
cp /path/to/extracting_pdf_pipeline/output/*.json data/
```

The app indexes these automatically on startup. Only new or changed nodes are re-embedded (MD5 dedup).

### 4. Pre-download models (recommended)

```bash
mkdir -p models
huggingface-cli download BAAI/bge-m3 --local-dir ./models/bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir ./models/bge-reranker-v2-m3
```

### 5. Start

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

Access the UI at **http://localhost:8501**.

---

## Services

| Service | Port | Description |
|---|---|---|
| `app` | 8501 | Streamlit web UI |
| `chatbot_api` | 9100 | FastAPI RAG service (REST + SSE) |
| `llm_proxy` | 5000 | LLM proxy (health-check, retry, Gemini fallback) |
| `qdrant` | 6333 | Vector database |
| `postgres` | 5432 | SQL database |

---

## LLM Backends

| `LLM_MODEL` | Backend | Requires |
|---|---|---|
| `gemini` | Google Gemini 2.5 Flash | `GOOGLE_API_KEY` |
| `qwen3-8b` | Qwen3-8B-AWQ via vLLM (Slurm cluster) | SSH tunnel on port 8001 |
| `groq-qwen3-32b` | Qwen3-32B via Groq | `GROQ_API_KEY` |

Switch without rebuilding:

```bash
sed -i 's/LLM_MODEL=.*/LLM_MODEL=gemini/' .env
docker compose -f docker/docker-compose.yml up -d llm_proxy app
```

`llm_proxy` automatically falls back to Gemini when the vLLM endpoint is unreachable.

---

## API Reference (port 9100)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health |
| `/subjects` | GET | List indexed subjects |
| `/query` | POST | Non-streaming Q&A |
| `/query/stream` | POST | Streaming Q&A (SSE) |
| `/query/debug` | POST | Q&A with per-stage retrieval scores (dense, BM25, hybrid, rerank) |
| `/reload` | POST | Re-index `data/` without restarting |

---

## Embeddable Widget

A self-contained JavaScript widget for embedding the chatbot on any webpage:

```html
<script src="https://your-domain.com/widget/loader.js"
        data-api-url="https://your-api-domain.com"
        data-subject="Calculus 1"
        data-position="bottom-right"
        data-primary-color="#1f77b4">
</script>
```

See [`widget/README.md`](./widget/README.md) for the full list of configuration attributes.

---

## Evaluation

Evaluation scripts are in `scripts/` (ground-truth files must be created separately):

```bash
# End-to-end retrieval + generation evaluation
python scripts/evaluate_rag_english.py

# Reranker backend benchmark
python scripts/benchmark_reranker_backends.py

# End-to-end latency benchmark
python scripts/benchmark_end_to_end_latency.py
```

Metrics: nDCG@k, MRR, Precision@k, RAGAS (faithfulness, answer relevancy, context precision).

---

## Database Schema

```
users ──► conversations ──► messages
  │              └──► shared_conversations
  └──► user_settings
```

Key fields: `conversations.subject` (course name), `messages.sources` (JSON array of retrieved chunks), `messages.processing_time`.

---

## License

Developed for academic purposes — Undergraduate Thesis, International University VNU-HCM, 2026.