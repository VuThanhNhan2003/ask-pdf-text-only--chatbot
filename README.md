# 📚 Advanced RAG for Educational Q&A

<p align="center">
  <strong>A production-oriented, advanced Retrieval-Augmented Generation (RAG) system designed for educational Q&A.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/FastAPI-0.110-05998b?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangChain-0.2-green?logo=langchain" alt="LangChain">
  <img src="https://img.shields.io/badge/Qdrant-Vector_DB-DC382D" alt="Qdrant">
  <img src="https://img.shields.io/badge/PostgreSQL-15-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker">
</p>

This project provides a complete, production-oriented RAG stack for answering questions grounded in educational documents. It features a sophisticated retrieval pipeline with hybrid search and reranking, a resilient LLM serving layer, and multiple user interfaces including a Streamlit app and a deployable web widget.

---

## 📖 Introduction

This system is an **advanced RAG implementation** that goes beyond basic vector search. It is designed for high accuracy and relevance in a real-world educational setting.

### Key Features

- **Advanced Retrieval Pipeline**:
  - **Hybrid Search**: Combines dense (semantic) and lexical (BM25 keyword) search for robustness against both conceptual and keyword-based queries.
  - **Configurable Fusion**: Supports weighted score fusion and Reciprocal Rank Fusion (RRF) to merge search results.
  - **Cross-Encoder Reranking**: A second-stage reranker (`BAAI/bge-reranker-v2-m3`) refines candidate chunks for maximum relevance before generation.
  - **Metadata-Aware Search**: Boosts chunks based on keyword and topic metadata during both lexical search and reranking.
- **Multi-Source Ingestion**: Indexes documents from raw PDFs and pre-chunked structured JSON files.
- **Resilient LLM Serving**:
  - **LLM Abstraction**: Supports multiple backends including Google Gemini, Groq, and local models via a vLLM proxy.
  - **Proxy with Fallback**: The proxy layer includes health checks, retries, and automatic fallback to the Gemini API if the local vLLM is unavailable.
- **Multiple Interfaces**:
  - **Streamlit Web App**: A full-featured chat interface for students and administrators.
  - **Embeddable Web Widget**: A self-contained JavaScript widget that can be embedded on any website for direct Q&A.
- **Developer-Focused Tooling**:
  - **Debug Endpoint**: An API endpoint (`/query/debug`) provides a detailed breakdown of retrieval scores (dense, BM25, hybrid, rerank) for analysis and tuning.
  - **Evaluation Suite**: A comprehensive script (`scripts/evaluate_rag_english.py`) for end-to-end evaluation of retrieval and generation quality using standard metrics (nDCG, MRR) and RAGAS.

### User-Facing Features

- **Multi-Subject Q&A** — Organizes documents by subject (e.g., Calculus 1, Business Statistics, Physics 2).
- **User Authentication** — Secure registration/login with bcrypt.
- **Conversation Management** — Saves chat history, pins, archives, and deletes conversations.
- **Streaming Response** — Real-time responses.
- **Share Conversation** — Shares conversations via a secure token.

---

## 🏗️ System Architecture

The application is composed of several services orchestrated by Docker Compose: a Streamlit web UI, a FastAPI backend for the RAG service, a PostgreSQL database, a Qdrant vector database, and a proxy for the LLM.

### Services

| Service | Port | Description |
|---|---|---|
| **app** | `8501` | Streamlit Web UI |
| **chatbot_api** | `9100` | FastAPI RAG service |
| **llm_proxy** | `5000` | FastAPI LLM Proxy |
| **qdrant** | `6333` | Vector Database |
| **postgres** | `5432` | SQL Database |

---

## ⚙️ RAG Pipeline Details

The query processing flow is designed to maximize relevance and accuracy.

```
User Question
    ↓
[Embedding] → BAAI/bge-m3
    ↓
┌───────────────────────────────────┐
│         Hybrid Retrieval          │
├───────────────────────────────────┤
│ 1. Dense Search (Qdrant)          │ → Top-k semantic results
│ 2. Lexical Search (BM25)          │ → Top-k keyword results
└───────────────────────────────────┘
    ↓
[Fusion: Score-based or RRF]
    ↓
[Candidate Pool Sizing]
    ↓
[Cross-Encoder Reranking] → BAAI/bge-reranker-v2-m3
    ↓
Top-N final chunks for context
    ↓
[Prompt Building + Chat History]
    ↓
[LLM Generation (Gemini / Groq / vLLM)]
    ↓
Answer + Sources
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit 1.49 |
| **Backend & API** | FastAPI, Uvicorn |
| **LLM & RAG** | LangChain 0.2, Google Gemini, Groq, Qwen3 (vLLM) |
| **Embedding & Reranking** | Sentence Transformers (`BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`) |
| **Vector DB** | Qdrant |
| **Database** | PostgreSQL 15 + SQLAlchemy ORM |
| **PDF Parsing** | PyMuPDF + pymupdf4llm |
| **Auth** | bcrypt |
| **ML Backend** | PyTorch 2.8 (CPU) / OnnxRuntime / Openvino |
| **DevOps** | Docker Compose |

---

## 📁 Project Structure

```
rag-pipeline/
├── src/                        # Core application
│   ├── app_v2.py               # Streamlit main app & UI
│   ├── api_service.py          # FastAPI RAG service
│   ├── config.py               # Configuration classes
│   ├── processor.py            # RAG pipeline (ingest, embed, search, generate)
│   ├── llm_manager.py          # LLM abstraction layer
│   └── llm_proxy.py            # FastAPI proxy server (vLLM + Gemini fallback)
├── auth/
│   └── authentication.py       # User auth service (signup, login, password hashing)
├── database/
│   ├── database.py             # SQLAlchemy session management
│   └── models.py               # ORM models (User, Conversation, Message, ...)
├── services/
│   └── conversation_service.py # Business logic for conversations & messages
├── data/                       # Directory for document by subject
│   ├── source_docs/
│   └── processed_chunks/
├── models/                     # Pre-downloaded models
├── docker/
│   ├── docker-compose.yml      # Multi-service orchestration
│   ├── Dockerfile              # Main app image
│   └── Dockerfile.proxy        # LLM proxy image
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Usage

### Prerequisites

- **Python** 3.11+
- **Docker** & **Docker Compose** (recommended)
- **Google API Key** (for Gemini fallback)

### 1. Clone & Install Dependencies

```bash
git clone <repository-url>
cd rag-pipeline

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the root directory:

```env
# Google Gemini API
GOOGLE_API_KEY=your_google_api_key

# Groq API
GROQ_API_KEY=your_groq_api_key

# PostgreSQL
POSTGRES_PASSWORD=your_postgres_password

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=documents

# LLM Proxy
LLM_PROXY_URL=http://localhost:5000
LLM_MODEL=qwen3-8b          # or "gemini" or "groq"

# LLM Parameters
LLM_TEMPERATURE=0.7
LLM_MAX_OUTPUT_TOKENS=2048
```

### 3. Launch with Docker (Recommended)

```bash
docker compose -f docker/docker-compose.yml up -d
```

Access the application at: **http://localhost:8501**

### 4. Add Documents

Place your documents into the `data/source_docs/` directory, organized by subject. The application supports PDF files and pre-chunked JSON files.

**For PDF files:**
```
data/source_docs/
├── Calculus 1/
│   ├── chapter1.pdf
│   └── chapter2.pdf
├── Business Statistics/
│   └── textbook.pdf
└── Physics 2/
    └── lecture_notes.pdf
```

**For pre-chunked JSON files:**
Place your JSON files in the `data/processed_chunks/` directory, following the same subject-based structure.

The documents will be automatically ingested when the application starts.

---

## ⚙️ Detailed Configuration

Configuration classes are defined in `src/config.py`:

| Class | Description | Key Parameters |
|---|---|---|
| `EmbeddingConfig` | Embedding model | `model_name`, `batch_size`, `local_path` |
| `QdrantConfig` | Vector DB | `host`, `port` (6333), `collection_name` |
| `ChunkingConfig` | Text chunking | `chunk_size` (1000), `chunk_overlap` (200) |
| `LLMConfig` | LLM selection | `current_model`, `temperature`, `proxy_url` |
| `RetrievalConfig` | RAG search | `top_k` (5), `score_threshold` (0.3) |
| `AppConfig` | UI & paths | `data_folder`, `log_folder`, `page_title` |

### Supported LLM Models

The application supports multiple LLMs, configured in `src/config.py`. The `LLMManager` class handles the selection and instantiation of the appropriate LLM.

```python
AVAILABLE_MODELS = {
    "gemini": {
        "type": "api",
        "provider": "google",
        "name": "gemini-2.5-flash"
    },
    "groq": {
        "type": "groq",
        "provider": "groq",
        "name": "qwen/qwen3-32b"
    },
    "qwen3-8b": {
        "type": "proxy",
        "provider": "vllm",
        "name": "Qwen/Qwen3-8B-AWQ"
    }
}
```

---

## 🗄️ Database Schema

The database schema is defined using SQLAlchemy ORM in `database/models.py`.

- **users**: Stores user account information.
- **user_settings**: Stores user preferences.
- **conversations**: Stores chat sessions.
- **messages**: Stores individual messages within a conversation.
- **shared_conversations**: Stores information about publicly shared conversations.

---

## 🔌 API Endpoints (FastAPI Service - Port 9100)

The `chatbot_api` service provides a set of endpoints to interact with the RAG pipeline.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/models` | GET | List available LLM models |
| `/subjects` | GET | List loaded subjects |
| `/query` | POST | Answer a question (non-streaming) |
| `/query/debug` | POST | Answer a question with retrieval debug information |
| `/query/stream` | POST | Answer a question (streaming via Server-Sent Events) |
| `/reload` | POST | Reload data into the vector store |

---

## 🐳 Docker

The `docker-compose.yml` file orchestrates the different services of the application.

### Services

- **postgres**: PostgreSQL database.
- **qdrant**: Qdrant vector database.
- **llm_proxy**: FastAPI proxy for vLLM with Gemini fallback.
- **app**: Streamlit web UI.
- **chatbot_api**: FastAPI RAG service.

### Build & Deploy

```bash
# Build and run all services
docker compose -f docker/docker-compose.yml up -d --build

# View logs
docker compose -f docker/docker-compose.yml logs -f app

# Stop all services
docker compose -f docker/docker-compose.yml down
```

---

## 📝 Logging

- **File log**: `logs/rag_YYYYMMDD.log` (daily rotation)
- **Console**: Displays WARNING level and above
- **Format**: `Timestamp | Logger | Level | Message`

---

## 🛣️ Roadmap

- [ ] Admin dashboard for user management
- [ ] Support for more file formats (DOCX, PPTX)
- [ ] Token-based authentication (JWT)
- [ ] Multi-language support
- [ ] Export conversation to PDF
- [ ] Analytics & usage statistics

---

## 📄 License

This project is developed for educational purposes.
