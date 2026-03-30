# 📚 Ask PDF — RAG Chatbot for Education

<p align="center">
  <strong>Hệ thống hỏi đáp tài liệu PDF, video thông minh cho sinh viên MOOC, sử dụng Retrieval-Augmented Generation (RAG)</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.49-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/LangChain-0.2-green?logo=langchain" alt="LangChain">
  <img src="https://img.shields.io/badge/Qdrant-Vector_DB-DC382D" alt="Qdrant">
  <img src="https://img.shields.io/badge/PostgreSQL-15-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker">
</p>

---

## 📖 Giới thiệu

**Ask PDF** là một chatbot RAG hỗ trợ sinh viên tra cứu và hỏi đáp nội dung từ tài liệu PDF theo nhiều môn học. Hệ thống trích xuất văn bản từ PDF, tạo embedding vector, lưu trữ vào Qdrant và sử dụng LLM để sinh câu trả lời chính xác dựa trên ngữ cảnh tài liệu gốc.

### Tính năng chính

- **Hỏi đáp đa môn học** — Tổ chức tài liệu theo từng môn (Giải tích 1, Thống kê trong kinh doanh, Vật lý 2, ...)
- **Xác thực người dùng** — Đăng ký / đăng nhập bảo mật với bcrypt
- **Quản lý hội thoại** — Lưu lịch sử chat, ghim, lưu trữ, xoá cuộc hội thoại
- **Hybrid LLM** — Hỗ trợ Google Gemini API và local vLLM (Qwen3-8B-AWQ) với cơ chế fallback tự động
- **Streaming Response** — Phản hồi theo thời gian thực
- **Chia sẻ hội thoại** — Chia sẻ cuộc trò chuyện qua token bảo mật

---

## 🏗️ Kiến trúc hệ thống

```
┌──────────────────────────────────────────────────────────┐
│                STREAMLIT WEB UI (app_v2.py)               │
│   Auth UI  ·  Chat Interface  ·  Conversation Sidebar     │
└──────────────────────┬───────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌───────────────────┐    ┌─────────────────────────┐
│   RAGProcessor    │    │      LLMManager         │
│  (processor.py)   │    │   (llm_manager.py)      │
│                   │    │                         │
│ · PDF ingestion   │    │ · Gemini API client     │
│ · Text chunking   │    │ · Proxy LLM client      │
│ · Embedding       │    │ · Model caching         │
│ · Vector search   │    │ · Streaming support     │
└────────┬──────────┘    └────────────┬────────────┘
         │                            │
         ▼                            ▼
┌────────────────┐       ┌─────────────────────────┐
│   Qdrant VDB   │       │  LLM Proxy (FastAPI)    │
│   (port 6333)  │       │     (port 5000)         │
└────────────────┘       │                         │
                         │  vLLM ──► Gemini (fallback)
                         └─────────────────────────┘
         │
         ▼
┌────────────────┐
│  PostgreSQL    │
│  (port 5432)   │
│                │
│ Users · Conversations · Messages · Settings
└────────────────┘
```

### Luồng xử lý câu hỏi

```
Câu hỏi → Embedding (sentence-transformers) → Tìm kiếm Qdrant (top-K)
    → Xây dựng context (tài liệu + lịch sử hội thoại)
    → Gửi đến LLM → Streaming response → Lưu vào PostgreSQL
```

---

## 🛠️ Tech Stack

| Layer | Công nghệ |
|-------|-----------|
| **Frontend** | Streamlit 1.49 |
| **LLM Framework** | LangChain 0.2, Google Gemini 2.5 Flash, Qwen3-8B-AWQ (vLLM) |
| **Embedding** | sentence-transformers (all-MiniLM-L6-v2) |
| **Vector DB** | Qdrant |
| **Database** | PostgreSQL 15 + SQLAlchemy ORM |
| **LLM Proxy** | FastAPI + Uvicorn |
| **PDF Parsing** | PyMuPDF + pymupdf4llm |
| **Auth** | bcrypt |
| **ML Backend** | PyTorch 2.8 (CPU) |
| **DevOps** | Docker Compose |

---

## 📁 Cấu trúc dự án

```
rag-pipeline/
├── src/                        # Core application
│   ├── app_v2.py               # Streamlit main app & UI
│   ├── config.py               # Configuration classes
│   ├── processor.py            # RAG pipeline (ingest, embed, search, generate)
│   ├── llm_manager.py          # LLM abstraction layer
│   └── llm_proxy.py            # FastAPI proxy server (vLLM + Gemini fallback)
├── auth/
│   └── authentication.py       # User auth service (signup, login, password hashing)
├── components/
│   └── auth_ui.py              # Streamlit auth UI components
├── database/
│   ├── database.py             # SQLAlchemy session management
│   └── models.py               # ORM models (User, Conversation, Message, ...)
├── services/
│   └── conversation_service.py # Business logic cho hội thoại & tin nhắn
├── data/                       # Thư mục chứa PDF theo môn học
│   ├── Giai tich 1/
│   ├── Môn Thống kê trong kinh doanh/
│   └── Vật lý 2/
├── models/                     # Pre-downloaded models
│   ├── all-MiniLM-L6-v2/      # Embedding model
│   └── llm/                    # LLM weights (Qwen3-8B-AWQ)
├── docker/
│   ├── docker-compose.yml      # Multi-service orchestration
│   ├── Dockerfile              # Main app image
│   └── Dockerfile.proxy        # LLM proxy image
├── vpn_config/                 # Scripts kết nối GPU server qua VPN
├── logs/                       # Application logs
├── requirements.txt
└── README.md
```

---

## 🚀 Cài đặt & Chạy

### Yêu cầu

- **Python** 3.11+
- **Docker** & **Docker Compose** (khuyến nghị)
- **Google API Key** (cho Gemini fallback)

### 1. Clone & cài đặt dependencies

```bash
git clone <repository-url>
cd rag-pipeline

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Cấu hình environment

Tạo file `.env` ở thư mục gốc:

```env
# Google Gemini API
GOOGLE_API_KEY=your_google_api_key

# PostgreSQL
DATABASE_URL=postgresql://chatbot_user:chatbot_pass_2024@localhost:5432/chatbot_db

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=documents

# LLM Proxy
LLM_PROXY_URL=http://localhost:5000
LLM_MODEL=qwen2-7b          # hoặc "gemini"

# LLM Parameters
LLM_TEMPERATURE=0.7
LLM_MAX_OUTPUT_TOKENS=2048
```

### 3. Khởi chạy với Docker (khuyến nghị)

```bash
docker compose -f docker/docker-compose.yml up -d
```

Các service sẽ chạy:

| Service | Port | Mô tả |
|---------|------|--------|
| **app** | `8501` | Streamlit Web UI |
| **llm_proxy** | `5000` | FastAPI LLM Proxy |
| **qdrant** | `6333` | Vector Database |
| **postgres** | `5432` | SQL Database |

Truy cập ứng dụng tại: **http://localhost:8501**

### 4. Khởi chạy thủ công (development)

```bash
# Khởi tạo database
python -c "from database.database import init_db; init_db()"

# Chạy LLM Proxy
uvicorn src.llm_proxy:app --host 0.0.0.0 --port 5000 &

# Chạy Streamlit
streamlit run src/app_v2.py
```

### 5. Thêm tài liệu PDF

Đặt file PDF vào thư mục `data/` theo cấu trúc môn học:

```
data/
├── Giải tích 1/
│   ├── chuong1.pdf
│   └── chuong2.pdf
├── Thống kê trong kinh doanh/
│   └── textbook.pdf
└── Vật lý 2/
    └── lecture_notes.pdf
```

Tài liệu sẽ được tự động ingest khi ứng dụng khởi chạy.

---

## ⚙️ Cấu hình chi tiết

Các class cấu hình được định nghĩa trong `src/config.py`:

| Class | Mô tả | Tham số quan trọng |
|-------|--------|-------------------|
| `EmbeddingConfig` | Embedding model | `model_name`, `batch_size`, `local_path` |
| `QdrantConfig` | Vector DB | `host`, `port` (6333), `collection_name` |
| `ChunkingConfig` | Chia nhỏ văn bản | `chunk_size` (1000), `chunk_overlap` (200) |
| `LLMConfig` | LLM selection | `current_model`, `temperature`, `proxy_url` |
| `RetrievalConfig` | Tìm kiếm RAG | `top_k` (5), `score_threshold` (0.3) |
| `AppConfig` | UI & đường dẫn | `data_folder`, `log_folder`, `page_title` |

### Các model LLM hỗ trợ

```python
AVAILABLE_MODELS = {
    "gemini": {          # Google Gemini 2.5 Flash (API)
        "type": "api",
        "provider": "google"
    },
    "qwen2-7b": {        # Qwen 2.5-7B Instruct (vLLM local)
        "type": "proxy",
        "provider": "vllm"
    }
}
```

---

## 🗄️ Database Schema

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│    users     │────►│  conversations   │────►│      messages        │
├──────────────┤     ├──────────────────┤     ├──────────────────────┤
│ id           │     │ id               │     │ id                   │
│ email        │     │ user_id (FK)     │     │ conversation_id (FK) │
│ username     │     │ title            │     │ role (user/assistant)│
│ password_hash│     │ subject          │     │ content              │
│ full_name    │     │ is_pinned        │     │ sources (JSON)       │
│ is_active    │     │ is_archived      │     │ tokens_used          │
│ is_admin     │     │ created_at       │     │ processing_time      │
│ created_at   │     │ updated_at       │     │ is_deleted           │
│ last_login   │     └──────────────────┘     │ created_at           │
└──────┬───────┘              │                └──────────────────────┘
       │              ┌───────┴────────────┐
       │              │shared_conversations│
       ▼              ├────────────────────┤
┌──────────────┐      │ id                 │
│user_settings │      │ conversation_id(FK)│
├──────────────┤      │ share_token        │
│ user_id (FK) │      │ is_active          │
│ theme        │      │ view_count         │
│ language     │      │ expires_at         │
│ temperature  │      └────────────────────┘
│ max_tokens   │
│ top_k        │
└──────────────┘
```

---

## 🔌 API Endpoints (LLM Proxy)

| Endpoint | Method | Mô tả |
|----------|--------|--------|
| `/` | GET | Trạng thái service |
| `/health` | GET | Health check (số server healthy) |
| `/servers` | GET | Danh sách server và trạng thái |
| `/v1/chat/completions` | POST | Chat completion (OpenAI-compatible, hỗ trợ streaming) |

---

## 🔐 Xác thực

- **Đăng ký**: Email + username + password (tối thiểu 6 ký tự)
- **Mật khẩu**: Mã hóa bằng bcrypt với salt
- **Phiên đăng nhập**: Quản lý qua Streamlit session state
- **Đổi mật khẩu**: Xác thực mật khẩu cũ trước khi thay đổi

---

## 🐳 Docker

### Services

```yaml
services:
  postgres:     # PostgreSQL 15-alpine — port 5432
  qdrant:       # Qdrant vector DB   — port 6333
  llm_proxy:    # FastAPI proxy       — port 5000
  app:          # Streamlit UI        — port 8501
```

### Health Checks

- **PostgreSQL**: `pg_isready` (10s interval)
- **LLM Proxy**: HTTP `/health` (30s interval, 3 retries)
- **App**: Phụ thuộc vào postgres, qdrant, llm_proxy healthy

### Build & Deploy

```bash
# Build và chạy tất cả services
docker compose -f docker/docker-compose.yml up -d --build

# Xem logs
docker compose -f docker/docker-compose.yml logs -f app

# Dừng tất cả
docker compose -f docker/docker-compose.yml down
```

---

## 📝 Logging

- **File log**: `logs/rag_YYYYMMDD.log` (xoay vòng theo ngày)
- **Console**: Chỉ hiển thị WARNING trở lên
- **Format**: `Timestamp | Logger | Level | Message`

---

## 🛣️ Roadmap

- [ ] Admin dashboard quản lý người dùng
- [ ] Hỗ trợ thêm định dạng file (DOCX, PPTX)
- [ ] Token-based authentication (JWT)
- [ ] Multi-language support
- [ ] Export hội thoại ra PDF
- [ ] Analytics & thống kê sử dụng

---

## 📄 License

Dự án này được phát triển cho mục đích giáo dục.