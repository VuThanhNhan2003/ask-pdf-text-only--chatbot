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

---

## 🎯 Gợi ý cấu trúc Report/Thesis:

### **Chương 2: System Design**
- Diagram 1, 2, 3 (Architecture, Query flow, Fusion strategy)

### **Chương 3: Implementation**
- Diagram 4, 5, 6, 7 (Ingestion, Reranking, LLM abstraction, DB schema)

### **Chương 4: Retrieval Strategy**
- Diagram 3 (Fusion comparison + formulas)
- Diagram 8 (Candidate pool tuning)

### **Chương 5: Deployment**
- Diagram 6 (Proxy strategy)
- Diagram 10 (Docker stack)

### **Chương 6: Evaluation/Results**
- Chart so sánh score fusion vs RRF
- Chart ảnh hưởng candidate multiplier đến recall/latency

---

## 💡 Công cụ vẽ diagram:

1. **Draw.io** / **Excalidraw** - Free, online, dễ dùng
2. **Mermaid** - Vẽ bằng code (tích hợp vào report nếu dùng Markdown/LaTeX)
3. **Lucidchart** - Professional
4. **TikZ** (LaTeX) - Cho thèsisTeX

Bạn muốn mình vẽ diagram nào bằng **Mermaid** luôn không? Tôi có thể render cho bạn xem.