# IEEE-Style Diagrams (Mermaid)

## Figure 1. System Architecture Overview

```mermaid
flowchart LR
  %% System-level architecture for the RAG chatbot deployment
  U["Users"] --> UI["UI Layer\nStreamlit App"]
  U --> API["API Layer\nFastAPI Service"]

  UI --> RAG["RAG Processor\n(retrieval + rerank + prompt)"]
  API --> RAG

  RAG --> QDR["Vector DB\nQdrant"]
  RAG --> SQL["Relational DB\nPostgreSQL"]
  RAG --> LLM["LLM Manager"]

  LLM --> GEM["Gemini API"]
  LLM --> PROXY["LLM Proxy\n(FastAPI)"]
  PROXY --> VLLM["vLLM Server\nGPU Node"]

  subgraph CPU["CPU Server (edx1)"]
    UI
    API
    RAG
    QDR
    SQL
    PROXY
  end

  subgraph GPU["Slurm GPU Cluster"]
    VLLM
  end

  classDef layer fill:#eef7ff,stroke:#2b4c7e,stroke-width:1px;
  classDef store fill:#fff6e6,stroke:#8a5a00,stroke-width:1px;
  classDef llm fill:#f0f7f0,stroke:#1f6f3b,stroke-width:1px;

  class UI,API,RAG layer;
  class QDR,SQL store;
  class LLM,PROXY,GEM,VLLM llm;
```
```

## Figure 2. End-to-End Query Processing Pipeline (IEEE Stepwise)

```mermaid
flowchart TD
  %% IEEE-style stepwise processing pipeline
  S0(["Start: User Query"]) --> S1["Step 1: Normalize input\n(language, subject, history flag)"]
  S1 --> S2["Step 2: Embed query\nSentence-Transformers"]
  S2 --> S3["Step 3: Dense retrieval\nQdrant cosine search"]
  S2 --> S4["Step 4: Lexical retrieval\nBM25 on payload text"]
  S3 --> S5["Step 5: Hybrid fusion\n(score or RRF)"]
  S4 --> S5
  S5 --> S6["Step 6: Candidate pool sizing\nmultiplier + cap"]
  S6 --> S7["Step 7: Cross-encoder rerank\nBGE reranker"]
  S7 --> S8["Step 8: Prompt assembly\ncontext + short history"]
  S8 --> S9["Step 9: LLM generation\nProxy or Gemini"]
  S9 --> S10(["End: Answer + Sources"])
```
```

## Figure 3. LLM Proxy Resilience Flow (IEEE Stepwise)

```mermaid
flowchart TD
  P0(["Start: LLM Request"]) --> P1["Step 1: Health check vLLM"]
  P1 -->|"healthy"| P2["Step 2: Forward request\nOpenAI-compatible API"]
  P1 -->|"unhealthy"| P3["Step 3: Retry with backoff\n(max N attempts)"]
  P3 -->|"recovered"| P2
  P3 -->|"failed"| P4["Step 4: Fallback to Gemini"]
  P2 --> P5(["End: Response to caller"])
  P4 --> P5
```

## Figure 4. Overall System Architecture (SoC: Offline vs Online)

```mermaid
flowchart TB
  %% High-level SoC view with minimal nodes
  subgraph OFF["Offline: Multimodal Ingestion"]
    A1["Extract + Normalize\nFaster-Whisper, Qwen-VL"] --> A2["Knowledge Artifacts"]
  end

  subgraph ON["Online: Hybrid RAG Core"]
    B1["Retrieval + Reranking\nBGE-M3, BM25"]
  end

  subgraph SERV["Serving Layer"]
    C1["FastAPI Gateway"] --> C2["Fault-Tolerant LLM\nvLLM + Gemini Fallback"]
  end

  A2 --> B1 --> C1
```
```
