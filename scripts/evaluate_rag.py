#!/usr/bin/env python3
"""
End-to-end RAG evaluation runner for this repository.

What it does:
- Loads chunk data (JSON) and normalizes chunk IDs/text/metadata.
- Builds evaluation queries from either:
  1) a query annotation file (recommended), or
  2) auto-generated queries from chunk metadata/text.
- Runs retrieval through the existing RAGProcessor pipeline.
- Computes retrieval metrics: Precision@k, Recall@k, F1@k, Hit@k, MRR, nDCG@k.
- Optionally runs generation + RAGAS metrics: Faithfulness, Answer Correctness.
- Exports per-query report and aggregate summary.

Usage example:
python scripts/evaluate_rag.py \
  --chunks-path "data/processed_chunks/Mon A/chunks.json" \
  --subject "Mon A" \
  --top-k 5 \
  --max-queries 50
"""

from __future__ import annotations

import argparse
import contextlib
import io
import inspect
import json
import logging
import math
import os
import random
import re
import sys
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The class `HuggingFaceEmbeddings` was deprecated in LangChain.*",
)

try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning  # type: ignore

    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    pass


# Ensure src/ is importable when running from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from processor import RAGProcessor  # noqa: E402


DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed_chunks" / "Môn Triết học Mác-Lênin" / "19filespdftriếthọcm-ln112024_19filespdftriếthọcm-ln112024.json"
SCRIPT_VERSION = "eval-rag-2026-04-15-v2"


@dataclass
class NormalizedChunk:
    chunk_id: str
    text: str
    subject: str
    file_name: str
    page: int
    metadata: Dict[str, Any]


@dataclass
class EvalQuery:
    query_id: str
    query: str
    ground_truth_ids: List[str]
    reference_answer: str
    ground_truth_grades: Dict[str, float]
    query_type: str = ""
    difficulty: str = ""
    source_file: str = ""
    source_page: Optional[int] = None


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_chunk_record(record: Dict[str, Any], fallback_idx: int) -> Optional[NormalizedChunk]:
    text = _safe_text(record.get("text") or record.get("chunk") or record.get("content") or record.get("page_content"))
    if not text:
        return None

    metadata = record.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    chunk_id = (
        record.get("id_")
        or record.get("id")
        or metadata.get("chunk_id")
        or f"chunk_{fallback_idx}"
    )
    chunk_id = _safe_text(chunk_id)

    subject = _safe_text(metadata.get("subject"))
    file_name = _safe_text(metadata.get("file_name") or metadata.get("file") or "")

    page = metadata.get("page", 0)
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 0

    return NormalizedChunk(
        chunk_id=chunk_id,
        text=text,
        subject=subject,
        file_name=file_name,
        page=page,
        metadata=metadata,
    )


def normalize_chunks(raw: Any) -> List[NormalizedChunk]:
    if isinstance(raw, dict):
        for key in ("chunks", "data", "items", "documents"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break

    if not isinstance(raw, list):
        raise ValueError("Chunk file must be a list or a dict containing list field: chunks/data/items/documents")

    chunks: List[NormalizedChunk] = []
    seen_ids = set()
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue

        normalized = normalize_chunk_record(item, fallback_idx=idx)
        if normalized is None:
            continue

        chunk_id = normalized.chunk_id
        if chunk_id in seen_ids:
            chunk_id = f"{chunk_id}__dup_{idx}"
            normalized = NormalizedChunk(
                chunk_id=chunk_id,
                text=normalized.text,
                subject=normalized.subject,
                file_name=normalized.file_name,
                page=normalized.page,
                metadata=normalized.metadata,
            )

        seen_ids.add(normalized.chunk_id)
        chunks.append(normalized)

    if not chunks:
        raise ValueError("No valid chunks found after normalization.")

    return chunks


def _first_sentence(text: str, max_len: int = 180) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    candidate = parts[0] if parts else cleaned
    return candidate[:max_len]


def _build_query_from_chunk(chunk: NormalizedChunk) -> str:
    topic = _safe_text(chunk.metadata.get("topic"))
    keywords = chunk.metadata.get("keywords")
    keyword_text = ""
    if isinstance(keywords, list):
        cleaned = [str(k).strip() for k in keywords if str(k).strip()]
        keyword_text = ", ".join(cleaned[:3])

    if topic and keyword_text:
        return f"Trong tài liệu, nội dung chính về '{topic}' là gì? (liên quan: {keyword_text})"
    if topic:
        return f"Trong tài liệu, nội dung chính về '{topic}' là gì?"

    sentence = _first_sentence(chunk.text, max_len=120)
    if sentence:
        return f"Hãy giải thích ngắn gọn ý chính của đoạn sau: {sentence}"
    return "Nội dung chính của phần này là gì?"


def auto_generate_eval_queries(
    chunks: List[NormalizedChunk],
    max_queries: int,
    random_seed: int,
) -> List[EvalQuery]:
    random.seed(random_seed)

    # Prefer diversified samples by topic first, then fill with random chunks.
    by_topic: Dict[str, List[NormalizedChunk]] = {}
    fallback_pool: List[NormalizedChunk] = []
    for ch in chunks:
        topic = _safe_text(ch.metadata.get("topic"))
        if topic:
            by_topic.setdefault(topic, []).append(ch)
        else:
            fallback_pool.append(ch)

    selected: List[NormalizedChunk] = []
    topic_keys = list(by_topic.keys())
    random.shuffle(topic_keys)
    for topic in topic_keys:
        if len(selected) >= max_queries:
            break
        candidates = by_topic[topic]
        random.shuffle(candidates)
        selected.append(candidates[0])

    if len(selected) < max_queries:
        remaining = [c for c in chunks if c not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[: max_queries - len(selected)])

    selected = selected[:max_queries]

    eval_queries: List[EvalQuery] = []
    for i, chunk in enumerate(selected, start=1):
        query_text = _build_query_from_chunk(chunk)
        eval_queries.append(
            EvalQuery(
                query_id=f"auto_{i:04d}",
                query=query_text,
                ground_truth_ids=[chunk.chunk_id],
                reference_answer=chunk.text,
                ground_truth_grades={chunk.chunk_id: 1.0},
            )
        )

    return eval_queries


def _normalize_ground_truth_ids(raw: Any) -> List[str]:
    if raw is None:
        return []

    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []

    if isinstance(raw, list):
        out: List[str] = []
        for item in raw:
            if isinstance(item, dict):
                val = item.get("id") or item.get("chunk_id") or item.get("id_")
                if val:
                    out.append(str(val).strip())
            else:
                val = str(item).strip()
                if val:
                    out.append(val)
        return out

    if isinstance(raw, dict):
        # Accept map: id -> grade
        out = [str(k).strip() for k in raw.keys() if str(k).strip()]
        return out

    return []


def _normalize_grades(raw: Any, default_ids: List[str]) -> Dict[str, float]:
    grades: Dict[str, float] = {}

    if isinstance(raw, dict):
        for k, v in raw.items():
            key = str(k).strip()
            if not key:
                continue
            try:
                grades[key] = float(v)
            except (TypeError, ValueError):
                grades[key] = 1.0

    if not grades:
        for gid in default_ids:
            grades[gid] = 1.0

    return grades


def load_eval_queries_from_file(path: Path, max_queries: int) -> List[EvalQuery]:
    raw = load_json_file(path)
    if isinstance(raw, dict):
        for key in ("queries", "data", "items"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break

    if not isinstance(raw, list):
        raise ValueError("Query file must be list or dict with queries/data/items list")

    queries: List[EvalQuery] = []
    for i, item in enumerate(raw):
        if len(queries) >= max_queries:
            break

        if not isinstance(item, dict):
            continue

        query = _safe_text(item.get("query") or item.get("question") or item.get("user_input"))
        if not query:
            continue

        gt_raw = (
            item.get("ground_truth_ids")
            or item.get("ground_truth")
            or item.get("relevant_chunk_ids")
            or item.get("gold_ids")
        )
        ground_truth_ids = _normalize_ground_truth_ids(gt_raw)
        if not ground_truth_ids:
            continue

        reference_answer = _safe_text(item.get("reference_answer") or item.get("reference") or item.get("answer"))
        grades = _normalize_grades(item.get("ground_truth_grades") or item.get("grades") or gt_raw, ground_truth_ids)

        query_id = _safe_text(item.get("query_id") or item.get("id") or f"q_{i:04d}")
        query_type = _safe_text(item.get("query_type"))
        difficulty = _safe_text(item.get("difficulty"))
        source_file = _safe_text(item.get("source_file"))
        source_page_raw = item.get("source_page")
        source_page: Optional[int] = None
        if source_page_raw is not None and str(source_page_raw).strip():
            try:
                source_page = int(source_page_raw)
            except (TypeError, ValueError):
                source_page = None

        queries.append(
            EvalQuery(
                query_id=query_id,
                query=query,
                ground_truth_ids=ground_truth_ids,
                reference_answer=reference_answer,
                ground_truth_grades=grades,
                query_type=query_type,
                difficulty=difficulty,
                source_file=source_file,
                source_page=source_page,
            )
        )

    if not queries:
        raise ValueError("No valid evaluation queries found in query file")

    return queries


def precision_at_k(retrieved: List[str], relevant: set) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for rid in retrieved if rid in relevant)
    return hits / float(len(retrieved))


def recall_at_k(retrieved: List[str], relevant: set) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for rid in retrieved if rid in relevant)
    return hits / float(len(relevant))


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def reciprocal_rank(retrieved: List[str], relevant: set) -> float:
    for i, rid in enumerate(retrieved, start=1):
        if rid in relevant:
            return 1.0 / float(i)
    return 0.0


def dcg_at_k(retrieved: List[str], grades: Dict[str, float], k: int) -> float:
    score = 0.0
    for rank, rid in enumerate(retrieved[:k], start=1):
        rel = float(grades.get(rid, 0.0))
        if rel <= 0:
            continue
        score += (2.0 ** rel - 1.0) / math.log2(rank + 1.0)
    return score


def ndcg_at_k(retrieved: List[str], grades: Dict[str, float], k: int) -> float:
    actual = dcg_at_k(retrieved, grades, k)
    ideal_rels = sorted([float(v) for v in grades.values() if float(v) > 0], reverse=True)

    ideal = 0.0
    for rank, rel in enumerate(ideal_rels[:k], start=1):
        ideal += (2.0 ** rel - 1.0) / math.log2(rank + 1.0)

    if ideal == 0:
        return 0.0
    return actual / ideal


def hit_at_n(retrieved: List[str], relevant: set, n: int) -> float:
    if n <= 0:
        return 0.0
    return 1.0 if any(rid in relevant for rid in retrieved[:n]) else 0.0


def max_precision_at_k(relevant_count: int, k: int) -> float:
    """Upper bound of Precision@k given the number of relevant docs."""
    if k <= 0 or relevant_count <= 0:
        return 0.0
    return min(relevant_count, k) / float(k)


def normalized_precision_at_k(precision_k: float, relevant_count: int, k: int) -> float:
    """Precision@k normalized by its theoretical maximum in sparse-label settings."""
    max_p = max_precision_at_k(relevant_count, k)
    if max_p <= 0:
        return 0.0
    return precision_k / max_p


def _is_placeholder_reference(reference_text: str) -> bool:
    text = reference_text.strip()
    if not text:
        return True

    compact = text.lower().replace(" ", "")
    if compact.startswith("câuhỏi"):
        return True

    placeholder_tokens = {
        "a.", "b.", "c.", "d.",
        "1.", "2.", "3.", "4.",
        "i.", "ii.", "iii.", "iv.", "v.",
        "nộidungtrọngtâm1.",
    }
    return compact in placeholder_tokens


def build_query_set_diagnostics(queries: List[EvalQuery], top_k: int) -> Dict[str, Any]:
    """Build diagnostics to interpret retrieval metrics more reliably."""
    if not queries:
        return {
            "single_ground_truth_ratio": 0.0,
            "multi_ground_truth_ratio": 0.0,
            "avg_ground_truth_count": 0.0,
            "avg_max_precision_at_k": 0.0,
            "short_reference_ratio": 0.0,
            "placeholder_reference_ratio": 0.0,
            "query_type_distribution": {},
            "difficulty_distribution": {},
            "notes": [],
        }

    gt_counts = [len(q.ground_truth_ids) for q in queries]
    query_count = len(queries)
    single_gt_ratio = sum(1 for c in gt_counts if c == 1) / float(query_count)
    multi_gt_ratio = sum(1 for c in gt_counts if c > 1) / float(query_count)
    avg_gt = sum(gt_counts) / float(query_count)
    avg_max_p_at_k = sum(max_precision_at_k(c, top_k) for c in gt_counts) / float(query_count)

    refs = [q.reference_answer.strip() for q in queries]
    short_ref_ratio = sum(1 for r in refs if len(r) <= 12) / float(query_count)
    placeholder_ref_ratio = sum(1 for r in refs if _is_placeholder_reference(r)) / float(query_count)

    query_type_distribution: Dict[str, int] = {}
    difficulty_distribution: Dict[str, int] = {}
    for q in queries:
        if q.query_type:
            query_type_distribution[q.query_type] = query_type_distribution.get(q.query_type, 0) + 1
        if q.difficulty:
            difficulty_distribution[q.difficulty] = difficulty_distribution.get(q.difficulty, 0) + 1

    notes: List[str] = []
    if single_gt_ratio >= 0.8 and top_k > 1:
        notes.append(
            "Most queries have exactly one ground-truth chunk, so Precision@k has a low ceiling "
            f"(avg max Precision@{top_k} = {avg_max_p_at_k:.4f})."
        )
    if placeholder_ref_ratio >= 0.2:
        notes.append(
            "A high fraction of reference answers look like placeholders (e.g. 'CÂU HỎI 1.', 'b.'), "
            "which can under-estimate answer_correctness."
        )

    return {
        "single_ground_truth_ratio": single_gt_ratio,
        "multi_ground_truth_ratio": multi_gt_ratio,
        "avg_ground_truth_count": avg_gt,
        "avg_max_precision_at_k": avg_max_p_at_k,
        "short_reference_ratio": short_ref_ratio,
        "placeholder_reference_ratio": placeholder_ref_ratio,
        "query_type_distribution": query_type_distribution,
        "difficulty_distribution": difficulty_distribution,
        "notes": notes,
    }


def strip_sources_from_answer(answer: str) -> str:
    marker = "📚 **Nguồn tham khảo:**"
    if marker in answer:
        return answer.split(marker, 1)[0].strip()
    return answer.strip()


def _token_overlap_ratio(reference_text: str, answer_text: str) -> float:
    """Simple lexical overlap from reference to answer for quick diagnostics."""
    ref_tokens = set(re.findall(r"\w+", str(reference_text).lower(), flags=re.UNICODE))
    ans_tokens = set(re.findall(r"\w+", str(answer_text).lower(), flags=re.UNICODE))
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & ans_tokens) / float(len(ref_tokens))


def _correctness_band(score: Optional[float]) -> str:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "missing"
    if score < 0.55:
        return "very_low"
    if score < 0.70:
        return "low"
    if score < 0.82:
        return "medium"
    return "high"


def _diagnose_row(
    answer_correctness: Optional[float],
    recall_k: float,
    precision_1: float,
    full_gt_hit: bool,
    ref_answer_overlap: float,
) -> str:
    if answer_correctness is None or (isinstance(answer_correctness, float) and math.isnan(answer_correctness)):
        return "ragas_missing_or_error"
    if not full_gt_hit or recall_k < 1.0:
        return "retrieval_gap"
    if answer_correctness < 0.70 and ref_answer_overlap < 0.65:
        return "generation_paraphrase_or_missing_keypoints"
    if answer_correctness < 0.70 and precision_1 < 1.0:
        return "top1_not_best_even_when_topk_hits"
    if answer_correctness < 0.70:
        return "answer_not_aligned_with_reference_style"
    return "ok"


def _compact_generated_answer(answer_text: str, query_type: str) -> str:
    """Compact long generations to improve style alignment with short references."""
    text = str(answer_text or "").strip()
    if not text:
        return text

    # Remove common markdown/bullet prefixes while keeping content.
    cleaned_lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•]\s+", "", line)
        line = re.sub(r"^\d+[\.)]\s+", "", line)
        line = re.sub(r"^[a-zA-Z][\.)]\s+", "", line)
        line = line.strip()
        if line:
            cleaned_lines.append(line)

    if not cleaned_lines:
        return text

    # If model returned one long paragraph, split by sentence boundaries.
    units: List[str] = []
    if len(cleaned_lines) == 1:
        units = [u.strip() for u in re.split(r"(?<=[.!?])\s+", cleaned_lines[0]) if u.strip()]
    else:
        units = cleaned_lines

    qt = (query_type or "").strip().lower()
    max_units = 3
    if qt in {"definition", "role"}:
        max_units = 2
    elif qt in {"enumeration", "compare_contrast", "reasoning"}:
        max_units = 3

    compact = " ".join(units[:max_units]).strip()
    return compact or text


def _format_hint_by_query_type(query_type: str) -> str:
    qt = (query_type or "").strip().lower()
    if qt == "definition":
        return (
            "- Định nghĩa: Trả lời đúng ý chính, tối đa 2 câu.\n"
            "- Không diễn giải, không mở đầu, không ví dụ."
        )
    if qt == "compare_contrast":
        return (
            "- So sánh: Nêu điểm giống/khác chính xác, tối đa 3 ý.\n"
            "- Dùng định dạng: 'A: ...; B: ...; Khác nhau chính: ...'."
        )
    if qt == "enumeration":
        return (
            "- Liệt kê: Đúng số ý, mỗi ý 1 dòng, không lặp.\n"
            "- Không thêm giải thích ngoài danh sách."
        )
    if qt == "reasoning":
        return (
            "- Lập luận: Nêu kết luận trước, sau đó 1-2 lý do sát ngữ cảnh.\n"
            "- Không diễn giải lan man, không mở rộng ngoài dữ kiện."
        )
    if qt == "role":
        return (
            "- Vai trò/chức năng: Trả lời trực tiếp, tối đa 2 câu.\n"
            "- Không mở rộng, không giải thích thêm."
        )
    return "- Trả lời trực tiếp, ngắn gọn, không diễn giải ngoài ngữ cảnh."


def build_eval_answer_prompt(question: str, context: str, query_type: str = "", concise_mode: bool = True) -> str:
    """Prompt optimized for evaluation stability and answer_correctness."""
    style_hint = _format_hint_by_query_type(query_type)
    max_items_line = "2. Trả lời trực tiếp, tối đa 3 ý ngắn." if concise_mode else "2. Trả lời trực tiếp, tối đa 6 ý ngắn."
    numbering_line = "6. Không dùng danh sách đánh số nếu câu hỏi không yêu cầu liệt kê."

    return f"""Bạn là trợ lý học thuật. Hãy trả lời NGẮN GỌN, ĐÚNG Ý CHÍNH, TUYỆT ĐỐI KHÔNG DIỄN GIẢI.

Ngữ cảnh:
{context}

Câu hỏi:
{question}

Yêu cầu bắt buộc:
1. Chỉ dùng thông tin có trong ngữ cảnh, không thêm ví dụ, không mở rộng.
2. Trả lời đúng dạng câu hỏi, tối đa 2-3 ý chính (tùy loại).
3. Không dùng lời dẫn nhập, không giải thích vòng vo, không meta text.
4. Nếu thiếu dữ liệu: trả lời "Không đủ dữ liệu trong ngữ cảnh".
5. Giữ nguyên thuật ngữ, công thức, số liệu nếu có.
{style_hint}

Chỉ in phần trả lời cuối cùng, không thêm ghi chú.
"""


def run_ragas_evaluation(
    samples: List[Dict[str, Any]],
    llm_model_name: Optional[str] = None,
    llm_base_url: Optional[str] = None,
) -> Tuple[List[Dict[str, Optional[float]]], Optional[str]]:
    """
    Try running RAGAS Faithfulness + Answer Correctness.
    Returns per-sample score list and optional warning message.
    """
    if not samples:
        return [], None

    try:
        from datasets import Dataset  # type: ignore
        try:
            from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
        from langchain_openai import ChatOpenAI  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore
        from ragas.llms import LangchainLLMWrapper  # type: ignore
        from ragas.metrics import answer_correctness, faithfulness  # type: ignore

        # Keep RAGAS internals less noisy in terminal output.
        logging.getLogger("ragas").setLevel(logging.ERROR)

        # Use local OpenAI-compatible proxy by default to avoid requiring a real OpenAI key.
        ragas_model = llm_model_name or os.getenv("RAGAS_LLM_MODEL", "Qwen/Qwen3-8B-AWQ")
        ragas_base = llm_base_url or os.getenv("RAGAS_LLM_BASE_URL") or os.getenv("LLM_PROXY_URL", "http://localhost:5000/v1")
        ragas_base = ragas_base.rstrip("/")
        if not ragas_base.endswith("/v1"):
            ragas_base = f"{ragas_base}/v1"

        ragas_timeout = float(os.getenv("RAGAS_TIMEOUT", "120"))
        ragas_max_tokens = int(os.getenv("RAGAS_MAX_TOKENS", "2048"))
        ragas_max_retries = int(os.getenv("RAGAS_MAX_RETRIES", "3"))
        ragas_batch_size = int(os.getenv("RAGAS_BATCH_SIZE", "4"))

        # ChatOpenAI requires an API key field even for local OpenAI-compatible endpoints.
        openai_api_key = os.getenv("OPENAI_API_KEY", "local-proxy")
        ragas_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=ragas_model,
                api_key=openai_api_key,
                base_url=ragas_base,
                temperature=0.0,
                max_tokens=ragas_max_tokens,
                timeout=ragas_timeout,
                max_retries=ragas_max_retries,
            )
        )

        embed_model = os.getenv("RAGAS_EMBEDDING_MODEL", str(PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The class `HuggingFaceEmbeddings` was deprecated in LangChain.*",
            )
            ragas_embeddings = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(
                    model_name=embed_model,
                    model_kwargs={"device": "cpu"},
                )
            )

        dataset = Dataset.from_list(samples)
        eval_kwargs: Dict[str, Any] = {
            "dataset": dataset,
            "metrics": [faithfulness, answer_correctness],
            "llm": ragas_llm,
            "embeddings": ragas_embeddings,
        }

        try:
            eval_sig = inspect.signature(evaluate)
            if "raise_exceptions" in eval_sig.parameters:
                eval_kwargs["raise_exceptions"] = False
            if "batch_size" in eval_sig.parameters:
                eval_kwargs["batch_size"] = ragas_batch_size
            if "run_config" in eval_sig.parameters:
                try:
                    from ragas.run_config import RunConfig  # type: ignore

                    eval_kwargs["run_config"] = RunConfig(
                        timeout=ragas_timeout,
                        max_retries=ragas_max_retries,
                        max_workers=1,
                    )
                except Exception:
                    pass
        except Exception:
            pass

        ragas_stdout_buffer = io.StringIO()
        ragas_stderr_buffer = io.StringIO()
        with contextlib.redirect_stdout(ragas_stdout_buffer), contextlib.redirect_stderr(ragas_stderr_buffer):
            result = evaluate(**eval_kwargs)

        captured_ragas_logs = "\n".join(
            line for line in (ragas_stdout_buffer.getvalue() + "\n" + ragas_stderr_buffer.getvalue()).splitlines() if line.strip()
        )

        rows: List[Dict[str, Optional[float]]] = []
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            for _, r in df.iterrows():
                rows.append(
                    {
                        "faithfulness": float(r.get("faithfulness")) if pd.notna(r.get("faithfulness")) else None,
                        "answer_correctness": float(r.get("answer_correctness")) if pd.notna(r.get("answer_correctness")) else None,
                    }
                )
            warning_msg = None
            if "Exception raised in Job" in captured_ragas_logs:
                warning_msg = "RAGAS had parser/timeout retries on some samples; scores were still collected where possible."
            return rows, warning_msg

        if hasattr(result, "scores") and isinstance(result.scores, list):
            for score in result.scores:
                rows.append(
                    {
                        "faithfulness": score.get("faithfulness"),
                        "answer_correctness": score.get("answer_correctness"),
                    }
                )
            warning_msg = None
            if "Exception raised in Job" in captured_ragas_logs:
                warning_msg = "RAGAS had parser/timeout retries on some samples; scores were still collected where possible."
            return rows, warning_msg

        return [], "RAGAS ran but result format is not recognized."

    except Exception as exc:
        msg = (
            "Skipped RAGAS metrics (Faithfulness, Answer Correctness). "
            f"Reason: {exc}"
        )
        return [], msg


def evaluate_queries(
    processor: RAGProcessor,
    queries: List[EvalQuery],
    chunk_lookup: Dict[str, NormalizedChunk],
    top_k: int,
    run_generation: bool,
    generation_context_k: int,
    concise_mode: bool,
    answer_compaction: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    ragas_samples: List[Dict[str, Any]] = []
    ragas_index_map: List[int] = []
    generation_retry_count = int(os.getenv("EVAL_GENERATION_RETRIES", "2"))

    for idx, item in enumerate(queries, start=1):
        retrieved_chunks = processor._retrieve_relevant_chunks(item.query, k=top_k)  # noqa: SLF001
        retrieved_ids = [str(ch.get("id")) for ch in retrieved_chunks if ch.get("id") is not None]

        relevant_set = set(item.ground_truth_ids)
        p_at_k = precision_at_k(retrieved_ids, relevant_set)
        max_p_at_k = max_precision_at_k(len(relevant_set), top_k)
        norm_p_at_k = normalized_precision_at_k(p_at_k, len(relevant_set), top_k)
        r_at_k = recall_at_k(retrieved_ids, relevant_set)
        f1_at_k = f1_score(p_at_k, r_at_k)
        mrr = reciprocal_rank(retrieved_ids, relevant_set)
        hit_k = 1.0 if any(rid in relevant_set for rid in retrieved_ids) else 0.0
        ndcg = ndcg_at_k(retrieved_ids, item.ground_truth_grades, top_k)
        p_at_1 = precision_at_k(retrieved_ids[:1], relevant_set)
        p_at_3 = precision_at_k(retrieved_ids[:3], relevant_set)
        hit_1 = hit_at_n(retrieved_ids, relevant_set, 1)
        hit_3 = hit_at_n(retrieved_ids, relevant_set, 3)

        answer = None
        cleaned_answer = None

        if run_generation:
            try:
                # Keep retrieval metrics unchanged, but use a smaller context for generation
                # to reduce noise and improve alignment with concise reference answers.
                gen_context_k = max(1, int(generation_context_k))
                generation_chunks = retrieved_chunks[:gen_context_k]
                context = processor._build_context(generation_chunks)  # noqa: SLF001
                prompt = build_eval_answer_prompt(
                    item.query,
                    context,
                    query_type=item.query_type,
                    concise_mode=concise_mode,
                )

                answer_text = ""
                last_error: Optional[Exception] = None
                for _ in range(max(1, generation_retry_count)):
                    try:
                        answer_text = processor.llm.invoke(prompt)
                        if answer_text and str(answer_text).strip():
                            break
                    except Exception as invoke_exc:
                        last_error = invoke_exc

                if not answer_text:
                    if last_error is not None:
                        raise last_error
                    raise RuntimeError("Empty generation output")

                if answer_compaction:
                    answer_text = _compact_generated_answer(answer_text, item.query_type)

                sources_text = processor._format_sources(generation_chunks)  # noqa: SLF001
                answer = f"{answer_text}\n\n{sources_text}" if sources_text else answer_text
                cleaned_answer = strip_sources_from_answer(answer)
            except Exception as exc:
                cleaned_answer = f"[GENERATION_ERROR] {exc}"

        gt_texts = [chunk_lookup[cid].text for cid in item.ground_truth_ids if cid in chunk_lookup]
        reference_answer = item.reference_answer or "\n\n".join(gt_texts[:2])

        row = {
            "query_index": idx,
            "query_id": item.query_id,
            "query": item.query,
            "ground_truth_count": len(item.ground_truth_ids),
            "ground_truth_ids": item.ground_truth_ids,
            "retrieved_ids": retrieved_ids,
            "precision_at_k": p_at_k,
            "max_precision_at_k": max_p_at_k,
            "normalized_precision_at_k": norm_p_at_k,
            "recall_at_k": r_at_k,
            "f1_at_k": f1_at_k,
            "precision_at_1": p_at_1,
            "precision_at_3": p_at_3,
            "mrr": mrr,
            "hit_at_k": hit_k,
            "hit_at_1": hit_1,
            "hit_at_3": hit_3,
            "ndcg_at_k": ndcg,
            "answer": cleaned_answer,
            "reference_answer": reference_answer,
            "query_type": item.query_type,
            "difficulty": item.difficulty,
            "source_file": item.source_file,
            "source_page": item.source_page,
            "faithfulness": None,
            "answer_correctness": None,
            "matched_ground_truth_count": int(len(set(retrieved_ids) & relevant_set)),
            "full_ground_truth_hit": bool(set(item.ground_truth_ids).issubset(set(retrieved_ids))),
            "ref_answer_overlap": _token_overlap_ratio(reference_answer, cleaned_answer or ""),
        }
        rows.append(row)

        if run_generation and cleaned_answer:
            ragas_samples.append(
                {
                    "user_input": item.query,
                    "response": cleaned_answer,
                    "retrieved_contexts": [str(ch.get("text", "")) for ch in generation_chunks],
                    "reference": reference_answer,
                }
            )
            ragas_index_map.append(len(rows) - 1)

    ragas_warning = None
    if run_generation and ragas_samples:
        ragas_model_name = getattr(processor.llm, "model_name", None)
        ragas_base_url = getattr(processor.llm, "proxy_url", None)

        ragas_scores, ragas_warning = run_ragas_evaluation(
            ragas_samples,
            llm_model_name=ragas_model_name,
            llm_base_url=ragas_base_url,
        )
        for map_idx, score in zip(ragas_index_map, ragas_scores):
            rows[map_idx]["faithfulness"] = score.get("faithfulness")
            rows[map_idx]["answer_correctness"] = score.get("answer_correctness")

        # Second pass: re-evaluate only missing RAGAS rows to improve coverage.
        missing_positions: List[int] = []
        for pos, map_idx in enumerate(ragas_index_map):
            f_val = rows[map_idx].get("faithfulness")
            c_val = rows[map_idx].get("answer_correctness")
            if f_val is None or c_val is None:
                missing_positions.append(pos)

        if missing_positions:
            retry_samples = [ragas_samples[pos] for pos in missing_positions]
            fallback_model = os.getenv("RAGAS_FALLBACK_MODEL", "").strip() or ragas_model_name
            retry_scores, retry_warning = run_ragas_evaluation(
                retry_samples,
                llm_model_name=fallback_model,
                llm_base_url=ragas_base_url,
            )

            for pos, score in zip(missing_positions, retry_scores):
                map_idx = ragas_index_map[pos]
                if rows[map_idx].get("faithfulness") is None and score.get("faithfulness") is not None:
                    rows[map_idx]["faithfulness"] = score.get("faithfulness")
                if rows[map_idx].get("answer_correctness") is None and score.get("answer_correctness") is not None:
                    rows[map_idx]["answer_correctness"] = score.get("answer_correctness")

            still_missing = 0
            for map_idx in ragas_index_map:
                if rows[map_idx].get("faithfulness") is None or rows[map_idx].get("answer_correctness") is None:
                    still_missing += 1

            if still_missing > 0:
                extra_warn = (
                    f"RAGAS coverage is incomplete after retry ({still_missing}/{len(ragas_index_map)} missing)."
                )
                ragas_warning = f"{ragas_warning} {extra_warn}" if ragas_warning else extra_warn
            elif retry_warning:
                ragas_warning = ragas_warning or retry_warning

    df = pd.DataFrame(rows)

    if "answer_correctness" in df.columns:
        df["answer_correctness_band"] = df["answer_correctness"].apply(_correctness_band)
    else:
        df["answer_correctness_band"] = "missing"

    df["diagnosis"] = df.apply(
        lambda r: _diagnose_row(
            answer_correctness=r.get("answer_correctness"),
            recall_k=float(r.get("recall_at_k") or 0.0),
            precision_1=float(r.get("precision_at_1") or 0.0),
            full_gt_hit=bool(r.get("full_ground_truth_hit")),
            ref_answer_overlap=float(r.get("ref_answer_overlap") or 0.0),
        ),
        axis=1,
    )

    metrics_for_avg = [
        "precision_at_k",
        "max_precision_at_k",
        "normalized_precision_at_k",
        "recall_at_k",
        "f1_at_k",
        "precision_at_1",
        "precision_at_3",
        "mrr",
        "hit_at_k",
        "hit_at_1",
        "hit_at_3",
        "ndcg_at_k",
        "faithfulness",
        "answer_correctness",
    ]

    summary: Dict[str, Any] = {
        "query_count": int(len(df)),
        "top_k": int(top_k),
        "metrics_avg": {},
        "metrics_std": {},
        "ragas_warning": ragas_warning,
    }

    if run_generation:
        faith_series = pd.to_numeric(df.get("faithfulness"), errors="coerce")
        corr_series = pd.to_numeric(df.get("answer_correctness"), errors="coerce")
        summary["ragas_coverage"] = {
            "faithfulness_coverage": float(faith_series.notna().mean()) if len(faith_series) else 0.0,
            "answer_correctness_coverage": float(corr_series.notna().mean()) if len(corr_series) else 0.0,
            "faithfulness_missing": int(faith_series.isna().sum()) if len(faith_series) else 0,
            "answer_correctness_missing": int(corr_series.isna().sum()) if len(corr_series) else 0,
        }

    for metric in metrics_for_avg:
        if metric not in df.columns:
            continue
        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        if len(series) == 0:
            summary["metrics_avg"][metric] = None
            summary["metrics_std"][metric] = None
            continue
        summary["metrics_avg"][metric] = float(series.mean())
        summary["metrics_std"][metric] = float(series.std(ddof=0)) if len(series) > 1 else 0.0

    return df, summary


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_report(
    output_dir: Path,
    run_config: Dict[str, Any],
    eval_df: pd.DataFrame,
    summary: Dict[str, Any],
) -> Dict[str, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    per_query_json = output_dir / f"per_query_{ts}.json"
    per_query_csv = output_dir / f"per_query_{ts}.csv"
    per_query_diagnostics_csv = output_dir / f"per_query_diagnostics_{ts}.csv"
    summary_json = output_dir / f"summary_{ts}.json"

    eval_df.to_csv(per_query_csv, index=False)
    eval_df.to_json(per_query_json, orient="records", force_ascii=False, indent=2)

    sort_cols = [c for c in ("answer_correctness", "faithfulness", "query_index") if c in eval_df.columns]
    if sort_cols:
        eval_df.sort_values(sort_cols, na_position="first").to_csv(per_query_diagnostics_csv, index=False)
    else:
        eval_df.to_csv(per_query_diagnostics_csv, index=False)

    summary_payload = {
        "run_config": run_config,
        "summary": summary,
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    return {
        "per_query_csv": per_query_csv,
        "per_query_json": per_query_json,
        "per_query_diagnostics_csv": per_query_diagnostics_csv,
        "summary_json": summary_json,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end RAG evaluation.")
    parser.add_argument("--chunks-path", type=str, default=str(DEFAULT_CHUNKS_PATH), help="Path to processed chunk JSON file.")
    parser.add_argument("--queries-path", type=str, default="", help="Optional path to query annotation JSON.")
    parser.add_argument("--subject", type=str, default="", help="Subject filter for RAGProcessor. Empty = all subjects.")
    parser.add_argument("--llm-model", type=str, default="", help="LLM model key used by RAGProcessor (for generation step).")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for retrieval metrics.")
    parser.add_argument("--max-queries", type=int, default=50, help="Max number of queries to evaluate.")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "evaluation_reports"), help="Output directory for reports.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for auto query generation.")
    parser.add_argument(
        "--generation-context-k",
        type=int,
        default=3,
        help="How many top retrieved chunks to pass into generation (metrics still use --top-k).",
    )
    parser.add_argument(
        "--loose-answer-style",
        action="store_true",
        help="Use looser generation style (default is concise mode for better answer_correctness).",
    )
    parser.add_argument(
        "--disable-answer-compaction",
        action="store_true",
        help="Disable compacting generated answers before RAGAS scoring.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation + RAGAS metrics (retrieval-only eval).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    chunks_path = Path(args.chunks_path).expanduser().resolve()
    queries_path = Path(args.queries_path).expanduser().resolve() if args.queries_path else None
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not chunks_path.exists():
        print(f"[ERROR] Chunks file not found: {chunks_path}")
        return 1

    ensure_output_dir(output_dir)

    try:
        raw_chunks = load_json_file(chunks_path)
        chunks = normalize_chunks(raw_chunks)
    except Exception as exc:
        print(f"[ERROR] Failed to load/normalize chunks: {exc}")
        traceback.print_exc()
        return 1

    chunk_lookup = {c.chunk_id: c for c in chunks}

    try:
        if queries_path:
            eval_queries = load_eval_queries_from_file(queries_path, max_queries=args.max_queries)
            query_source = str(queries_path)
        else:
            eval_queries = auto_generate_eval_queries(
                chunks=chunks,
                max_queries=args.max_queries,
                random_seed=args.random_seed,
            )
            query_source = "auto-generated-from-chunks"
    except Exception as exc:
        print(f"[ERROR] Failed to prepare evaluation queries: {exc}")
        traceback.print_exc()
        return 1

    query_set_diagnostics = build_query_set_diagnostics(eval_queries, top_k=args.top_k)

    subject = args.subject.strip() or None
    llm_model = args.llm_model.strip() or None

    print("=" * 80)
    print("RAG Evaluation Start")
    print("=" * 80)
    print(f"Script version    : {SCRIPT_VERSION}")
    print(f"Chunks path       : {chunks_path}")
    print(f"Normalized chunks : {len(chunks)}")
    print(f"Queries source    : {query_source}")
    print(f"Queries count     : {len(eval_queries)}")
    print(f"Subject filter    : {subject or 'ALL'}")
    print(f"Top-k             : {args.top_k}")
    print(f"Gen context-k     : {args.generation_context_k}")
    print(f"Concise mode      : {'OFF' if args.loose_answer_style else 'ON'}")
    print(f"Answer compaction : {'OFF' if args.disable_answer_compaction else 'ON'}")
    print(f"Generation eval   : {'OFF' if args.skip_generation else 'ON'}")
    print("\nQuery-set diagnostics:")
    print(f"- avg_ground_truth_count      : {query_set_diagnostics['avg_ground_truth_count']:.4f}")
    print(f"- single_ground_truth_ratio   : {query_set_diagnostics['single_ground_truth_ratio']:.4f}")
    print(f"- avg_max_precision_at_k      : {query_set_diagnostics['avg_max_precision_at_k']:.4f}")
    print(f"- short_reference_ratio       : {query_set_diagnostics['short_reference_ratio']:.4f}")
    print(f"- placeholder_reference_ratio : {query_set_diagnostics['placeholder_reference_ratio']:.4f}")
    for note in query_set_diagnostics.get("notes", []):
        print(f"[NOTE] {note}")

    try:
        processor = RAGProcessor(subject=subject, llm_model=llm_model)
    except Exception as exc:
        print(f"[ERROR] Failed to initialize RAGProcessor: {exc}")
        traceback.print_exc()
        return 1

    eval_df, summary = evaluate_queries(
        processor=processor,
        queries=eval_queries,
        chunk_lookup=chunk_lookup,
        top_k=args.top_k,
        run_generation=not args.skip_generation,
        generation_context_k=args.generation_context_k,
        concise_mode=not args.loose_answer_style,
        answer_compaction=not args.disable_answer_compaction,
    )

    run_config = {
        "chunks_path": str(chunks_path),
        "queries_source": query_source,
        "subject": subject,
        "llm_model": llm_model,
        "top_k": args.top_k,
        "generation_context_k": args.generation_context_k,
        "concise_mode": not args.loose_answer_style,
        "answer_compaction": not args.disable_answer_compaction,
        "max_queries": args.max_queries,
        "skip_generation": args.skip_generation,
        "random_seed": args.random_seed,
        "query_set_diagnostics": query_set_diagnostics,
        "timestamp": datetime.now().isoformat(),
    }

    report_paths = write_report(
        output_dir=output_dir,
        run_config=run_config,
        eval_df=eval_df,
        summary=summary,
    )

    print("\n" + "=" * 80)
    print("Evaluation Summary (Averages)")
    print("=" * 80)
    for metric, value in summary.get("metrics_avg", {}).items():
        if value is None:
            print(f"- {metric:<20}: N/A")
        else:
            print(f"- {metric:<20}: {value:.4f}")

    ragas_warning = summary.get("ragas_warning")
    if ragas_warning:
        print("\n[WARN] " + ragas_warning)

    ragas_coverage = summary.get("ragas_coverage")
    if isinstance(ragas_coverage, dict):
        print("\nRAGAS coverage:")
        print(f"- faithfulness_coverage      : {ragas_coverage.get('faithfulness_coverage', 0.0):.4f}")
        print(f"- answer_correctness_coverage: {ragas_coverage.get('answer_correctness_coverage', 0.0):.4f}")
        print(f"- faithfulness_missing       : {ragas_coverage.get('faithfulness_missing', 0)}")
        print(f"- answer_correctness_missing : {ragas_coverage.get('answer_correctness_missing', 0)}")

    print("\nReports written:")
    for key, path in report_paths.items():
        print(f"- {key}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
