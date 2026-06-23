#!/usr/bin/env python3
"""
End-to-end RAG evaluation runner with ablation study support.

Ablation modes:
  dense_only          – dense retrieval only (no BM25, no rerank)
  hybrid_no_rerank    – dense + BM25, no reranking
  dense_with_rerank   – dense + reranker, no BM25
  full                – hybrid (dense + BM25) + reranker  [default]

Usage examples:
  # Run only the full pipeline (original behaviour)
  python scripts/evaluate_rag.py --chunks-path ... --mode full

  # Run all ablation modes and print a comparison table
  python scripts/evaluate_rag.py --chunks-path ... --mode all
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
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="Using `TRANSFORMERS_CACHE` is deprecated.*",
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from processor import (  # noqa: E402
    RAGProcessor,
    HYBRID_TOP_K,
    DENSE_TOP_K,
    BM25_TOP_K,
    RERANK_TOP_N_DEFAULT,
    RERANK_CANDIDATE_MULTIPLIER,
    RERANK_CANDIDATE_CAP,
)

DEFAULT_CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed_chunks"
    / "Môn Triết học Mác-Lênin"
    / "19filespdftriếthọcm-ln112024_19filespdftriếthọcm-ln112024.json"
)
SCRIPT_VERSION = "eval-rag-2026-04-15-v3"

# ─────────────────────────────────────────────────────────────────────────────
# ABLATION CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AblationConfig:
    """Controls which retrieval components are active during evaluation."""
    name: str
    use_dense: bool = True
    use_bm25: bool = True
    use_rerank: bool = True
    description: str = ""

    def __post_init__(self):
        if not self.description:
            parts = []
            if self.use_dense:
                parts.append("dense")
            if self.use_bm25:
                parts.append("BM25")
            if self.use_rerank:
                parts.append("rerank")
            self.description = " + ".join(parts) if parts else "none"


# All supported ablation modes.
ABLATION_MODES: Dict[str, AblationConfig] = {
    "dense_only": AblationConfig(
        name="dense_only",
        use_dense=True,
        use_bm25=False,
        use_rerank=False,
    ),
    "hybrid_no_rerank": AblationConfig(
        name="hybrid_no_rerank",
        use_dense=True,
        use_bm25=True,
        use_rerank=False,
    ),
    "dense_with_rerank": AblationConfig(
        name="dense_with_rerank",
        use_dense=True,
        use_bm25=False,
        use_rerank=True,
    ),
    "full": AblationConfig(
        name="full",
        use_dense=True,
        use_bm25=True,
        use_rerank=True,
    ),
}

# Metrics shown in the comparison table (in display order).
COMPARISON_METRICS = [
    "normalized_precision_at_k",
    "mrr",
    "hit_at_k",
    "hit_at_1",
    "hit_at_3",
    "ndcg_at_k",
    "faithfulness",
    "answer_correctness",
]


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION-AWARE RETRIEVAL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_with_config(
    processor: RAGProcessor,
    query: str,
    top_k: int,
    cfg: AblationConfig,
) -> List[Dict[str, Any]]:
    """
    Run retrieval according to the ablation config.

    This wraps processor's lower-level retrieval primitives so we can test
    every combination without touching processor.py.

      use_dense + use_bm25  → hybrid_retrieve()  (score or RRF fusion)
      use_dense only        → retrieve_dense()
      use_bm25  only        → retrieve_bm25()
      use_rerank            → rerank() on top of whichever candidates we got
    """
    # ── candidate generation ──────────────────────────────────────────────────
    multiplier   = max(1, RERANK_CANDIDATE_MULTIPLIER)
    cap          = max(top_k, RERANK_CANDIDATE_CAP)
    candidate_k  = max(HYBRID_TOP_K, min(cap, top_k * multiplier))

    if cfg.use_dense and cfg.use_bm25:
        candidates = processor.hybrid_retrieve(query, top_k=candidate_k)
    elif cfg.use_dense:
        candidates = processor.retrieve_dense(query, top_k=candidate_k)
    elif cfg.use_bm25:
        candidates = processor.retrieve_bm25(query, top_k=candidate_k)
    else:
        return []

    if not candidates:
        return []

    # ── optional reranking ────────────────────────────────────────────────────
    if cfg.use_rerank:
        return processor.rerank(query, candidates, top_n=top_k)

    return candidates[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES  (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS  (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_chunk_record(record: Dict[str, Any], fallback_idx: int) -> Optional[NormalizedChunk]:
    text = _safe_text(
        record.get("text") or record.get("chunk") or record.get("content") or record.get("page_content")
    )
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
    subject  = _safe_text(metadata.get("subject"))
    file_name = _safe_text(metadata.get("file_name") or metadata.get("file") or "")

    page = metadata.get("page", 0)
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 0

    return NormalizedChunk(
        chunk_id=chunk_id, text=text, subject=subject,
        file_name=file_name, page=page, metadata=metadata,
    )


def normalize_chunks(raw: Any) -> List[NormalizedChunk]:
    if isinstance(raw, dict):
        for key in ("chunks", "data", "items", "documents"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break

    if not isinstance(raw, list):
        raise ValueError("Chunk file must be a list or a dict containing list field.")

    chunks: List[NormalizedChunk] = []
    seen_ids: set = set()
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
                chunk_id=chunk_id, text=normalized.text, subject=normalized.subject,
                file_name=normalized.file_name, page=normalized.page, metadata=normalized.metadata,
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
        eval_queries.append(
            EvalQuery(
                query_id=f"auto_{i:04d}",
                query=_build_query_from_chunk(chunk),
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
        return [str(k).strip() for k in raw.keys() if str(k).strip()]
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

        reference_answer = _safe_text(
            item.get("reference_answer") or item.get("reference") or item.get("answer")
        )
        grades = _normalize_grades(
            item.get("ground_truth_grades") or item.get("grades") or gt_raw,
            ground_truth_ids,
        )

        source_page_raw = item.get("source_page")
        source_page: Optional[int] = None
        if source_page_raw is not None and str(source_page_raw).strip():
            try:
                source_page = int(source_page_raw)
            except (TypeError, ValueError):
                pass

        queries.append(
            EvalQuery(
                query_id=_safe_text(item.get("query_id") or item.get("id") or f"q_{i:04d}"),
                query=query,
                ground_truth_ids=ground_truth_ids,
                reference_answer=reference_answer,
                ground_truth_grades=grades,
                query_type=_safe_text(item.get("query_type")),
                difficulty=_safe_text(item.get("difficulty")),
                source_file=_safe_text(item.get("source_file")),
                source_page=source_page,
            )
        )

    if not queries:
        raise ValueError("No valid evaluation queries found in query file")
    return queries


# ─────────────────────────────────────────────────────────────────────────────
# METRICS  (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(retrieved: List[str], relevant: set) -> float:
    if not retrieved:
        return 0.0
    return sum(1 for rid in retrieved if rid in relevant) / float(len(retrieved))


def recall_at_k(retrieved: List[str], relevant: set) -> float:
    if not relevant:
        return 0.0
    return sum(1 for rid in retrieved if rid in relevant) / float(len(relevant))


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
        if rel > 0:
            score += (2.0 ** rel - 1.0) / math.log2(rank + 1.0)
    return score


def ndcg_at_k(retrieved: List[str], grades: Dict[str, float], k: int) -> float:
    actual = dcg_at_k(retrieved, grades, k)
    ideal_rels = sorted([float(v) for v in grades.values() if float(v) > 0], reverse=True)
    ideal = sum(
        (2.0 ** rel - 1.0) / math.log2(rank + 1.0)
        for rank, rel in enumerate(ideal_rels[:k], start=1)
    )
    return 0.0 if ideal == 0 else actual / ideal


def hit_at_n(retrieved: List[str], relevant: set, n: int) -> float:
    if n <= 0:
        return 0.0
    return 1.0 if any(rid in relevant for rid in retrieved[:n]) else 0.0


def max_precision_at_k(relevant_count: int, k: int) -> float:
    if k <= 0 or relevant_count <= 0:
        return 0.0
    return min(relevant_count, k) / float(k)


def normalized_precision_at_k(precision_k: float, relevant_count: int, k: int) -> float:
    max_p = max_precision_at_k(relevant_count, k)
    return 0.0 if max_p <= 0 else precision_k / max_p


def _is_placeholder_reference(reference_text: str) -> bool:
    text = reference_text.strip()
    if not text:
        return True
    compact = text.lower().replace(" ", "")
    if compact.startswith("câuhỏi"):
        return True
    return compact in {"a.", "b.", "c.", "d.", "1.", "2.", "3.", "4.", "i.", "ii.", "iii.", "iv.", "v."}


def build_query_set_diagnostics(queries: List[EvalQuery], top_k: int) -> Dict[str, Any]:
    if not queries:
        return {
            "single_ground_truth_ratio": 0.0, "multi_ground_truth_ratio": 0.0,
            "avg_ground_truth_count": 0.0, "avg_max_precision_at_k": 0.0,
            "short_reference_ratio": 0.0, "placeholder_reference_ratio": 0.0,
            "query_type_distribution": {}, "difficulty_distribution": {}, "notes": [],
        }

    gt_counts = [len(q.ground_truth_ids) for q in queries]
    query_count = len(queries)
    avg_gt = sum(gt_counts) / float(query_count)
    avg_max_p_at_k = sum(max_precision_at_k(c, top_k) for c in gt_counts) / float(query_count)
    refs = [q.reference_answer.strip() for q in queries]
    placeholder_ref_ratio = sum(1 for r in refs if _is_placeholder_reference(r)) / float(query_count)

    notes: List[str] = []
    single_gt_ratio = sum(1 for c in gt_counts if c == 1) / float(query_count)
    if single_gt_ratio >= 0.8 and top_k > 1:
        notes.append(
            f"Most queries have exactly one ground-truth chunk (avg max Precision@{top_k} = {avg_max_p_at_k:.4f})."
        )
    if placeholder_ref_ratio >= 0.2:
        notes.append("High fraction of reference answers look like placeholders.")

    query_type_dist: Dict[str, int] = {}
    difficulty_dist: Dict[str, int] = {}
    for q in queries:
        if q.query_type:
            query_type_dist[q.query_type] = query_type_dist.get(q.query_type, 0) + 1
        if q.difficulty:
            difficulty_dist[q.difficulty] = difficulty_dist.get(q.difficulty, 0) + 1

    return {
        "single_ground_truth_ratio": single_gt_ratio,
        "multi_ground_truth_ratio": sum(1 for c in gt_counts if c > 1) / float(query_count),
        "avg_ground_truth_count": avg_gt,
        "avg_max_precision_at_k": avg_max_p_at_k,
        "short_reference_ratio": sum(1 for r in refs if len(r) <= 12) / float(query_count),
        "placeholder_reference_ratio": placeholder_ref_ratio,
        "query_type_distribution": query_type_dist,
        "difficulty_distribution": difficulty_dist,
        "notes": notes,
    }


def strip_sources_from_answer(answer: str) -> str:
    marker = "📚 **Nguồn tham khảo:**"
    if marker in answer:
        return answer.split(marker, 1)[0].strip()
    return answer.strip()


def _token_overlap_ratio(reference_text: str, answer_text: str) -> float:
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


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT / GENERATION HELPERS  (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

def _needs_synthesis(text: str) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return False
    bullet_lines = sum(
        1 for l in lines
        if re.match(r"^(\d+[\.)]\s|[-*•]\s|[a-zA-Z][\.)]\s)", l)
    )
    return bullet_lines >= 2


def _strip_bullets_regex(text: str) -> str:
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
    return " ".join(cleaned_lines).strip() if cleaned_lines else text.strip()


def _compact_generated_answer(answer_text: str, query_type: str, llm: Any = None) -> str:
    text = str(answer_text or "").strip()
    if not text or not _needs_synthesis(text):
        return text
    if llm is not None:
        try:
            prompt = _build_synthesis_prompt(text, query_type)
            synthesized = str(llm.invoke(prompt) or "").strip()
            if synthesized and not synthesized.startswith("[") and len(synthesized) > 20:
                return synthesized
        except Exception:
            pass
    return _strip_bullets_regex(text)


def _build_synthesis_prompt(answer_text: str, query_type: str) -> str:
    qt = (query_type or "").strip().lower()
    if qt == "enumeration":
        style_hint = "Nối các mục thành văn xuôi bằng dấu phẩy, chấm phẩy hoặc liên từ."
    elif qt == "compare_contrast":
        style_hint = "Trình bày điểm giống/khác trong một đoạn liền mạch."
    elif qt == "reasoning":
        style_hint = "Nêu kết luận trước, sau đó trình bày các lý do."
    else:
        style_hint = "Viết thành đoạn văn xuôi tự nhiên, dùng liên từ và câu chuyển tiếp."

    return (
        f"Viết lại nội dung dưới đây thành một đoạn văn xuôi liền mạch, "
        f"không dùng bullet hay số thứ tự.\n{style_hint}\n\nNội dung gốc:\n{answer_text}\n\nĐoạn văn xuôi:"
    )


def _estimate_reference_complexity(reference: str, query_type: str = "") -> Dict[str, Any]:
    text = (reference or "").strip()
    if not text or _is_placeholder_reference(text):
        return {"words": 0, "key_points": 1, "length_tier": "short", "target_range": "1-2 câu"}

    words = len(re.findall(r"\w+", text, flags=re.UNICODE))
    bullet_items = len(re.findall(r"(?m)^[\-\*•]\s+|\d+[\.)]\s+[^\d]|[a-zA-Z][\.)]\s+", text))
    sentences = len([s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()])
    key_points = min(max(bullet_items, sentences, 1), 8)
    qt = (query_type or "").strip().lower()

    if words <= 40:
        tier = "short"
    elif words <= 100:
        tier = "medium"
    elif words <= 220:
        tier = "long"
    else:
        tier = "very_long"

    if qt in {"enumeration", "compare_contrast"} and tier == "short":
        tier = "medium"

    range_map = {
        "short":     "1-2 câu ngắn",
        "medium":    f"khoảng {min(key_points + 1, 4)} ý chính",
        "long":      f"khoảng {min(key_points, 6)} ý — đủ ý, không bỏ sót",
        "very_long": f"toàn bộ ý chính (khoảng {key_points} ý) — ưu tiên đầy đủ hơn ngắn gọn",
    }
    return {"words": words, "key_points": key_points, "length_tier": tier, "target_range": range_map[tier]}


def _build_adaptive_length_guidance(concise_mode: bool, ref_complexity: Optional[Dict[str, Any]]) -> str:
    if ref_complexity is None:
        return (
            "Trả lời súc tích, đúng trọng tâm." if concise_mode
            else "Trả lời đầy đủ các ý chính."
        )
    tier   = ref_complexity.get("length_tier", "medium")
    target = ref_complexity.get("target_range", "2-3 ý")

    if tier == "short":
        return f"Trả lời ngắn gọn: {target}."
    if tier == "medium":
        return (
            f"Trả lời súc tích: {target}." if concise_mode
            else f"Trả lời đầy đủ: {target}."
        )
    if tier == "long":
        return f"Trả lời đầy đủ: {target}. Đảm bảo bao gồm TẤT CẢ ý quan trọng."
    return f"Trả lời toàn diện: {target}. Ưu tiên đầy đủ ý hơn ngắn gọn."


def _format_hint_by_query_type(query_type: str) -> str:
    qt = (query_type or "").strip().lower()
    if qt == "definition":
        return "- Định nghĩa: Nêu khái niệm cốt lõi trong một đoạn văn liền mạch."
    if qt == "compare_contrast":
        return "- So sánh: Trình bày điểm giống và khác trong văn xuôi."
    if qt == "enumeration":
        return "- Liệt kê: Nối các mục bằng dấu phẩy hoặc liên từ."
    if qt == "reasoning":
        return "- Lập luận: Nêu kết luận trước, sau đó giải thích lý do."
    return "- Trả lời rõ ràng bằng văn xuôi, bám sát ngữ cảnh."


def build_eval_answer_prompt(
    question: str,
    context: str,
    query_type: str = "",
    concise_mode: bool = True,
    reference_complexity: Optional[Dict[str, Any]] = None,
) -> str:
    style_hint      = _format_hint_by_query_type(query_type)
    length_guidance = _build_adaptive_length_guidance(concise_mode, reference_complexity)

    return f"""Bạn là trợ lý học thuật. Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.

Ngữ cảnh:
{context}

Câu hỏi:
{question}

Yêu cầu:
1. Chỉ sử dụng thông tin có trong ngữ cảnh.
2. {length_guidance}
3. Viết thành văn xuôi liền mạch — không dùng bullet hay số thứ tự.
4. Giữ nguyên thuật ngữ, công thức, số liệu nếu có.
5. Không dùng lời dẫn nhập thừa.
6. Nếu ngữ cảnh không đủ thông tin: trả lời "Không đủ dữ liệu trong ngữ cảnh".
{style_hint}

Trả lời (văn xuôi):"""


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS  (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

def run_ragas_evaluation(
    samples: List[Dict[str, Any]],
    llm_model_name: Optional[str] = None,
    llm_base_url: Optional[str] = None,
) -> Tuple[List[Dict[str, Optional[float]]], Optional[str]]:
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

        logging.getLogger("ragas").setLevel(logging.ERROR)

        ragas_model  = llm_model_name or os.getenv("RAGAS_LLM_MODEL", "Qwen/Qwen3-8B-AWQ")
        ragas_base   = llm_base_url or os.getenv("RAGAS_LLM_BASE_URL") or os.getenv("LLM_PROXY_URL", "http://localhost:5000/v1")
        ragas_base   = ragas_base.rstrip("/")
        if not ragas_base.endswith("/v1"):
            ragas_base = f"{ragas_base}/v1"

        ragas_timeout    = float(os.getenv("RAGAS_TIMEOUT", "120"))
        ragas_max_tokens = int(os.getenv("RAGAS_MAX_TOKENS", "2048"))
        ragas_max_retries = int(os.getenv("RAGAS_MAX_RETRIES", "3"))
        ragas_batch_size  = int(os.getenv("RAGAS_BATCH_SIZE", "4"))
        openai_api_key    = os.getenv("OPENAI_API_KEY", "local-proxy")

        ragas_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=ragas_model, api_key=openai_api_key, base_url=ragas_base,
                temperature=0.0, max_tokens=ragas_max_tokens,
                timeout=ragas_timeout, max_retries=ragas_max_retries,
            )
        )
        embed_model = os.getenv(
            "RAGAS_EMBEDDING_MODEL",
            str(PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The class `HuggingFaceEmbeddings`.*")
            ragas_embeddings = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name=embed_model, model_kwargs={"device": "cpu"})
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
                        timeout=ragas_timeout, max_retries=ragas_max_retries, max_workers=1,
                    )
                except Exception:
                    pass
        except Exception:
            pass

        buf_out, buf_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            result = evaluate(**eval_kwargs)

        captured = "\n".join(
            line for line in (buf_out.getvalue() + "\n" + buf_err.getvalue()).splitlines() if line.strip()
        )
        rows: List[Dict[str, Optional[float]]] = []

        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            for _, r in df.iterrows():
                rows.append({
                    "faithfulness":       float(r.get("faithfulness"))       if pd.notna(r.get("faithfulness"))       else None,
                    "answer_correctness": float(r.get("answer_correctness")) if pd.notna(r.get("answer_correctness")) else None,
                })
        elif hasattr(result, "scores") and isinstance(result.scores, list):
            for score in result.scores:
                rows.append({
                    "faithfulness":       score.get("faithfulness"),
                    "answer_correctness": score.get("answer_correctness"),
                })
        else:
            return [], "RAGAS ran but result format is not recognized."

        warning_msg = (
            "RAGAS had parser/timeout retries on some samples."
            if "Exception raised in Job" in captured else None
        )
        return rows, warning_msg

    except Exception as exc:
        return [], f"Skipped RAGAS metrics. Reason: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# CORE EVALUATION  (now accepts AblationConfig)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_queries(
    processor: RAGProcessor,
    queries: List[EvalQuery],
    chunk_lookup: Dict[str, NormalizedChunk],
    top_k: int,
    run_generation: bool,
    generation_context_k: int,
    concise_mode: bool,
    answer_compaction: bool,
    ablation_cfg: AblationConfig,                  # ← NEW
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate a list of queries under a given ablation configuration.

    The only change vs v2: retrieval is delegated to ``retrieve_with_config()``
    instead of calling ``processor._retrieve_relevant_chunks()`` directly.
    Everything else (metrics, generation, RAGAS) is identical.
    """
    rows: List[Dict[str, Any]] = []
    ragas_samples: List[Dict[str, Any]] = []
    ragas_index_map: List[int] = []
    generation_retry_count = int(os.getenv("EVAL_GENERATION_RETRIES", "2"))

    for idx, item in enumerate(queries, start=1):
        # ── retrieval (ablation-aware) ────────────────────────────────────────
        retrieved_chunks = retrieve_with_config(
            processor, item.query, top_k, ablation_cfg
        )
        retrieved_ids = [str(ch.get("id")) for ch in retrieved_chunks if ch.get("id") is not None]

        relevant_set   = set(item.ground_truth_ids)
        p_at_k         = precision_at_k(retrieved_ids, relevant_set)
        norm_p_at_k    = normalized_precision_at_k(p_at_k, len(relevant_set), top_k)
        r_at_k         = recall_at_k(retrieved_ids, relevant_set)
        f1_at_k        = f1_score(norm_p_at_k, r_at_k)
        f1_at_k_raw    = f1_score(p_at_k, r_at_k)
        mrr            = reciprocal_rank(retrieved_ids, relevant_set)
        hit_k          = 1.0 if any(rid in relevant_set for rid in retrieved_ids) else 0.0
        ndcg           = ndcg_at_k(retrieved_ids, item.ground_truth_grades, top_k)
        p_at_1         = precision_at_k(retrieved_ids[:1], relevant_set)
        p_at_3         = precision_at_k(retrieved_ids[:3], relevant_set)
        hit_1          = hit_at_n(retrieved_ids, relevant_set, 1)
        hit_3          = hit_at_n(retrieved_ids, relevant_set, 3)

        gt_texts = [chunk_lookup[cid].text for cid in item.ground_truth_ids if cid in chunk_lookup]
        reference_answer = item.reference_answer or "\n\n".join(gt_texts[:2])
        ref_complexity   = _estimate_reference_complexity(reference_answer, query_type=item.query_type)

        answer = None
        cleaned_answer = None

        if run_generation:
            try:
                gen_context_k   = max(1, int(generation_context_k))
                generation_chunks = retrieved_chunks[:gen_context_k]
                context = processor._build_context(generation_chunks)
                prompt  = build_eval_answer_prompt(
                    item.query, context,
                    query_type=item.query_type,
                    concise_mode=concise_mode,
                    reference_complexity=ref_complexity,
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
                    answer_text = _compact_generated_answer(
                        answer_text, item.query_type, llm=processor.llm
                    )

                sources_text = processor._format_sources(generation_chunks)
                answer       = f"{answer_text}\n\n{sources_text}" if sources_text else answer_text
                cleaned_answer = strip_sources_from_answer(answer)
            except Exception as exc:
                cleaned_answer = f"[GENERATION_ERROR] {exc}"

        rows.append({
            "query_index":              idx,
            "query_id":                 item.query_id,
            "query":                    item.query,
            "ablation_mode":            ablation_cfg.name,          # ← NEW
            "ground_truth_count":       len(item.ground_truth_ids),
            "ground_truth_ids":         item.ground_truth_ids,
            "retrieved_ids":            retrieved_ids,
            "precision_at_k":           p_at_k,
            "max_precision_at_k":       max_precision_at_k(len(relevant_set), top_k),
            "normalized_precision_at_k": norm_p_at_k,
            "recall_at_k":              r_at_k,
            "f1_at_k":                  f1_at_k,
            "f1_at_k_raw":              f1_at_k_raw,
            "precision_at_1":           p_at_1,
            "precision_at_3":           p_at_3,
            "mrr":                      mrr,
            "hit_at_k":                 hit_k,
            "hit_at_1":                 hit_1,
            "hit_at_3":                 hit_3,
            "ndcg_at_k":                ndcg,
            "answer":                   cleaned_answer,
            "reference_answer":         reference_answer,
            "query_type":               item.query_type,
            "difficulty":               item.difficulty,
            "source_file":              item.source_file,
            "source_page":              item.source_page,
            "faithfulness":             None,
            "answer_correctness":       None,
            "matched_ground_truth_count": int(len(set(retrieved_ids) & relevant_set)),
            "full_ground_truth_hit":    bool(set(item.ground_truth_ids).issubset(set(retrieved_ids))),
            "ref_answer_overlap":       _token_overlap_ratio(reference_answer, cleaned_answer or ""),
            "ref_length_tier":          ref_complexity.get("length_tier", ""),
            "ref_key_points":           ref_complexity.get("key_points", 0),
            "ref_words":                ref_complexity.get("words", 0),
        })

        if run_generation and cleaned_answer:
            ragas_samples.append({
                "user_input":         item.query,
                "response":           cleaned_answer,
                "retrieved_contexts": [str(ch.get("text", "")) for ch in generation_chunks],
                "reference":          reference_answer,
            })
            ragas_index_map.append(len(rows) - 1)

    # ── RAGAS scoring (unchanged) ─────────────────────────────────────────────
    ragas_warning = None
    if run_generation and ragas_samples:
        ragas_model_name = getattr(processor.llm, "model_name", None)
        ragas_base_url   = getattr(processor.llm, "proxy_url", None)

        ragas_scores, ragas_warning = run_ragas_evaluation(
            ragas_samples, llm_model_name=ragas_model_name, llm_base_url=ragas_base_url,
        )
        for map_idx, score in zip(ragas_index_map, ragas_scores):
            rows[map_idx]["faithfulness"]       = score.get("faithfulness")
            rows[map_idx]["answer_correctness"] = score.get("answer_correctness")

        # retry missing
        missing_positions = [
            pos for pos, map_idx in enumerate(ragas_index_map)
            if rows[map_idx].get("faithfulness") is None or rows[map_idx].get("answer_correctness") is None
        ]
        if missing_positions:
            retry_samples  = [ragas_samples[pos] for pos in missing_positions]
            fallback_model = os.getenv("RAGAS_FALLBACK_MODEL", "").strip() or ragas_model_name
            retry_scores, retry_warning = run_ragas_evaluation(
                retry_samples, llm_model_name=fallback_model, llm_base_url=ragas_base_url,
            )
            for pos, score in zip(missing_positions, retry_scores):
                map_idx = ragas_index_map[pos]
                if rows[map_idx].get("faithfulness") is None and score.get("faithfulness") is not None:
                    rows[map_idx]["faithfulness"] = score.get("faithfulness")
                if rows[map_idx].get("answer_correctness") is None and score.get("answer_correctness") is not None:
                    rows[map_idx]["answer_correctness"] = score.get("answer_correctness")

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
        "precision_at_k", "max_precision_at_k", "normalized_precision_at_k",
        "recall_at_k", "f1_at_k", "f1_at_k_raw", "precision_at_1", "precision_at_3",
        "mrr", "hit_at_k", "hit_at_1", "hit_at_3", "ndcg_at_k",
        "faithfulness", "answer_correctness",
    ]

    summary: Dict[str, Any] = {
        "ablation_mode":  ablation_cfg.name,
        "description":    ablation_cfg.description,
        "query_count":    int(len(df)),
        "top_k":          int(top_k),
        "metrics_avg":    {},
        "metrics_std":    {},
        "ragas_warning":  ragas_warning,
    }

    if run_generation:
        faith_series = pd.to_numeric(df.get("faithfulness"), errors="coerce")
        corr_series  = pd.to_numeric(df.get("answer_correctness"), errors="coerce")
        summary["ragas_coverage"] = {
            "faithfulness_coverage":        float(faith_series.notna().mean()) if len(faith_series) else 0.0,
            "answer_correctness_coverage":  float(corr_series.notna().mean())  if len(corr_series)  else 0.0,
            "faithfulness_missing":         int(faith_series.isna().sum())      if len(faith_series) else 0,
            "answer_correctness_missing":   int(corr_series.isna().sum())       if len(corr_series)  else 0,
        }

    for metric in metrics_for_avg:
        if metric not in df.columns:
            continue
        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        summary["metrics_avg"][metric] = float(series.mean()) if len(series) else None
        summary["metrics_std"][metric] = float(series.std(ddof=0)) if len(series) > 1 else 0.0

    return df, summary


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION STUDY ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_study(
    processor: RAGProcessor,
    queries: List[EvalQuery],
    chunk_lookup: Dict[str, NormalizedChunk],
    top_k: int,
    run_generation: bool,
    generation_context_k: int,
    concise_mode: bool,
    answer_compaction: bool,
    modes: List[str],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Run evaluation for each requested ablation mode and collect results.

    Returns:
        dfs     – dict[mode_name → per-query DataFrame]
        results – dict[mode_name → summary dict]  (also serialisable as JSON)
    """
    dfs:     Dict[str, pd.DataFrame] = {}
    results: Dict[str, Any]           = {}

    for mode_name in modes:
        cfg = ABLATION_MODES[mode_name]
        print(f"\n{'='*80}")
        print(f"  Ablation mode: {mode_name.upper()}  ({cfg.description})")
        print(f"{'='*80}")

        df, summary = evaluate_queries(
            processor=processor,
            queries=queries,
            chunk_lookup=chunk_lookup,
            top_k=top_k,
            run_generation=run_generation,
            generation_context_k=generation_context_k,
            concise_mode=concise_mode,
            answer_compaction=answer_compaction,
            ablation_cfg=cfg,
        )

        dfs[mode_name]     = df
        results[mode_name] = summary

        _print_mode_summary(mode_name, summary)

    return dfs, results


def _print_mode_summary(mode_name: str, summary: Dict[str, Any]) -> None:
    print(f"\nSummary for [{mode_name}] ({summary.get('description', '')}):")
    for metric, value in summary.get("metrics_avg", {}).items():
        display = f"{value:.4f}" if value is not None else "N/A"
        print(f"  - {metric:<30}: {display}")
    if summary.get("ragas_warning"):
        print(f"  [WARN] {summary['ragas_warning']}")


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(results: Dict[str, Any]) -> None:
    """Print a side-by-side comparison of all ablation modes."""
    if not results:
        return

    col_width    = 12
    name_width   = 22
    mode_names   = list(results.keys())

    header_cells = [f"{'Config':<{name_width}}"] + [f"{m:>{col_width}}" for m in mode_names]
    sep          = "-" * (name_width + (col_width + 1) * len(mode_names))

    print(f"\n{'='*80}")
    print("  ABLATION COMPARISON TABLE")
    print(f"{'='*80}")
    print(" | ".join(header_cells))
    print(sep)

    for metric in COMPARISON_METRICS:
        cells = [f"{metric:<{name_width}}"]
        for mode_name in mode_names:
            value = results[mode_name].get("metrics_avg", {}).get(metric)
            cells.append(f"{value:>{col_width}.4f}" if value is not None else f"{'N/A':>{col_width}}")
        print(" | ".join(cells))

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# FILE I/O
# ─────────────────────────────────────────────────────────────────────────────

def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_ablation_report(
    output_dir: Path,
    run_config: Dict[str, Any],
    dfs: Dict[str, pd.DataFrame],
    results: Dict[str, Any],
) -> Dict[str, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    written: Dict[str, Path] = {}

    for mode_name, df in dfs.items():
        csv_path  = output_dir / f"per_query_{mode_name}_{ts}.csv"
        json_path = output_dir / f"per_query_{mode_name}_{ts}.json"
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        written[f"{mode_name}_csv"]  = csv_path
        written[f"{mode_name}_json"] = json_path

    # Combined ablation JSON
    ablation_json = output_dir / f"ablation_results_{ts}.json"
    with ablation_json.open("w", encoding="utf-8") as f:
        json.dump({"run_config": run_config, "results": results}, f, ensure_ascii=False, indent=2)
    written["ablation_results"] = ablation_json

    return written


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    valid_modes = list(ABLATION_MODES.keys()) + ["all"]

    parser = argparse.ArgumentParser(description="Run end-to-end RAG evaluation with ablation study.")
    parser.add_argument("--chunks-path",  type=str, default=str(DEFAULT_CHUNKS_PATH))
    parser.add_argument("--queries-path", type=str, default="")
    parser.add_argument("--subject",      type=str, default="")
    parser.add_argument("--llm-model",    type=str, default="")
    parser.add_argument("--top-k",        type=int, default=5)
    parser.add_argument("--max-queries",  type=int, default=50)
    parser.add_argument("--output-dir",   type=str, default=str(PROJECT_ROOT / "evaluation_reports"))
    parser.add_argument("--random-seed",  type=int, default=42)
    parser.add_argument("--generation-context-k", type=int, default=3)
    parser.add_argument("--loose-answer-style",   action="store_true")
    parser.add_argument("--disable-answer-compaction", action="store_true")
    parser.add_argument("--skip-generation",       action="store_true")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=valid_modes,
        help=(
            "Ablation mode(s) to run. "
            "'all' runs every mode: " + ", ".join(ABLATION_MODES.keys()) + ". "
            "Default: 'full' (original behaviour)."
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    chunks_path  = Path(args.chunks_path).expanduser().resolve()
    queries_path = Path(args.queries_path).expanduser().resolve() if args.queries_path else None
    output_dir   = Path(args.output_dir).expanduser().resolve()

    if not chunks_path.exists():
        print(f"[ERROR] Chunks file not found: {chunks_path}")
        return 1

    ensure_output_dir(output_dir)

    # ── load chunks ──────────────────────────────────────────────────────────
    try:
        chunks = normalize_chunks(load_json_file(chunks_path))
    except Exception as exc:
        print(f"[ERROR] Failed to load/normalize chunks: {exc}")
        traceback.print_exc()
        return 1

    chunk_lookup = {c.chunk_id: c for c in chunks}

    # ── load / generate queries ──────────────────────────────────────────────
    try:
        if queries_path:
            eval_queries  = load_eval_queries_from_file(queries_path, max_queries=args.max_queries)
            query_source  = str(queries_path)
        else:
            eval_queries  = auto_generate_eval_queries(chunks, args.max_queries, args.random_seed)
            query_source  = "auto-generated-from-chunks"
    except Exception as exc:
        print(f"[ERROR] Failed to prepare evaluation queries: {exc}")
        traceback.print_exc()
        return 1

    query_set_diagnostics = build_query_set_diagnostics(eval_queries, top_k=args.top_k)

    subject   = args.subject.strip()   or None
    llm_model = args.llm_model.strip() or None

    # ── resolve modes ────────────────────────────────────────────────────────
    modes_to_run = list(ABLATION_MODES.keys()) if args.mode == "all" else [args.mode]

    # ── print header ─────────────────────────────────────────────────────────
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
    print(f"Ablation modes    : {', '.join(modes_to_run)}")
    print("\nQuery-set diagnostics:")
    print(f"  avg_ground_truth_count      : {query_set_diagnostics['avg_ground_truth_count']:.4f}")
    print(f"  single_ground_truth_ratio   : {query_set_diagnostics['single_ground_truth_ratio']:.4f}")
    print(f"  avg_max_precision_at_k      : {query_set_diagnostics['avg_max_precision_at_k']:.4f}")
    print(f"  placeholder_reference_ratio : {query_set_diagnostics['placeholder_reference_ratio']:.4f}")
    for note in query_set_diagnostics.get("notes", []):
        print(f"[NOTE] {note}")

    # ── init processor ────────────────────────────────────────────────────────
    try:
        processor = RAGProcessor(subject=subject, llm_model=llm_model)
    except Exception as exc:
        print(f"[ERROR] Failed to initialize RAGProcessor: {exc}")
        traceback.print_exc()
        return 1

    # ── run ablation study ────────────────────────────────────────────────────
    dfs, results = run_ablation_study(
        processor=processor,
        queries=eval_queries,
        chunk_lookup=chunk_lookup,
        top_k=args.top_k,
        run_generation=not args.skip_generation,
        generation_context_k=args.generation_context_k,
        concise_mode=not args.loose_answer_style,
        answer_compaction=not args.disable_answer_compaction,
        modes=modes_to_run,
    )

    # ── comparison table (only meaningful when > 1 mode) ────────────────────
    if len(modes_to_run) > 1:
        print_comparison_table(results)

    # ── write reports ────────────────────────────────────────────────────────
    run_config = {
        "chunks_path":          str(chunks_path),
        "queries_source":       query_source,
        "subject":              subject,
        "llm_model":            llm_model,
        "top_k":                args.top_k,
        "generation_context_k": args.generation_context_k,
        "concise_mode":         not args.loose_answer_style,
        "answer_compaction":    not args.disable_answer_compaction,
        "max_queries":          args.max_queries,
        "skip_generation":      args.skip_generation,
        "random_seed":          args.random_seed,
        "ablation_modes":       modes_to_run,
        "query_set_diagnostics": query_set_diagnostics,
        "timestamp":            datetime.now().isoformat(),
        "script_version":       SCRIPT_VERSION,
    }

    report_paths = write_ablation_report(output_dir, run_config, dfs, results)

    print("\nReports written:")
    for key, path in report_paths.items():
        print(f"  - {key}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())