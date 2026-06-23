#!/usr/bin/env python3
"""
End-to-end latency benchmark for the full RAG pipeline.
Measures: dense retrieval + BM25 + fusion + rerank + prompt build + generation.
Batch size is always 1 (production-realistic).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from processor import RAGProcessor  # noqa: E402


DEFAULT_QUERIES_PATH = PROJECT_ROOT / "evaluation" / "sample_queries.json"


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_queries(path: Path, max_queries: int) -> List[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for key in ("queries", "data", "items"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break

    if not isinstance(raw, list):
        raise ValueError("Query file must be a list or dict with queries/data/items list")

    queries: List[str] = []
    for item in raw:
        if len(queries) >= max_queries:
            break
        if isinstance(item, str):
            q = _safe_text(item)
        elif isinstance(item, dict):
            q = _safe_text(item.get("query") or item.get("question") or item.get("user_input"))
        else:
            q = ""
        if q:
            queries.append(q)

    if not queries:
        raise ValueError("No valid queries found")

    return queries


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1)))))
    return values[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end latency benchmark for RAG pipeline")
    parser.add_argument("--queries-path", type=str, default=str(DEFAULT_QUERIES_PATH))
    parser.add_argument("--subject", type=str, default="", help="Subject filter; empty = all")
    parser.add_argument("--llm-model", type=str, default="", help="LLM model key")
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-history", action="store_true", help="Include history in prompt")
    parser.add_argument("--save-json", type=str, default="", help="Optional output JSON file")
    args = parser.parse_args()

    queries_path = Path(args.queries_path).expanduser().resolve()
    if not queries_path.exists():
        print(f"[ERROR] Query file not found: {queries_path}")
        return 1

    queries = load_queries(queries_path, max_queries=args.max_queries)
    random.seed(args.seed)

    subject = args.subject.strip() or None
    llm_model = args.llm_model.strip() or None

    processor = RAGProcessor(subject=subject, llm_model=llm_model)

    # Warm-up runs to stabilize caches.
    for _ in range(max(0, args.warmup)):
        _ = processor.get_response(queries[0], use_history=args.use_history)

    latencies_ms: List[float] = []
    results: List[Dict[str, Any]] = []

    for q in queries:
        start = time.perf_counter()
        _ = processor.get_response(q, use_history=args.use_history)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)
        results.append({"query": q, "latency_ms": elapsed_ms})

    p50 = percentile(latencies_ms, 50)
    p95 = percentile(latencies_ms, 95)
    avg = statistics.mean(latencies_ms) if latencies_ms else 0.0

    print("End-to-end latency (batch_size=1)")
    print(f"Queries: {len(latencies_ms)}")
    print(f"Average: {avg:.2f} ms")
    print(f"P50: {p50:.2f} ms")
    print(f"P95: {p95:.2f} ms")

    if args.save_json:
        out_path = Path(args.save_json).expanduser().resolve()
        payload = {
            "queries": len(latencies_ms),
            "avg_ms": avg,
            "p50_ms": p50,
            "p95_ms": p95,
            "subject": subject or "all",
            "llm_model": llm_model or "default",
            "use_history": args.use_history,
            "results": results,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
