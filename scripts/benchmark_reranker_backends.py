"""
Quick reranker backend benchmark for CPU environments.
Usage:
  python3 scripts/benchmark_reranker_backends.py \
    --model cross-encoder/ms-marco-MiniLM-L6-v2 \
    --batch-size 8 --pairs 48 --runs 5
"""

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from sentence_transformers import CrossEncoder

try:
    from tabulate import tabulate
except Exception:  # pragma: no cover - optional dependency
    tabulate = None


@dataclass
class StageStats:
    avg_ms: float
    p50_ms: float
    p95_ms: float


@dataclass
class BackendBenchmarkResult:
    backend: str
    Hybrid_search: StageStats
    reranking: StageStats
    generation: StageStats
    total: StageStats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES_PATH = PROJECT_ROOT / "evaluation" / "sample_queries.json"
DEFAULT_CHUNKS_DIR = PROJECT_ROOT / "data" / "processed_chunks"


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_corpus(n: int) -> List[str]:
    base_doc = (
        "Chu nghia duy vat bien chung khang dinh vat chat la tinh thu nhat, y thuc la tinh thu hai. "
        "Y thuc xa hoi phan anh ton tai xa hoi va co tinh doc lap tuong doi."
    )
    return [f"{base_doc} [doc-{idx + 1}]" for idx in range(n)]


def build_query() -> str:
    return "Triet hoc Mac-Lenin thuc hien nhung chuc nang co ban nao?"


def load_queries(path: Path, max_queries: int) -> List[str]:
    if not path.exists():
        return [build_query()]
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

    return queries or [build_query()]


def load_corpus_from_subject(subject: Optional[str], max_docs: int) -> List[str]:
    if not subject:
        return build_corpus(max_docs)

    subject_dir = DEFAULT_CHUNKS_DIR / subject
    if not subject_dir.exists():
        raise ValueError(f"Subject folder not found: {subject_dir}")

    json_files = sorted(subject_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No chunk files found in: {subject_dir}")

    raw = json.loads(json_files[0].read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Chunk file must be a list of objects")

    corpus: List[str] = []
    for item in raw:
        if len(corpus) >= max_docs:
            break
        if isinstance(item, dict):
            text = _safe_text(item.get("text"))
        else:
            text = ""
        if text:
            corpus.append(text)

    return corpus or build_corpus(max_docs)


def compute_stats(latencies_ms: Sequence[float]) -> StageStats:
    if not latencies_ms:
        return StageStats(avg_ms=0.0, p50_ms=0.0, p95_ms=0.0)
    sorted_vals = sorted(latencies_ms)
    p50 = statistics.median(sorted_vals)
    p95_index = max(0, int(0.95 * len(sorted_vals)) - 1)
    p95 = sorted_vals[p95_index]
    avg = statistics.mean(sorted_vals)
    return StageStats(avg_ms=avg, p50_ms=p50, p95_ms=p95)


def run_Hybrid_search(
    query: str,
    corpus: Sequence[str],
    top_k: int,
    delay_ms: float,
) -> Tuple[List[str], float]:
    start = time.perf_counter()
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)
    retrieved = list(corpus[: max(0, min(top_k, len(corpus)))])
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return retrieved, elapsed_ms


def run_reranking(
    model: CrossEncoder,
    query: str,
    docs: Sequence[str],
    batch_size: int,
) -> Tuple[List[float], float]:
    pairs = [(query, doc) for doc in docs]
    start = time.perf_counter()
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return list(scores), elapsed_ms


def run_generation(
    query: str,
    docs: Sequence[str],
    delay_ms: float,
) -> Tuple[str, float]:
    start = time.perf_counter()
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)
    top_doc = docs[0] if docs else ""
    generated = f"Answer for: {query} | context: {top_doc[:64]}"
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return generated, elapsed_ms


def benchmark_pipeline(
    model: CrossEncoder,
    queries: Sequence[str],
    corpus: Sequence[str],
    top_k: int,
    batch_size: int,
    Hybrid_search_delay_ms: float,
    generation_delay_ms: float,
    runs: int,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    Hybrid_search_latencies: List[float] = []
    reranking_latencies: List[float] = []
    generation_latencies: List[float] = []
    total_latencies: List[float] = []

    # Warm-up
    warmup_query = queries[0] if queries else build_query()
    _docs, _ = run_Hybrid_search(warmup_query, corpus, top_k, Hybrid_search_delay_ms)
    _scores, _ = run_reranking(model, warmup_query, _docs, batch_size)
    _text, _ = run_generation(warmup_query, _docs, generation_delay_ms)

    for idx in range(runs):
        query = queries[idx % len(queries)] if queries else build_query()
        docs, Hybrid_search_ms = run_Hybrid_search(query, corpus, top_k, Hybrid_search_delay_ms)
        _scores, reranking_ms = run_reranking(model, query, docs, batch_size)
        _text, generation_ms = run_generation(query, docs, generation_delay_ms)
        total_ms = Hybrid_search_ms + reranking_ms + generation_ms

        Hybrid_search_latencies.append(Hybrid_search_ms)
        reranking_latencies.append(reranking_ms)
        generation_latencies.append(generation_ms)
        total_latencies.append(total_ms)

    return Hybrid_search_latencies, reranking_latencies, generation_latencies, total_latencies


def print_results_table(results: Sequence[BackendBenchmarkResult]) -> None:
    headers = ["Backend", "Hybrid_search", "Reranking", "Generation", "Total"]
    rows = [
        [
            result.backend,
            f"{result.Hybrid_search.avg_ms:.2f} ms",
            f"{result.reranking.avg_ms:.2f} ms",
            f"{result.generation.avg_ms:.2f} ms",
            f"{result.total.avg_ms:.2f} ms",
        ]
        for result in results
    ]

    if tabulate is not None:
        print(tabulate(rows, headers=headers, tablefmt="github"))
        return

    col_widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    def format_row(values: Iterable[str]) -> str:
        return "| " + " | ".join(
            value.ljust(col_widths[idx]) for idx, value in enumerate(values)
        ) + " |"

    print(format_row(headers))
    print("| " + " | ".join("-" * width for width in col_widths) + " |")
    for row in rows:
        print(format_row(row))


def run_backend(
    model_name: str,
    backend: str,
    batch_size: int,
    queries: Sequence[str],
    corpus: Sequence[str],
    top_k: int,
    Hybrid_search_delay_ms: float,
    generation_delay_ms: float,
    runs: int,
) -> Optional[BackendBenchmarkResult]:
    print(f"\n=== Backend: {backend} ===")
    try:
        model = CrossEncoder(
            model_name,
            backend=backend,
            device="cpu",
            max_length=512,
        )
    except Exception as exc:
        print(f"Load failed: {exc}")
        return None

    try:
        Hybrid_search_latencies, reranking_latencies, generation_latencies, total_latencies = (
            benchmark_pipeline(
                model,
                queries,
                corpus,
                top_k,
                batch_size,
                Hybrid_search_delay_ms,
                generation_delay_ms,
                runs,
            )
        )
    except Exception as exc:
        print(f"Benchmark failed: {exc}")
        return None

    Hybrid_search_stats = compute_stats(Hybrid_search_latencies)
    reranking_stats = compute_stats(reranking_latencies)
    generation_stats = compute_stats(generation_latencies)
    total_stats = compute_stats(total_latencies)

    print(
        "Summary -> "
        f"Hybrid_search avg={Hybrid_search_stats.avg_ms:.2f} ms | "
        f"reranking avg={reranking_stats.avg_ms:.2f} ms | "
        f"generation avg={generation_stats.avg_ms:.2f} ms | "
        f"total avg={total_stats.avg_ms:.2f} ms"
    )

    return BackendBenchmarkResult(
        backend=backend,
        Hybrid_search=Hybrid_search_stats,
        reranking=reranking_stats,
        generation=generation_stats,
        total=total_stats,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark CrossEncoder backends on CPU")
    parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pairs", type=int, default=48)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--queries-path", type=str, default=str(DEFAULT_QUERIES_PATH))
    parser.add_argument("--subject", type=str, default="", help="Subject folder under data/processed_chunks")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--Hybrid_search-delay-ms", type=float, default=10.0)
    parser.add_argument("--generation-delay-ms", type=float, default=200.0)
    parser.add_argument(
        "--backend",
        default="all",
        choices=["all", "torch", "onnx", "openvino"],
        help="Run only one backend or all",
    )
    args = parser.parse_args()

    queries_path = Path(args.queries_path).expanduser().resolve()
    queries = load_queries(queries_path, max_queries=args.max_queries)
    subject = args.subject.strip() or None
    corpus = load_corpus_from_subject(subject, max_docs=args.pairs)
    top_k = max(1, min(args.top_k, len(corpus)))

    print(f"Model: {args.model}")
    print(
        "Config -> "
        f"Queries: {len(queries)} | Corpus: {len(corpus)} | Top-k: {top_k} | Batch size: {args.batch_size} | "
        f"Runs: {args.runs} | Hybrid_search delay: {args.Hybrid_search_delay_ms:.2f} ms | "
        f"Generation delay: {args.generation_delay_ms:.2f} ms | Subject: {subject or 'default'}"
    )
    if queries_path.exists():
        print(f"Queries path: {queries_path}")

    backends = ("torch", "onnx", "openvino") if args.backend == "all" else (args.backend,)
    results: List[BackendBenchmarkResult] = []
    for backend in backends:
        result = run_backend(
            args.model,
            backend,
            args.batch_size,
            queries,
            corpus,
            top_k,
            args.Hybrid_search_delay_ms,
            args.generation_delay_ms,
            args.runs,
        )
        if result is not None:
            results.append(result)

    if results:
        print("\n=== Results Table (avg) ===")
        print_results_table(results)

        for result in results:
            print(f"\nBackend: {result.backend}")
            print(
                "Hybrid_search avg/p50/p95 -> "
                f"{result.Hybrid_search.avg_ms:.2f} / {result.Hybrid_search.p50_ms:.2f} / {result.Hybrid_search.p95_ms:.2f} ms"
            )
            print(
                "Reranking avg/p50/p95 -> "
                f"{result.reranking.avg_ms:.2f} / {result.reranking.p50_ms:.2f} / {result.reranking.p95_ms:.2f} ms"
            )
            print(
                "Generation avg/p50/p95 -> "
                f"{result.generation.avg_ms:.2f} / {result.generation.p50_ms:.2f} / {result.generation.p95_ms:.2f} ms"
            )
            print(
                "Total avg/p50/p95 -> "
                f"{result.total.avg_ms:.2f} / {result.total.p50_ms:.2f} / {result.total.p95_ms:.2f} ms"
            )


if __name__ == "__main__":
    main()
