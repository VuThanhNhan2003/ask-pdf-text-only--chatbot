"""
Quick reranker backend benchmark for CPU environments.
Usage:
  python3 scripts/benchmark_reranker_backends.py \
    --model cross-encoder/ms-marco-MiniLM-L6-v2 \
    --batch-size 8 --pairs 48 --runs 5
"""

import argparse
import statistics
import time
from typing import List, Tuple

from sentence_transformers import CrossEncoder


def build_pairs(n: int) -> List[Tuple[str, str]]:
    query = "Triet hoc Mac-Lenin thuc hien nhung chuc nang co ban nao?"
    doc = (
        "Triet hoc Mac-Lenin co hai chuc nang co ban: the gioi quan va phuong phap luan. "
        "No dinh huong nhan thuc va thuc tien cho con nguoi trong boi canh doi moi."
    )
    return [(query, doc) for _ in range(n)]


def run_backend(model_name: str, backend: str, batch_size: int, pairs: List[Tuple[str, str]], runs: int):
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
        return

    # Warm-up
    try:
        model.predict(pairs[: min(8, len(pairs))], batch_size=min(batch_size, 8), show_progress_bar=False)
    except Exception as exc:
        print(f"Warm-up failed: {exc}")
        return

    latencies = []
    for idx in range(runs):
        start = time.perf_counter()
        _ = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        elapsed = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed)
        print(f"Run {idx + 1}: {elapsed:.2f} ms")

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)]
    avg = statistics.mean(latencies)
    qps = (len(pairs) / (avg / 1000.0)) if avg > 0 else 0.0

    print(f"Summary -> avg={avg:.2f} ms | p50={p50:.2f} ms | p95={p95:.2f} ms | throughput={qps:.2f} pairs/s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CrossEncoder backends on CPU")
    parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pairs", type=int, default=48)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--backend",
        default="all",
        choices=["all", "torch", "onnx", "openvino"],
        help="Run only one backend or all",
    )
    args = parser.parse_args()

    pairs = build_pairs(args.pairs)
    print(f"Model: {args.model}")
    print(f"Pairs: {len(pairs)} | Batch size: {args.batch_size} | Runs: {args.runs}")

    backends = ("torch", "onnx", "openvino") if args.backend == "all" else (args.backend,)
    for backend in backends:
        run_backend(args.model, backend, args.batch_size, pairs, args.runs)


if __name__ == "__main__":
    main()
