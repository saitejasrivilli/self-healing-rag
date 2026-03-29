"""
scripts/benchmark.py
=====================
Measures and reports latency, throughput, and quality metrics
for each pipeline configuration. Generates a benchmark report
with p50/p95/p99 latencies and comparison tables.

Usage:
  python scripts/benchmark.py --docs ./data/raw --runs 20
  python scripts/benchmark.py --docs ./data/raw --runs 50 --out ./data/logs/benchmark.json

Configurations benchmarked:
  1. Dense-only retrieval (no BM25, no rerank)
  2. Dense + rerank (no BM25)
  3. Hybrid BM25+Dense + rerank (no healing)
  4. Full system (hybrid + rerank + HyDE + self-healing)
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean, median, quantiles, stdev

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Test queries covering different retrieval difficulty levels
BENCHMARK_QUERIES = [
    # Easy — exact keyword match (BM25 should excel)
    ("easy",   "What is BM25?"),
    ("easy",   "What is ChromaDB?"),
    ("easy",   "What is LoRA?"),
    # Medium — paraphrase match (dense should excel)
    ("medium", "How do transformers process sequential data?"),
    ("medium", "What technique reduces memory during LLM inference?"),
    ("medium", "How is relevance measured in information retrieval?"),
    # Hard — multi-hop / compositional
    ("hard",   "Compare parameter-efficient fine-tuning methods and their GPU memory requirements"),
    ("hard",   "What is the relationship between HyDE and retrieval quality for complex questions?"),
    ("hard",   "How do sparse and dense retrieval complement each other in hybrid search systems?"),
]


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f, c = int(k), int(k) + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def run_benchmark(
    docs_path: str,
    num_runs: int = 20,
    output_path: str | None = None,
    configs: list[str] | None = None,
) -> dict:
    from orchestration.workflow_engine import AsyncSelfHealingRAG
    from rag_pipeline import RAGConfig

    configs_to_run = configs or ["full"]

    # Build config variants
    config_specs = {
        "full": RAGConfig(top_k_retrieve=10, top_k_rerank=4, max_retries=3,
                          confidence_threshold=0.65),
        "no_healing": RAGConfig(top_k_retrieve=10, top_k_rerank=4, max_retries=1,
                                confidence_threshold=0.0),
        "no_rerank": RAGConfig(top_k_retrieve=4, top_k_rerank=4, max_retries=1,
                               confidence_threshold=0.0),
        "dense_only": RAGConfig(top_k_retrieve=10, top_k_rerank=4, max_retries=1,
                                confidence_threshold=0.0),
    }

    results = {}
    queries = [q for _, q in BENCHMARK_QUERIES[:num_runs]] \
        if num_runs <= len(BENCHMARK_QUERIES) \
        else [q for _, q in BENCHMARK_QUERIES] * (num_runs // len(BENCHMARK_QUERIES) + 1)
    queries = queries[:num_runs]

    for config_name in configs_to_run:
        cfg = config_specs.get(config_name, config_specs["full"])
        print(f"\n{'─'*50}")
        print(f"  Config: {config_name.upper()} ({num_runs} queries)")
        print(f"{'─'*50}")

        enable_hyde = config_name == "full"
        rag = AsyncSelfHealingRAG(config=cfg, enable_hyde=enable_hyde,
                                   enable_observability=False)
        rag.ingest(docs_path)

        latencies, confidences, verified_count, heal_count = [], [], 0, 0

        for i, query in enumerate(queries):
            t0 = time.time()
            try:
                resp = rag.query(query)
                latency = (time.time() - t0) * 1000
                latencies.append(latency)
                confidences.append(resp.confidence)
                if resp.verified:
                    verified_count += 1
                if resp.attempts > 1:
                    heal_count += 1
                status = "✅" if resp.verified else "⚠️"
                print(f"  [{i+1:02d}] {status} {latency:.0f}ms conf={resp.confidence:.2f} "
                      f"att={resp.attempts} | {query[:50]}")
            except Exception as e:
                logger.warning("Query failed: %s — %s", query, e)

        n = len(latencies)
        results[config_name] = {
            "n": n,
            "latency_p50_ms":   round(percentile(latencies, 50), 1),
            "latency_p95_ms":   round(percentile(latencies, 95), 1),
            "latency_p99_ms":   round(percentile(latencies, 99), 1),
            "latency_mean_ms":  round(mean(latencies) if latencies else 0, 1),
            "latency_stdev_ms": round(stdev(latencies) if len(latencies) > 1 else 0, 1),
            "avg_confidence":   round(mean(confidences) if confidences else 0, 3),
            "verified_rate":    round(verified_count / max(n, 1), 3),
            "self_heal_rate":   round(heal_count / max(n, 1), 3),
        }

    # Print comparison table
    print(f"\n{'═'*70}")
    print("  BENCHMARK RESULTS")
    print(f"{'═'*70}")
    print(f"  {'Config':<16} {'p50':>6} {'p95':>6} {'p99':>7} {'Conf':>6} {'Verified':>9} {'Healed':>7}")
    print(f"  {'─'*16} {'─'*6} {'─'*6} {'─'*7} {'─'*6} {'─'*9} {'─'*7}")
    for name, r in results.items():
        print(f"  {name:<16} {r['latency_p50_ms']:>5.0f}ms {r['latency_p95_ms']:>5.0f}ms "
              f"{r['latency_p99_ms']:>6.0f}ms {r['avg_confidence']:>6.2f} "
              f"{r['verified_rate']:>8.0%} {r['self_heal_rate']:>6.0%}")
    print(f"{'═'*70}\n")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"configs": results, "num_runs": num_runs,
                       "docs_path": docs_path}, f, indent=2)
        print(f"✅ Benchmark saved → {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark the RAG pipeline configurations")
    parser.add_argument("--docs",    default="./data/raw",  help="Document path to ingest")
    parser.add_argument("--runs",    type=int, default=9,   help="Number of benchmark queries")
    parser.add_argument("--out",     default=None,          help="Output JSON path")
    parser.add_argument("--configs", nargs="+",
                        choices=["full", "no_healing", "no_rerank", "dense_only"],
                        default=["full", "no_healing"],
                        help="Configurations to benchmark")
    args = parser.parse_args()
    run_benchmark(docs_path=args.docs, num_runs=args.runs,
                  output_path=args.out, configs=args.configs)


if __name__ == "__main__":
    main()
