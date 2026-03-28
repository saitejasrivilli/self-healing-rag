"""
apps/cli/main.py
=================
Command-line interface for the Agentic AI Platform.

Commands:
  ingest   <path>             Index documents into the knowledge base
  query    "<question>"       Ask a question via the RAG pipeline
  eval                        Run RAGAS evaluation on the golden dataset
  status                      Show platform status and collection stats
  history  [--last N]         Show last N queries from observability log
  clear-db                    Reset the vector store

Usage:
  python apps/cli/main.py ingest ./data/raw
  python apps/cli/main.py query "What is LoRA?"
  python apps/cli/main.py status
  python apps/cli/main.py history --last 5
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))      # keep compat with src/ modules


def cmd_ingest(args):
    from orchestration.workflow_engine import AsyncSelfHealingRAG
    rag = AsyncSelfHealingRAG()
    print(f"📄 Ingesting documents from: {args.path}")
    n = rag.ingest(args.path)
    print(f"✅ {n} chunks indexed.")


def cmd_query(args):
    from orchestration.workflow_engine import AsyncSelfHealingRAG
    rag = AsyncSelfHealingRAG()
    print(f"🔍 Query: {args.question}")
    t0 = time.time()
    resp = rag.query(args.question)
    elapsed = (time.time() - t0) * 1000

    badge = "✅ VERIFIED" if resp.verified else "⚠️  LOW CONFIDENCE"
    healed = f" | 🔄 Self-healed ×{resp.attempts}" if resp.attempts > 1 else ""

    print(f"\n{badge}{healed}")
    print(f"Confidence: {resp.confidence:.0%} | Latency: {elapsed:.0f}ms")
    print(f"\n{resp.answer}\n")
    print(f"Verifier: {resp.reasoning}")
    print(f"\nSources ({len(resp.sources)}):")
    for src in resp.sources:
        print(f"  [{src.score:.3f}] {os.path.basename(src.source)}")


def cmd_status(args):
    from orchestration.workflow_engine import AsyncSelfHealingRAG
    rag = AsyncSelfHealingRAG()
    n = rag.retriever.collection.count()
    api_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    print(f"\n{'─'*40}")
    print(f"  Agentic AI Platform — Status")
    print(f"{'─'*40}")
    print(f"  Knowledge base: {n} chunks")
    print(f"  Gemini API:     {'✅ configured' if api_key else '❌ not set'}")
    print(f"  ChromaDB:       ./chroma_db")
    print(f"{'─'*40}\n")


def cmd_history(args):
    log_path = Path("./data/logs/pipeline_obs.jsonl")
    if not log_path.exists():
        print("No query history found.")
        return
    records = []
    with log_path.open() as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    records = list(reversed(records))[:args.last]
    print(f"\nLast {len(records)} queries:\n")
    for i, r in enumerate(records):
        badge = "✅" if r.get("verified") else "⚠️"
        heal  = f" 🔄×{r['attempts']}" if r.get("attempts", 1) > 1 else ""
        print(f"  [{i+1}] {badge}{heal} {r.get('confidence', 0):.0%} — {r.get('query','')[:70]}")


def cmd_eval(args):
    import subprocess
    print("🧪 Running RAGAS evaluation...")
    result = subprocess.run(
        [sys.executable, "examples/evaluate.py",
         "--docs", "./data/raw",
         "--golden", "./data/processed/qa_pairs.json",
         "--out", "./data/logs/eval_results.json"],
        capture_output=False,
    )
    if result.returncode == 0:
        print("✅ Evaluation complete. Results saved to ./data/logs/eval_results.json")
    else:
        print("❌ Evaluation failed.")


def cmd_clear_db(args):
    import shutil
    confirm = input("⚠️  This will delete all indexed documents. Type 'yes' to confirm: ")
    if confirm.strip().lower() == "yes":
        shutil.rmtree("./chroma_db", ignore_errors=True)
        print("✅ ChromaDB cleared.")
    else:
        print("Aborted.")


def main():
    parser = argparse.ArgumentParser(
        description="Agentic AI Platform CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Index documents")
    p_ingest.add_argument("path", help="Path to document file or directory")

    # query
    p_query = sub.add_parser("query", help="Ask a question")
    p_query.add_argument("question", help="The question to ask")

    # status
    sub.add_parser("status", help="Show platform status")

    # history
    p_hist = sub.add_parser("history", help="Show recent query history")
    p_hist.add_argument("--last", type=int, default=10, metavar="N")

    # eval
    sub.add_parser("eval", help="Run RAGAS evaluation")

    # clear-db
    sub.add_parser("clear-db", help="Reset the vector store")

    args = parser.parse_args()
    dispatch = {
        "ingest":   cmd_ingest,
        "query":    cmd_query,
        "status":   cmd_status,
        "history":  cmd_history,
        "eval":     cmd_eval,
        "clear-db": cmd_clear_db,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
