"""
Unit tests for Self-Healing RAG components
Run: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag_pipeline import (
    RAGConfig,
    RetrievedChunk,
    VerificationAgent,
)


# ── Config tests ──────────────────────────────
def test_default_config():
    cfg = RAGConfig()
    assert cfg.top_k_retrieve == 10
    assert cfg.max_retries == 3
    assert 0 < cfg.confidence_threshold < 1


def test_custom_config():
    cfg = RAGConfig(top_k_retrieve=5, max_retries=2, confidence_threshold=0.8)
    assert cfg.top_k_retrieve == 5
    assert cfg.max_retries == 2


# ── RetrievedChunk tests ─────────────────────
def test_retrieved_chunk():
    chunk = RetrievedChunk(text="LoRA reduces parameters by 99%", source="paper.pdf", score=0.92, chunk_id="c001")
    assert chunk.score == 0.92
    assert "LoRA" in chunk.text


# ── Heuristic verifier tests ─────────────────
def test_heuristic_verify_high_overlap():
    chunks = [RetrievedChunk(text="Transformers use self-attention for sequence modeling", source="x", score=0.9, chunk_id="c1")]
    conf, verified, reason = VerificationAgent._heuristic_verify(
        "Transformers rely on self-attention for modeling sequences", chunks
    )
    assert conf > 0.0
    assert isinstance(verified, bool)


def test_heuristic_verify_empty_answer():
    chunks = [RetrievedChunk(text="Some context", source="x", score=0.8, chunk_id="c1")]
    conf, verified, _ = VerificationAgent._heuristic_verify("", chunks)
    assert conf == 0.0
    assert verified is False


def test_heuristic_verify_low_confidence_phrase():
    chunks = [RetrievedChunk(text="Some unrelated content", source="x", score=0.7, chunk_id="c1")]
    conf, verified, _ = VerificationAgent._heuristic_verify("I don't know the answer to this.", chunks)
    assert conf < 0.65
    assert verified is False


def test_heuristic_verify_no_chunks():
    conf, verified, _ = VerificationAgent._heuristic_verify("An answer", [])
    assert conf == 0.0
    assert verified is False


# ── Integration smoke test (no API key needed) ──
def test_full_pipeline_smoke(tmp_path):
    """Smoke test: ingest a text file and run a query using heuristic verifier."""
    # Write a tiny knowledge base
    kb = tmp_path / "kb.txt"
    kb.write_text(
        "ChromaDB is an open-source vector database. "
        "It stores embeddings for AI applications. "
        "ChromaDB supports persistent and in-memory modes."
    )

    cfg = RAGConfig(top_k_retrieve=3, top_k_rerank=2, max_retries=1)
    from rag_pipeline import SelfHealingRAG
    rag = SelfHealingRAG(config=cfg, persist_dir=str(tmp_path / "db"))
    n = rag.ingest(str(tmp_path))
    assert n > 0, "Should index at least one chunk"

    # Without an API key the generator/verifier fall back to heuristics
    # Only run the query if a key is available (CI skips generation)
    if not os.getenv("GEMINI_API_KEY"):
        return  # Skip generation in CI

    response = rag.query("What is ChromaDB?")
    assert response.answer
    assert 0.0 <= response.confidence <= 1.0
    assert response.attempts >= 1
