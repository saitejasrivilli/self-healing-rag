"""
tests/unit/test_streaming.py
=============================
Tests for the SSE streaming pipeline — event format, ordering, error handling.
All LLM/retrieval calls are mocked; no API key required.
"""

import sys
import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from api.streaming import _sse, StreamQueryRequest


# ── SSE formatting ────────────────────────────────────────────────────────────
def test_sse_format_basic():
    result = _sse("stage", {"stage": "retrieve", "message": "Retrieving..."})
    assert result.startswith("event: stage\n")
    assert "data: " in result
    assert result.endswith("\n\n")


def test_sse_data_is_valid_json():
    result = _sse("done", {"confidence": 0.9, "verified": True})
    lines = result.strip().split("\n")
    data_line = next(l for l in lines if l.startswith("data: "))
    payload = json.loads(data_line[len("data: "):])
    assert payload["confidence"] == 0.9
    assert payload["verified"] is True


def test_sse_token_event():
    result = _sse("token", {"token": "Hello "})
    assert "event: token" in result
    assert '"token": "Hello "' in result


def test_sse_error_event():
    result = _sse("error", {"message": "Knowledge base is empty."})
    assert "event: error" in result
    assert "Knowledge base is empty" in result


def test_sse_empty_data():
    result = _sse("stage", {})
    assert "event: stage" in result
    assert "data: {}" in result


# ── StreamQueryRequest validation ────────────────────────────────────────────
def test_stream_request_valid():
    req = StreamQueryRequest(query="What is LoRA?")
    assert req.query == "What is LoRA?"
    assert req.enable_hyde is True
    assert req.max_retries == 3
    assert req.confidence_threshold == 0.65


def test_stream_request_query_too_short():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        StreamQueryRequest(query="Hi")


def test_stream_request_query_too_long():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        StreamQueryRequest(query="x" * 2001)


def test_stream_request_custom_params():
    req = StreamQueryRequest(
        query="What is hybrid search?",
        confidence_threshold=0.8,
        max_retries=5,
        enable_hyde=False,
        session_id="sess_123",
    )
    assert req.confidence_threshold == 0.8
    assert req.max_retries == 5
    assert req.enable_hyde is False
    assert req.session_id == "sess_123"


def test_stream_request_confidence_bounds():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        StreamQueryRequest(query="test query", confidence_threshold=1.5)
    with pytest.raises(ValidationError):
        StreamQueryRequest(query="test query", confidence_threshold=-0.1)


# ── SSE event sequence ────────────────────────────────────────────────────────
def test_stage_events_are_ordered():
    """Verify stage names match expected pipeline order."""
    expected_stages = ["hyde", "retrieve", "rerank", "generate", "verify"]
    seen = []
    for stage in expected_stages:
        msg = _sse("stage", {"stage": stage, "message": f"Running {stage}..."})
        data = json.loads(msg.split("data: ")[1])
        seen.append(data["stage"])
    assert seen == expected_stages


def test_done_event_has_required_fields():
    payload = {
        "confidence": 0.87,
        "verified": True,
        "attempts": 1,
        "latency_ms": 1234.5,
        "reasoning": "All claims supported.",
        "query_used": "What is LoRA?",
        "sources": [],
    }
    msg = _sse("done", payload)
    data = json.loads(msg.split("data: ")[1])
    for field in ["confidence", "verified", "attempts", "latency_ms", "sources"]:
        assert field in data


# ── Stream generator with mocked pipeline ────────────────────────────────────
@pytest.mark.asyncio
async def test_stream_pipeline_empty_kb():
    """Empty knowledge base should yield error event immediately."""
    from api.streaming import _stream_pipeline

    mock_rag = MagicMock()
    mock_rag.retriever.collection.count.return_value = 0

    with patch("api.streaming.AsyncSelfHealingRAG", return_value=mock_rag), \
         patch("api.streaming.RAGConfig"):
        events = []
        async for chunk in _stream_pipeline(StreamQueryRequest(query="What is RAG?")):
            events.append(chunk)

        event_types = [e.split("\n")[0].replace("event: ", "") for e in events]
        assert "error" in event_types


@pytest.mark.asyncio
async def test_stream_pipeline_yields_stage_events():
    """Full pipeline run should emit stage events in order."""
    from api.streaming import _stream_pipeline

    mock_chunk = MagicMock()
    mock_chunk.text = "RAG combines retrieval and generation."
    mock_chunk.source = "doc.txt"
    mock_chunk.score = 0.9
    mock_chunk.retrieval_method = "hybrid"

    mock_rag = MagicMock()
    mock_rag.retriever.collection.count.return_value = 5
    mock_rag.hyde.expand = MagicMock(return_value=("expanded query", "hypothetical doc"))
    mock_rag.retriever.retrieve = MagicMock(return_value=[mock_chunk])
    mock_rag.reranker.rerank = MagicMock(return_value=[mock_chunk])
    mock_rag.generator.generate = MagicMock(return_value="RAG combines retrieval.")
    mock_rag.verifier.verify = MagicMock(return_value=(0.9, True, "Supported."))
    mock_rag.config.top_k_rerank = 4

    with patch("api.streaming.AsyncSelfHealingRAG", return_value=mock_rag), \
         patch("api.streaming.RAGConfig"):
        events = []
        async for chunk in _stream_pipeline(StreamQueryRequest(query="What is RAG?")):
            events.append(chunk)

        event_types = []
        for e in events:
            lines = e.strip().split("\n")
            for line in lines:
                if line.startswith("event: "):
                    event_types.append(line.replace("event: ", ""))

        assert "stage" in event_types
        assert "token" in event_types
        assert "done" in event_types
        # No errors
        assert "error" not in event_types


@pytest.mark.asyncio
async def test_stream_tokens_spell_out_answer():
    """Token events should reconstruct the full answer when joined."""
    from api.streaming import _stream_pipeline

    answer = "LoRA reduces trainable parameters."
    mock_chunk = MagicMock()
    mock_chunk.text = answer
    mock_chunk.source = "doc.txt"
    mock_chunk.score = 0.9
    mock_chunk.retrieval_method = "hybrid"

    mock_rag = MagicMock()
    mock_rag.retriever.collection.count.return_value = 3
    mock_rag.hyde.expand = MagicMock(return_value=("query", ""))
    mock_rag.retriever.retrieve = MagicMock(return_value=[mock_chunk])
    mock_rag.reranker.rerank = MagicMock(return_value=[mock_chunk])
    mock_rag.generator.generate = MagicMock(return_value=answer)
    mock_rag.verifier.verify = MagicMock(return_value=(0.88, True, "Supported."))
    mock_rag.config.top_k_rerank = 4

    with patch("api.streaming.AsyncSelfHealingRAG", return_value=mock_rag), \
         patch("api.streaming.RAGConfig"):
        tokens = []
        async for chunk in _stream_pipeline(StreamQueryRequest(query="What is LoRA?")):
            if "event: token" in chunk:
                data = json.loads(chunk.split("data: ")[1])
                tokens.append(data["token"])

        reconstructed = "".join(tokens).strip()
        assert reconstructed == answer
