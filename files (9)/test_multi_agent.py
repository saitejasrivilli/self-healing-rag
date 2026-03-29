"""
tests/agent_tests/test_multi_agent.py
======================================
Tests for MultiAgentCoordinator, RAGAgent, ToolAgent, SynthesisAgent.
All LLM/retrieval calls mocked — no API key required.
"""

import sys
import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from orchestration.multi_agent import (
    AgentResult,
    RAGAgent,
    ToolAgent,
    SynthesisAgent,
    MultiAgentCoordinator,
    MultiAgentResponse,
)


# ── AgentResult ───────────────────────────────────────────────────────────────
def test_agent_result_success():
    r = AgentResult(agent_name="rag", output="some answer", confidence=0.9, latency_ms=100)
    assert r.success is True


def test_agent_result_failure_on_empty_output():
    r = AgentResult(agent_name="rag", output="", confidence=0.0, latency_ms=50)
    assert r.success is False


def test_agent_result_failure_on_error():
    r = AgentResult(agent_name="rag", output="text", confidence=0.0,
                    latency_ms=50, error="API failure")
    assert r.success is False


def test_agent_result_fields():
    r = AgentResult(agent_name="tool", output="data", confidence=0.7,
                    latency_ms=200, metadata={"tool": "web_search"})
    assert r.agent_name == "tool"
    assert r.metadata["tool"] == "web_search"


# ── ToolAgent ─────────────────────────────────────────────────────────────────
def test_tool_agent_name():
    agent = ToolAgent()
    assert agent.name == "tool_agent"


@pytest.mark.asyncio
async def test_tool_agent_success():
    from tools.tool_registry import ToolResult
    mock_result = ToolResult(
        tool_name="web_search",
        output=[{"title": "Result", "snippet": "Hybrid search combines BM25 and dense retrieval."}],
        success=True,
        latency_ms=50,
    )
    with patch("orchestration.multi_agent.registry") as mock_reg:
        mock_reg.invoke.return_value = mock_result
        agent = ToolAgent()
        result = await agent.run("hybrid search", {"tool_hint": "web_search"})
        assert result.success is True
        assert "Hybrid search" in result.output
        assert result.agent_name == "tool_agent"


@pytest.mark.asyncio
async def test_tool_agent_failure_propagates():
    from tools.tool_registry import ToolResult
    mock_result = ToolResult(
        tool_name="web_search", output=None, success=False, error="API timeout"
    )
    with patch("orchestration.multi_agent.registry") as mock_reg:
        mock_reg.invoke.return_value = mock_result
        agent = ToolAgent()
        result = await agent.run("query", {"tool_hint": "web_search"})
        assert result.success is False
        assert result.error == "API timeout"


@pytest.mark.asyncio
async def test_tool_agent_formats_list_output():
    from tools.tool_registry import ToolResult
    items = [
        {"title": "A", "snippet": "snippet A"},
        {"title": "B", "snippet": "snippet B"},
    ]
    mock_result = ToolResult(tool_name="web_search", output=items, success=True)
    with patch("orchestration.multi_agent.registry") as mock_reg:
        mock_reg.invoke.return_value = mock_result
        agent = ToolAgent()
        result = await agent.run("query", {})
        assert "snippet A" in result.output
        assert "snippet B" in result.output


# ── SynthesisAgent ────────────────────────────────────────────────────────────
def test_synthesis_agent_name():
    assert SynthesisAgent().name == "synthesis_agent"


@pytest.mark.asyncio
async def test_synthesis_agent_empty_input():
    agent = SynthesisAgent()
    result = await agent.run("query", {"agent_results": []})
    assert result.success is False or "No agent results" in result.output


@pytest.mark.asyncio
async def test_synthesis_agent_extractive_fallback(monkeypatch):
    """When no API key, should use extractive synthesis."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    results = [
        AgentResult("rag_agent", "LoRA reduces parameters.", 0.85, 500),
        AgentResult("tool_agent", "Recent benchmarks show 10x speedup.", 0.7, 300),
    ]
    agent = SynthesisAgent()
    result = await agent.run("Tell me about LoRA", {"agent_results": results})
    assert result.success is True
    assert "LoRA reduces parameters" in result.output
    assert "Recent benchmarks" in result.output


@pytest.mark.asyncio
async def test_synthesis_agent_all_failed():
    results = [
        AgentResult("rag_agent", "", 0.0, 100, error="KB empty"),
        AgentResult("tool_agent", "", 0.0, 50, error="API down"),
    ]
    agent = SynthesisAgent()
    result = await agent.run("query", {"agent_results": results})
    assert "failed" in result.output.lower() or result.confidence == 0.0


@pytest.mark.asyncio
async def test_synthesis_confidence_is_average():
    """Extractive synthesis confidence should be avg of successful agents."""
    results = [
        AgentResult("rag_agent", "Answer A", 0.8, 100),
        AgentResult("tool_agent", "Answer B", 0.6, 50),
    ]
    agent = SynthesisAgent()
    # Force extractive path
    import os
    with patch.dict(os.environ, {}, clear=True):
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]
        result = agent._extractive_synthesize("query", results, __import__("time").time())
    assert abs(result.confidence - 0.7) < 0.01


# ── MultiAgentCoordinator ─────────────────────────────────────────────────────
@pytest.fixture
def coordinator():
    return MultiAgentCoordinator()


def test_coordinator_needs_rag_for_normal_query(coordinator):
    assert coordinator._needs_rag("What is transformer architecture?") is True


def test_coordinator_needs_tool_for_search_query(coordinator):
    assert coordinator._needs_tool("search for latest AI papers") is True
    assert coordinator._needs_tool("what is the latest news on AI") is True


def test_coordinator_rag_only_for_definition(coordinator):
    assert coordinator._needs_rag("What is LoRA?") is True
    assert coordinator._needs_tool("What is LoRA?") is False


def test_coordinator_both_for_mixed_query(coordinator):
    assert coordinator._needs_rag("explain LoRA and search for recent benchmarks") is True
    assert coordinator._needs_tool("explain LoRA and search for recent benchmarks") is True


@pytest.mark.asyncio
async def test_coordinator_rag_only_returns_rag_result():
    """Pure knowledge query: only RAGAgent should run."""
    mock_rag_result = AgentResult("rag_agent", "LoRA answer", 0.85, 500)

    coordinator = MultiAgentCoordinator()
    coordinator.rag_agent.run = AsyncMock(return_value=mock_rag_result)
    coordinator.tool_agent.run = AsyncMock(return_value=AgentResult("tool_agent","",0,0))
    coordinator.synthesizer.run = AsyncMock()

    response = await coordinator.run("What is LoRA fine-tuning?")
    assert response.final_answer == "LoRA answer"
    assert "rag_agent" in response.agents_used
    assert response.confidence == 0.85
    # Synthesizer should NOT have been called (single agent)
    coordinator.synthesizer.run.assert_not_called()


@pytest.mark.asyncio
async def test_coordinator_parallel_runs_both_agents():
    """Search query: both RAG and tool agents should run in parallel."""
    rag_result  = AgentResult("rag_agent",  "LoRA from KB",  0.80, 400)
    tool_result = AgentResult("tool_agent", "Recent benchmarks", 0.70, 200)
    synth_result = AgentResult("synthesis_agent", "Combined answer", 0.82, 100)

    coordinator = MultiAgentCoordinator()
    coordinator.rag_agent.run  = AsyncMock(return_value=rag_result)
    coordinator.tool_agent.run = AsyncMock(return_value=tool_result)
    coordinator.synthesizer.run = AsyncMock(return_value=synth_result)

    response = await coordinator.run("Explain LoRA and search for recent benchmarks")
    assert response.final_answer == "Combined answer"
    assert "rag_agent"  in response.agents_used
    assert "tool_agent" in response.agents_used
    assert "synthesis_agent" in response.agents_used
    assert response.strategy == "parallel"


@pytest.mark.asyncio
async def test_coordinator_response_has_all_fields():
    rag_result = AgentResult("rag_agent", "Answer", 0.85, 300)
    coordinator = MultiAgentCoordinator()
    coordinator.rag_agent.run = AsyncMock(return_value=rag_result)

    response = await coordinator.run("What is attention?")
    assert hasattr(response, "final_answer")
    assert hasattr(response, "confidence")
    assert hasattr(response, "agents_used")
    assert hasattr(response, "agent_results")
    assert hasattr(response, "total_latency_ms")
    assert hasattr(response, "strategy")
    assert response.total_latency_ms > 0


@pytest.mark.asyncio
async def test_coordinator_fallback_to_rag_on_unknown_query():
    """Unrecognized query should fall back to RAGAgent."""
    rag_result = AgentResult("rag_agent", "Fallback answer", 0.6, 200)
    coordinator = MultiAgentCoordinator()
    coordinator.rag_agent.run = AsyncMock(return_value=rag_result)

    response = await coordinator.run("xyzzy incomprehensible query")
    assert "rag_agent" in response.agents_used
    assert response.final_answer == "Fallback answer"
