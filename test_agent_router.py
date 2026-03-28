"""
tests/agent_tests/test_agent_router.py
=======================================
Tests for AgentRouter — routing decisions, fallbacks, policy loading.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from orchestration.agent_router import AgentRouter, RoutingDecision


MOCK_POLICY = """
routing:
  default_agent: "rag_agent"
  agents:
    rag_agent:
      description: "RAG for document Q&A"
      triggers: ["question", "explain", "what is", "how does"]
    tool_agent:
      description: "External tool execution"
      triggers: ["search", "fetch", "calculate", "lookup"]
    reflection_agent:
      description: "Answer validation"
      triggers: ["verify", "check", "validate"]
  fallback_agent: "rag_agent"
  confidence_threshold: 0.65
"""


@pytest.fixture
def router(tmp_path):
    policy_file = tmp_path / "routing_policy.yaml"
    policy_file.write_text(MOCK_POLICY)
    return AgentRouter(policy_path=str(policy_file))


# ── Routing decisions ─────────────────────────────────────────────────────────
def test_routes_to_rag_on_question(router):
    decision = router.route("What is transformer architecture?")
    assert decision.agent_name == "rag_agent"
    assert not decision.fallback_used


def test_routes_to_tool_on_search(router):
    decision = router.route("Please search for recent AI papers")
    assert decision.agent_name == "tool_agent"
    assert not decision.fallback_used


def test_routes_to_reflection_on_verify(router):
    decision = router.route("Can you verify this answer?")
    assert decision.agent_name == "reflection_agent"
    assert not decision.fallback_used


def test_fallback_on_no_trigger_match(router):
    decision = router.route("blah blah unrecognized gibberish xyz")
    assert decision.agent_name == "rag_agent"
    assert decision.fallback_used is True


def test_routing_decision_has_confidence(router):
    decision = router.route("explain how RAG works")
    assert 0.0 < decision.confidence <= 1.0


def test_fallback_has_lower_confidence(router):
    matched = router.route("what is LoRA?")
    fallback = router.route("xyzzy foo bar")
    assert matched.confidence > fallback.confidence


def test_longer_trigger_wins(router):
    """'how does' (2 words) should score higher than 'question' (1 word)."""
    d = router.route("how does attention work?")
    assert d.matched_trigger is not None
    assert len(d.matched_trigger) >= 3


# ── List agents ───────────────────────────────────────────────────────────────
def test_list_agents_returns_all(router):
    agents = router.list_agents()
    assert "rag_agent" in agents
    assert "tool_agent" in agents
    assert "reflection_agent" in agents


def test_agent_description(router):
    desc = router.agent_description("rag_agent")
    assert "RAG" in desc or len(desc) > 0


def test_unknown_agent_description(router):
    desc = router.agent_description("nonexistent_agent")
    assert desc == "No description."


# ── Policy loading ────────────────────────────────────────────────────────────
def test_missing_policy_uses_defaults(tmp_path):
    router = AgentRouter(policy_path=str(tmp_path / "nonexistent.yaml"))
    assert "rag_agent" in router.list_agents()


# ── RoutingDecision fields ────────────────────────────────────────────────────
def test_routing_decision_fields(router):
    d = router.route("what is BERT?")
    assert hasattr(d, "agent_name")
    assert hasattr(d, "confidence")
    assert hasattr(d, "matched_trigger")
    assert hasattr(d, "fallback_used")


def test_case_insensitive_routing(router):
    d1 = router.route("WHAT IS BERT?")
    d2 = router.route("what is bert?")
    assert d1.agent_name == d2.agent_name
