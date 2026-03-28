"""
orchestration/agent_router.py
==============================
Routes incoming requests to the appropriate agent based on
routing_policy.yaml rules and intent classification.

Routing logic:
  1. Keyword trigger matching against routing_policy.yaml
  2. Confidence-based fallback
  3. Returns agent name + metadata for workflow_engine to execute
"""

from __future__ import annotations
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

POLICY_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "routing_policy.yaml")


@dataclass
class RoutingDecision:
    agent_name: str
    confidence: float
    matched_trigger: Optional[str]
    fallback_used: bool


class AgentRouter:
    """
    Keyword + policy-based router.
    Maps user queries → agent names using routing_policy.yaml.
    """

    def __init__(self, policy_path: str = POLICY_PATH):
        self.policy = self._load_policy(policy_path)
        self.agents = self.policy.get("routing", {}).get("agents", {})
        self.fallback = self.policy.get("routing", {}).get("fallback_agent", "rag_agent")
        logger.info("AgentRouter loaded %d agents", len(self.agents))

    @staticmethod
    def _load_policy(path: str) -> dict:
        try:
            with open(path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("routing_policy.yaml not found — using defaults")
            return {
                "routing": {
                    "agents": {
                        "rag_agent": {"triggers": ["question", "what", "how", "explain"]},
                    },
                    "fallback_agent": "rag_agent",
                }
            }

    def route(self, query: str) -> RoutingDecision:
        """Return the best agent for this query."""
        query_lower = query.lower()
        best_agent = None
        best_trigger = None
        best_score = 0

        for agent_name, agent_cfg in self.agents.items():
            triggers = agent_cfg.get("triggers", [])
            for trigger in triggers:
                if re.search(r"\b" + re.escape(trigger) + r"\b", query_lower):
                    # Simple scoring: longer trigger = more specific = higher score
                    score = len(trigger)
                    if score > best_score:
                        best_score = score
                        best_agent = agent_name
                        best_trigger = trigger

        if best_agent:
            confidence = min(0.5 + best_score * 0.05, 0.99)
            return RoutingDecision(
                agent_name=best_agent,
                confidence=confidence,
                matched_trigger=best_trigger,
                fallback_used=False,
            )

        logger.info("No trigger matched — routing to fallback: %s", self.fallback)
        return RoutingDecision(
            agent_name=self.fallback,
            confidence=0.4,
            matched_trigger=None,
            fallback_used=True,
        )

    def list_agents(self) -> list[str]:
        return list(self.agents.keys())

    def agent_description(self, agent_name: str) -> str:
        return self.agents.get(agent_name, {}).get("description", "No description.")
