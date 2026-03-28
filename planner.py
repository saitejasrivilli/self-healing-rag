"""
cognition/planner.py
=====================
Decomposes complex user queries into a sequence of sub-tasks
that the orchestration layer can execute step-by-step.

For simple queries → single-step plan (direct RAG)
For complex queries → multi-step plan (decompose → retrieve each → synthesize)

Plan schema:
  {
    "steps": [
      {"id": 1, "action": "retrieve", "query": "...", "depends_on": []},
      {"id": 2, "action": "generate", "depends_on": [1]},
      {"id": 3, "action": "verify",   "depends_on": [2]},
    ],
    "strategy": "sequential" | "parallel",
    "complexity": "simple" | "complex"
  }
"""

from __future__ import annotations
import json
import logging
import os
from typing import Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)

PLANNER_SYSTEM = """You are a task planner for a RAG-based AI system.
Given a user query, decompose it into the minimal set of retrieval + reasoning steps.
Return ONLY valid JSON. No markdown, no explanation."""

PLANNER_PROMPT = """Query: {query}

Classify this query and return a JSON plan:
{{
  "complexity": "simple" or "complex",
  "strategy": "sequential" or "parallel",
  "steps": [
    {{"id": 1, "action": "retrieve", "query": "<what to retrieve>", "depends_on": []}},
    {{"id": 2, "action": "generate", "depends_on": [1]}},
    {{"id": 3, "action": "verify",   "depends_on": [2]}}
  ]
}}

For simple single-topic questions use 3 steps (retrieve→generate→verify).
For complex multi-part questions decompose into multiple retrieve steps first."""


SIMPLE_PLAN_TEMPLATE = {
    "complexity": "simple",
    "strategy": "sequential",
    "steps": [
        {"id": 1, "action": "retrieve", "query": "{query}", "depends_on": []},
        {"id": 2, "action": "generate", "depends_on": [1]},
        {"id": 3, "action": "verify",   "depends_on": [2]},
    ],
}


class QueryPlanner:
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key and use_llm:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                "gemini-1.5-flash",
                system_instruction=PLANNER_SYSTEM,
                generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=400),
            )
            self._ready = True
        else:
            self._ready = False

    def plan(self, query: str) -> dict:
        """Return an execution plan for the query."""
        if self._ready and self.use_llm:
            return self._llm_plan(query)
        return self._heuristic_plan(query)

    def _llm_plan(self, query: str) -> dict:
        try:
            resp = self.model.generate_content(PLANNER_PROMPT.format(query=query))
            text = resp.text.strip().replace("```json", "").replace("```", "")
            plan = json.loads(text)
            logger.info("LLM plan: %s | %d steps", plan.get("complexity"), len(plan.get("steps", [])))
            return plan
        except Exception as e:
            logger.warning("LLM planner failed (%s) — falling back to heuristic", e)
            return self._heuristic_plan(query)

    @staticmethod
    def _heuristic_plan(query: str) -> dict:
        """Rule-based plan — complex if query has multiple clauses."""
        is_complex = any(kw in query.lower() for kw in [" and ", " also ", " additionally ", "compare", "difference between", "versus"])
        if is_complex:
            parts = [s.strip() for s in query.replace(" and ", "|").replace(",", "|").split("|") if s.strip()]
            steps = [{"id": i+1, "action": "retrieve", "query": p, "depends_on": []} for i, p in enumerate(parts[:3])]
            dep_ids = [s["id"] for s in steps]
            steps.append({"id": len(steps)+1, "action": "generate", "depends_on": dep_ids})
            steps.append({"id": len(steps)+1, "action": "verify",   "depends_on": [len(steps)]})
            return {"complexity": "complex", "strategy": "parallel", "steps": steps}

        plan = dict(SIMPLE_PLAN_TEMPLATE)
        plan["steps"] = [
            {"id": 1, "action": "retrieve", "query": query, "depends_on": []},
            {"id": 2, "action": "generate", "depends_on": [1]},
            {"id": 3, "action": "verify",   "depends_on": [2]},
        ]
        return plan
