"""
llm/prompt_templates.py
========================
Centralized prompt template library.
All prompts used across the platform are defined here
so they can be versioned, A/B tested, and swapped without
touching business logic.

Templates use Python's str.format() style placeholders.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    name: str
    system: str
    user: str
    version: str = "1.0"
    description: str = ""

    def format(self, **kwargs) -> dict[str, str]:
        """Returns {"system": str, "user": str} with placeholders filled."""
        try:
            return {
                "system": self.system,
                "user": self.user.format(**kwargs),
            }
        except KeyError as e:
            raise ValueError(f"PromptTemplate '{self.name}' missing variable: {e}") from e


# ── Template registry ─────────────────────────────────────────────────────────
_REGISTRY: dict[str, PromptTemplate] = {}


def register(template: PromptTemplate) -> PromptTemplate:
    _REGISTRY[template.name] = template
    return template


def get(name: str) -> PromptTemplate:
    if name not in _REGISTRY:
        raise KeyError(f"No prompt template named '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_templates() -> list[str]:
    return list(_REGISTRY.keys())


# ── Built-in templates ────────────────────────────────────────────────────────

RAG_ANSWER = register(PromptTemplate(
    name="rag_answer",
    description="Standard grounded RAG answer generation.",
    system=(
        "You are a precise, grounded AI assistant. "
        "Answer ONLY using the provided context. "
        "If the context is insufficient, say so clearly. "
        "Do not fabricate information."
    ),
    user=(
        "Context:\n{context}\n\n"
        "Conversation history:\n{history}\n\n"
        "Question: {query}\n\n"
        "Provide a clear, factual answer grounded strictly in the context above."
    ),
))

VERIFICATION = register(PromptTemplate(
    name="verification",
    description="LLM-based answer verification against retrieved context.",
    system=(
        "You are a strict factual verifier. "
        "Assess whether the answer is supported by the context. "
        "Return only valid JSON."
    ),
    user=(
        "Context:\n{context}\n\n"
        "Question: {query}\n"
        "Answer: {answer}\n\n"
        "Return JSON:\n"
        '{{"confidence": <0.0-1.0>, "verified": <true|false>, "reasoning": "<one sentence>"}}'
    ),
))

QUERY_EXPANSION = register(PromptTemplate(
    name="query_expansion",
    description="Expand a low-confidence query with different vocabulary.",
    system="You are a search query optimizer. Rephrase queries to improve retrieval.",
    user=(
        "The following query did not retrieve sufficient context.\n"
        "Original query: {query}\n"
        "Previous answer (low confidence): {previous_answer}\n\n"
        "Return ONLY the rephrased query, nothing else."
    ),
))

HYDE = register(PromptTemplate(
    name="hyde",
    description="Generate a hypothetical document for HyDE retrieval.",
    system=(
        "You are a technical expert. Generate a concise factual passage "
        "(2-4 sentences) that would directly answer the question. "
        "Write as if you are a document in a knowledge base."
    ),
    user="Question: {query}\n\nWrite a hypothetical passage that answers this question:",
))

PLANNING = register(PromptTemplate(
    name="planning",
    description="Decompose complex queries into retrieval sub-tasks.",
    system=(
        "You are a task planner for a RAG-based AI system. "
        "Decompose user queries into minimal retrieval + reasoning steps. "
        "Return ONLY valid JSON."
    ),
    user=(
        "Query: {query}\n\n"
        "Return a JSON plan with steps (retrieve/generate/verify)."
    ),
))

REFLECTION = register(PromptTemplate(
    name="reflection",
    description="Agent self-critique of generated answers.",
    system=(
        "You are a critical self-reviewer for an AI assistant. "
        "Evaluate answers honestly and return structured JSON feedback."
    ),
    user=(
        "Question: {query}\n"
        "Answer: {answer}\n"
        "Context preview: {context_preview}\n\n"
        "Score completeness, clarity, groundedness, conciseness (0-1 each). "
        "Return JSON with scores + critique + needs_revision boolean."
    ),
))

COT_REASONING = register(PromptTemplate(
    name="cot_reasoning",
    description="Chain-of-thought step-by-step reasoning.",
    system=(
        "You are a careful, step-by-step reasoner. "
        "Ground every claim in the provided context. Never fabricate facts."
    ),
    user=(
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Think step by step:\n"
        "1. What key facts does the context provide?\n"
        "2. How do they relate to the question?\n"
        "3. What is the best supported answer?\n\n"
        "Final Answer:"
    ),
))
