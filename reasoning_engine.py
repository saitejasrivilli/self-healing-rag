"""
cognition/reasoning_engine.py
===============================
Chain-of-thought reasoning engine.
Wraps LLM generation with explicit step-by-step reasoning
before producing the final answer (reduces hallucination on complex queries).

Modes:
  - direct      : Standard single-shot generation
  - cot         : Chain-of-thought (reason then answer)
  - self_ask    : Generate sub-questions, answer each, synthesize
"""

from __future__ import annotations
import logging
import os
from enum import Enum
from typing import Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)


class ReasoningMode(str, Enum):
    DIRECT   = "direct"
    COT      = "cot"
    SELF_ASK = "self_ask"


COT_SYSTEM = """You are a careful, step-by-step reasoner. 
When answering, first lay out your reasoning explicitly, then give a final answer.
Ground every claim in the provided context. Never fabricate facts."""

COT_PROMPT = """Context:
{context}

Question: {query}

Think step by step:
1. What key facts does the context provide?
2. How do they relate to the question?
3. What is the best supported answer?

Final Answer:"""

SELF_ASK_PROMPT = """Context:
{context}

Main Question: {query}

Break this into sub-questions, answer each from the context, then synthesize.
Format:
Sub-question 1: ...
Answer 1: ...
Sub-question 2: ...
Answer 2: ...
Final Answer: ..."""


class ReasoningEngine:
    def __init__(self, mode: ReasoningMode = ReasoningMode.COT):
        self.mode = mode
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                "gemini-1.5-flash",
                system_instruction=COT_SYSTEM,
                generation_config=genai.GenerationConfig(temperature=0.2, max_output_tokens=600),
            )
            self._ready = True
        else:
            self._ready = False
            logger.warning("ReasoningEngine: no API key — will return placeholder")

    def reason(self, query: str, context: str) -> tuple[str, str]:
        """
        Returns (answer, reasoning_trace).
        reasoning_trace is the CoT steps (empty for direct mode).
        """
        if not self._ready:
            return "API key required for LLM reasoning.", ""

        if self.mode == ReasoningMode.DIRECT:
            return self._direct(query, context)
        elif self.mode == ReasoningMode.COT:
            return self._cot(query, context)
        elif self.mode == ReasoningMode.SELF_ASK:
            return self._self_ask(query, context)

        return self._cot(query, context)

    def _direct(self, query: str, context: str) -> tuple[str, str]:
        resp = self.model.generate_content(f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
        return resp.text.strip(), ""

    def _cot(self, query: str, context: str) -> tuple[str, str]:
        prompt = COT_PROMPT.format(context=context, query=query)
        resp = self.model.generate_content(prompt)
        full = resp.text.strip()
        # Split reasoning trace from final answer
        if "Final Answer:" in full:
            parts = full.split("Final Answer:", 1)
            return parts[1].strip(), parts[0].strip()
        return full, ""

    def _self_ask(self, query: str, context: str) -> tuple[str, str]:
        prompt = SELF_ASK_PROMPT.format(context=context, query=query)
        resp = self.model.generate_content(prompt)
        full = resp.text.strip()
        if "Final Answer:" in full:
            parts = full.split("Final Answer:", 1)
            return parts[1].strip(), parts[0].strip()
        return full, ""
