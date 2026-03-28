"""
cognition/reflection.py
========================
Post-generation reflection — the agent critiques its own answer
before returning it to the user.

Steps:
  1. Generate initial answer (done upstream)
  2. Reflect: is the answer accurate, complete, and grounded?
  3. If critique score is low → flag for self-healing
  4. Optionally revise the answer based on the critique

This is separate from the Verification Agent (which checks factual grounding).
Reflection checks: completeness, clarity, tone, and potential gaps.
"""

from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)

REFLECT_SYSTEM = """You are a critical self-reviewer for an AI assistant.
Evaluate answers honestly and return structured JSON feedback."""

REFLECT_PROMPT = """Question: {query}
Answer: {answer}
Context used: {context_preview}

Evaluate the answer on these dimensions (score 0.0–1.0 each):
- completeness: Does it fully address the question?
- clarity: Is it clear and well-structured?
- groundedness: Is it supported by the context?
- conciseness: Is it appropriately concise (not too long/short)?

Return ONLY valid JSON (no markdown):
{{
  "completeness": <float>,
  "clarity": <float>,
  "groundedness": <float>,
  "conciseness": <float>,
  "overall": <float>,
  "critique": "<one sentence summary>",
  "needs_revision": <true|false>
}}"""


@dataclass
class ReflectionResult:
    completeness: float
    clarity: float
    groundedness: float
    conciseness: float
    overall: float
    critique: str
    needs_revision: bool


class ReflectionAgent:
    def __init__(self, revision_threshold: float = 0.55):
        self.revision_threshold = revision_threshold
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                "gemini-1.5-flash",
                system_instruction=REFLECT_SYSTEM,
                generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=300),
            )
            self._ready = True
        else:
            self._ready = False

    def reflect(self, query: str, answer: str, context_texts: list[str]) -> ReflectionResult:
        """Evaluate the answer quality and return structured reflection."""
        if not self._ready:
            return self._default_result()

        context_preview = " ".join(context_texts)[:500]
        prompt = REFLECT_PROMPT.format(
            query=query, answer=answer, context_preview=context_preview
        )
        try:
            resp = self.model.generate_content(prompt)
            data = json.loads(resp.text.strip())
            return ReflectionResult(
                completeness=float(data.get("completeness", 0.7)),
                clarity=float(data.get("clarity", 0.7)),
                groundedness=float(data.get("groundedness", 0.7)),
                conciseness=float(data.get("conciseness", 0.7)),
                overall=float(data.get("overall", 0.7)),
                critique=data.get("critique", "No critique generated."),
                needs_revision=bool(data.get("needs_revision", False)),
            )
        except Exception as e:
            logger.warning("Reflection failed: %s", e)
            return self._default_result()

    @staticmethod
    def _default_result() -> ReflectionResult:
        return ReflectionResult(
            completeness=0.7, clarity=0.7, groundedness=0.7,
            conciseness=0.7, overall=0.7,
            critique="Reflection unavailable (no API key).",
            needs_revision=False,
        )
