"""
llm/output_parser.py
=====================
Structured output parsing for LLM responses.
Handles JSON extraction, field validation, and fallback defaults.
Used by: VerificationAgent, QueryPlanner, ReflectionAgent.
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ParseResult:
    success: bool
    data: Any
    raw: str
    error: Optional[str] = None


class OutputParser:
    """
    Robust parser for LLM text → structured Python objects.
    Handles:
      - Raw JSON
      - JSON wrapped in markdown fences (```json ... ```)
      - Partial JSON with fallback defaults
    """

    @staticmethod
    def parse_json(text: str, required_keys: list[str] | None = None,
                   defaults: dict | None = None) -> ParseResult:
        """
        Extract and parse JSON from LLM output.
        Tries multiple strategies before giving up.
        """
        raw = text.strip()

        # Strategy 1: Strip markdown fences
        cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

        # Strategy 2: Extract first {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Strategy 3: Apply defaults if provided
            if defaults:
                logger.warning("JSON parse failed (%s) — using defaults", e)
                return ParseResult(success=False, data=defaults, raw=raw,
                                   error=str(e))
            return ParseResult(success=False, data=None, raw=raw, error=str(e))

        # Merge missing keys with defaults
        if defaults:
            for key, val in defaults.items():
                data.setdefault(key, val)

        # Validate required keys
        if required_keys:
            missing = [k for k in required_keys if k not in data]
            if missing:
                err = f"Missing required keys: {missing}"
                if defaults:
                    for k in missing:
                        data[k] = defaults.get(k)
                    return ParseResult(success=False, data=data, raw=raw, error=err)
                return ParseResult(success=False, data=data, raw=raw, error=err)

        return ParseResult(success=True, data=data, raw=raw)

    @staticmethod
    def parse_verification(text: str) -> dict:
        """Parse verification agent output → {confidence, verified, reasoning}."""
        defaults = {"confidence": 0.5, "verified": False, "reasoning": "Parse error."}
        result = OutputParser.parse_json(
            text,
            required_keys=["confidence", "verified", "reasoning"],
            defaults=defaults,
        )
        data = result.data or defaults
        # Type coerce
        try:
            data["confidence"] = float(data["confidence"])
            data["verified"] = bool(data["verified"])
            data["reasoning"] = str(data["reasoning"])
        except (TypeError, ValueError):
            return defaults
        return data

    @staticmethod
    def parse_plan(text: str) -> dict:
        """Parse planner output → {complexity, strategy, steps}."""
        defaults = {
            "complexity": "simple",
            "strategy": "sequential",
            "steps": [
                {"id": 1, "action": "retrieve", "query": "", "depends_on": []},
                {"id": 2, "action": "generate", "depends_on": [1]},
                {"id": 3, "action": "verify", "depends_on": [2]},
            ],
        }
        result = OutputParser.parse_json(text, defaults=defaults)
        return result.data or defaults

    @staticmethod
    def extract_final_answer(text: str, marker: str = "Final Answer:") -> str:
        """Extract the final answer from chain-of-thought output."""
        if marker in text:
            return text.split(marker, 1)[1].strip()
        return text.strip()

    @staticmethod
    def clean_text(text: str) -> str:
        """Remove markdown formatting and excessive whitespace."""
        text = re.sub(r"[*_`#]+", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
