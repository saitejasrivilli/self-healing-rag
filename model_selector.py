"""
llm/model_selector.py
======================
Dynamically selects the optimal LLM for each task based on:
  - Task type (generation, verification, planning, embedding)
  - Available API keys
  - Cost / latency trade-offs
  - Fallback chain on failure

Config is driven by config/model_registry.yaml.
"""

from __future__ import annotations
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    GENERATION   = "generation"
    VERIFICATION = "verification"
    PLANNING     = "planning"
    REFLECTION   = "reflection"
    HYDE         = "hyde"
    EMBEDDING    = "embedding"
    RERANKING    = "reranking"


@dataclass
class ModelSpec:
    provider: str          # "google" | "openai" | "huggingface" | "local"
    name: str
    temperature: float = 0.2
    max_tokens: int = 512
    available: bool = True


# Default task → model mapping
_TASK_MODELS: dict[TaskType, list[ModelSpec]] = {
    TaskType.GENERATION: [
        ModelSpec("google", "gemini-1.5-flash", temperature=0.2, max_tokens=512),
        ModelSpec("huggingface", "microsoft/Phi-3-mini-4k-instruct", temperature=0.2, max_tokens=512),
    ],
    TaskType.VERIFICATION: [
        ModelSpec("google", "gemini-1.5-flash", temperature=0.0, max_tokens=200),
    ],
    TaskType.PLANNING: [
        ModelSpec("google", "gemini-1.5-flash", temperature=0.0, max_tokens=400),
    ],
    TaskType.REFLECTION: [
        ModelSpec("google", "gemini-1.5-flash", temperature=0.0, max_tokens=300),
    ],
    TaskType.HYDE: [
        ModelSpec("google", "gemini-1.5-flash", temperature=0.3, max_tokens=200),
    ],
    TaskType.EMBEDDING: [
        ModelSpec("huggingface", "BAAI/bge-base-en-v1.5"),
    ],
    TaskType.RERANKING: [
        ModelSpec("huggingface", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    ],
}


class ModelSelector:
    """
    Selects the best available model for each task type.
    Checks API key availability and marks models unavailable on failure.
    """

    def __init__(self):
        self._availability: dict[str, bool] = {}
        self._check_providers()

    def _check_providers(self) -> None:
        self._availability["google"] = bool(
            os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        self._availability["openai"] = bool(os.getenv("OPENAI_API_KEY"))
        self._availability["huggingface"] = True   # local models always available
        self._availability["local"] = True
        logger.info("ModelSelector: providers available: %s", self._availability)

    def select(self, task: TaskType) -> ModelSpec:
        """Return the first available model for the given task type."""
        candidates = _TASK_MODELS.get(task, [])
        for spec in candidates:
            if self._availability.get(spec.provider, False):
                logger.debug("ModelSelector: %s → %s/%s", task, spec.provider, spec.name)
                return spec
        # Absolute fallback: local HF model
        logger.warning("ModelSelector: no preferred model available for %s — using HF fallback", task)
        return ModelSpec("huggingface", "microsoft/Phi-3-mini-4k-instruct")

    def mark_failed(self, provider: str) -> None:
        """Called when a model returns an error — deprioritises that provider."""
        self._availability[provider] = False
        logger.warning("ModelSelector: provider '%s' marked unavailable", provider)

    def available_providers(self) -> dict[str, bool]:
        return dict(self._availability)


# Module-level singleton
selector = ModelSelector()
