"""
tests/unit/test_llm.py
=======================
Tests for LiteLLMGateway, ModelSelector, PromptTemplates, and OutputParser.
No real API calls — all model interactions mocked.
"""

import sys
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from llm.litellm_adapter import LiteLLMGateway
from llm.model_selector import ModelSelector, TaskType, ModelSpec
from llm.prompt_templates import get, list_templates, PromptTemplate
from llm.output_parser import OutputParser


# ══════════════════════════════════════════════
# LiteLLMGateway
# ══════════════════════════════════════════════

def test_litellm_gateway_init_no_litellm(monkeypatch):
    """Should handle missing litellm gracefully."""
    import builtins
    real_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "litellm":
            raise ImportError("No module named 'litellm'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        gw = LiteLLMGateway()
        assert gw._ready is False


def test_litellm_gateway_heuristic_fallback():
    gw = LiteLLMGateway.__new__(LiteLLMGateway)
    gw._ready = False
    chunk = MagicMock()
    chunk.text = "ChromaDB is a vector database."
    result = gw._heuristic_answer("What is ChromaDB?", [chunk])
    assert "ChromaDB" in result


def test_litellm_gateway_heuristic_no_chunks():
    gw = LiteLLMGateway.__new__(LiteLLMGateway)
    gw._ready = False
    result = gw._heuristic_answer("What?", [])
    assert "No relevant context" in result


def test_litellm_gateway_available_models_with_gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test_key")
    gw = LiteLLMGateway.__new__(LiteLLMGateway)
    gw._ready = True
    models = gw.available_models()
    assert any("gemini" in m for m in models)


def test_litellm_gateway_available_models_local_always_present(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    gw = LiteLLMGateway.__new__(LiteLLMGateway)
    models = gw.available_models()
    assert any("ollama" in m for m in models)


def test_litellm_generate_mocked():
    """Mock litellm.completion and verify generate() calls it."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "LoRA is parameter efficient."

    gw = LiteLLMGateway.__new__(LiteLLMGateway)
    gw._ready = True
    gw.model = "gemini/gemini-1.5-flash"
    gw.temperature = 0.2
    gw.max_tokens = 512
    gw.fallback_models = []

    chunk = MagicMock()
    chunk.text = "LoRA reduces params."
    chunk.source = "doc.txt"

    with patch("litellm.completion", return_value=mock_response):
        result = gw.generate("What is LoRA?", [chunk])
    assert result == "LoRA is parameter efficient."


def test_litellm_fallback_on_primary_failure():
    """Should try fallback model when primary fails."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Fallback answer."
    call_count = {"n": 0}

    def mock_completion(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("Primary model failed")
        return mock_response

    gw = LiteLLMGateway.__new__(LiteLLMGateway)
    gw._ready = True
    gw.model = "gemini/gemini-1.5-flash"
    gw.temperature = 0.2
    gw.max_tokens = 512
    gw.fallback_models = ["gpt-4o-mini"]

    chunk = MagicMock()
    chunk.text = "Context."
    chunk.source = "x"

    with patch("litellm.completion", side_effect=mock_completion):
        result = gw._call_with_fallback([{"role": "user", "content": "test"}])
    assert result == "Fallback answer."
    assert call_count["n"] == 2


def test_litellm_all_models_fail():
    """Returns failure message when all models fail."""
    gw = LiteLLMGateway.__new__(LiteLLMGateway)
    gw._ready = True
    gw.model = "model_a"
    gw.fallback_models = ["model_b"]

    with patch("litellm.completion", side_effect=Exception("all down")):
        result = gw._call_with_fallback([{"role": "user", "content": "test"}])
    assert "Unable to generate" in result or "failed" in result.lower()


# ══════════════════════════════════════════════
# ModelSelector
# ══════════════════════════════════════════════

def test_model_selector_gemini_when_key_present(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test_key")
    sel = ModelSelector()
    spec = sel.select(TaskType.GENERATION)
    assert spec.provider == "google"
    assert "gemini" in spec.name


def test_model_selector_hf_fallback_no_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    sel = ModelSelector()
    spec = sel.select(TaskType.GENERATION)
    assert spec.provider == "huggingface"


def test_model_selector_embedding_always_hf():
    sel = ModelSelector()
    spec = sel.select(TaskType.EMBEDDING)
    assert spec.provider == "huggingface"
    assert "bge" in spec.name.lower()


def test_model_selector_reranker_always_hf():
    sel = ModelSelector()
    spec = sel.select(TaskType.RERANKING)
    assert spec.provider == "huggingface"


def test_model_selector_mark_failed():
    sel = ModelSelector()
    sel._availability["google"] = True
    sel.mark_failed("google")
    assert sel._availability["google"] is False


def test_model_selector_available_providers():
    sel = ModelSelector()
    providers = sel.available_providers()
    assert "google" in providers
    assert "huggingface" in providers


def test_model_selector_all_task_types():
    sel = ModelSelector()
    for task in TaskType:
        spec = sel.select(task)
        assert spec is not None
        assert spec.name


# ══════════════════════════════════════════════
# PromptTemplates
# ══════════════════════════════════════════════

def test_list_templates_returns_builtin_names():
    names = list_templates()
    assert "rag_answer" in names
    assert "verification" in names
    assert "hyde" in names
    assert "query_expansion" in names
    assert "planning" in names
    assert "reflection" in names
    assert "cot_reasoning" in names


def test_get_template_by_name():
    tmpl = get("rag_answer")
    assert isinstance(tmpl, PromptTemplate)
    assert tmpl.name == "rag_answer"


def test_get_unknown_template_raises():
    with pytest.raises(KeyError):
        get("nonexistent_template")


def test_template_format_fills_placeholders():
    tmpl = get("rag_answer")
    result = tmpl.format(context="ChromaDB is a vector DB.", history="None", query="What is ChromaDB?")
    assert "ChromaDB is a vector DB." in result["user"]
    assert "What is ChromaDB?" in result["user"]


def test_template_format_missing_variable_raises():
    tmpl = get("rag_answer")
    with pytest.raises(ValueError):
        tmpl.format(context="only context provided")   # missing history, query


def test_template_has_system_and_user():
    for name in list_templates():
        tmpl = get(name)
        assert tmpl.system
        assert tmpl.user
        assert len(tmpl.system) > 10
        assert len(tmpl.user) > 10


def test_hyde_template_format():
    tmpl = get("hyde")
    result = tmpl.format(query="What is LoRA?")
    assert "LoRA" in result["user"]


def test_verification_template_format():
    tmpl = get("verification")
    result = tmpl.format(
        context="LoRA reduces parameters.", query="What is LoRA?", answer="LoRA is efficient."
    )
    assert "LoRA" in result["user"]
    assert "JSON" in result["user"]
