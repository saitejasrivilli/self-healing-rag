"""
tests/unit/test_memory.py
==========================
Tests for ShortTermMemory, LongTermMemory (mocked), EpisodicMemory,
VectorStore (mocked), and OutputParser.
"""

import sys
import json
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from memory.short_term import ShortTermMemory, MemoryEntry
from memory.episodic_memory import EpisodicMemory, Episode
from llm.output_parser import OutputParser


# ══════════════════════════════════════════════
# ShortTermMemory
# ══════════════════════════════════════════════
@pytest.fixture
def stm():
    return ShortTermMemory(max_turns=5, max_chunks=10)


def test_add_and_get_turn(stm):
    stm.add_turn("user", "What is RAG?")
    turns = stm.get_turns()
    assert len(turns) == 1
    assert turns[0].role == "user"
    assert turns[0].content == "What is RAG?"


def test_max_turns_evicts_oldest(stm):
    for i in range(7):
        stm.add_turn("user", f"message {i}")
    turns = stm.get_turns()
    assert len(turns) == 5   # max_turns=5
    assert "message 6" in turns[-1].content


def test_get_turns_last_n(stm):
    for i in range(5):
        stm.add_turn("user", f"msg {i}")
    turns = stm.get_turns(last_n=2)
    assert len(turns) == 2
    assert "msg 4" in turns[-1].content


def test_format_history(stm):
    stm.add_turn("user", "Hello")
    stm.add_turn("assistant", "Hi!")
    history = stm.format_history()
    assert "User: Hello" in history
    assert "Assistant: Hi!" in history


def test_cache_and_recall_chunks(stm):
    chunk = MagicMock()
    chunk.text = "ChromaDB is an embedding database."
    chunk.source = "doc.txt"
    stm.cache_chunks("vector database query", [chunk])
    cached = stm.get_cached_chunks("vector database query")
    assert len(cached) == 1
    assert "ChromaDB" in cached[0].content


def test_get_cached_chunks_no_match(stm):
    chunk = MagicMock()
    chunk.text = "some text"
    chunk.source = "x"
    stm.cache_chunks("query A", [chunk])
    assert stm.get_cached_chunks("query B") == []


def test_clear_resets_memory(stm):
    stm.add_turn("user", "test")
    stm.clear()
    assert stm.get_turns() == []
    assert stm.stats()["total_turns_seen"] == 0


def test_stats(stm):
    stm.add_turn("user", "msg1")
    stm.add_turn("assistant", "reply1")
    s = stm.stats()
    assert s["turns"] == 2
    assert s["total_turns_seen"] == 2


def test_turn_counter_increments(stm):
    stm.add_turn("user", "a")
    stm.add_turn("user", "b")
    turns = stm.get_turns()
    assert turns[0].turn == 1
    assert turns[1].turn == 2


def test_turn_has_metadata(stm):
    stm.add_turn("user", "hello", metadata={"session": "abc"})
    turn = stm.get_turns()[0]
    assert turn.metadata["session"] == "abc"


# ══════════════════════════════════════════════
# EpisodicMemory
# ══════════════════════════════════════════════
@pytest.fixture
def episode_mem(tmp_path):
    return EpisodicMemory(store_path=str(tmp_path / "episodes.jsonl"))


def make_episode(query="What is LoRA?", confidence=0.85, verified=True, attempts=1):
    import uuid
    return Episode(
        episode_id=uuid.uuid4().hex[:8],
        query=query,
        answer="LoRA reduces parameters by 99%.",
        confidence=confidence,
        verified=verified,
        attempts=attempts,
        latency_ms=500.0,
        sources=["ml_kb.txt"],
        query_used=query,
        reasoning="All claims supported.",
    )


def test_record_and_load(episode_mem, tmp_path):
    episode_mem.record(make_episode())
    mem2 = EpisodicMemory(store_path=str(tmp_path / "episodes.jsonl"))
    assert len(mem2._episodes) == 1


def test_recall_similar(episode_mem):
    episode_mem.record(make_episode(query="What is LoRA fine-tuning?"))
    episode_mem.record(make_episode(query="What is RLHF?"))
    results = episode_mem.recall_similar("LoRA fine-tuning methods")
    assert results[0].query == "What is LoRA fine-tuning?"


def test_recall_failures(episode_mem):
    episode_mem.record(make_episode(attempts=1))
    episode_mem.record(make_episode(attempts=3))
    failures = episode_mem.recall_failures(min_attempts=2)
    assert len(failures) == 1
    assert failures[0].attempts == 3


def test_recall_verified(episode_mem):
    episode_mem.record(make_episode(confidence=0.9, verified=True))
    episode_mem.record(make_episode(confidence=0.4, verified=False))
    verified = episode_mem.recall_verified(min_confidence=0.7)
    assert len(verified) == 1


def test_to_training_dataset(episode_mem):
    episode_mem.record(make_episode(confidence=0.9, verified=True))
    episode_mem.record(make_episode(confidence=0.3, verified=False))
    dataset = episode_mem.to_training_dataset(min_confidence=0.7)
    assert len(dataset) == 1
    assert "prompt" in dataset[0]
    assert "completion" in dataset[0]


def test_stats_empty(episode_mem):
    assert episode_mem.stats() == {"total": 0}


def test_stats_populated(episode_mem):
    episode_mem.record(make_episode(confidence=0.8, verified=True, attempts=1))
    episode_mem.record(make_episode(confidence=0.6, verified=False, attempts=2))
    s = episode_mem.stats()
    assert s["total"] == 2
    assert abs(s["avg_confidence"] - 0.7) < 0.01
    assert s["self_heal_rate"] == 0.5


# ══════════════════════════════════════════════
# OutputParser
# ══════════════════════════════════════════════
def test_parse_json_clean():
    result = OutputParser.parse_json('{"confidence": 0.9, "verified": true}')
    assert result.success is True
    assert result.data["confidence"] == 0.9


def test_parse_json_with_markdown_fence():
    text = '```json\n{"confidence": 0.8, "verified": false}\n```'
    result = OutputParser.parse_json(text)
    assert result.success is True
    assert result.data["verified"] is False


def test_parse_json_applies_defaults():
    result = OutputParser.parse_json('{}', defaults={"confidence": 0.5, "verified": False})
    assert result.data["confidence"] == 0.5


def test_parse_json_invalid_returns_defaults():
    result = OutputParser.parse_json("not json at all", defaults={"key": "value"})
    assert result.success is False
    assert result.data["key"] == "value"


def test_parse_verification_coerces_types():
    text = '{"confidence": "0.85", "verified": "true", "reasoning": "ok"}'
    data = OutputParser.parse_verification(text)
    assert isinstance(data["confidence"], float)
    assert isinstance(data["verified"], bool)
    assert isinstance(data["reasoning"], str)


def test_parse_verification_fallback_on_bad_json():
    data = OutputParser.parse_verification("bad json !!")
    assert "confidence" in data
    assert "verified" in data
    assert "reasoning" in data


def test_extract_final_answer():
    text = "Step 1: think...\nStep 2: reason...\nFinal Answer: LoRA is parameter efficient."
    answer = OutputParser.extract_final_answer(text)
    assert answer == "LoRA is parameter efficient."


def test_extract_final_answer_no_marker():
    text = "Just a plain answer."
    assert OutputParser.extract_final_answer(text) == "Just a plain answer."


def test_clean_text():
    text = "**Bold** and `code` and ## Header\n\n\n\nextra newlines"
    cleaned = OutputParser.clean_text(text)
    assert "**" not in cleaned
    assert "`" not in cleaned
    assert "\n\n\n" not in cleaned


def test_parse_plan_returns_defaults_on_bad_json():
    plan = OutputParser.parse_plan("not a plan")
    assert "steps" in plan
    assert "complexity" in plan
    assert len(plan["steps"]) >= 3
