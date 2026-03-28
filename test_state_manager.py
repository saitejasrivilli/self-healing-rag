"""
tests/agent_tests/test_state_manager.py
========================================
Tests for StateManager — session lifecycle, history, context, persistence, TTL.
"""

import sys
import time
import json
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from orchestration.state_manager import StateManager


@pytest.fixture
def mgr():
    return StateManager(ttl_seconds=60)


# ── Session lifecycle ─────────────────────────────────────────────────────────
def test_create_session(mgr):
    sid = mgr.create_session("sess1")
    assert sid == "sess1"
    assert mgr.get_session("sess1") is not None


def test_get_nonexistent_session_returns_none(mgr):
    assert mgr.get_session("ghost") is None


def test_delete_session(mgr):
    mgr.create_session("to_delete")
    assert mgr.delete_session("to_delete") is True
    assert mgr.get_session("to_delete") is None


def test_delete_nonexistent_returns_false(mgr):
    assert mgr.delete_session("nope") is False


def test_session_has_required_fields(mgr):
    mgr.create_session("s")
    s = mgr.get_session("s")
    assert "history" in s
    assert "context" in s
    assert "created_at" in s
    assert "updated_at" in s


def test_session_with_initial_context(mgr):
    mgr.create_session("ctx_sess", context={"user": "sai"})
    s = mgr.get_session("ctx_sess")
    assert s["context"]["user"] == "sai"


def test_active_sessions_count(mgr):
    mgr.create_session("a")
    mgr.create_session("b")
    assert mgr.active_sessions() >= 2


# ── History management ────────────────────────────────────────────────────────
def test_append_and_get_history(mgr):
    mgr.create_session("h")
    mgr.append_message("h", "user", "Hello")
    mgr.append_message("h", "assistant", "Hi there!")
    history = mgr.get_history("h")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "Hi there!"


def test_get_history_last_n(mgr):
    mgr.create_session("long")
    for i in range(10):
        mgr.append_message("long", "user", f"message {i}")
    history = mgr.get_history("long", last_n=3)
    assert len(history) == 3
    assert "message 9" in history[-1]["content"]


def test_clear_history(mgr):
    mgr.create_session("clear_me")
    mgr.append_message("clear_me", "user", "test")
    mgr.clear_history("clear_me")
    assert mgr.get_history("clear_me") == []


def test_append_creates_session_if_missing(mgr):
    mgr.append_message("auto_created", "user", "test")
    assert mgr.get_session("auto_created") is not None


# ── Context ───────────────────────────────────────────────────────────────────
def test_set_and_get_context(mgr):
    mgr.create_session("ctx")
    mgr.set_context("ctx", "language", "python")
    assert mgr.get_context("ctx", "language") == "python"


def test_get_context_default(mgr):
    mgr.create_session("ctx2")
    assert mgr.get_context("ctx2", "missing_key", default="default_val") == "default_val"


def test_get_context_nonexistent_session(mgr):
    assert mgr.get_context("ghost", "key", default=42) == 42


# ── TTL / Expiry ──────────────────────────────────────────────────────────────
def test_expired_session_returns_none():
    mgr = StateManager(ttl_seconds=0)   # expires immediately
    mgr.create_session("exp")
    time.sleep(0.01)
    assert mgr.get_session("exp") is None


def test_purge_expired():
    mgr = StateManager(ttl_seconds=0)
    mgr.create_session("exp1")
    mgr.create_session("exp2")
    time.sleep(0.01)
    count = mgr.purge_expired()
    assert count >= 2


# ── Persistence ───────────────────────────────────────────────────────────────
def test_save_and_load_from_disk(tmp_path):
    persist = tmp_path / "state.json"
    mgr1 = StateManager(persist_path=str(persist))
    mgr1.create_session("persisted")
    mgr1.append_message("persisted", "user", "hello")
    mgr1.save_to_disk()

    mgr2 = StateManager(persist_path=str(persist))
    s = mgr2.get_session("persisted")
    assert s is not None
    assert len(s["history"]) == 1
