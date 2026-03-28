"""
tests/agent_tests/test_task_graph.py
=====================================
Tests for TaskGraph — DAG execution, dependency ordering, concurrency, error handling.
"""

import sys
import asyncio
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from orchestration.task_graph import TaskGraph, TaskStatus


# ── Async helpers ─────────────────────────────────────────────────────────────
async def make_task(return_val):
    async def fn(ctx, results):
        return return_val
    return fn


async def failing_task(ctx, results):
    raise RuntimeError("Intentional failure")


async def dependent_task(ctx, results):
    """Returns sum of all upstream results."""
    return sum(v for v in results.values() if isinstance(v, (int, float)))


# ── Basic execution ───────────────────────────────────────────────────────────
def test_single_task_runs():
    graph = TaskGraph()
    graph.add("step1", lambda ctx, r: asyncio.coroutine(lambda: 42)())

    async def _run():
        graph.add("t", make_value(10))
        return await graph.run({})

    async def make_value(v):
        return v

    async def runner():
        g = TaskGraph()
        g.add("a", make_value(42))
        return await g.run({})

    results = asyncio.run(runner())
    assert results["a"] == 42


def test_sequential_dependency():
    async def step1(ctx, r): return 10
    async def step2(ctx, r): return r["step1"] * 2

    async def runner():
        g = TaskGraph()
        g.add("step1", step1)
        g.add("step2", step2, depends_on=["step1"])
        return await g.run({})

    results = asyncio.run(runner())
    assert results["step1"] == 10
    assert results["step2"] == 20


def test_parallel_independent_tasks():
    import time

    async def slow_a(ctx, r):
        await asyncio.sleep(0.05)
        return "A"

    async def slow_b(ctx, r):
        await asyncio.sleep(0.05)
        return "B"

    async def runner():
        g = TaskGraph()
        g.add("a", slow_a)
        g.add("b", slow_b)
        t0 = asyncio.get_event_loop().time()
        results = await g.run({})
        elapsed = asyncio.get_event_loop().time() - t0
        return results, elapsed

    results, elapsed = asyncio.run(runner())
    assert results["a"] == "A"
    assert results["b"] == "B"
    # Parallel: should finish in ~50ms not ~100ms
    assert elapsed < 0.15


def test_three_step_chain():
    async def s1(ctx, r): return 1
    async def s2(ctx, r): return r["s1"] + 1
    async def s3(ctx, r): return r["s2"] + 1

    async def runner():
        g = TaskGraph()
        g.add("s1", s1)
        g.add("s2", s2, depends_on=["s1"])
        g.add("s3", s3, depends_on=["s2"])
        return await g.run({})

    results = asyncio.run(runner())
    assert results == {"s1": 1, "s2": 2, "s3": 3}


def test_context_passed_to_tasks():
    async def uses_ctx(ctx, r):
        return ctx["value"] * 2

    async def runner():
        g = TaskGraph()
        g.add("t", uses_ctx)
        return await g.run({"value": 7})

    results = asyncio.run(runner())
    assert results["t"] == 14


# ── Error handling ────────────────────────────────────────────────────────────
def test_failed_task_marks_status():
    async def runner():
        g = TaskGraph()
        g.add("bad", failing_task)
        results = await g.run({})
        return g, results

    graph, results = asyncio.run(runner())
    assert graph.tasks["bad"].status == TaskStatus.FAILED
    assert "bad" not in results


def test_dependent_task_skipped_on_upstream_failure():
    async def runner():
        g = TaskGraph()
        g.add("bad", failing_task)
        g.add("dep", dependent_task, depends_on=["bad"])
        results = await g.run({})
        return g, results

    graph, results = asyncio.run(runner())
    assert graph.tasks["dep"].status == TaskStatus.SKIPPED
    assert "dep" not in results


# ── Summary ───────────────────────────────────────────────────────────────────
def test_summary_contains_all_tasks():
    async def t(ctx, r): return 1

    async def runner():
        g = TaskGraph()
        g.add("x", t)
        g.add("y", t)
        await g.run({})
        return g.summary()

    summary = asyncio.run(runner())
    assert "x" in summary
    assert "y" in summary
    assert summary["x"]["status"] == TaskStatus.DONE


def test_latency_tracked_in_summary():
    async def slow(ctx, r):
        await asyncio.sleep(0.02)
        return True

    async def runner():
        g = TaskGraph()
        g.add("slow", slow)
        await g.run({})
        return g.summary()

    summary = asyncio.run(runner())
    assert summary["slow"]["latency_ms"] >= 15
