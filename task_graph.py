"""
orchestration/task_graph.py
============================
Represents the RAG pipeline as a DAG of tasks.
Each node is an async step; edges encode dependencies.
Enables parallel execution of independent steps (e.g. BM25 + dense retrieval).

Graph for Self-Healing RAG:
  hyde_expand ──► bm25_retrieve ──┐
                                  ├──► rrf_fuse ──► rerank ──► generate ──► verify
  hyde_expand ──► dense_retrieve ─┘

If verify FAILS → query_expand ──► [restart from bm25/dense]
"""

from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    SKIPPED   = "skipped"


@dataclass
class Task:
    name: str
    fn: Callable[..., Coroutine]
    depends_on: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    latency_ms: float = 0.0


class TaskGraph:
    """
    Simple async DAG executor.
    Tasks with no unresolved dependencies run concurrently.
    """

    def __init__(self):
        self.tasks: dict[str, Task] = {}

    def add(self, name: str, fn: Callable, depends_on: list[str] | None = None) -> "TaskGraph":
        self.tasks[name] = Task(name=name, fn=fn, depends_on=depends_on or [])
        return self

    async def run(self, context: dict) -> dict[str, Any]:
        """
        Execute all tasks respecting dependency order.
        Returns a dict of {task_name: result}.
        """
        results: dict[str, Any] = {}
        completed: set[str] = set()
        failed: set[str] = set()

        remaining = set(self.tasks.keys())

        while remaining:
            # Find tasks ready to run (all deps completed)
            ready = [
                name for name in remaining
                if all(dep in completed for dep in self.tasks[name].depends_on)
                and not any(dep in failed for dep in self.tasks[name].depends_on)
            ]

            if not ready:
                # Deadlock or all remaining tasks have failed deps
                for name in remaining:
                    if any(dep in failed for dep in self.tasks[name].depends_on):
                        self.tasks[name].status = TaskStatus.SKIPPED
                        failed.add(name)
                        remaining.discard(name)
                if not remaining:
                    break
                raise RuntimeError(f"Task graph deadlock — stuck on: {remaining}")

            # Run ready tasks concurrently
            async def run_task(name: str):
                task = self.tasks[name]
                task.status = TaskStatus.RUNNING
                t0 = time.time()
                try:
                    task.result = await task.fn(context, results)
                    task.status = TaskStatus.DONE
                    task.latency_ms = (time.time() - t0) * 1000
                    logger.debug("Task '%s' done in %.0fms", name, task.latency_ms)
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = e
                    task.latency_ms = (time.time() - t0) * 1000
                    logger.error("Task '%s' failed: %s", name, e)

            await asyncio.gather(*[run_task(name) for name in ready])

            for name in ready:
                remaining.discard(name)
                if self.tasks[name].status == TaskStatus.DONE:
                    completed.add(name)
                    results[name] = self.tasks[name].result
                else:
                    failed.add(name)

        return results

    def summary(self) -> dict:
        return {
            name: {"status": t.status, "latency_ms": t.latency_ms}
            for name, t in self.tasks.items()
        }
