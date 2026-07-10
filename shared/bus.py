"""Shared event bus (v0-contract).

A single priority queue. Producers call ``event_bus.publish(event)``; the one
audio consumer thread calls ``event_bus.get()`` in a loop. Do NOT modify
without a team-wide agreement.
"""
from __future__ import annotations

import itertools
import queue
from typing import Optional

from shared.events import Event


class EventBus:
    def __init__(self) -> None:
        self._q: "queue.PriorityQueue" = queue.PriorityQueue()
        # Monotonic tie-breaker: two events of equal priority must never be
        # compared to each other (Event isn't orderable). Also enforces FIFO
        # ordering within a single priority level.
        self._seq = itertools.count()

    def publish(self, event: Event) -> None:
        """Enqueue an event. Lower Priority value is dequeued first."""
        self._q.put((int(event.priority), next(self._seq), event))

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Event:
        """Pop the most-urgent event. Blocks by default (consumer loop)."""
        _, _, event = self._q.get(block=block, timeout=timeout)
        return event

    def task_done(self) -> None:
        self._q.task_done()

    def qsize(self) -> int:
        return self._q.qsize()

    def empty(self) -> bool:
        return self._q.empty()


# The single global instance every module imports.
event_bus = EventBus()
