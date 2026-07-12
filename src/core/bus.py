"""Shared event bus: one priority queue, drained by the single audio consumer.

Frozen: do not change without team agreement.
"""
from __future__ import annotations

import itertools
import queue
from typing import Optional

from src.core.events import Event


class EventBus:
    def __init__(self) -> None:
        self._q: "queue.PriorityQueue" = queue.PriorityQueue()
        # Tie-breaker so equal-priority events stay FIFO and Event objects
        # (not orderable) are never compared.
        self._seq = itertools.count()

    def publish(self, event: Event) -> None:
        self._q.put((int(event.priority), next(self._seq), event))

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Event:
        _, _, event = self._q.get(block=block, timeout=timeout)
        return event

    def task_done(self) -> None:
        self._q.task_done()

    def qsize(self) -> int:
        return self._q.qsize()

    def empty(self) -> bool:
        return self._q.empty()


event_bus = EventBus()
