"""Event contract shared by every producer and the single audio consumer.

Frozen: do not change the Event/Priority shape without team agreement.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class Priority(IntEnum):
    """Lower value is dequeued (and spoken) first."""

    CRITICAL = 0  # imminent danger: collision
    NORMAL = 1    # standard guidance: obstacle, path
    LOW = 2       # ambient: beep, OCR, spoken answers


@dataclass
class Event:
    """A spoken-guidance event on the bus.

    `type` is a free-form tag (e.g. "obstacle", "collision", "ocr",
    "heartbeat") so new producers can add categories without touching this file.
    """

    message: str
    priority: Priority = Priority.NORMAL
    type: str = "generic"
    source: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
