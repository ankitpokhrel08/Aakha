"""Frozen event contract shared by every module (v0-contract).

Do NOT modify `Event` or `Priority` without a team-wide agreement — every
producer (vision, scene, ocr, voice) and the single audio consumer depend on
this exact shape.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class Priority(IntEnum):
    """Lower value = more urgent.

    The bus delivers lower values first, and the audio consumer is expected to
    let a CRITICAL event interrupt whatever NORMAL/LOW speech is in progress.
    """

    CRITICAL = 0  # imminent danger: collision / approaching fast
    NORMAL = 1    # standard guidance: obstacle ahead, path clearer to the left
    LOW = 2       # ambient narration: scene captions, OCR, spoken answers


@dataclass
class Event:
    """A single spoken-guidance event flowing through the bus.

    message   -- text the TTS consumer should speak
    priority  -- see Priority
    type      -- free-form category tag, e.g. "obstacle", "collision", "path",
                 "crosswalk", "traffic_light", "distance", "caption", "ocr",
                 "held_object". Kept a plain string on purpose so new producers
                 can add types without touching this frozen file.
    source    -- producing module, e.g. "vision", "scene", "ocr", "voice"
    data      -- optional structured payload (bbox, distance tier, corridor...)
    timestamp -- epoch seconds when the event was created
    """

    message: str
    priority: Priority = Priority.NORMAL
    type: str = "generic"
    source: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
