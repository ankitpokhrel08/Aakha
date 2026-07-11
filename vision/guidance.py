"""Tier 1 guidance — walking corridor + single-slot alert arbiter.

Turns a flood of correct detections into a calm, in-path, prioritized voice,
following the electronic-travel-aid consensus:

  * Corridor: a trapezoid anchored at the bottom-centre of the frame (where the
    user's next steps land). An object only matters if its GROUND CONTACT point
    (bottom-centre of its box) falls inside the corridor — so a bus across the
    road, whose feet are on the tarmac, is silently ignored.

  * GuidanceArbiter: one speech slot. Every producer proposes Candidate alerts;
    the arbiter emits at most one, ranked by priority then urgency (time-to-
    collision / proximity), with a global rate limit + per-key repeat
    suppression. CRITICAL (looming collision) preempts and has its own cooldown.

This module is intentionally free of any bus/detection imports so it stays
trivially testable; the caller builds Candidates and publishes the winner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from shared.events import Event, Priority

# Per-class danger weight — ranks which in-corridor obstacle to name first.
DANGER = {
    "person": 1.0, "bicycle": 1.2, "motorcycle": 1.4,
    "car": 1.5, "bus": 1.5, "truck": 1.5,
}
DANGER_DEFAULT = 0.8

# Classes we name in speech. Anything detected but NOT here is spoken as a
# generic "object" — we don't guess uncertain labels (product decision).
SPEAK_BY_NAME = {"person", "bicycle", "motorcycle", "car", "bus", "truck",
                 "dog", "cat"}


def display_name(name: str) -> str:
    """Spoken label: the class name if it's one we name, else 'object'."""
    return name if name in SPEAK_BY_NAME else "object"


@dataclass
class Corridor:
    """Trapezoid over the walkable region ahead (fractions of frame size)."""

    bottom_width_frac: float = 0.72   # half-open width at the very bottom
    top_width_frac: float = 0.20      # width where the corridor starts (far)
    top_frac: float = 0.45            # corridor begins at this fraction of height

    def contains(self, px: float, py: float, w: int, h: int) -> bool:
        """Is ground point (px, py) inside the corridor? py is the bbox bottom."""
        top_y = h * self.top_frac
        if py < top_y:
            return False                      # too far up the frame == too far away
        t = (py - top_y) / max(1e-6, (h - top_y))     # 0 at far end, 1 at feet
        half = (self.top_width_frac + t * (self.bottom_width_frac
                                           - self.top_width_frac)) * w / 2.0
        return abs(px - w / 2.0) <= half

    def polygon(self, w: int, h: int) -> list[tuple[int, int]]:
        top_y = int(h * self.top_frac)
        th = int(self.top_width_frac * w / 2)
        bh = int(self.bottom_width_frac * w / 2)
        cx = w // 2
        return [(cx - th, top_y), (cx + th, top_y), (cx + bh, h), (cx - bh, h)]


@dataclass
class Candidate:
    priority: Priority
    urgency: float           # tie-breaker within a priority (higher = sooner)
    message: str
    type: str
    key: str                 # repeat-suppression key, e.g. "obstacle:left:person"
    cooldown: float          # min seconds before this key may repeat
    data: dict = field(default_factory=dict)

    def to_event(self) -> Event:
        return Event(message=self.message, priority=self.priority,
                     type=self.type, source="vision", data=self.data)


class GuidanceArbiter:
    """One speech slot with a steady cadence.

    - CRITICAL fires immediately, throttled only per-track (critical_gap) so a
      looming hazard is never delayed by the timer.
    - NORMAL/LOW: exactly one cue every `min_gap` seconds — the single most
      relevant thing right now (nearest obstacle, or a "path is clear" filler
      the caller supplies). This guarantees the user hears *something* on a
      steady beat instead of unpredictable silence.
    """

    def __init__(self, min_gap: float = 2.0, critical_gap: float = 1.5) -> None:
        self.min_gap = min_gap
        self.critical_gap = critical_gap
        self._last_emit = float("-inf")          # last emit of anything
        self._last_crit: dict[str, float] = {}   # per-track critical throttle

    def select(self, candidates: list[Candidate], now: float) -> Optional[Candidate]:
        # 1) CRITICAL — no cadence gate; only a short per-track throttle.
        crits = [c for c in candidates if int(c.priority) == int(Priority.CRITICAL)]
        if crits:
            c = max(crits, key=lambda c: c.urgency)
            if now - self._last_crit.get(c.key, float("-inf")) >= self.critical_gap:
                self._last_crit[c.key] = now
                self._last_emit = now
                return c
        # 2) NORMAL/LOW — one cue per min_gap; pick highest priority then urgency.
        if now - self._last_emit >= self.min_gap:
            others = [c for c in candidates if int(c.priority) != int(Priority.CRITICAL)]
            if others:
                c = min(others, key=lambda c: (int(c.priority), -c.urgency))
                self._last_emit = now
                return c
        return None
