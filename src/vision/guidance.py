"""Guidance — walking corridor + single-slot alert arbiter.

Corridor: a trapezoid at the frame bottom; an object matters only if its ground
contact (bbox bottom-centre) falls inside it, so a bus across the road is
ignored. GuidanceArbiter: one speech slot; producers propose Candidates and it
emits at most one, ranked by priority then urgency, with per-key repeat
suppression. No bus/detection imports here, so it stays trivially testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from src.core.events import Event, Priority

# Far edge of the shorter ahead-only corridor (see Corridor.ahead()).
AHEAD_TOP_FRAC = 0.63

# Per-class danger weight — ranks which in-corridor obstacle to name first.
DANGER = {
    "person": 1.0, "bicycle": 1.2, "motorcycle": 1.4,
    "car": 1.5, "bus": 1.5, "truck": 1.5,
}
DANGER_DEFAULT = 0.8

# Classes named in speech; anything else is spoken as a generic "object".
SPEAK_BY_NAME = {"person", "bicycle", "motorcycle", "car", "bus", "truck", "train",
                 "dog", "cat", "bench", "chair", "potted plant", "fire hydrant",
                 "stop sign", "parking meter", "backpack", "suitcase", "umbrella"}


def display_name(name: str) -> str:
    """Spoken label: the class name if it's one we name, else 'object'."""
    return name if name in SPEAK_BY_NAME else "object"


@dataclass
class Corridor:
    """Trapezoid over the walkable region ahead (fractions of frame size)."""

    bottom_width_frac: float = 0.72   # width at the very bottom (at your feet)
    top_width_frac: float = 0.36      # width at the far edge
    top_frac: float = 0.52            # corridor begins at this fraction of height
    # Tuned to preview only the ~2-3 m local-navigation zone: a shorter, roughly
    # constant ~1 m lane. contains(), polygon(), and path.clearest_path() all read
    # these fields, so they stay aligned.

    def half_width_at(self, py: float, w: int, h: int) -> Optional[float]:
        """Half-width (px) of the corridor at image row py, or None if py is above
        the far edge (too far up the frame == beyond the corridor depth)."""
        top_y = h * self.top_frac
        if py < top_y:
            return None
        t = (py - top_y) / max(1e-6, (h - top_y))     # 0 at far end, 1 at feet
        return (self.top_width_frac + t * (self.bottom_width_frac
                                           - self.top_width_frac)) * w / 2.0

    def contains(self, px: float, py: float, w: int, h: int) -> bool:
        """Is ground point (px, py) inside the corridor? py is the bbox bottom."""
        half = self.half_width_at(py, w, h)
        if half is None:
            return False                      # too far up the frame == too far away
        return abs(px - w / 2.0) <= half

    def ahead(self, top_frac: float = AHEAD_TOP_FRAC) -> "Corridor":
        """A shorter copy (same width taper) used only for the straight-ahead
        obstacle/collision cue, so 'X ahead' fires nearer than the side (left/
        right) cues and path guidance, which keep this full-length corridor."""
        return replace(self, top_frac=top_frac)

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

    CRITICAL fires immediately, throttled only per-track (critical_gap).
    NORMAL/LOW: at most one cue per `min_gap`, and no key more often than its own
    Candidate.cooldown. If the best candidate is still cooling down we fall
    through to the next eligible one; if none are, we stay silent.
    """

    def __init__(self, min_gap: float = 1.4, critical_gap: float = 1.5) -> None:
        self.min_gap = min_gap
        self.critical_gap = critical_gap
        self._last_emit = float("-inf")          # last emit of anything
        self._last_crit: dict[str, float] = {}   # per-track critical throttle
        self._last_key: dict[str, float] = {}    # per-key throttle (Candidate.cooldown)

    def select(self, candidates: list[Candidate], now: float) -> Optional[Candidate]:
        # 1) CRITICAL — no cadence gate; only a short per-track throttle.
        crits = [c for c in candidates if int(c.priority) == int(Priority.CRITICAL)]
        if crits:
            c = max(crits, key=lambda c: c.urgency)
            if now - self._last_crit.get(c.key, float("-inf")) >= self.critical_gap:
                self._last_crit[c.key] = now
                self._last_emit = now
                return c
        # 2) NORMAL/LOW — one cue per min_gap, and each key no more often than its
        #    own cooldown; pick highest priority then urgency among the eligible.
        if now - self._last_emit >= self.min_gap:
            others = [c for c in candidates
                      if int(c.priority) != int(Priority.CRITICAL)
                      and now - self._last_key.get(c.key, float("-inf")) >= c.cooldown]
            if others:
                c = min(others, key=lambda c: (int(c.priority), -c.urgency))
                self._last_emit = now
                self._last_key[c.key] = now
                return c
        return None
