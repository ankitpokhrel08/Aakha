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

from dataclasses import dataclass, field, replace
from typing import Optional

from shared.events import Event, Priority

# The straight-ahead "X ahead" obstacle cue uses a SHORTER corridor than the side
# (left/right) cues and path guidance: an in-path warning should only fire for
# something genuinely near, while side/path detection keep a longer reach. This is
# the far edge of that shorter ahead-only corridor (see Corridor.ahead()).
AHEAD_TOP_FRAC = 0.63

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
    top_width_frac: float = 0.36      # width where the corridor starts (far)
    top_frac: float = 0.52            # corridor begins at this fraction of height
    #  top_frac lowered from the original 0.45 to shorten the over-long corridor
    #    (far-off objects were treated as in-path; the ETA literature puts the
    #    useful local-navigation zone at ~2-3 m ahead — preview beyond that only
    #    slows walkers and adds noise, "The Cost of Knowing", ACM 2022). We first
    #    tried 0.63 (a ~40% cut) but that clipped the near field too aggressively;
    #    0.52 backs ~30% of that length off (0.37 -> 0.48 of frame height) for a
    #    corridor that's shorter than the original yet still previews enough ground.
    #  top_width_frac raised 0.20 -> 0.36: the old 0.20 taper was tuned for the
    #    longer 0.45 corridor; on the shortened one it pinched the lane to a point
    #    right where coverage is still needed. A shorter corridor spans less depth,
    #    so it needs less perspective narrowing. 0.36 keeps a roughly constant
    #    ~0.9-1.0 m walkable lane (ADA min clear width) from feet to far edge.
    #  contains(), the polygon() overlay, and path.clearest_path() all read these
    #    same fields, so they stay aligned automatically.

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

    - CRITICAL fires immediately, throttled only per-track (critical_gap) so a
      looming hazard is never delayed by the timer.
    - NORMAL/LOW: at most one cue every `min_gap` seconds AND no single key more
      often than its own Candidate.cooldown. The global gap keeps the voice calm;
      the per-key cooldown stops any one message (a persistent "path is clear"
      filler, a standing obstacle, a lingering crosswalk) from nagging on every
      beat. When the best candidate is still cooling down we fall through to the
      next eligible one, and if nothing is eligible we stay silent — silence
      itself reads as "nothing new", which is calmer than repeating "path clear".
    """

    def __init__(self, min_gap: float = 2.0, critical_gap: float = 1.5) -> None:
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
