"""Tier 1 collision / approach warnings from bbox growth.

Monocular "looming" heuristic: an object whose bounding box is both (a) large
in the frame — i.e. close — and (b) growing quickly — i.e. approaching — is a
collision risk. We track each ByteTrack id's smoothed bbox area over time and
fire when the fractional growth-per-second crosses a threshold while the object
already occupies a meaningful share of the frame.

No ML here — pure arithmetic on the areas the tracker already gives us, which
is why it costs ~nothing on top of detection (per the Tier 1 budget).

Feed it observations via update(); it returns a growth-rate when a warning
should fire (else None). The caller turns that into a CRITICAL Event.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class _Track:
    ema_area: float   # smoothed bbox area
    last_t: float     # timestamp of last observation
    # -inf so the first qualifying observation always clears the cooldown,
    # regardless of the clock's origin (wall clock vs monotonic-from-zero).
    last_alert_t: float = float("-inf")


class CollisionMonitor:
    def __init__(self, growth_per_sec: float = 0.35, min_area_frac: float = 0.04,
                 cooldown: float = 2.5, alpha: float = 0.5,
                 forget_after: float = 1.5) -> None:
        """
        growth_per_sec -- fractional area growth/sec that counts as "approaching"
                          (0.35 == box area growing 35% per second)
        min_area_frac  -- object must fill at least this fraction of the frame
                          before we warn (filters distant jitter)
        cooldown       -- min seconds between repeat alerts for the same track
        alpha          -- EMA smoothing factor for area (higher == less smoothing)
        forget_after   -- seconds of silence after which a track id is treated
                          as new (id was reused / object left and returned)
        """
        self.growth_per_sec = growth_per_sec
        self.min_area_frac = min_area_frac
        self.cooldown = cooldown
        self.alpha = alpha
        self.forget_after = forget_after
        self._tracks: dict[int, _Track] = {}

    def update(self, track_id: int, area: float, frame_area: float,
               now: float) -> Optional[float]:
        """Feed one bbox observation for a track.

        Returns the growth-rate (fraction/sec) when this observation should
        trigger a collision warning, otherwise None. The first observation for
        a track always returns None (need two samples to measure growth).
        """
        st = self._tracks.get(track_id)
        if st is None or (now - st.last_t) > self.forget_after:
            # new or stale track — seed it, can't measure growth yet
            self._tracks[track_id] = _Track(ema_area=area, last_t=now)
            self._prune(now)
            return None

        dt = now - st.last_t
        if dt <= 0:
            return None

        prev = st.ema_area
        st.ema_area = self.alpha * area + (1.0 - self.alpha) * st.ema_area
        st.last_t = now

        growth = (st.ema_area - prev) / prev / dt if prev > 0 else 0.0
        occupies = st.ema_area / frame_area if frame_area > 0 else 0.0

        if growth >= self.growth_per_sec and occupies >= self.min_area_frac:
            if (now - st.last_alert_t) >= self.cooldown:
                st.last_alert_t = now
                return growth
        return None

    def _prune(self, now: float) -> None:
        """Drop tracks unseen for a while so the dict doesn't grow unbounded."""
        cutoff = self.forget_after * 4
        stale = [tid for tid, s in self._tracks.items()
                 if (now - s.last_t) > cutoff]
        for tid in stale:
            del self._tracks[tid]
