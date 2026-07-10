"""Tier 1 traffic-light state — HSV colour threshold, no ML.

YOLO already gives us the "traffic light" box (COCO class 9). We crop it,
convert to HSV, and count bright, saturated pixels falling in the red / amber /
green hue ranges — the lit lamp dominates. Whichever colour wins is the state.
Costs ~nothing; it's a threshold on a box we already have.

A short per-light persistence + cooldown (TrafficLightMonitor) avoids
announcing every flickery frame and re-announcing an unchanged state.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

TRAFFIC_LIGHT_ID = 9  # COCO class id for "traffic light"

# HSV bands (OpenCV hue is 0-179). Only bright + saturated pixels count as "lit".
_RED_1 = ((0, 90, 120), (10, 255, 255))
_RED_2 = ((160, 90, 120), (179, 255, 255))
_YELLOW = ((15, 90, 120), (35, 255, 255))
_GREEN = ((40, 60, 110), (90, 255, 255))


def _count(hsv, lo, hi) -> int:
    mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return int(cv2.countNonZero(mask))


def classify_light(frame, box, min_lit_frac: float = 0.02) -> str:
    """Return 'red' | 'yellow' | 'green' | 'unknown' for a traffic-light box."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    counts = {
        "red": _count(hsv, *_RED_1) + _count(hsv, *_RED_2),
        "yellow": _count(hsv, *_YELLOW),
        "green": _count(hsv, *_GREEN),
    }
    lit = sum(counts.values())
    area = roi.shape[0] * roi.shape[1]
    if area == 0 or lit < min_lit_frac * area:
        return "unknown"
    return max(counts, key=counts.get)


@dataclass
class _LightState:
    recent: deque
    announced: Optional[str] = None
    last_t: float = float("-inf")


class TrafficLightMonitor:
    def __init__(self, persist: int = 3, window: int = 5,
                 cooldown: float = 6.0) -> None:
        """
        persist  -- a state must appear this many times in the recent window
                    before it's announced (debounces flicker)
        window   -- size of the recent-states ring buffer
        cooldown -- re-announce the same unchanged state at most this often
        """
        self.persist = persist
        self.window = window
        self.cooldown = cooldown
        self._states: dict[int, _LightState] = {}

    def update(self, key: int, state: str, now: float) -> Optional[str]:
        """Feed one observation for a light (keyed by track id).

        Returns the state string to announce now, or None. Announces on a
        stable state change, and re-announces an unchanged state only after
        the cooldown.
        """
        st = self._states.get(key)
        if st is None:
            st = _LightState(recent=deque(maxlen=self.window))
            self._states[key] = st
        st.recent.append(state)

        if state == "unknown" or st.recent.count(state) < self.persist:
            return None
        if state != st.announced or (now - st.last_t) >= self.cooldown:
            st.announced = state
            st.last_t = now
            return state
        return None
