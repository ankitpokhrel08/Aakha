"""Crosswalk (zebra-stripe) detection — classic CV, no ML.

A zebra crossing is a stack of bright parallel bars near the frame bottom. On a
lower-central ROI: Canny -> probabilistic Hough -> keep near-horizontal segments
-> declare a crosswalk when they span enough distinct y-bands and the ROI reads
black-and-white. Persistence + cooldown avoid flicker and TTS spam.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class CrosswalkResult:
    found: bool
    lines: list = field(default_factory=list)  # (x1,y1,x2,y2) full-frame coords
    roi_top: int = 0
    n_bands: int = 0


def _near_horizontal(x1: int, y1: int, x2: int, y2: int, max_angle: float) -> bool:
    """True if the segment is within max_angle degrees of horizontal."""
    ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))  # 0 or 180 == flat
    return ang <= max_angle or ang >= 180 - max_angle


def _looks_zebra(roi_bgr, bright_frac_min: float = 0.18,
                 std_min: float = 28.0, max_sat: float = 80.0) -> bool:
    """Black-and-white gate: bright white bands present (a real zebra has lots
    of white paint), high brightness contrast (alternating light/dark bands),
    and achromatic (low saturation). Rejects coloured or low-contrast patterns
    like door panels, tiles or faint road paint."""
    if roi_bgr.size == 0:
        return False
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2]
    sat = hsv[:, :, 1]
    bright = float((val > 170).sum()) / float(val.size)
    return (bright >= bright_frac_min
            and float(val.std()) >= std_min
            and float(sat.mean()) < max_sat)


class CrosswalkDetector:
    def __init__(self, roi_top_frac: float = 0.5, roi_side_frac: float = 0.66,
                 max_angle: float = 20.0, min_bands: int = 8,
                 n_bands_grid: int = 12, canny_lo: int = 50, canny_hi: int = 150,
                 hough_thresh: int = 60, min_len_frac: float = 0.28,
                 max_gap: int = 20, persist_frames: int = 8,
                 persist_hits: int = 6, cooldown: float = 8.0) -> None:
        """
        roi_top_frac  -- analyze the frame below this fraction of height
        max_angle     -- max degrees off horizontal for a stripe edge
        min_bands     -- distinct y-bands that must contain a stripe edge
        n_bands_grid  -- how many horizontal bands the ROI is split into
        min_len_frac  -- min line length as a fraction of frame width
        persist_frames/persist_hits -- publish only if >= hits of the last
                         frames were positive (debounce flicker)
        cooldown      -- min seconds between published crosswalk events
        """
        self.roi_top_frac = roi_top_frac
        self.roi_side_frac = roi_side_frac    # analyze only the central column
        self.max_angle = max_angle
        self.min_bands = min_bands
        self.n_bands_grid = n_bands_grid
        self.canny_lo = canny_lo
        self.canny_hi = canny_hi
        self.hough_thresh = hough_thresh
        self.min_len_frac = min_len_frac
        self.max_gap = max_gap
        self.cooldown = cooldown
        self.persist_hits = persist_hits
        self._recent: deque[bool] = deque(maxlen=persist_frames)
        self._last_event_t = float("-inf")

    def analyze(self, frame) -> CrosswalkResult:
        """Pure per-frame detection (no temporal state)."""
        h, w = frame.shape[:2]
        roi_top = int(h * self.roi_top_frac)
        # only the central column — a crosswalk you're about to step on fills the
        # lower-centre; road lane-markings off to the sides no longer count.
        x0 = int(w * (1 - self.roi_side_frac) / 2)
        x1 = w - x0
        roi = frame[roi_top:h, x0:x1]
        if roi.size == 0:
            return CrosswalkResult(found=False, roi_top=roi_top)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, self.canny_lo, self.canny_hi)
        segments = cv2.HoughLinesP(
            edges, 1, math.pi / 180, self.hough_thresh,
            minLineLength=int((x1 - x0) * self.min_len_frac), maxLineGap=self.max_gap)

        lines: list = []
        bands: set[int] = set()
        roi_h = h - roi_top
        band_h = max(1, roi_h // self.n_bands_grid)
        if segments is not None:
            for seg in segments:
                # OpenCV 4 returns (1,4) rows, OpenCV 5 returns (4,); ravel handles both
                sx1, sy1, sx2, sy2 = (int(v) for v in np.ravel(seg)[:4])
                if not _near_horizontal(sx1, sy1, sx2, sy2, self.max_angle):
                    continue
                # back to full-frame coordinates (add the central-crop x offset)
                lines.append((sx1 + x0, sy1 + roi_top, sx2 + x0, sy2 + roi_top))
                bands.add(min(sy1, sy2) // band_h)

        # parallel horizontal bands AND a black-and-white zebra appearance
        found = len(bands) >= self.min_bands and _looks_zebra(roi)
        return CrosswalkResult(found=found, lines=lines,
                               roi_top=roi_top, n_bands=len(bands))

    def update(self, frame, now: float) -> tuple[CrosswalkResult, bool]:
        """Per-frame detection + temporal gating.

        Returns (result, should_publish). should_publish is True at most once
        per cooldown, and only when the recent window is mostly positive.
        """
        res = self.analyze(frame)
        self._recent.append(res.found)
        stable = sum(self._recent) >= self.persist_hits
        should_publish = False
        if stable and (now - self._last_event_t) >= self.cooldown:
            self._last_event_t = now
            should_publish = True
        return res, should_publish
