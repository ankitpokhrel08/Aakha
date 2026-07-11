"""Tier 1 path guidance — clearest-path free-space + off-path drift (classic CV).

Two cheap, model-free signals that answer "where should I go / am I leaving the
path", built on things we already compute:

  clearest_path(dets, w, h)
      Split the lower frame into left / ahead / right corridors, score each by
      nearby-obstacle occupancy (YOLO boxes weighted by how close/low they are),
      and suggest the emptiest side when straight-ahead is clearly more blocked.

  path_drift(frame)
      Hough line edges in the lower frame -> dominant left/right path boundaries
      -> where the walkable path sits at your feet vs the frame centre. If it's
      off-centre with confidence, cue "ease left/right".

PathGuide bundles both with debounce + hysteresis and returns the messages to
publish. Advisory only, NORMAL priority — collision (CRITICAL) always wins.
It prefers the drift ("stay on path") cue over the clearer-side suggestion so
the two never contradict each other in the same breath.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# --- region of interest ---
ROI_TOP_FRAC = 0.5          # analyse the frame below this fraction of height

# --- clearest-path (free-space) ---
NEAR_FRAC = 0.45            # only obstacles whose box bottom is below this count
CLEAR_MARGIN = 0.35         # a side must be this much emptier (relative) than ahead
CLEAR_COOLDOWN = 3.0        # min seconds between clearer-side suggestions

# --- off-path drift (Hough boundaries) ---
CANNY_LO, CANNY_HI = 50, 150
HOUGH_THRESH = 40
MIN_LEN_FRAC = 0.20         # min line length as a fraction of frame width
MAX_GAP = 30
MIN_SLOPE = 0.36            # ~20 deg from horizontal (reject near-flat curbs)
MAX_SLOPE = 2.75           # ~70 deg (reject near-vertical poles/walls)
MIN_SUPPORT = 2            # min boundary lines needed on each side
DRIFT_THRESH = 0.18        # |offset| (fraction of half-width) before we cue
DRIFT_COOLDOWN = 3.0


@dataclass
class PathResult:
    scores: dict = field(default_factory=dict)      # corridor occupancy
    suggest: Optional[str] = None                   # "left"/"right"/None
    drift: dict = field(default_factory=dict)       # path_drift() output


# --------------------------------------------------------------------------- #
# Layer 1 — clearest-path free-space
# --------------------------------------------------------------------------- #
def clearest_path(dets: list[dict], w: int, h: int, corridor=None) -> dict:
    """Score left/ahead/right by nearby-obstacle occupancy and suggest the
    emptiest side only when straight-ahead is clearly more blocked.

    With a corridor, ONLY obstacles whose ground contact is inside the walking
    corridor are counted, and left/ahead/right are sub-columns *within* the
    corridor — so a suggestion is always a small nudge onto walkable ground,
    never "go left" into an empty road that just happens to have no objects.
    """
    score = {"left": 0.0, "ahead": 0.0, "right": 0.0}
    near_y = h * NEAR_FRAC
    if corridor is not None:
        half = corridor.bottom_width_frac * w / 2.0
        left_b, right_b = w / 2 - half / 3.0, w / 2 + half / 3.0
    else:
        left_b, right_b = w / 3.0, 2 * w / 3.0

    for d in dets:
        _, _, _, y2 = d["box"]
        if y2 < near_y:                     # too far up the frame = too far away
            continue
        if corridor is not None and not corridor.contains(d["cx"], y2, w, h):
            continue                        # off the walkable corridor — ignore
        cx = d["cx"]
        zone = "left" if cx < left_b else "right" if cx > right_b else "ahead"
        prox = min(1.0, y2 / h)             # lower in frame == closer
        area_frac = d["area"] / float(w * h)
        score[zone] += area_frac * (0.5 + prox)

    best = min(score, key=score.get)
    ahead = score["ahead"]
    suggest = None
    if best != "ahead" and (ahead - score[best]) > CLEAR_MARGIN * max(ahead, 0.05):
        suggest = best
    return {"scores": score, "suggest": suggest}


# --------------------------------------------------------------------------- #
# Layer 2 — off-path drift from path-edge Hough lines
# --------------------------------------------------------------------------- #
def path_drift(frame) -> dict:
    """Estimate lateral offset of the walkable path at your feet vs frame centre.

    Returns dict: found, offset (-1..1, +=path is to your right), center_x (px),
    confidence, left_lines / right_lines (full-frame coords for overlay),
    roi_top. found is False (offset 0) unless both boundaries have support.
    """
    h, w = frame.shape[:2]
    roi_top = int(h * ROI_TOP_FRAC)
    roi = frame[roi_top:h, :]
    out = {"found": False, "offset": 0.0, "center_x": w // 2,
           "confidence": 0, "left_lines": [], "right_lines": [], "roi_top": roi_top}
    if roi.size == 0:
        return out
    roi_h = h - roi_top

    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, CANNY_LO, CANNY_HI)
    segs = cv2.HoughLinesP(edges, 1, math.pi / 180, HOUGH_THRESH,
                           minLineLength=int(w * MIN_LEN_FRAC), maxLineGap=MAX_GAP)
    if segs is None:
        return out

    left_xb, right_xb = [], []          # x where a boundary crosses the bottom row
    left_lines, right_lines = [], []
    for seg in segs:
        x1, y1, x2, y2 = (int(v) for v in np.ravel(seg)[:4])
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if not (MIN_SLOPE <= abs(slope) <= MAX_SLOPE):
            continue
        x_bottom = x1 + (roi_h - y1) / slope     # extrapolate to ROI bottom row
        if not (-w <= x_bottom <= 2 * w):        # sanity clamp
            continue
        full = (x1, y1 + roi_top, x2, y2 + roi_top)
        if slope < 0:                            # bottom-left -> up-right = left edge
            left_xb.append(x_bottom); left_lines.append(full)
        else:                                    # bottom-right -> up-left = right edge
            right_xb.append(x_bottom); right_lines.append(full)

    out["left_lines"], out["right_lines"] = left_lines, right_lines
    if len(left_xb) < MIN_SUPPORT or len(right_xb) < MIN_SUPPORT:
        return out                               # need both boundaries

    center_x = (float(np.median(left_xb)) + float(np.median(right_xb))) / 2.0
    out.update(found=True,
               center_x=int(center_x),
               offset=(center_x - w / 2) / (w / 2),
               confidence=min(len(left_xb), len(right_xb)))
    return out


# --------------------------------------------------------------------------- #
# Monitor — bundle both signals with debounce + hysteresis
# --------------------------------------------------------------------------- #
class PathGuide:
    def __init__(self, emit_drift: bool = False) -> None:
        # emit_drift: the "ease left/right" off-path cue. OFF by default — it's
        # noisy on wide-angle bodycams (fisheye bends the edge lines) and
        # confusing as speech. Steering belongs on a tone/haptic channel.
        self.emit_drift = emit_drift
        self._last_drift = float("-inf")
        self._last_clear = float("-inf")

    def update(self, frame, dets: list[dict], now: float, corridor=None):
        """Return (PathResult, messages). messages is a list of dicts
        {message, type, data} for the caller to publish.

        corridor, when given, constrains clearest-path to walkable ground."""
        h, w = frame.shape[:2]
        cp = clearest_path(dets, w, h, corridor)
        dr = (path_drift(frame) if self.emit_drift
              else {"found": False, "left_lines": [], "right_lines": []})
        result = PathResult(scores=cp["scores"], suggest=cp["suggest"], drift=dr)

        messages: list[dict] = []
        if self.emit_drift and dr.get("found") and abs(dr["offset"]) >= DRIFT_THRESH \
                and (now - self._last_drift) >= DRIFT_COOLDOWN:
            self._last_drift = now
            direction = "right" if dr["offset"] > 0 else "left"
            messages.append({"message": f"ease {direction}", "type": "path_drift",
                             "data": {"offset": round(dr["offset"], 2)}})
        elif cp["suggest"] and (now - self._last_clear) >= CLEAR_COOLDOWN:
            self._last_clear = now
            messages.append({"message": f"path is clearer to your {cp['suggest']}",
                             "type": "path", "data": {"suggest": cp["suggest"]}})
        return result, messages


# --------------------------------------------------------------------------- #
# Overlay
# --------------------------------------------------------------------------- #
def annotate_path(frame, result: PathResult) -> None:
    """Draw path boundaries, the estimated path centre, and any cue."""
    h, w = frame.shape[:2]
    dr = result.drift or {}
    for (x1, y1, x2, y2) in dr.get("left_lines", []):
        cv2.line(frame, (x1, y1), (x2, y2), (255, 120, 0), 2)     # left edge (blue)
    for (x1, y1, x2, y2) in dr.get("right_lines", []):
        cv2.line(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)     # right edge (orange)

    if dr.get("found"):
        cx = int(dr["center_x"])
        cv2.line(frame, (w // 2, h), (w // 2, h - 40), (150, 150, 150), 2)  # you
        cv2.line(frame, (cx, h), (cx, h - 60), (0, 255, 0), 3)             # path ctr
        direction = "right" if dr["offset"] > 0 else "left"
        if abs(dr["offset"]) >= DRIFT_THRESH:
            cv2.putText(frame, f"ease {direction}", (w // 2 - 70, h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    if result.suggest:
        cv2.putText(frame, f"clearer: {result.suggest}", (10, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
