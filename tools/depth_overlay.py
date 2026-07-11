"""Visualize what the monocular-depth free-space monitor actually sees.

For every clip in assets/wall/ (or any --source), render a side-by-side video:

    [ camera frame ]  [ depth heat-map ]

with the exact regions the B1 metric samples drawn on both panels — the central
walking corridor column, the "at your feet" NEAR band, and the "a few steps
ahead" AHEAD band — plus a live readout of the raw near-fraction, its smoothed
EMA, and the WALL / free verdict (same EMA + hysteresis as vision/depth.py).

Depth colour: red = near, blue = far (larger depth value = closer). An open path
shows a smooth near(red, bottom) -> far(blue, top) gradient; a wall paints the
AHEAD band red (as near as your feet) and trips the WALL verdict.

Outputs one <clip>_depth.mp4 per clip into a single folder at the repo root
(default: depth_overlays/).

Usage (from the repo root):
    .venv/bin/python tools/depth_overlay.py                 # all assets/wall/*.mp4
    .venv/bin/python tools/depth_overlay.py --source assets/clean_road.mp4
    .venv/bin/python tools/depth_overlay.py --out depth_overlays --open
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

import cv2
import numpy as np

# repo root on path so `vision` imports regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.depth import (  # noqa: E402
    EMA_ALPHA, OFF_THRESHOLD, ON_THRESHOLD, DepthEstimator, near_fraction)

PANEL_H = 720            # height of each side-by-side panel

# Sample regions (must match near_fraction in vision/depth.py), as fractions.
COL_X0, COL_X1 = 0.30, 0.70        # central corridor column
NEAR_Y0 = 0.70                     # at-your-feet band (rows below this)
AHEAD_Y0, AHEAD_Y1 = 0.30, 0.60    # a-few-steps-ahead band


def depth_heatmap(depth: np.ndarray) -> np.ndarray:
    """Colour a relative-depth map: red = near, blue = far."""
    lo, hi = np.percentile(depth, 2), np.percentile(depth, 98)
    norm = np.clip((depth - lo) / (hi - lo + 1e-6), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def draw_regions(img: np.ndarray) -> None:
    """Draw the corridor column + NEAR/AHEAD sample bands on a panel."""
    h, w = img.shape[:2]
    x0, x1 = int(w * COL_X0), int(w * COL_X1)
    cv2.rectangle(img, (x0, int(h * AHEAD_Y0)), (x1, int(h * AHEAD_Y1)),
                  (0, 255, 255), 2)                                   # AHEAD (cyan)
    cv2.putText(img, "AHEAD", (x0 + 4, int(h * AHEAD_Y0) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, int(h * NEAR_Y0)), (x1, h - 2),
                  (0, 255, 0), 2)                                     # NEAR (green)
    cv2.putText(img, "NEAR", (x0 + 4, int(h * NEAR_Y0) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


def fit(img: np.ndarray, height: int) -> np.ndarray:
    scale = height / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), height))


def render(source: str, out_path: str, est: DepthEstimator) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"  skip (cannot open): {source}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    score, blocked = 0.0, False
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        depth = est.infer(frame)
        nf = near_fraction(depth)
        score += EMA_ALPHA * (nf - score)                # same EMA as the monitor
        if not blocked and score >= ON_THRESHOLD:
            blocked = True
        elif blocked and score <= OFF_THRESHOLD:
            blocked = False

        left = fit(frame.copy(), PANEL_H)
        right = fit(depth_heatmap(depth), PANEL_H)
        draw_regions(left)
        draw_regions(right)
        combo = np.hstack([left, right])

        verdict = "WALL" if blocked else "free"
        col = (0, 0, 255) if blocked else (0, 200, 0)
        cv2.rectangle(combo, (0, 0), (combo.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(combo, f"near-frac={nf:.2f}  ema={score:.2f}  [{verdict}]",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        cv2.putText(combo, "red=near  blue=far",
                    (combo.shape[1] // 2 + 10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)

        if writer is None:
            h, w = combo.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (w, h))
        writer.write(combo)
    cap.release()
    if writer is not None:
        writer.release()
        print(f"  -> {out_path}  ({n} frames)")
    else:
        print(f"  no frames: {source}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=None,
                    help="a single clip (default: all assets/wall/*.mp4)")
    ap.add_argument("--out", default="depth_overlays",
                    help="output folder at the repo root (default: depth_overlays)")
    ap.add_argument("--open", action="store_true",
                    help="open the output folder when done (macOS)")
    args = ap.parse_args()

    sources = [args.source] if args.source else sorted(glob.glob("assets/wall/*.mp4"))
    if not sources:
        sys.exit("no clips found (assets/wall/*.mp4 empty; pass --source)")

    os.makedirs(args.out, exist_ok=True)
    est = DepthEstimator()
    for src in sources:
        name = os.path.splitext(os.path.basename(src))[0]
        out = os.path.join(args.out, f"{name}_depth.mp4")
        print(f"[depth-overlay] {src}")
        render(src, out, est)

    print(f"[depth-overlay] done -> {args.out}/")
    if args.open:
        subprocess.run(["open", args.out], check=False)


if __name__ == "__main__":
    main()
