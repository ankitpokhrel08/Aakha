"""End-to-end validation video: object + audio + wall detection on real footage.

Runs the real pipeline (YOLO, collision, guidance arbiter, depth wall metric)
over a clip and renders a side-by-side [annotated frame | depth heat-map] with
the spoken guidance muxed in as narration (macOS `say`), so you see the boxes and
wall verdict and hear exactly what the user would. Depth is sampled on a
video-time cadence (it's the expensive stage) and held between samples; this is a
validation artifact, not the real-time path.

Usage (from the repo root):
    .venv/bin/python tools/validate_wall.py                    # all assets/wall/*
    .venv/bin/python tools/validate_wall.py --source assets/clean_road.mp4 --open
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import render_narrated as narr  # noqa: E402  (audio synth/placement/mux helpers)
from ultralytics import YOLO  # noqa: E402

from src.vision.collision import CollisionMonitor  # noqa: E402
from src.vision.crosswalk import CrosswalkDetector  # noqa: E402
from src.vision.depth import (  # noqa: E402
    OFF_THRESHOLD, ON_THRESHOLD, TAU, DepthEstimator, near_fraction)
from src.vision.detect import (  # noqa: E402
    CONF, IMG_SIZE, _build_candidates, detections_from, draw_overlay,
    ensure_onnx_model)
from src.vision.guidance import Corridor, GuidanceArbiter  # noqa: E402
from src.vision.path import PathGuide  # noqa: E402
from src.vision.traffic_light import (  # noqa: E402
    TRAFFIC_LIGHT_ID, TrafficLightMonitor, classify_light)

from depth_overlay import PANEL_H, depth_heatmap, draw_regions, fit  # noqa: E402

# Offline-render speed knobs (affect only temporal resolution): jump frames, and
# sample the expensive depth on a video-time cadence (a wall is static).
FRAME_STRIDE = 2         # process/write every Nth frame (output plays at fps/stride)
DEPTH_PERIOD = 0.4       # min seconds of video-time between depth samples (~2.5 Hz)


def _process(source: str, video_out: str, model: YOLO, est: DepthEstimator,
             stride: int = FRAME_STRIDE, depth_period: float = DEPTH_PERIOD,
             label: str = ""):
    """Render the annotated+depth video (no audio yet); return (events, fps, dur)."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"  skip (cannot open): {source}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = fps / stride                  # we keep only every `stride`-th frame

    corridor, arbiter = Corridor(), GuidanceArbiter()
    collision = CollisionMonitor()
    crosswalk_det = CrosswalkDetector()
    light_monitor = TrafficLightMonitor()
    path_guide = PathGuide()
    score, blocked = 0.0, False
    depth = None                           # last computed depth map (held between samples)
    last_depth_t = -1e9
    last_banner, last_banner_t = "", -1e9
    last_beep_t = -1e9                      # 1 Hz on-track beep while path is clear
    events: list[tuple[float, str, str]] = []
    writer = None
    read = 0                               # frames read from the video

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if read % stride != 0:             # frame-jump: skip, but keep real time
            read += 1
            continue
        now = read / fps                       # video-time drives cadence + audio
        read += 1
        h, w = frame.shape[:2]
        frame_area = h * w

        results = model.track(frame, persist=True, conf=CONF, imgsz=IMG_SIZE,
                              verbose=False)[0]
        dets = detections_from(results, w)

        growths, alert_ids = {}, set()
        for d in dets:
            if d["id"] is None:
                continue
            g = collision.update(d["id"], d["area"], frame_area, now)
            if g is not None:
                growths[d["id"]] = g
                alert_ids.add(d["id"])

        # sample depth on a video-time cadence; hold the map + verdict between samples
        if depth is None or (now - last_depth_t) >= depth_period:
            depth = est.infer(frame)
            # rate-independent smoothing on video-time dt (same TAU as the monitor)
            dt = (now - last_depth_t) if last_depth_t > -1e8 else depth_period
            score += (1.0 - math.exp(-dt / TAU)) * (near_fraction(depth) - score)
            if not blocked and score >= ON_THRESHOLD:
                blocked = True
            elif blocked and score <= OFF_THRESHOLD:
                blocked = False
            last_depth_t = now

        light_anns = []
        for d in dets:
            if d["cls"] != TRAFFIC_LIGHT_ID:
                continue
            st = classify_light(frame, d["box"])
            d["light_state"] = st
            ann = light_monitor.update(d["id"] if d["id"] is not None else 0, st, now)
            if ann:
                light_anns.append((ann, d["id"]))

        xres, xpub = crosswalk_det.update(frame, now)
        path_info, path_msgs = path_guide.update(frame, dets, now, corridor=corridor)

        chosen = arbiter.select(
            _build_candidates(dets, corridor, w, h, growths,
                              xres if xpub else None, light_anns, path_msgs,
                              blocked=blocked),
            now)
        if chosen is not None:
            events.append((now, chosen.priority.name, chosen.message))
            last_banner, last_banner_t = chosen.message, now

        # 1 Hz on-track beep while the path is clear (mirrors the live heartbeat)
        in_ahead = any(d["cls"] != TRAFFIC_LIGHT_ID
                       and corridor.contains(d["cx"], d["box"][3], w, h)
                       for d in dets)
        if (not blocked and not in_ahead) and (now - last_beep_t) >= 1.0:
            events.append((now, "LOW", narr.BEEP_MARKER))
            last_beep_t = now

        banner = last_banner if (now - last_banner_t) < 2.0 else ""

        vis = frame.copy()
        fs = SimpleNamespace(available=True, blocked=blocked, score=score)
        draw_overlay(vis, dets, frozenset(alert_ids), crosswalk=xres,
                     path=path_info, corridor=corridor, banner=banner, freespace=fs)

        right = depth_heatmap(depth)
        draw_regions(right)
        combo = np.hstack([fit(vis, PANEL_H), fit(right, PANEL_H)])
        if label:                              # self-identify (depth rate / stride)
            cv2.rectangle(combo, (0, 0), (270, 26), (0, 0, 0), -1)
            cv2.putText(combo, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
        if writer is None:
            ch, cw = combo.shape[:2]
            writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     out_fps, (cw, ch))
        writer.write(combo)

    cap.release()
    if writer is not None:
        writer.release()
    return events, out_fps, read / fps        # real duration for audio alignment


def render(source: str, out_path: str, model: YOLO, est: DepthEstimator,
           stride: int = FRAME_STRIDE, depth_period: float = DEPTH_PERIOD,
           label: str = "") -> None:
    with tempfile.TemporaryDirectory() as tmp:
        silent = os.path.join(tmp, "video.mp4")
        narration = os.path.join(tmp, "narration.wav")
        res = _process(source, silent, model, est, stride, depth_period, label)
        if res is None:
            return
        events, fps, dur = res
        placed = narr._place_clips(events, tmp)
        narr._build_track(placed, dur, narration)
        narr._mux(silent, narration, out_path)
    spoken = ", ".join(sorted({m for _, _, m in events})) or "(nothing spoken)"
    bt = next((t for t, _, m in events if m == "path blocked"), None)
    when = f"   first 'path blocked' @ {bt:.2f}s" if bt is not None else ""
    print(f"  -> {out_path}\n     spoke: {spoken}{when}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=None,
                    help="a single clip (default: all assets/wall/*.mp4)")
    ap.add_argument("--out", default="wall_validated",
                    help="output folder at the repo root (default: wall_validated)")
    ap.add_argument("--stride", type=int, default=FRAME_STRIDE,
                    help=f"process/write every Nth frame (default {FRAME_STRIDE}; "
                         f"higher = faster render, choppier video)")
    ap.add_argument("--depth-hz", type=float, default=1.0 / DEPTH_PERIOD,
                    help=f"depth-sampling rate in video-time (default "
                         f"{1.0 / DEPTH_PERIOD:g} Hz; lower = laggier wall verdict)")
    ap.add_argument("--suffix", default="_validated",
                    help="output filename suffix (default _validated)")
    ap.add_argument("--open", action="store_true", help="open the folder when done")
    args = ap.parse_args()

    narr._require("say")
    narr._require("ffmpeg")
    sources = [args.source] if args.source else sorted(glob.glob("assets/wall/*.mp4"))
    if not sources:
        sys.exit("no clips found (assets/wall/*.mp4 empty; pass --source)")

    os.makedirs(args.out, exist_ok=True)
    model = YOLO(ensure_onnx_model())
    est = DepthEstimator()
    depth_period = 1.0 / max(1e-3, args.depth_hz)
    label = f"depth {args.depth_hz:g}Hz  stride {args.stride}"
    for src in sources:
        name = os.path.splitext(os.path.basename(src))[0]
        print(f"[validate] {src}  ({label})")
        render(src, os.path.join(args.out, f"{name}{args.suffix}.mp4"), model, est,
               args.stride, depth_period, label)

    print(f"[validate] done -> {args.out}/")
    if args.open:
        subprocess.run(["open", args.out], check=False)


if __name__ == "__main__":
    main()
