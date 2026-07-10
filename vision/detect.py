"""Tier 1 vision — YOLO11n detection -> directional obstacle events.

Pipeline: read a frame -> YOLO11n (ONNX) inference -> for each relevant box
compute its horizontal zone (left / ahead / right) -> print to console and
publish an Event onto the shared bus. Proximity is approximated by bbox area
(bigger box == closer), so the single most-prominent obstacle is the one we
speak about, debounced so we don't flood the TTS queue.

Standalone (run from the repo root):
    python -m vision.detect                       # default webcam (source 0)
    python -m vision.detect --source 1            # a different camera index
    python -m vision.detect --source clip.mp4     # a video file
    python -m vision.detect --source frame.jpg    # a single image
    python -m vision.detect --no-bus              # print only, don't publish

For integration, main.run() can start `vision_loop(...)` in a thread.
Tracking (ByteTrack) + collision logic land next on dev1/tracking.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO

from shared.bus import event_bus
from shared.events import Event, Priority

# --- Tier 1 config (kept local; a global settings module is Dev 3's job) ---
MODEL_PT = "yolo11n.pt"
MODEL_ONNX = "yolo11n.onnx"
CONF = 0.35          # detection confidence floor
IMG_SIZE = 640       # inference / export resolution
DEBOUNCE_SECONDS = 2.0  # min gap between repeated alerts for the same thing

# COCO class ids worth alerting on for street navigation. Set to None for all.
# 0 person  1 bicycle  2 car  3 motorcycle  5 bus  6 train  7 truck
# 9 traffic light  10 fire hydrant  11 stop sign  12 parking meter  13 bench
# 15 cat  16 dog
RELEVANT_CLASS_IDS: Optional[set[int]] = {0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_onnx_model(pt: str = MODEL_PT, onnx: str = MODEL_ONNX,
                      imgsz: int = IMG_SIZE) -> str:
    """Return a path to the ONNX model, exporting it from the .pt once.

    The first call downloads yolo11n.pt (if absent) and writes yolo11n.onnx
    next to it. Both are gitignored (weights are per-machine).
    """
    if not Path(onnx).exists():
        print(f"[vision] exporting {pt} -> {onnx} (one-time)...")
        YOLO(pt).export(format="onnx", imgsz=imgsz)
    return onnx


def zone_for(center_x: float, width: int) -> str:
    """Map a bbox center-x to a horizontal corridor: left / ahead / right."""
    if center_x < width / 3:
        return "left"
    if center_x > 2 * width / 3:
        return "right"
    return "ahead"


def phrase_for(name: str, zone: str) -> str:
    """Human/TTS phrase for an obstacle in a given zone."""
    if zone == "ahead":
        return f"{name} ahead"
    return f"{name} on your {zone}"


class Debounce:
    """Suppress repeat alerts: allow only when the key changes or cools down."""

    def __init__(self, cooldown: float = DEBOUNCE_SECONDS) -> None:
        self.cooldown = cooldown
        self._last_key: Optional[tuple] = None
        self._last_t = 0.0

    def allow(self, *key) -> bool:
        now = time.time()
        if key != self._last_key or (now - self._last_t) >= self.cooldown:
            self._last_key = key
            self._last_t = now
            return True
        return False


def detections_from(results, width: int,
                    class_ids: Optional[set[int]] = RELEVANT_CLASS_IDS) -> list[dict]:
    """Extract detections from an Ultralytics result, with zone/area.

    class_ids filters to those COCO ids; pass None to keep all 80 classes.
    """
    dets: list[dict] = []
    names = results.names
    for box in results.boxes:
        cls = int(box.cls[0])
        if class_ids is not None and cls not in class_ids:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        cx = (x1 + x2) / 2.0
        dets.append({
            "name": names[cls],
            "cls": cls,
            "conf": float(box.conf[0]),
            "cx": cx,
            "area": (x2 - x1) * (y2 - y1),
            "zone": zone_for(cx, width),
            "box": (x1, y1, x2, y2),
        })
    return dets


def pick_nearest(dets: list[dict]) -> Optional[dict]:
    """The obstacle we speak about: closest ~= largest bbox area."""
    return max(dets, key=lambda d: d["area"]) if dets else None


def draw_overlay(frame, dets: list[dict]):
    """Draw corridor dividers, boxes+labels, and the announced phrase.

    The nearest (announced) obstacle is drawn in red; the rest in green.
    """
    h, w = frame.shape[:2]
    # left | ahead | right corridor boundaries
    for x in (w // 3, 2 * w // 3):
        cv2.line(frame, (x, 0), (x, h), (80, 80, 80), 1)

    nearest = pick_nearest(dets)
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        is_near = d is nearest
        color = (0, 0, 255) if is_near else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if is_near else 1)
        label = f"{d['name']} {d['conf']:.2f} {d['zone']}"
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if nearest:
        cv2.putText(frame, phrase_for(nearest["name"], nearest["zone"]),
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2, cv2.LINE_AA)
    return frame


def process_frame(frame, model: YOLO, *, publish: bool = True,
                  debounce: Optional[Debounce] = None,
                  class_ids: Optional[set[int]] = RELEVANT_CLASS_IDS) -> list[dict]:
    """Run detection on one frame: print boxes, publish the closest obstacle."""
    width = frame.shape[1]
    results = model(frame, conf=CONF, imgsz=IMG_SIZE, verbose=False)[0]
    dets = detections_from(results, width, class_ids)

    for d in dets:
        print(f"  {d['name']:13} conf={d['conf']:.2f} "
              f"zone={d['zone']:5} area={d['area']:.0f}")
    if not dets:
        print("  (no relevant objects)")
        return dets

    # Closest ~= largest bbox. Speak that one.
    nearest = pick_nearest(dets)
    if publish and (debounce is None or debounce.allow(nearest["zone"], nearest["name"])):
        event_bus.publish(Event(
            message=phrase_for(nearest["name"], nearest["zone"]),
            priority=Priority.NORMAL,
            type="obstacle",
            source="vision",
            data={"class": nearest["name"], "zone": nearest["zone"],
                  "conf": nearest["conf"], "area": nearest["area"]},
        ))
    return dets


WINDOW = "Aakha — Tier 1 vision (press q to quit)"


def _show(frame) -> bool:
    """Display a frame. Returns False if the window couldn't open or the user
    pressed q/ESC, True to keep going. Never raises (headless-safe)."""
    try:
        cv2.imshow(WINDOW, frame)
    except cv2.error as exc:
        print(f"[vision] display unavailable ({exc}); continuing headless")
        return False
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), 27)  # 27 == ESC


def vision_loop(source=0, *, publish: bool = True, show: bool = False,
                class_ids: Optional[set[int]] = RELEVANT_CLASS_IDS,
                stop_event=None) -> None:
    """Continuous capture loop for live sources (webcam / video).

    main.run() can start this in a daemon thread (use show=False there — GUI
    windows must live on the main thread on macOS). Stops when the source ends,
    stop_event is set, or the user presses q/ESC in the window.
    """
    model = YOLO(ensure_onnx_model())
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[vision] ERROR: could not open source {source!r} "
              f"(camera permission? wrong index?)")
        return
    debounce = Debounce()
    frame_no = 0
    try:
        while stop_event is None or not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            print(f"[frame {frame_no}]")
            dets = process_frame(frame, model, publish=publish,
                                 debounce=debounce, class_ids=class_ids)
            if show:
                draw_overlay(frame, dets)
                if not _show(frame):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_on_image(path: str, *, publish: bool = True, show: bool = False,
                 class_ids: Optional[set[int]] = RELEVANT_CLASS_IDS) -> None:
    """Single-shot detection on one image (for standalone testing)."""
    model = YOLO(ensure_onnx_model())
    frame = cv2.imread(path)
    if frame is None:
        print(f"[vision] ERROR: could not read image {path!r}")
        return
    print(f"[image {path}]")
    dets = process_frame(frame, model, publish=publish, debounce=None,
                         class_ids=class_ids)
    if show:
        draw_overlay(frame, dets)
        try:
            cv2.imshow(WINDOW, frame)
            print("[vision] press any key in the window to close")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as exc:
            print(f"[vision] display unavailable ({exc})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tier 1 YOLO11n detection")
    ap.add_argument("--source", default="0",
                    help="webcam index (0), video file, or image path")
    ap.add_argument("--no-bus", action="store_true",
                    help="print detections only, do not publish events")
    ap.add_argument("--no-show", action="store_true",
                    help="disable the preview window (console output only)")
    ap.add_argument("--all-classes", action="store_true",
                    help="detect all 80 COCO classes (e.g. bottle, cell phone) "
                         "instead of only the navigation-obstacle subset; "
                         "useful for indoor testing")
    args = ap.parse_args()
    publish = not args.no_bus
    show = not args.no_show
    class_ids = None if args.all_classes else RELEVANT_CLASS_IDS

    src = args.source
    if src.isdigit():
        vision_loop(int(src), publish=publish, show=show, class_ids=class_ids)
    elif Path(src).suffix.lower() in IMAGE_SUFFIXES:
        run_on_image(src, publish=publish, show=show, class_ids=class_ids)
    else:
        vision_loop(src, publish=publish, show=show, class_ids=class_ids)

    # Standalone: show what actually landed on the bus.
    print(f"[vision] events on bus after run: {event_bus.qsize()}")


if __name__ == "__main__":
    main()
