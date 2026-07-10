"""Tier 1 vision — YOLO11n detection, tracking, collision, crosswalk, lights.

Pipeline per frame: read frame -> YOLO11n (ONNX) detection (+ ByteTrack when
tracking) -> for each relevant box compute its horizontal zone
(left / ahead / right) -> publish Events onto the shared bus:

  * obstacle      (NORMAL)   nearest object + direction
  * collision     (CRITICAL) a tracked object looming / approaching fast
  * crosswalk     (NORMAL)   zebra stripes detected ahead (Canny + Hough)
  * traffic_light (NORMAL)   red / amber / green state of a traffic-light box

Proximity is approximated by bbox area (bigger == closer); alerts are debounced
so we don't flood the TTS queue.

Standalone (run from the repo root):
    python -m vision.detect                       # default webcam (source 0)
    python -m vision.detect --source clip.mp4     # a video file
    python -m vision.detect --source frame.jpg    # a single image
    python -m vision.detect --save out.mp4        # write annotated video
    python -m vision.detect --no-show             # console only
    # toggles: --no-track --no-crosswalk --no-traffic-light --all-classes --no-bus

For integration, main.run() can start `vision_loop(...)` in a thread
(use show=False there — GUI windows must live on the main thread on macOS).
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
from vision.collision import CollisionMonitor
from vision.crosswalk import CrosswalkDetector
from vision.traffic_light import (
    TRAFFIC_LIGHT_ID, TrafficLightMonitor, classify_light)

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
            "id": int(box.id[0]) if box.id is not None else None,
        })
    return dets


def pick_nearest(dets: list[dict]) -> Optional[dict]:
    """The obstacle we speak about: closest ~= largest bbox area."""
    return max(dets, key=lambda d: d["area"]) if dets else None


def print_dets(dets: list[dict]) -> None:
    if not dets:
        print("  (no relevant objects)")
        return
    for d in dets:
        tid = "" if d["id"] is None else f"#{d['id']} "
        print(f"  {tid}{d['name']:13} conf={d['conf']:.2f} "
              f"zone={d['zone']:5} area={d['area']:.0f}")


def announce_directional(dets: list[dict], *, publish: bool,
                         debounce: Optional[Debounce]) -> Optional[dict]:
    """Publish a NORMAL directional alert for the nearest obstacle."""
    nearest = pick_nearest(dets)
    if nearest is None or not publish:
        return nearest
    if debounce is None or debounce.allow(nearest["zone"], nearest["name"]):
        event_bus.publish(Event(
            message=phrase_for(nearest["name"], nearest["zone"]),
            priority=Priority.NORMAL,
            type="obstacle",
            source="vision",
            data={"class": nearest["name"], "zone": nearest["zone"],
                  "conf": nearest["conf"], "area": nearest["area"]},
        ))
    return nearest


def announce_collisions(dets: list[dict], frame_area: float,
                        monitor: CollisionMonitor, *, publish: bool,
                        now: float) -> set[int]:
    """Feed tracked dets to the collision monitor; publish CRITICAL warnings.

    Returns the set of track ids that triggered a warning this frame so the
    overlay can highlight them.
    """
    alerted: set[int] = set()
    for d in dets:
        if d["id"] is None:
            continue
        growth = monitor.update(d["id"], d["area"], frame_area, now)
        if growth is None:
            continue
        alerted.add(d["id"])
        if publish:
            event_bus.publish(Event(
                message=f"{d['name']} approaching fast",
                priority=Priority.CRITICAL,
                type="collision",
                source="vision",
                data={"class": d["name"], "zone": d["zone"], "id": d["id"],
                      "growth_per_sec": round(growth, 2), "area": d["area"]},
            ))
    return alerted


def announce_traffic_lights(frame, dets: list[dict],
                            monitor: Optional[TrafficLightMonitor], *,
                            publish: bool, now: float) -> None:
    """Classify each traffic-light box, tag it (d['light_state']) for the
    overlay, and publish NORMAL state-change events.

    monitor=None means stateless (single image): announce any known state once.
    """
    for d in dets:
        if d["cls"] != TRAFFIC_LIGHT_ID:
            continue
        state = classify_light(frame, d["box"])
        d["light_state"] = state
        if monitor is not None:
            key = d["id"] if d["id"] is not None else 0
            ann = monitor.update(key, state, now)
        else:
            ann = state if state != "unknown" else None
        if ann and publish:
            event_bus.publish(Event(
                message=f"{ann} light", priority=Priority.NORMAL,
                type="traffic_light", source="vision",
                data={"state": ann, "id": d["id"]}))


def process_frame(frame, model: YOLO, *, publish: bool = True,
                  debounce: Optional[Debounce] = None,
                  class_ids: Optional[set[int]] = RELEVANT_CLASS_IDS) -> list[dict]:
    """Stateless single-frame detection (for images): print + directional alert.

    No tracking/collision here — those need temporal state (see vision_loop).
    """
    width = frame.shape[1]
    results = model(frame, conf=CONF, imgsz=IMG_SIZE, verbose=False)[0]
    dets = detections_from(results, width, class_ids)
    print_dets(dets)
    announce_directional(dets, publish=publish, debounce=debounce)
    return dets


def draw_overlay(frame, dets: list[dict], alert_ids: frozenset = frozenset(),
                 crosswalk=None):
    """Draw corridor dividers, boxes+labels, crosswalk stripes, and phrases.

    Collision-alerted tracks are drawn thick red with an APPROACHING tag; the
    nearest obstacle red; everything else green. Crosswalk stripe edges are
    yellow, with a banner when a crossing is detected. A traffic-light box is
    coloured by its lamp state.
    """
    h, w = frame.shape[:2]
    # left | ahead | right corridor boundaries
    for x in (w // 3, 2 * w // 3):
        cv2.line(frame, (x, 0), (x, h), (80, 80, 80), 1)

    # crosswalk stripe edges (drawn under the boxes)
    if crosswalk is not None:
        for (x1, y1, x2, y2) in crosswalk.lines:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if crosswalk.found:
            cv2.putText(frame, "crosswalk ahead", (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
                        cv2.LINE_AA)

    light_colors = {"red": (0, 0, 255), "yellow": (0, 255, 255),
                    "green": (0, 200, 0)}
    nearest = pick_nearest(dets)
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        alerting = d["id"] is not None and d["id"] in alert_ids
        is_near = d is nearest
        color = (0, 0, 255) if (alerting or is_near) else (0, 200, 0)
        thickness = 3 if alerting else (2 if is_near else 1)
        light = d.get("light_state")
        if light in light_colors:
            color = light_colors[light]  # colour the box by lamp state
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        tid = "" if d["id"] is None else f"#{d['id']} "
        tag = " APPROACHING" if alerting else ""
        ltag = f" [{light}]" if light in light_colors else ""
        label = f"{tid}{d['name']} {d['conf']:.2f} {d['zone']}{tag}{ltag}"
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if nearest:
        cv2.putText(frame, phrase_for(nearest["name"], nearest["zone"]),
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2, cv2.LINE_AA)
    return frame


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
                track: bool = True, save: Optional[str] = None,
                crosswalk: bool = True, traffic_light: bool = True,
                class_ids: Optional[set[int]] = RELEVANT_CLASS_IDS,
                stop_event=None) -> None:
    """Continuous capture loop for live sources (webcam / video).

    With track=True (default) each frame runs ByteTrack (model.track,
    persist=True) so objects keep stable ids, and a CollisionMonitor turns
    bbox growth into CRITICAL "approaching fast" warnings. Directional (NORMAL)
    alerts fire in both modes. Crosswalk + traffic-light detection run when
    enabled.

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
    collision = CollisionMonitor() if track else None
    crosswalk_det = CrosswalkDetector() if crosswalk else None
    light_monitor = TrafficLightMonitor() if traffic_light else None
    writer = None
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_no = 0
    try:
        while stop_event is None or not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            now = time.time()
            width = frame.shape[1]
            frame_area = frame.shape[0] * frame.shape[1]
            if track:
                results = model.track(frame, persist=True, conf=CONF,
                                      imgsz=IMG_SIZE, verbose=False)[0]
            else:
                results = model(frame, conf=CONF, imgsz=IMG_SIZE,
                                verbose=False)[0]
            dets = detections_from(results, width, class_ids)

            print(f"[frame {frame_no}]")
            print_dets(dets)
            announce_directional(dets, publish=publish, debounce=debounce)
            alert_ids: frozenset = frozenset()
            if collision is not None:
                alert_ids = frozenset(announce_collisions(
                    dets, frame_area, collision, publish=publish, now=now))

            if light_monitor is not None:
                announce_traffic_lights(frame, dets, light_monitor,
                                        publish=publish, now=now)

            xres = None
            if crosswalk_det is not None:
                xres, xpub = crosswalk_det.update(frame, now)
                if xpub and publish:
                    event_bus.publish(Event(
                        message="crosswalk ahead", priority=Priority.NORMAL,
                        type="crosswalk", source="vision",
                        data={"n_bands": xres.n_bands}))

            if show or save is not None:
                draw_overlay(frame, dets, alert_ids, crosswalk=xres)
            if save is not None:
                if writer is None:
                    fh, fw = frame.shape[:2]
                    fps = src_fps if src_fps and src_fps > 0 else 20.0
                    writer = cv2.VideoWriter(
                        save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
                writer.write(frame)
            if show and not _show(frame):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[vision] saved annotated video -> {save}")
        cv2.destroyAllWindows()


def run_on_image(path: str, *, publish: bool = True, show: bool = False,
                 crosswalk: bool = True, traffic_light: bool = True,
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
    if traffic_light:
        announce_traffic_lights(frame, dets, None, publish=publish, now=0.0)
        for d in dets:
            if d.get("light_state") and d["light_state"] != "unknown":
                print(f"[traffic light] {d['light_state']}")
    xres = None
    if crosswalk:
        xres = CrosswalkDetector().analyze(frame)
        print(f"[crosswalk] {'DETECTED' if xres.found else 'none'} "
              f"(bands={xres.n_bands}, lines={len(xres.lines)})")
        if xres.found and publish:
            event_bus.publish(Event(
                message="crosswalk ahead", priority=Priority.NORMAL,
                type="crosswalk", source="vision",
                data={"n_bands": xres.n_bands}))
    if show:
        draw_overlay(frame, dets, crosswalk=xres)
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
    ap.add_argument("--no-track", action="store_true",
                    help="disable ByteTrack + collision warnings (detection only)")
    ap.add_argument("--save", metavar="PATH", default=None,
                    help="write the annotated video to PATH (e.g. out.mp4)")
    ap.add_argument("--no-crosswalk", action="store_true",
                    help="disable crosswalk (zebra-stripe) detection")
    ap.add_argument("--no-traffic-light", action="store_true",
                    help="disable traffic-light (red/amber/green) detection")
    args = ap.parse_args()
    publish = not args.no_bus
    show = not args.no_show
    track = not args.no_track
    crosswalk = not args.no_crosswalk
    traffic_light = not args.no_traffic_light
    class_ids = None if args.all_classes else RELEVANT_CLASS_IDS

    src = args.source
    if src.isdigit():
        vision_loop(int(src), publish=publish, show=show, track=track,
                    save=args.save, crosswalk=crosswalk,
                    traffic_light=traffic_light, class_ids=class_ids)
    elif Path(src).suffix.lower() in IMAGE_SUFFIXES:
        run_on_image(src, publish=publish, show=show, crosswalk=crosswalk,
                     traffic_light=traffic_light, class_ids=class_ids)
    else:
        vision_loop(src, publish=publish, show=show, track=track,
                    save=args.save, crosswalk=crosswalk,
                    traffic_light=traffic_light, class_ids=class_ids)

    # Standalone: show what actually landed on the bus.
    print(f"[vision] events on bus after run: {event_bus.qsize()}")


if __name__ == "__main__":
    main()
