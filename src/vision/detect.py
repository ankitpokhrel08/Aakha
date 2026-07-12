"""Vision pipeline — YOLO11n detection, tracking, collision, crosswalk, lights.

Per frame: YOLO11n (ONNX) detection (+ ByteTrack when tracking) -> per box
compute its horizontal zone -> publish Events. Proximity is approximated by
bbox area (bigger == closer); alerts are debounced to avoid flooding TTS.

Standalone: python -m src.vision.detect [--source clip.mp4|frame.jpg]
[--save out.mp4] [--no-show] [--no-track --no-crosswalk --no-traffic-light
--all-classes --no-bus]. main.run() starts vision_loop() in a thread with
show=False (GUI windows must live on the main thread on macOS).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
from ultralytics import YOLO

from src.core.bus import event_bus
from src.core.events import Event, Priority
from src.vision.collision import CollisionMonitor
from src.vision.crosswalk import CrosswalkDetector
from src.vision.depth import FreeSpaceMonitor
from src.vision.guidance import (
    DANGER, DANGER_DEFAULT, Candidate, Corridor, GuidanceArbiter, display_name)
from src.vision.path import PathGuide, annotate_path
from src.vision.traffic_light import (
    TRAFFIC_LIGHT_ID, TrafficLightMonitor, classify_light)

MODEL_PT = "yolo11n.pt"
MODEL_ONNX = "yolo11n.onnx"
CONF = 0.35          # detection confidence floor
IMG_SIZE = 640
DEBOUNCE_SECONDS = 2.0
# Push mode: keep the ~2s beat alive across short frame stalls by re-running the
# arbiter on the last scene; past this many stale seconds, go silent.
STALE_SCENE_LIMIT = 4.0
# Objects this fraction of frame width beside the corridor get a left/right cue.
SIDE_BAND_FRAC = 0.15
ON_TRACK_BEEP_GAP = 1.0     # seconds between on-track beeps
CLEAR_FILLER_GAP = 8.0      # spoken "path is clear" cadence for non-heartbeat callers

# None = detect all 80 COCO classes; unnamed ones are announced as "object" so
# out-of-vocabulary obstacles (carts, vendors) still get flagged.
RELEVANT_CLASS_IDS: Optional[set[int]] = None
SUPPRESSED_CLASS_IDS: set[int] = {6}  # 6 = train (never relevant here)

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


def collision_phrase(name: str, zone: str) -> str:
    """CRITICAL closing-warning phrase. Trigger is bbox growth (gap closing),
    so "closing on X" stays neutral to whether the object or the user is moving."""
    dirn = "" if zone == "ahead" else f" on your {zone}"
    return f"closing on {name}{dirn}"


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
        if cls in SUPPRESSED_CLASS_IDS:          # never relevant (e.g. train)
            continue
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
                message=collision_phrase(d["name"], d["zone"]),
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


def _build_candidates(dets, corridor, w, h, growths, xres, light_anns, path_msgs,
                      blocked=False, ahead_corridor=None, announce_clear=False,
                      heartbeat=False):
    """Turn this frame's signals into Candidate alerts for the arbiter.

    Obstacles are corridor-filtered by ground contact (bbox bottom-centre).
    `blocked` is the depth free-space verdict (a wall YOLO can't see). The
    straight-ahead cue uses the shorter `ahead_corridor`; side/path cues use the
    full `corridor`. `ahead_corridor=None` falls back to `corridor`.
    """
    ahead = ahead_corridor if ahead_corridor is not None else corridor
    # traffic lights are handled by the light detector, not as obstacles
    in_corr = [d for d in dets
               if d["cls"] != TRAFFIC_LIGHT_ID
               and ahead.contains(d["cx"], d["box"][3], w, h)]
    cands: list = []
    # CRITICAL — looming obstacles that are in the corridor
    for d in in_corr:
        g = growths.get(d["id"])
        if g is not None:
            cands.append(Candidate(
                Priority.CRITICAL, 10.0 + g,
                collision_phrase(display_name(d['name']), d['zone']), "collision",
                f"collision:{d['id']}", 3.0,
                {"class": d["name"], "id": d["id"], "growth_per_sec": round(g, 2)}))
    # NORMAL — the single most-pressing in-corridor obstacle (closeness * danger)
    if in_corr:
        nd = max(in_corr, key=lambda d: (d["box"][3] / h)
                 * DANGER.get(d["name"], DANGER_DEFAULT))
        urg = (nd["box"][3] / h) * DANGER.get(nd["name"], DANGER_DEFAULT)
        cands.append(Candidate(
            Priority.NORMAL, urg, phrase_for(display_name(nd["name"]), nd["zone"]),
            "obstacle", f"obstacle:{nd['zone']}", 2.0,
            {"class": nd["name"], "zone": nd["zone"]}))
    elif blocked:
        # corridor clear of objects but depth sees a wall ahead — warn, don't reassure
        cands.append(Candidate(Priority.NORMAL, 1.0, "path blocked", "path_state",
                               "blocked", 3.0, {}))
    elif heartbeat:
        # live app: speak "path is clear" once on the transition; the steady beep
        # (emitted directly in vision_loop) covers the ongoing clear state.
        if announce_clear:
            cands.append(Candidate(Priority.LOW, 0.2, "path is clear",
                                   "path_state", "clear", 2.0, {}))
    else:
        # offline callers (no beep channel): throttled spoken "path is clear"
        cands.append(Candidate(Priority.LOW, 0.1, "path is clear", "path_state",
                               "clear", CLEAR_FILLER_GAP, {}))
    # LOW — one side object just outside the corridor, ranked by proximity*danger
    side = []
    for d in dets:
        if d["cls"] == TRAFFIC_LIGHT_ID:
            continue
        if corridor.contains(d["cx"], d["box"][3], w, h):
            continue                          # already an in-corridor obstacle
        half = corridor.half_width_at(d["box"][3], w, h)
        if half is None:                      # beyond corridor depth = too far
            continue
        if abs(d["cx"] - w / 2.0) <= half + SIDE_BAND_FRAC * w:
            side.append(d)
    if side:
        sd = max(side, key=lambda d: (d["box"][3] / h)
                 * DANGER.get(d["name"], DANGER_DEFAULT))
        side_zone = "left" if sd["cx"] < w / 2.0 else "right"
        cands.append(Candidate(
            Priority.LOW, 0.5, phrase_for(display_name(sd["name"]), side_zone),
            "obstacle_side", f"side:{side_zone}", 2.0,
            {"class": sd["name"], "zone": side_zone}))
    # NORMAL — crosswalk (only when the detector's persistence already fired)
    if xres is not None:
        cands.append(Candidate(Priority.NORMAL, 0.5, "zebra crossing ahead",
                               "crosswalk", "crosswalk", 8.0,
                               {"n_bands": xres.n_bands}))
    # NORMAL — traffic-light state changes
    for st, idv in light_anns:
        cands.append(Candidate(Priority.NORMAL, 0.6, f"{st} light",
                               "traffic_light", f"light:{st}", 6.0,
                               {"state": st, "id": idv}))
    # LOW — path steering hints
    for m in path_msgs:
        cands.append(Candidate(Priority.LOW, 0.3, m["message"], m["type"],
                               m["type"], 4.0, m["data"]))
    return cands


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
                 crosswalk=None, path=None, corridor=None, banner=None,
                 freespace=None, ahead_corridor=None):
    """Draw corridor dividers, boxes+labels, crosswalk stripes, and phrases.

    Collision tracks are thick red, the nearest obstacle red, else green; a
    traffic-light box is coloured by its lamp state.
    """
    h, w = frame.shape[:2]
    # left | ahead | right corridor boundaries
    for x in (w // 3, 2 * w // 3):
        cv2.line(frame, (x, 0), (x, h), (80, 80, 80), 1)

    # walking corridor (guidance) — the region that can trigger obstacle alerts
    if corridor is not None:
        import numpy as _np
        pts = _np.array(corridor.polygon(w, h), dtype=_np.int32)
        cv2.polylines(frame, [pts], True, (255, 255, 0), 2)     # full corridor: yellow
        ahead = ahead_corridor if ahead_corridor is not None else corridor
        if ahead_corridor is not None:                          # shorter ahead-only
            apts = _np.array(ahead_corridor.polygon(w, h), dtype=_np.int32)
            cv2.polylines(frame, [apts], True, (0, 165, 255), 1)   # ahead: orange, thin
        for d in dets:                     # mark ground points by corridor relation
            gx, gy = int(d["cx"]), int(d["box"][3])
            if ahead.contains(d["cx"], d["box"][3], w, h):
                cv2.circle(frame, (gx, gy), 5, (0, 255, 255), -1)   # ahead obstacle
            elif not corridor.contains(d["cx"], d["box"][3], w, h):
                half = corridor.half_width_at(d["box"][3], w, h)   # side-band:
                if half is not None and \
                        abs(d["cx"] - w / 2.0) <= half + SIDE_BAND_FRAC * w:
                    cv2.circle(frame, (gx, gy), 5, (255, 0, 255), -1)  # "on your L/R"

    # free-space (monocular-depth) verdict readout, top-right
    if freespace is not None and getattr(freespace, "available", False):
        blocked = freespace.blocked
        txt = f"{'WALL' if blocked else 'free'} {freespace.score:.2f}"
        col = (0, 0, 255) if blocked else (150, 150, 150)
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, txt, (w - tw - 10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, col, 2, cv2.LINE_AA)

    # path guidance (boundaries, path centre, drift cue) — under the boxes
    if path is not None:
        annotate_path(frame, path)

    # crosswalk stripe edges (drawn under the boxes)
    if crosswalk is not None:
        for (x1, y1, x2, y2) in crosswalk.lines:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if crosswalk.found:
            cv2.putText(frame, "zebra crossing ahead", (10, 34),
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
        tag = " CLOSING" if alerting else ""
        ltag = f" [{light}]" if light in light_colors else ""
        label = f"{tid}{d['name']} {d['conf']:.2f} {d['zone']}{tag}{ltag}"
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # bottom banner: what the guidance actually SPOKE (arbiter output), so what
    # you see matches what you'd hear. banner="" means nothing spoken recently.
    if banner is not None:
        if banner:
            cv2.putText(frame, banner, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)
    elif nearest:                            # legacy fallback (image / no arbiter)
        cv2.putText(frame, phrase_for(nearest["name"], nearest["zone"]),
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2, cv2.LINE_AA)
    return frame


WINDOW = "Aakha vision (press q to quit)"


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
                path: bool = True, guidance: bool = True, freespace: bool = True,
                frames: Optional[Callable] = None,
                on_frame: Optional[Callable] = None,
                on_detections: Optional[Callable] = None,
                on_annotated: Optional[Callable] = None,
                voice_active: Optional[Callable] = None,
                nav_active: Optional[Callable] = None,
                class_ids: Optional[set[int]] = RELEVANT_CLASS_IDS,
                stop_event=None) -> None:
    """Continuous capture loop for live sources (webcam / video).

    track=True runs ByteTrack for stable ids so the CollisionMonitor can turn
    bbox growth into CRITICAL warnings.

    frames: a callable returning the latest BGR frame (push mode, e.g. phone over
    the /camera websocket); `source` is ignored when given, else a local
    cv2.VideoCapture(source) is opened. on_frame/on_detections/on_annotated feed
    main's shared slots. Stops when the source ends, stop_event is set, or q/ESC.
    """
    model = YOLO(ensure_onnx_model())
    cap = None
    if frames is None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[vision] ERROR: could not open source {source!r} "
                  f"(camera permission? wrong index?)")
            return
    else:
        print("[vision] running on pushed frames (server / phone camera)")
    debounce = Debounce()
    collision = CollisionMonitor() if track else None
    crosswalk_det = CrosswalkDetector() if crosswalk else None
    light_monitor = TrafficLightMonitor() if traffic_light else None
    path_guide = PathGuide() if path else None
    corridor = Corridor() if guidance else None
    # shorter corridor for the straight-ahead cue only (side/path use the full one)
    ahead_corridor = corridor.ahead() if corridor is not None else None
    arbiter = GuidanceArbiter() if guidance else None
    freespace_mon = FreeSpaceMonitor() if freespace else None
    if freespace_mon is not None:
        freespace_mon.start()
    writer = None
    src_fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else 0.0
    frame_no = 0
    last_banner = ""            # last message the arbiter actually spoke
    last_banner_t = float("-inf")
    last_cands = None           # candidates from the most recent real frame
    last_scene_t = float("-inf")  # wall-clock of that frame (for stall sustain)
    clear_announced = False     # spoke "path is clear" for the current clear run?
    last_beep_t = float("-inf")   # last on-track heartbeat beep (1 Hz metronome)
    try:
        while stop_event is None or not stop_event.is_set():
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    break
            else:
                frame = frames()
                if frame is None:          # no pushed frame yet — wait briefly
                    # Sustain the ~2s beat across short stalls, but not while muted.
                    quiet = (voice_active() if voice_active is not None else False) \
                        or (nav_active is not None and not nav_active())
                    if (publish and arbiter is not None and last_cands is not None
                            and not quiet):
                        tnow = time.time()
                        if tnow - last_scene_t <= STALE_SCENE_LIMIT:
                            chosen = arbiter.select(last_cands, tnow)
                            if chosen is not None:
                                event_bus.publish(chosen.to_event())
                                last_banner, last_banner_t = chosen.message, tnow
                    time.sleep(0.01)
                    continue
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

            # Feed the shared frame/detection slots for the OCR/voice threads.
            if on_frame is not None:
                on_frame(frame)
            if on_detections is not None:
                on_detections(dets)
            if freespace_mon is not None:
                freespace_mon.submit(frame)  # non-blocking, to the depth thread

            print(f"[frame {frame_no}]")
            print_dets(dets)

            # --- compute all signals once (advance the temporal monitors) ---
            h_ = frame.shape[0]
            alert_ids_set: set = set()
            growths: dict = {}
            if collision is not None:
                for d in dets:
                    if d["id"] is None:
                        continue
                    g = collision.update(d["id"], d["area"], frame_area, now)
                    if g is not None:
                        alert_ids_set.add(d["id"])
                        growths[d["id"]] = g
            alert_ids = frozenset(alert_ids_set)

            light_anns: list = []            # (state, id) to announce this frame
            if light_monitor is not None:
                for d in dets:
                    if d["cls"] != TRAFFIC_LIGHT_ID:
                        continue
                    st = classify_light(frame, d["box"])
                    d["light_state"] = st
                    ann = light_monitor.update(
                        d["id"] if d["id"] is not None else 0, st, now)
                    if ann:
                        light_anns.append((ann, d["id"]))

            xres = None
            xpub = False
            if crosswalk_det is not None:
                xres, xpub = crosswalk_det.update(frame, now)

            path_info = None
            path_msgs: list = []
            if path_guide is not None:
                path_info, path_msgs = path_guide.update(frame, dets, now,
                                                         corridor=corridor)

            # --- decide what to say ---
            if publish and arbiter is not None:
                blocked = freespace_mon.blocked if freespace_mon is not None else False
                # Mute all guidance (no cue, no beep) during a voice command or
                # while navigation is paused. Muting at the source stops it
                # uniformly on every consumer (laptop TTS, dashboard, phone).
                voice_on = voice_active() if voice_active is not None else False
                nav_on = nav_active() if nav_active is not None else True
                muted = voice_on or not nav_on
                # Is the straight-ahead path clear now? Speak "path is clear" once
                # per clear run; the beep covers the steady state. `announce` stays
                # sticky until the arbiter emits it, so a transition isn't lost.
                ahead_clear = not blocked and not any(
                    d["cls"] != TRAFFIC_LIGHT_ID
                    and ahead_corridor.contains(d["cx"], d["box"][3], width, h_)
                    for d in dets)
                if not ahead_clear:
                    clear_announced = False           # arm for the next clear run
                announce_clear = ahead_clear and not clear_announced
                last_cands = _build_candidates(dets, corridor, width, h_, growths,
                                               xres if xpub else None, light_anns,
                                               path_msgs, blocked=blocked,
                                               ahead_corridor=ahead_corridor,
                                               announce_clear=announce_clear,
                                               heartbeat=True)
                if muted:
                    # Clearing the list (vs. calling the arbiter) leaves its
                    # cadence/cooldown untouched, so guidance resumes cleanly.
                    last_cands = []
                last_scene_t = now
                chosen = arbiter.select(last_cands, now)
                if chosen is not None:
                    event_bus.publish(chosen.to_event())
                    last_banner, last_banner_t = chosen.message, now
                    if chosen.key == "clear":     # the one-shot actually went out
                        clear_announced = True

                # Steady 1 Hz on-track beep, emitted directly (bypassing the
                # arbiter's 2s beat). It means genuinely clear, so it's suppressed
                # if depth says blocked, anything stands in the FULL corridor (not
                # just the shorter ahead zone), any real cue is queued, or muted.
                in_path = any(
                    d["cls"] != TRAFFIC_LIGHT_ID
                    and corridor.contains(d["cx"], d["box"][3], width, h_)
                    for d in dets)
                has_cue = any(c.type != "path_state" for c in last_cands)
                path_clear = not blocked and not in_path and not has_cue and not muted
                if path_clear and (now - last_beep_t) >= ON_TRACK_BEEP_GAP:
                    event_bus.publish(Event(
                        message="", priority=Priority.LOW, type="heartbeat",
                        source="vision", data={"beep": True}))
                    last_beep_t = now
            elif publish:                    # legacy flood (--no-guidance), for A/B
                announce_directional(dets, publish=True, debounce=debounce)
                for idv, g in growths.items():
                    d = next((x for x in dets if x["id"] == idv), None)
                    if d is not None:
                        event_bus.publish(Event(
                            message=collision_phrase(d["name"], d["zone"]),
                            priority=Priority.CRITICAL, type="collision",
                            source="vision", data={"class": d["name"], "id": idv,
                                                   "growth_per_sec": round(g, 2)}))
                if xpub:
                    event_bus.publish(Event(
                        message="zebra crossing ahead", priority=Priority.NORMAL,
                        type="crosswalk", source="vision",
                        data={"n_bands": xres.n_bands}))
                for st, idv in light_anns:
                    event_bus.publish(Event(
                        message=f"{st} light", priority=Priority.NORMAL,
                        type="traffic_light", source="vision",
                        data={"state": st, "id": idv}))
                for m in path_msgs:
                    event_bus.publish(Event(
                        message=m["message"], priority=Priority.NORMAL,
                        type=m["type"], source="vision", data=m["data"]))

            if show or save is not None or on_annotated is not None:
                if arbiter is not None:
                    banner = last_banner if (now - last_banner_t) < 3.0 else ""
                else:
                    banner = None            # legacy: fall back to nearest phrase
                # draw on a copy so the clean frame kept for OCR/voice isn't
                # polluted with boxes.
                vis = frame.copy()
                draw_overlay(vis, dets, alert_ids, crosswalk=xres,
                             path=path_info, corridor=corridor, banner=banner,
                             freespace=freespace_mon, ahead_corridor=ahead_corridor)
                if on_annotated is not None:
                    on_annotated(vis)
                if save is not None:
                    if writer is None:
                        fh, fw = vis.shape[:2]
                        fps = src_fps if src_fps and src_fps > 0 else 20.0
                        writer = cv2.VideoWriter(
                            save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
                    writer.write(vis)
                if show and not _show(vis):
                    break
    finally:
        if freespace_mon is not None:
            freespace_mon.stop()
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
            print(f"[vision] saved annotated video -> {save}")
        cv2.destroyAllWindows()


def run_on_image(path: str, *, publish: bool = True, show: bool = False,
                 crosswalk: bool = True, traffic_light: bool = True,
                 path_guidance: bool = True,
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
    pinfo = None
    if path_guidance:
        pinfo, pmsgs = PathGuide().update(frame, dets, 0.0)
        for m in pmsgs:
            print(f"[path] {m['message']}")
            if publish:
                event_bus.publish(Event(
                    message=m["message"], priority=Priority.NORMAL,
                    type=m["type"], source="vision", data=m["data"]))
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
                message="zebra crossing ahead", priority=Priority.NORMAL,
                type="crosswalk", source="vision",
                data={"n_bands": xres.n_bands}))
    if show:
        draw_overlay(frame, dets, crosswalk=xres, path=pinfo)
        try:
            cv2.imshow(WINDOW, frame)
            print("[vision] press any key in the window to close")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as exc:
            print(f"[vision] display unavailable ({exc})")


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO11n detection")
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
    ap.add_argument("--no-path", action="store_true",
                    help="disable path guidance (clearest-path + off-path drift)")
    ap.add_argument("--no-guidance", action="store_true",
                    help="disable the corridor + single-slot arbiter (legacy flood)")
    ap.add_argument("--no-freespace", action="store_true",
                    help="disable the monocular-depth wall / dead-end detector")
    args = ap.parse_args()
    publish = not args.no_bus
    show = not args.no_show
    track = not args.no_track
    crosswalk = not args.no_crosswalk
    traffic_light = not args.no_traffic_light
    path_guidance = not args.no_path
    guidance = not args.no_guidance
    freespace = not args.no_freespace
    class_ids = None if args.all_classes else RELEVANT_CLASS_IDS

    src = args.source
    if src.isdigit():
        vision_loop(int(src), publish=publish, show=show, track=track,
                    save=args.save, crosswalk=crosswalk,
                    traffic_light=traffic_light, path=path_guidance,
                    guidance=guidance, freespace=freespace, class_ids=class_ids)
    elif Path(src).suffix.lower() in IMAGE_SUFFIXES:
        run_on_image(src, publish=publish, show=show, crosswalk=crosswalk,
                     traffic_light=traffic_light, path_guidance=path_guidance,
                     class_ids=class_ids)
    else:
        vision_loop(src, publish=publish, show=show, track=track,
                    save=args.save, crosswalk=crosswalk,
                    traffic_light=traffic_light, path=path_guidance,
                    guidance=guidance, freespace=freespace, class_ids=class_ids)

    # Standalone: show what actually landed on the bus.
    print(f"[vision] events on bus after run: {event_bus.qsize()}")


if __name__ == "__main__":
    main()
