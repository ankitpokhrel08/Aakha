"""App entry point. ``run()`` starts every worker thread and then blocks.

Workers: _vision_producer (detection), _audio_consumer (TTS/beep), _ocr,
_voice_trigger. The vision loop writes the shared frame/detection slots below;
the OCR and voice threads read them and idle until a frame is available.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

import numpy as np

from src.core.bus import event_bus  # noqa: F401  keep the contract live
from src.core.events import Event, Priority

# Populated by run(); smoke_test inspects this to confirm threads are alive.
WORKER_THREADS: list[threading.Thread] = []

# --- shared frame slot: written by vision, read by OCR/voice ---
_frame_lock = threading.Lock()
_latest_frame: Optional[np.ndarray] = None


def get_latest_frame() -> Optional[np.ndarray]:
    with _frame_lock:
        return _latest_frame


def set_latest_frame(frame: np.ndarray) -> None:
    global _latest_frame
    with _frame_lock:
        _latest_frame = frame


# --- shared detections slot: written by vision, read by voice ("what am I holding") ---
_detections_lock = threading.Lock()
_latest_detections: Optional[list] = None


def get_latest_detections() -> Optional[list]:
    """Latest detections as [{"label": str, "bbox": (x1, y1, x2, y2)}, ...]."""
    with _detections_lock:
        return _latest_detections


def set_latest_detections(detections: list) -> None:
    global _latest_detections
    with _detections_lock:
        _latest_detections = detections


# --- incoming frame slot: written by the server (phone camera), read by vision ---
_incoming_lock = threading.Lock()
_incoming_frame: Optional[np.ndarray] = None

# When set by the server on startup, vision consumes pushed frames instead of
# opening a local camera. None for the `python main.py` webcam/VISION_SOURCE path.
FRAME_SOURCE = None


def push_frame(jpeg_or_array) -> None:
    """Feed a frame in. Accepts raw JPEG bytes (decoded here) or a BGR array."""
    global _incoming_frame
    frame = jpeg_or_array
    if isinstance(jpeg_or_array, (bytes, bytearray)):
        import cv2
        frame = cv2.imdecode(np.frombuffer(jpeg_or_array, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
    if frame is None:
        return
    with _incoming_lock:
        _incoming_frame = frame


def get_incoming_frame() -> Optional[np.ndarray]:
    with _incoming_lock:
        return _incoming_frame


def clear_incoming_frame() -> None:
    """Empty the pushed-frame slot so the vision loop idles (server pauses nav)."""
    global _incoming_frame
    with _incoming_lock:
        _incoming_frame = None


# --- voice-session flag: True while a voice command is being handled ---
# Set by the server over /control. Vision reads it to hush the on-track beep so
# it doesn't tick over a spoken command. Kept here so vision needn't import the
# web layer.
_voice_lock = threading.Lock()
_voice_active = False


def set_voice_active(on: bool) -> None:
    global _voice_active
    with _voice_lock:
        _voice_active = bool(on)


def get_voice_active() -> bool:
    with _voice_lock:
        return _voice_active


# --- navigation gate: True while navigating, False while paused ---
# When paused the camera keeps streaming (so a voice command still sees a frame)
# but no guidance/beep is published. Defaults True so `python main.py` navigates
# without a server; the server sets it False on boot.
_nav_lock = threading.Lock()
_nav_active = True


def set_nav_active(on: bool) -> None:
    global _nav_active
    with _nav_lock:
        _nav_active = bool(on)


def get_nav_active() -> bool:
    with _nav_lock:
        return _nav_active


# --- annotated frame slot: written by vision, read by the server /monitor stream ---
_annotated_lock = threading.Lock()
_annotated_frame: Optional[np.ndarray] = None


def set_annotated_frame(frame: np.ndarray) -> None:
    global _annotated_frame
    with _annotated_lock:
        _annotated_frame = frame


def get_annotated_frame() -> Optional[np.ndarray]:
    with _annotated_lock:
        return _annotated_frame


def _vision_producer() -> None:
    """Run the vision loop, feeding the shared slots. Idles (thread stays alive)
    if the camera can't be opened, so headless machines don't crash the app."""
    event_bus.publish(Event("Vision thread started.", Priority.LOW, "system", "vision"))

    def _on_detections(dets: list) -> None:
        set_latest_detections([
            {"label": d["name"], "bbox": tuple(int(v) for v in d["box"])}
            for d in dets
        ])

    # Webcam (0) by default; VISION_SOURCE=clip.mp4 to run on a file.
    src_env = os.environ.get("VISION_SOURCE", "0")
    source = int(src_env) if src_env.isdigit() else src_env

    try:
        from config import config
        from src.vision.detect import vision_loop
        vision_loop(
            source=source,
            frames=FRAME_SOURCE,  # push mode when the server feeds phone frames
            show=False,           # GUI must stay on the main thread on macOS
            publish=True,
            crosswalk=config.crosswalk_detection,
            traffic_light=config.traffic_light_detection,
            on_frame=set_latest_frame,
            on_detections=_on_detections,
            on_annotated=set_annotated_frame,
            voice_active=get_voice_active,
            nav_active=get_nav_active,
        )
    except Exception as exc:  # camera missing, permission denied, source ended
        print(f"[vision] loop stopped ({exc}); thread idling")

    while True:
        time.sleep(1.0)


def start_workers() -> list[threading.Thread]:
    """Start all daemon workers and record them in WORKER_THREADS."""
    from src.audio.consumer import start_consumer
    from src.audio.voice_trigger import start_voice_trigger_thread
    from src.vision.ocr import start_ocr_thread

    WORKER_THREADS.clear()

    vision_thread = threading.Thread(
        target=_vision_producer, name="_vision_producer", daemon=True
    )
    vision_thread.start()
    WORKER_THREADS.append(vision_thread)
    WORKER_THREADS.append(start_consumer())
    WORKER_THREADS.append(start_ocr_thread(get_latest_frame))
    WORKER_THREADS.append(start_voice_trigger_thread(
        get_latest_frame, get_detections=get_latest_detections))
    return WORKER_THREADS


def run(block: bool = True) -> list[threading.Thread]:
    """Start workers. Blocks by default; pass block=False to just get the handles."""
    threads = start_workers()
    if block:
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    return threads


if __name__ == "__main__":
    run()
