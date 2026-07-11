"""App entry point. ``run()`` starts every worker thread and then blocks.

Thread layout after Dev 2 merges:
  _vision_producer  — Tier 1 stub (dev1/detection will replace this)
  _audio_consumer   — pyttsx3 priority-queue consumer (audio/consumer.py)
  _scene_caption    — Moondream2 background captioning (visuals/scene_caption.py)
  _ocr              — Tesseract on Enter keypress (visuals/ocr.py)
  _voice_trigger    — Vosk wake-word + MediaPipe Hands (audio/voice_trigger.py)

Shared frame slot:
  The scene-caption, OCR, and voice-trigger threads call ``get_latest_frame()``,
  which reads a BGR numpy array written by the vision thread. Until dev1/detection
  merges the stub leaves it None, so those threads idle without processing.
  When dev1 merges, its loop should call ``set_latest_frame(frame)`` each tick.

Shared detections slot:
  The voice-trigger thread calls ``get_latest_detections()`` to match the held
  object to a YOLO label. Stays None until dev1/detection merges; voice_trigger
  handles this gracefully (reports hand visible but object unlabelled).
  When dev1 merges, its loop should call ``set_latest_detections(dets)`` each tick.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

import numpy as np

from shared.bus import event_bus  # noqa: F401  wired so the contract is live
from shared.events import Event, Priority

# Populated by run(); smoke_test.py inspects this to confirm threads are alive.
WORKER_THREADS: list[threading.Thread] = []

# ---------------------------------------------------------------------------
# Shared frame slot — written by Tier 1, read by Tier 2/3
# ---------------------------------------------------------------------------
_frame_lock = threading.Lock()
_latest_frame: Optional[np.ndarray] = None


def get_latest_frame() -> Optional[np.ndarray]:
    """Return the most recent BGR frame from the vision thread, or None."""
    with _frame_lock:
        return _latest_frame


def set_latest_frame(frame: np.ndarray) -> None:
    """Called by the Tier 1 vision thread each time a new frame is processed."""
    global _latest_frame
    with _frame_lock:
        _latest_frame = frame


# ---------------------------------------------------------------------------
# Shared detections slot — written by Tier 1, read by voice_trigger (Tier 3)
# ---------------------------------------------------------------------------
_detections_lock = threading.Lock()
_latest_detections: Optional[list] = None


def get_latest_detections() -> Optional[list]:
    """Return the latest list of YOLO detections, or None until Tier 1 merges.

    Each entry: {"label": str, "bbox": (x1, y1, x2, y2)}
    """
    with _detections_lock:
        return _latest_detections


def set_latest_detections(detections: list) -> None:
    """Called by the Tier 1 vision thread after each YOLO inference pass."""
    global _latest_detections
    with _detections_lock:
        _latest_detections = detections


# ---------------------------------------------------------------------------
# Incoming frame slot — written by the server (phone camera), read by Tier 1
# ---------------------------------------------------------------------------
_incoming_lock = threading.Lock()
_incoming_frame: Optional[np.ndarray] = None

# When set (server.py does this on startup), the vision thread consumes pushed
# frames from here instead of opening a local camera. Left None for the
# `python main.py` local-webcam / VISION_SOURCE path.
FRAME_SOURCE = None


def push_frame(jpeg_or_array) -> None:
    """Feed a frame into the pipeline. Accepts raw JPEG bytes (decoded here) or a
    BGR numpy array. server.py wires its /camera websocket to this."""
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
    """Latest frame pushed by the server (phone camera), or None."""
    with _incoming_lock:
        return _incoming_frame


def clear_incoming_frame() -> None:
    """Empty the pushed-frame slot so the vision loop goes idle.

    Called by the server when navigation is toggled OFF. The vision loop reads
    frames via ``get_incoming_frame`` (push mode); once this is None it hits the
    ``frame is None`` branch in ``vision_loop`` and sleeps instead of
    re-processing the last frame forever. No vision-side change needed."""
    global _incoming_frame
    with _incoming_lock:
        _incoming_frame = None


# ---------------------------------------------------------------------------
# Annotated frame slot — written by Tier 1 (overlay drawn), read by the server
# /monitor MJPEG stream so you can watch detections live on the laptop.
# ---------------------------------------------------------------------------
_annotated_lock = threading.Lock()
_annotated_frame: Optional[np.ndarray] = None


def set_annotated_frame(frame: np.ndarray) -> None:
    global _annotated_frame
    with _annotated_lock:
        _annotated_frame = frame


def get_annotated_frame() -> Optional[np.ndarray]:
    """Latest frame with detection overlay drawn, or None."""
    with _annotated_lock:
        return _annotated_frame


# ---------------------------------------------------------------------------
# Tier 1 producer — real YOLO11n vision loop (dev1/detection)
# ---------------------------------------------------------------------------
def _vision_producer() -> None:
    """Run the real Tier 1 vision loop: YOLO11n detection + ByteTrack collision
    + crosswalk + traffic-light, publishing events to the bus and feeding the
    shared frame/detection slots for Tier 2/3.

    Falls back to idling (thread stays alive) if the camera can't be opened, so
    the smoke test and the Tier 2/3 threads that read get_latest_frame() don't
    die on a headless machine.
    """
    # Startup marker so smoke_test sees qsize() > 0 immediately, and the
    # dashboard shows the vision thread came up.
    event_bus.publish(Event(
        message="Vision thread started.",
        priority=Priority.LOW,
        type="system",
        source="vision",
    ))

    def _on_detections(dets: list) -> None:
        # Adapt Tier 1 dicts to the shape voice_trigger expects.
        set_latest_detections([
            {"label": d["name"], "bbox": tuple(int(v) for v in d["box"])}
            for d in dets
        ])

    # Source is the webcam (0) by default; point it at a file for testing with
    #   VISION_SOURCE=assets/cars.mp4 python main.py
    src_env = os.environ.get("VISION_SOURCE", "0")
    source = int(src_env) if src_env.isdigit() else src_env

    try:
        from config import config
        from vision.detect import vision_loop
        vision_loop(
            source=source,
            frames=FRAME_SOURCE,  # push mode when the server fed phone frames
            show=False,           # GUI must stay on the main thread on macOS
            publish=True,
            crosswalk=config.crosswalk_detection,
            traffic_light=config.traffic_light_detection,
            on_frame=set_latest_frame,
            on_detections=_on_detections,
            on_annotated=set_annotated_frame,  # feed the /monitor live view
        )
    except Exception as exc:  # camera missing, permission denied, source ended
        print(f"[vision] loop stopped ({exc}); thread idling")

    while True:
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------
def start_workers() -> list[threading.Thread]:
    """Start all daemon workers and record them in WORKER_THREADS."""
    from audio.consumer import start_consumer
    from audio.voice_trigger import start_voice_trigger_thread
    from visuals.ocr import start_ocr_thread
    from visuals.scene_caption import start_scene_caption_thread

    WORKER_THREADS.clear()

    # Tier 1 real vision loop (local camera, VISION_SOURCE file, or pushed
    # frames when FRAME_SOURCE was set by the server).
    vision_thread = threading.Thread(
        target=_vision_producer, name="_vision_producer", daemon=True
    )
    vision_thread.start()
    WORKER_THREADS.append(vision_thread)

    # Tier 2 audio consumer.
    WORKER_THREADS.append(start_consumer())

    # Tier 2 scene captioning (idles gracefully when get_latest_frame() → None).
    WORKER_THREADS.append(start_scene_caption_thread(get_latest_frame))

    # Tier 3 OCR — triggered by pressing Enter; idles when no frame available.
    WORKER_THREADS.append(start_ocr_thread(get_latest_frame))

    # Tier 3 voice trigger — idles if Vosk model not present; get_detections
    # is None until dev1/detection merges, handled gracefully inside the thread.
    WORKER_THREADS.append(start_voice_trigger_thread(
        get_latest_frame,
        get_detections=get_latest_detections,
    ))

    return WORKER_THREADS


def run(block: bool = True) -> list[threading.Thread]:
    """Start workers. Blocks the calling thread by default (real app use);
    pass block=False to start workers and return their handles."""
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
