"""App entry point. ``run()`` starts every worker thread and then blocks.

Thread layout after Dev 2 merges:
  _vision_producer  — Tier 1 stub (dev1/detection will replace this)
  _audio_consumer   — pyttsx3 priority-queue consumer (audio/consumer.py)
  _scene_caption    — Moondream2 background captioning (vision/scene_caption.py)
  _ocr              — Tesseract on Enter keypress (vision/ocr.py)
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
# Tier 1 stub — replaced by dev1/detection
# ---------------------------------------------------------------------------
def _vision_producer() -> None:
    """Tier 1 stub. dev1/detection replaces this with live YOLO11n + ByteTrack.

    Publishes one test event on startup so the smoke test sees qsize() > 0 even
    before the real vision branch merges.
    """
    event_bus.publish(Event(
        message="Vision thread started.",
        priority=Priority.LOW,
        type="system",
        source="vision_stub",
    ))
    while True:
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------
def start_workers() -> list[threading.Thread]:
    """Start all daemon workers and record them in WORKER_THREADS."""
    from audio.consumer import start_consumer
    from audio.voice_trigger import start_voice_trigger_thread
    from vision.ocr import start_ocr_thread
    from vision.scene_caption import start_scene_caption_thread

    WORKER_THREADS.clear()

    # Tier 1 stub (keeps smoke test happy until dev1 lands).
    vision_stub = threading.Thread(
        target=_vision_producer, name="_vision_producer", daemon=True
    )
    vision_stub.start()
    WORKER_THREADS.append(vision_stub)

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
