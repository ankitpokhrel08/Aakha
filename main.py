"""App entry point. ``run()`` starts every worker thread and then blocks.

Setup skeleton: the worker bodies are intentionally empty stubs so the call
contract exists and ``shared/smoke_test.py`` can import and exercise it. Feature
branches replace the stub bodies:

  - vision producer  -> dev1/detection     (YOLO11n + ByteTrack, Tier 1)
  - audio consumer   -> dev2/tts-consumer   (single pyttsx3 consumer)
  - scene captions   -> dev2/scene-caption  (Moondream2, Tier 2)

Keep ``run()`` importable at all times.
"""
from __future__ import annotations

import threading
import time

from shared.bus import event_bus  # noqa: F401  wired so the contract is live

# Populated by run(); smoke_test.py inspects this to confirm threads are alive.
WORKER_THREADS: list[threading.Thread] = []


def _vision_producer() -> None:
    """Tier 1 stub. dev1/detection replaces this with live YOLO11n + ByteTrack
    detection that publishes obstacle / collision / path events via
    ``event_bus.publish(...)``."""
    while True:
        time.sleep(1.0)


def _audio_consumer() -> None:
    """TTS stub. dev2/tts-consumer replaces this with the single pyttsx3
    consumer that drains ``event_bus`` by priority and speaks, letting CRITICAL
    interrupt NORMAL/LOW speech. Leaving it a no-op for now means detections
    accumulate in the queue (qsize > 0) once dev1 merges — which is exactly
    what the Merge 1 smoke test checks for."""
    while True:
        time.sleep(1.0)


def start_workers() -> list[threading.Thread]:
    """Start all daemon workers and record them in WORKER_THREADS."""
    WORKER_THREADS.clear()
    for target in (_vision_producer, _audio_consumer):
        t = threading.Thread(target=target, name=target.__name__, daemon=True)
        t.start()
        WORKER_THREADS.append(t)
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
