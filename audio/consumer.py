"""Single TTS consumer thread — the only thing that ever calls pyttsx3.

Architecture contract:
- There is EXACTLY ONE consumer thread. Never start a second one.
- CRITICAL events are always dequeued before NORMAL/LOW (priority queue guarantee).
- All other modules publish to event_bus; they never call pyttsx3 directly.
"""
from __future__ import annotations

import os
import sys

# Ensure repo root is on sys.path so `shared` is importable whether this file
# is run directly (python audio/consumer.py) or imported from the root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import queue
import threading
import time

import pyttsx3

from shared.bus import event_bus
from shared.events import Event, Priority


def _consumer_loop() -> None:
    """Drain event_bus by priority and speak each message via pyttsx3.

    pyttsx3 engines are not thread-safe to init outside the thread that calls
    runAndWait(), so we create the engine here inside the daemon thread.

    Interruption model: the priority queue guarantees CRITICAL is dequeued
    before NORMAL/LOW on every get() call. Messages are kept short, so the
    worst-case wait (finishing a NORMAL sentence before speaking CRITICAL) is
    ~1–2 s. True mid-sentence preemption would require a second thread calling
    engine.stop() — out of scope for Tier 1/2 but documented here for later.
    """
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)  # slightly faster than default for nav use

    while True:
        try:
            event = event_bus.get(block=True, timeout=0.5)
        except queue.Empty:
            continue

        try:
            engine.say(event.message)
            engine.runAndWait()
        except Exception:
            # Never let a pyttsx3 failure kill the consumer thread.
            pass
        finally:
            event_bus.task_done()


def start_consumer() -> threading.Thread:
    """Start the TTS consumer daemon thread and return it.

    Call once from main.py; append the returned thread to WORKER_THREADS.
    """
    t = threading.Thread(target=_consumer_loop, name="_audio_consumer", daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    """Standalone test: publish LOW → NORMAL → CRITICAL in that order.
    Expected speech order: CRITICAL → NORMAL → LOW (priority queue reorders).
    """

    print("Starting TTS consumer...")
    start_consumer()
    time.sleep(0.5)  # give the engine a moment to init before first event arrives

    print("Publishing LOW → NORMAL → CRITICAL (should be spoken in reverse priority order)...")
    event_bus.publish(Event("Scene caption: a corridor ahead", Priority.LOW, "caption", "scene"))
    time.sleep(0.05)
    event_bus.publish(Event("Obstacle ahead", Priority.NORMAL, "obstacle", "vision"))
    time.sleep(0.05)
    event_bus.publish(Event("Warning: object approaching fast", Priority.CRITICAL, "collision", "vision"))

    print("Waiting 25 s for speech to finish. Listen for order: CRITICAL, NORMAL, LOW.")
    time.sleep(25)
    print("Standalone test done.")
