"""Single TTS consumer thread — the only code that calls pyttsx3.

Never start a second consumer. Other modules publish to event_bus; CRITICAL is
always dequeued before NORMAL/LOW. type="repeat" re-speaks `last_spoken`;
type="heartbeat" plays the on-track beep instead of speaking.
"""
from __future__ import annotations

import os
import sys

# Repo root on sys.path so `src` resolves when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import queue
import subprocess
import threading
import time

import pyttsx3

from src.core.bus import event_bus
from src.core.events import Event, Priority

# On-track beep (vision publishes type="heartbeat" while the path is clear),
# played on this thread so it never overlaps speech.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BEEP_SOUND = os.path.join(_REPO_ROOT, "sounds", "Purr.mp3")

# Reply types that END a voice interaction: after speaking one, the consumer
# clears the voice-session flag itself so the beep resumes on the laptop's clock.
_VOICE_REPLY_TERMINAL = {"ocr", "held_object", "voice_no_match",
                         "voice_error"}


def _voice_active() -> bool:
    """True while a voice command is being handled. Imports main lazily (main
    imports us). Never raises."""
    try:
        import main
        return main.get_voice_active()
    except Exception:
        return False


def _clear_voice_active() -> None:
    """Mark the voice session over (laptop finished speaking the reply)."""
    try:
        import main
        main.set_voice_active(False)
    except Exception:
        pass


def _play_beep() -> None:
    """Play the on-track beep; fall back to the terminal bell off macOS. Never raises."""
    try:
        subprocess.run(["afplay", _BEEP_SOUND], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, OSError):
        print("\a", end="", flush=True)


# --- "repeat that" buffer: last spoken message, read by voice_trigger dispatch ---
_last_spoken_lock = threading.Lock()
last_spoken: str = ""


def get_last_spoken() -> str:
    with _last_spoken_lock:
        return last_spoken


def _set_last_spoken(msg: str) -> None:
    global last_spoken
    with _last_spoken_lock:
        last_spoken = msg


def _consumer_loop() -> None:
    """Drain event_bus by priority and speak each message via pyttsx3.

    The engine is created in this thread (pyttsx3 isn't safe to init elsewhere).
    CRITICAL is dequeued first but does NOT preempt a sentence already playing;
    messages are kept short so the worst-case wait is ~1-2 s.
    """
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)  # slightly faster than default for nav use

    while True:
        try:
            event = event_bus.get(block=True, timeout=0.5)
        except queue.Empty:
            continue

        try:
            # "Repeat that" — re-speak the last spoken message.
            if event.type == "repeat":
                msg = get_last_spoken()
                if msg:
                    engine.say(msg)
                    engine.runAndWait()
            elif event.type == "heartbeat":
                # Not speech, so don't record as last_spoken. Re-check at play
                # time: drop a beep queued just before a voice session started.
                if not _voice_active():
                    _play_beep()
            else:
                _set_last_spoken(event.message)
                engine.say(event.message)
                engine.runAndWait()
                # A voice-command reply is done speaking — end the session so the
                # beep resumes on the laptop's clock.
                if event.type in _VOICE_REPLY_TERMINAL:
                    _clear_voice_active()
        except Exception:
            # Never let a pyttsx3 failure kill the consumer thread.
            pass
        finally:
            event_bus.task_done()


def start_consumer() -> threading.Thread:
    """Start the TTS consumer daemon thread and return it. Call once from main."""
    t = threading.Thread(target=_consumer_loop, name="_audio_consumer", daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    # Publish LOW/NORMAL/CRITICAL; expected speech order is CRITICAL, NORMAL, LOW.
    print("Starting TTS consumer...")
    start_consumer()
    time.sleep(0.5)
    event_bus.publish(Event("Text reads: exit ahead", Priority.LOW, "ocr", "ocr"))
    event_bus.publish(Event("Obstacle ahead", Priority.NORMAL, "obstacle", "vision"))
    time.sleep(0.05)
    event_bus.publish(Event("Warning: object approaching fast", Priority.CRITICAL, "collision", "vision"))
    time.sleep(10)
    print("Standalone test done.")
