"""Single TTS consumer thread — the only thing that ever calls pyttsx3.

Architecture contract:
- There is EXACTLY ONE consumer thread. Never start a second one.
- CRITICAL events are always dequeued before NORMAL/LOW (priority queue guarantee).
- All other modules publish to event_bus; they never call pyttsx3 directly.

"Repeat that" support:
- The consumer keeps the last spoken message in `last_spoken` (module-level).
- voice_trigger dispatches a "repeat" command by publishing an Event with
  type="repeat"; the consumer detects it and re-speaks `last_spoken`.
"""
from __future__ import annotations

import os
import sys

# Ensure repo root is on sys.path so `shared` is importable whether this file
# is run directly (python audio/consumer.py) or imported from the root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import queue
import subprocess
import threading
import time

import pyttsx3

from shared.bus import event_bus
from shared.events import Event, Priority

# On-track heartbeat beep (vision publishes type="heartbeat" while the path is
# clear — see vision/detect.py). Rendered as a short, calm system sound instead
# of speech, on this same single audio thread so it never overlaps the voice.
# On-track beep sound, kept in the repo (sound_asset/) so it's portable and easy
# to swap — candidates live there; Purr is the current pick (~800 Hz, soft rounded
# timbre, short) per navigation-audio research (mid-range, gentle, non-alarming).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BEEP_SOUND = os.path.join(_REPO_ROOT, "sound_asset", "Purr.mp3")

# Voice-command reply types that END a voice interaction. After the consumer
# finishes SPEAKING one of these, the laptop is done talking, so it clears the
# voice-session flag itself — resuming the beep exactly when it should, instead
# of depending on the phone's (differently-timed) TTS to signal "done". Mirrors
# the phone client's REPLY_TERMINAL set.
_VOICE_REPLY_TERMINAL = {"ocr", "held_object", "voice_no_match",
                         "voice_error"}


def _voice_active() -> bool:
    """True while a voice command is being recorded / handled (set by the server
    via main.set_voice_active). Read lazily so this module doesn't import main at
    load time (main imports us). Never raises."""
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
    """Play the calm 'you're on track' beep. Falls back to the terminal bell if
    afplay / the sound file isn't available (e.g. non-macOS). Never raises."""
    try:
        subprocess.run(["afplay", _BEEP_SOUND], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, OSError):
        print("\a", end="", flush=True)

# ---------------------------------------------------------------------------
# Shared "repeat that" buffer — written by consumer, read by anyone who needs
# to know what was last spoken (currently only voice_trigger dispatch).
# ---------------------------------------------------------------------------
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
            # "Repeat that" — re-speak the last spoken message.
            if event.type == "repeat":
                msg = get_last_spoken()
                if msg:
                    engine.say(msg)
                    engine.runAndWait()
            elif event.type == "heartbeat":
                # on-track reassurance beep — not speech, so don't record it as
                # last_spoken (a "repeat that" should replay real guidance).
                # Gate at PLAY time too (not just at publish): a heartbeat queued
                # in the ~1 s before the voice session was flagged would otherwise
                # tick over the command. Drop it if a voice session is now active.
                if not _voice_active():
                    _play_beep()
            else:
                _set_last_spoken(event.message)
                engine.say(event.message)
                engine.runAndWait()
                # The laptop just finished speaking. If that was a voice-command
                # reply, the interaction is over — clear the session flag so the
                # beep resumes now, on the laptop's own clock (not the phone's).
                if event.type in _VOICE_REPLY_TERMINAL:
                    _clear_voice_active()
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
    event_bus.publish(Event("Text reads: exit ahead", Priority.LOW, "ocr", "ocr"))
    # time.sleep(0.05)
    event_bus.publish(Event("Obstacle ahead", Priority.NORMAL, "obstacle", "vision"))
    time.sleep(0.05)
    event_bus.publish(Event("Warning: object approaching fast", Priority.CRITICAL, "collision", "vision"))
    event_bus.publish(Event("Text reads: exit ahead", Priority.LOW, "ocr", "ocr"))
    # time.sleep(0.05)
    event_bus.publish(Event("Obstacle ahead", Priority.NORMAL, "obstacle", "vision"))
    
    print("Waiting 25 s for speech to finish. Listen for order: CRITICAL, NORMAL, LOW.")
    time.sleep(10)
    print("Standalone test done.")
