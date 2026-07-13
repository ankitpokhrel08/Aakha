"""One-shot Vosk transcription + 3-command keyword dispatch.

The server drops a short PCM clip onto `clip_queue` when the user holds to talk;
this thread transcribes it and dispatches one of three commands:
  "what am I holding"  -> most prominent object over a ~2 s scan
  "read this"          -> OCR the current frame
  "repeat that"        -> re-speak the last TTS message (type="repeat")

VOSK_MODEL_PATH overrides the model dir (default: <repo_root>/
vosk-model-small-en-us-0.15, ~40 MB from https://alphacephei.com/vosk/models).
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import queue
import threading
import time
from typing import Callable, List, Optional

import numpy as np

from src.core.bus import event_bus
from src.core.events import Event, Priority

# Public clip queue: the server puts raw 16kHz mono PCM bytes here.
clip_queue: queue.Queue = queue.Queue()

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "vosk-model-small-en-us-0.15",
)

# Command matching: a set of regex patterns per command (stems, synonyms, whole
# phrases) to tolerate Vosk mis-hearings and varied phrasing. Any pattern hit
# fires the command; first command with a hit wins.
_COMMAND_PATTERNS = [
    ("holding", [
        r"\bhold",
        r"\bcarry",
        r"in (my|the|your) hand",
        r"\bhand(s)?\b",
        r"what.*this (object|thing|item)",
        r"what('s| is) this thing",
    ]),
    ("read", [
        r"\bread",
        r"what does (it|this|that|the).* say",
        r"what('s| is) (it|this|that) say",
        r"\btext\b",
        r"\bwritten\b",
    ]),
    ("repeat", [
        r"\brepeat",
        r"(say|come) again",
        r"\bagain\b",
        r"what did you say",
        r"one more time",
        r"\bpardon\b",
    ]),
]

# Compile once at import (patterns never change) — matching is then just search.
_COMPILED_COMMANDS = [
    (key, [re.compile(p) for p in pats]) for key, pats in _COMMAND_PATTERNS
]


def _load_vosk_model():
    """Load and return a Vosk model, or None if not present."""
    model_path = os.environ.get("VOSK_MODEL_PATH", _DEFAULT_MODEL_PATH)
    if not os.path.isdir(model_path):
        print(
            f"[voice_trigger] Vosk model not found at: {model_path}\n"
            f"  Download vosk-model-small-en-us-0.15.zip from:\n"
            f"  https://alphacephei.com/vosk/models\n"
            f"  Unzip into the repo root so the path above exists."
        )
        return None
    import vosk
    vosk.SetLogLevel(-1)
    model = vosk.Model(model_path)
    print("[voice_trigger] Vosk model loaded.")
    return model


def transcribe_clip(audio_bytes: bytes, model=None) -> str:
    """Run Vosk on raw 16kHz mono PCM bytes; return transcript string.

    Args:
        audio_bytes: Raw PCM16 bytes at 16000 Hz mono.
        model:       A loaded vosk.Model instance. If None, loads from
                     VOSK_MODEL_PATH (slow — pass a cached model in the thread).
    """
    import vosk
    if model is None:
        model = _load_vosk_model()
        if model is None:
            return ""
    rec = vosk.KaldiRecognizer(model, 16000)
    rec.AcceptWaveform(audio_bytes)
    result = json.loads(rec.FinalResult())
    return result.get("text", "").lower().strip()


def _match_command(transcript: str) -> Optional[str]:
    """Return the command key whose patterns first match the transcript, or None."""
    t = transcript.lower().strip()
    if not t:
        return None
    for key, patterns in _COMPILED_COMMANDS:
        if any(p.search(t) for p in patterns):
            return key
    return None


# "what am I holding": no hand detection (it was unreliable). The held object is
# taken to be the most prominent one (biggest, most central), scanned over ~2 s
# and majority-voted so a single-frame miss can't flip the result.
_HOLD_SCAN_SECONDS = 2.0
_HOLD_SAMPLE_GAP = 0.12     # ~8 samples/sec
_NOT_HELD = {"person"}      # never reported as a held object


def _prominence(det, w: int, h: int) -> float:
    """How 'held up to the camera' a detection looks: mostly its size (a closer
    object fills more of the frame), biased toward the centre."""
    x1, y1, x2, y2 = det["bbox"]
    area = max(0, x2 - x1) * max(0, y2 - y1)
    if area <= 0:
        return 0.0
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    maxd = ((w / 2.0) ** 2 + (h / 2.0) ** 2) ** 0.5 or 1.0
    dist = ((cx - w / 2.0) ** 2 + (cy - h / 2.0) ** 2) ** 0.5
    centrality = 1.0 - min(1.0, dist / maxd)      # 1 at centre, 0 at a corner
    return area * (0.5 + 0.5 * centrality)         # size-dominant, centre-biased


def _most_prominent(detections, w: int, h: int) -> Optional[str]:
    """Label of the most prominent held-candidate object this frame, or None."""
    best_score, best = 0.0, None
    for det in detections:
        if det.get("label") in _NOT_HELD:
            continue
        s = _prominence(det, w, h)
        if s > best_score:
            best_score, best = s, det.get("label")
    return best


def _scan_held_object(get_frame, get_detections,
                      duration: float = _HOLD_SCAN_SECONDS) -> str:
    """Report the most prominent object over a ~`duration` s window, majority-voted."""
    from collections import Counter

    votes: "Counter[str]" = Counter()
    deadline = time.time() + duration
    while time.time() < deadline:
        frame = get_frame()
        dets = get_detections() if get_detections is not None else None
        if frame is not None and dets:
            h, w = frame.shape[:2]
            label = _most_prominent(dets, w, h)
            if label:
                votes[label] += 1
        time.sleep(_HOLD_SAMPLE_GAP)

    if not votes:
        return "I don't see anything clearly. Point the camera at it."
    return f"In front of you is a {votes.most_common(1)[0][0]}."


def dispatch_command(
    transcript: str,
    get_frame: Callable[[], Optional[np.ndarray]],
    get_detections: Optional[Callable[[], Optional[List[dict]]]] = None,
) -> None:
    """Match transcript -> command -> publish the appropriate Event(s)."""
    print(f"[voice_trigger] transcript: {transcript!r}")
    cmd = _match_command(transcript)
    if cmd is None:
        # Distinguish "heard nothing" (brief cue) from "heard an unknown command"
        # (offer the command list). Both are terminal, so navigation resumes after.
        if not transcript.strip():
            print("[voice_trigger] Nothing heard.")
            message = "I didn't catch that."
        else:
            print("[voice_trigger] No command matched.")
            message = ("I didn't understand. You can say: what am I holding, "
                       "read this, or repeat that.")
        event_bus.publish(Event(
            message=message,
            priority=Priority.NORMAL,
            type="voice_no_match",
            source="voice",
        ))
        return

    print(f"[voice_trigger] command: {cmd}")

    if cmd == "repeat":
        # Consumer re-speaks last_spoken when it sees type="repeat".
        event_bus.publish(Event(
            message="",
            priority=Priority.NORMAL,
            type="repeat",
            source="voice",
        ))

    elif cmd == "read":
        frame = get_frame()
        if frame is None:
            event_bus.publish(Event(
                message="Camera not ready. Try again in a moment.",
                priority=Priority.NORMAL,
                type="ocr",
                source="voice",
            ))
            return
        from src.vision.ocr import _run_ocr
        text = _run_ocr(frame)
        message = f"Text reads: {text}" if text else "No text found in the frame."
        event_bus.publish(Event(
            message=message,
            priority=Priority.NORMAL,
            type="ocr",
            source="voice",
        ))

    elif cmd == "holding":
        if get_frame() is None:
            event_bus.publish(Event(
                message="Camera not ready. Try again in a moment.",
                priority=Priority.NORMAL,
                type="held_object",
                source="voice",
            ))
            return

        try:
            message = _scan_held_object(get_frame, get_detections)
        except Exception as e:
            print(f"[voice_trigger] holding scan error: {e}")
            message = "Could not identify what is in front of you."
        event_bus.publish(Event(
            message=message,
            priority=Priority.NORMAL,
            type="held_object",
            source="voice",
        ))


# --- background thread: monitor clip_queue, transcribe, dispatch ---
def _voice_trigger_thread(
    get_frame: Callable[[], Optional[np.ndarray]],
    get_detections: Optional[Callable[[], Optional[List[dict]]]],
) -> None:
    model = _load_vosk_model()
    if model is None:
        print("[voice_trigger] Idling — no Vosk model. Put model at VOSK_MODEL_PATH to activate.")
        while True:
            time.sleep(10.0)

    print("[voice_trigger] Ready — waiting for audio clips on clip_queue.")
    while True:
        try:
            audio_bytes = clip_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            transcript = transcribe_clip(audio_bytes, model=model)
            dispatch_command(transcript, get_frame, get_detections)
        except Exception as exc:
            # A dispatch failure must never kill this thread (that would silence
            # every future command). Log, speak a fallback, keep going.
            print(f"[voice_trigger] dispatch failed: {exc}")
            try:
                event_bus.publish(Event(
                    message="Sorry, something went wrong with that command.",
                    priority=Priority.NORMAL,
                    type="voice_error",
                    source="voice",
                ))
            except Exception:
                pass
        finally:
            clip_queue.task_done()


def start_voice_trigger_thread(
    get_frame: Callable[[], Optional[np.ndarray]],
    get_detections: Optional[Callable[[], Optional[List[dict]]]] = None,
) -> threading.Thread:
    """Start the voice-trigger daemon thread and return it. get_frame is
    main.get_latest_frame; get_detections yields [{"label", "bbox"}, ...] or None."""
    t = threading.Thread(
        target=_voice_trigger_thread,
        args=(get_frame, get_detections),
        name="_voice_trigger",
        daemon=True,
    )
    t.start()
    return t


if __name__ == "__main__":
    # Standalone: exercise keyword dispatch directly (bypassing Vosk).
    from src.audio.consumer import start_consumer

    model_path = os.environ.get("VOSK_MODEL_PATH", _DEFAULT_MODEL_PATH)
    if not os.path.isdir(model_path):
        print(f"[test] Vosk model not found at: {model_path}")
        sys.exit(1)

    start_consumer()
    time.sleep(0.3)

    def _fake_frame():
        return np.zeros((480, 640, 3), dtype=np.uint8)

    for phrase in ["repeat that", "read this", "what am I holding"]:
        print(f"\n[test] Dispatching: {phrase!r}")
        dispatch_command(phrase, _fake_frame, get_detections=None)
        time.sleep(4)

    print("\n[test] Standalone test done.")
