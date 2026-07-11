"""Tier 3 — One-shot Vosk transcription + 4-command keyword dispatch.

Updated scope (per PROGRESS.md):
  Wake-word continuous listening is CUT. Instead, Dev 3 sends a short WAV
  clip over WiFi when the user presses the volume button on their phone.
  This module receives that clip (as raw bytes or a file path), runs Vosk
  on it, matches against 4 fixed phrases, and dispatches the right action.

The 4 commands and their actions:
  "what am I holding"  → most prominent object over a ~2 s scan → publish
  "read this"          → trigger OCR on current frame → publish
  "describe the scene" → trigger Moondream2 caption on demand → publish
  "repeat that"        → re-speak last TTS message → publish type="repeat"

Public API:
    transcribe_clip(audio_bytes: bytes) -> str
        Run Vosk on raw 16kHz mono PCM bytes; return transcribed text.

    dispatch_command(transcript: str, get_frame, get_detections=None) -> None
        Match transcript to the 4 commands and publish the right Event(s).

    start_voice_trigger_thread(get_frame, get_detections=None) -> threading.Thread
        Background thread that monitors a shared clip queue. Dev 3 drops
        audio clips into `clip_queue`; this thread processes them.

    clip_queue: queue.Queue
        Put raw 16kHz mono PCM bytes here to trigger transcription.
        Dev 3's WiFi transport should put received clips here directly.

Environment:
    VOSK_MODEL_PATH — path to a downloaded Vosk model directory.
                      Default: <repo_root>/vosk-model-small-en-us-0.15
                      Download: https://alphacephei.com/vosk/models
                        → vosk-model-small-en-us-0.15.zip (~40 MB)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import queue
import threading
import time
from typing import Callable, List, Optional

import numpy as np

from shared.bus import event_bus
from shared.events import Event, Priority

# ---------------------------------------------------------------------------
# Public clip queue — Dev 3 puts raw 16kHz mono PCM bytes here.
# ---------------------------------------------------------------------------
clip_queue: queue.Queue = queue.Queue()

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vosk-model-small-en-us-0.15",
)

# ---------------------------------------------------------------------------
# Keyword matching — keep vocabulary small for outdoor STT accuracy.
# Each entry: (keywords_that_must_appear, command_key)
# First match wins; order matters.
# ---------------------------------------------------------------------------
_COMMANDS = [
    ({"holding"},                     "holding"),
    ({"read"},                        "read"),
    ({"describe"},                    "describe"),
    ({"repeat"},                      "repeat"),
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
    """Return the command key for the first keyword set that matches."""
    words = set(transcript.lower().split())
    for keywords, key in _COMMANDS:
        if keywords.issubset(words):
            return key
    return None


# --- "what am I holding" — scan a short window, not one instant (V5) ----------
# The old code sampled a SINGLE frame: if the hand happened to be missing (or the
# object was between YOLO detections) in that exact frame, it wrongly answered
# "I can't see your hand" even though the object was plainly visible. Hand
# detection is dropped entirely — it was unreliable (kept failing to locate the
# hand). Instead we assume the object held up to the camera is simply the most
# PROMINENT one (closest => biggest box, and near the frame centre), scanned over
# ~2 s and decided by majority vote so a single-frame miss can't flip the result.
_HOLD_SCAN_SECONDS = 2.0
_HOLD_SAMPLE_GAP = 0.12     # ~8 samples/sec of the pushed stream
# Classes that are never a "held object" — the user / bystanders, so a body in
# view can't be reported as what he's holding.
_NOT_HELD = {"person"}


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
    """Decide what the user is holding by watching the live frame for ~`duration`
    s and reporting the most prominent object over the window (V5).

    No hand detection: the held object is taken to be the closest/most-central
    thing in view, majority-voted across frames so a one-frame miss can't flip
    the answer.
    """
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
        return "I don't see anything you might be holding. Hold it up to the camera."
    return f"You are probably holding a {votes.most_common(1)[0][0]}."


def dispatch_command(
    transcript: str,
    get_frame: Callable[[], Optional[np.ndarray]],
    get_detections: Optional[Callable[[], Optional[List[dict]]]] = None,
) -> None:
    """Match transcript → command → publish the appropriate Event(s).

    This is pure dispatch logic with no I/O of its own; the background thread
    calls it after transcription. Can also be called directly in tests.
    """
    print(f"[voice_trigger] transcript: {transcript!r}")
    cmd = _match_command(transcript)
    if cmd is None:
        # V4: distinguish "heard nothing" from "heard an unknown command".
        #   - empty transcript -> the user held but said nothing intelligible;
        #     just a brief cue (don't dump the whole command list every time).
        #   - real speech, no match -> offer the list of available commands.
        # Both are terminal replies, so navigation resumes once they finish.
        if not transcript.strip():
            print("[voice_trigger] Nothing heard.")
            message = "I didn't catch that."
        else:
            print("[voice_trigger] No command matched.")
            message = ("I didn't understand. You can say: what am I holding, "
                       "read this, describe the scene, or repeat that.")
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
        # Trigger OCR on the current frame inline (Tesseract is fast enough).
        frame = get_frame()
        if frame is None:
            event_bus.publish(Event(
                message="Camera not ready. Try again in a moment.",
                priority=Priority.NORMAL,
                type="ocr",
                source="voice",
            ))
            return
        from visuals.ocr import _run_ocr
        text = _run_ocr(frame)
        message = f"Text reads: {text}" if text else "No text found in the frame."
        event_bus.publish(Event(
            message=message,
            priority=Priority.NORMAL,
            type="ocr",
            source="voice",
        ))

    elif cmd == "describe":
        # Trigger Moondream2 caption on demand by clearing the cooldown.
        # scene_caption thread will pick it up on its next wake cycle.
        import visuals.scene_caption as sc
        sc._last_caption_requested = True  # sentinel checked in scene_caption
        event_bus.publish(Event(
            message="Describing the scene now.",
            priority=Priority.NORMAL,
            type="caption_request",
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
            message = "Could not check what you are holding."
        event_bus.publish(Event(
            message=message,
            priority=Priority.NORMAL,
            type="held_object",
            source="voice",
        ))


# ---------------------------------------------------------------------------
# Background thread — monitors clip_queue, transcribes, dispatches.
# ---------------------------------------------------------------------------
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
            # A dispatch failure (bad clip, OCR/object-scan error) must never kill
            # this thread — if it dies, every future voice command is silent,
            # which is the exact A7 symptom. Log, speak a fallback, keep going.
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
    """Start the voice-trigger daemon thread and return it.

    Args:
        get_frame:       Latest BGR frame callable (same as scene_caption).
        get_detections:  Optional callable returning Tier 1 YOLO detections
                         [{"label": str, "bbox": (x1,y1,x2,y2)}, ...].
                         Pass None until dev1/detection merges.
    """
    t = threading.Thread(
        target=_voice_trigger_thread,
        args=(get_frame, get_detections),
        name="_voice_trigger",
        daemon=True,
    )
    t.start()
    return t


if __name__ == "__main__":
    """Standalone test: simulate a phone clip arriving on clip_queue.

    Usage:
        env_aakha/bin/python audio/voice_trigger.py
        VOSK_MODEL_PATH=/path/to/model env_aakha/bin/python audio/voice_trigger.py

    Requires the Vosk model to be present. Simulates each of the 4 commands
    by injecting pre-recorded phrases as PCM bytes (silence used here — you
    can replace with a real WAV file).
    """
    import struct
    from audio.consumer import start_consumer

    model_path = os.environ.get("VOSK_MODEL_PATH", _DEFAULT_MODEL_PATH)
    if not os.path.isdir(model_path):
        print(f"[test] Vosk model not found at: {model_path}")
        print("[test] Download vosk-model-small-en-us-0.15.zip from:")
        print("[test]   https://alphacephei.com/vosk/models")
        sys.exit(1)

    print("[test] Starting TTS consumer...")
    start_consumer()
    time.sleep(0.3)

    # Test keyword dispatch directly (bypassing Vosk transcription).
    print("[test] Testing keyword dispatch directly...")

    def _fake_frame():
        return np.zeros((480, 640, 3), dtype=np.uint8)

    for phrase in ["repeat that", "read this", "describe the scene"]:
        print(f"\n[test] Dispatching: {phrase!r}")
        dispatch_command(phrase, _fake_frame, get_detections=None)
        time.sleep(4)

    print("\n[test] Standalone test done.")
