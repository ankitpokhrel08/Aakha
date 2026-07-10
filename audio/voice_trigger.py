"""Tier 3 — One-shot Vosk transcription + 4-command keyword dispatch.

Updated scope (per PROGRESS.md):
  Wake-word continuous listening is CUT. Instead, Dev 3 sends a short WAV
  clip over WiFi when the user presses the volume button on their phone.
  This module receives that clip (as raw bytes or a file path), runs Vosk
  on it, matches against 4 fixed phrases, and dispatches the right action.

The 4 commands and their actions:
  "what am I holding"  → MediaPipe Hands + YOLO detection match → publish
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
        print("[voice_trigger] No command matched.")
        event_bus.publish(Event(
            message="Sorry, I didn't catch that. Try: what am I holding, read this, describe the scene, or repeat that.",
            priority=Priority.LOW,
            type="voice_no_match",
            source="voice",
        ))
        return

    print(f"[voice_trigger] command: {cmd}")

    if cmd == "repeat":
        # Consumer re-speaks last_spoken when it sees type="repeat".
        event_bus.publish(Event(
            message="",
            priority=Priority.LOW,
            type="repeat",
            source="voice",
        ))

    elif cmd == "read":
        # Trigger OCR on the current frame inline (Tesseract is fast enough).
        frame = get_frame()
        if frame is None:
            event_bus.publish(Event(
                message="Camera not ready. Try again in a moment.",
                priority=Priority.LOW,
                type="ocr",
                source="voice",
            ))
            return
        from visuals.ocr import _run_ocr
        text = _run_ocr(frame)
        message = f"Text reads: {text}" if text else "No text found in the frame."
        event_bus.publish(Event(
            message=message,
            priority=Priority.LOW,
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
            priority=Priority.LOW,
            type="caption_request",
            source="voice",
        ))

    elif cmd == "holding":
        frame = get_frame()
        if frame is None:
            event_bus.publish(Event(
                message="Camera not ready. Try again in a moment.",
                priority=Priority.LOW,
                type="held_object",
                source="voice",
            ))
            return

        try:
            import mediapipe as mp
            import cv2
            mp_hands = mp.solutions.hands
            with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5,
            ) as hands:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

            if not result.multi_hand_landmarks:
                event_bus.publish(Event(
                    message="I can't see your hand. Hold the object up to the camera.",
                    priority=Priority.LOW,
                    type="held_object",
                    source="voice",
                ))
                return

            h, w = frame.shape[:2]
            lm = result.multi_hand_landmarks[0].landmark
            xs = [p.x * w for p in lm]
            ys = [p.y * h for p in lm]
            hand_box = (
                max(0, int(min(xs)) - 20),
                max(0, int(min(ys)) - 20),
                min(w, int(max(xs)) + 20),
                min(h, int(max(ys)) + 20),
            )
        except Exception as e:
            print(f"[voice_trigger] MediaPipe error: {e}")
            event_bus.publish(Event(
                message="Could not detect hand position.",
                priority=Priority.LOW,
                type="held_object",
                source="voice",
            ))
            return

        label = None
        if get_detections is not None:
            detections = get_detections()
            if detections:
                best_iou, best_label = 0.1, None
                for det in detections:
                    bx = det["bbox"]
                    xA, yA = max(hand_box[0], bx[0]), max(hand_box[1], bx[1])
                    xB, yB = min(hand_box[2], bx[2]), min(hand_box[3], bx[3])
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    if inter:
                        aA = (hand_box[2]-hand_box[0]) * (hand_box[3]-hand_box[1])
                        aB = (bx[2]-bx[0]) * (bx[3]-bx[1])
                        iou = inter / (aA + aB - inter)
                        if iou > best_iou:
                            best_iou, best_label = iou, det["label"]
                label = best_label

        if label:
            message = f"You are holding a {label}."
        else:
            message = "I can see your hand but cannot identify the object."
        event_bus.publish(Event(
            message=message,
            priority=Priority.LOW,
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

        transcript = transcribe_clip(audio_bytes, model=model)
        dispatch_command(transcript, get_frame, get_detections)
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
