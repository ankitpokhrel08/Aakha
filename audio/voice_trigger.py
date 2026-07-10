"""Tier 3 — Vosk wake-word detection + MediaPipe Hands for "what am I holding?".

Flow:
    1. Vosk listens continuously on the microphone.
    2. On hearing the wake phrase "what am I holding" (or just "holding"),
       it grabs the latest frame from the vision thread.
    3. MediaPipe Hands finds the hand bounding box in that frame.
    4. If get_detections() is provided (Tier 1 merged), the nearest YOLO
       detection box is matched to the hand region and named.
    5. The answer is published as a LOW priority event and spoken by TTS.

Public API:
    start_voice_trigger_thread(get_frame, get_detections=None) -> threading.Thread
        get_frame:       Callable[[], np.ndarray | None]  — same as scene_caption
        get_detections:  Callable[[], list[dict] | None]  — optional; provided by
                         the Tier 1 vision thread once dev1/detection merges.
                         Each dict: {"label": str, "bbox": (x1, y1, x2, y2)}

Environment:
    VOSK_MODEL_PATH — path to a downloaded Vosk model directory.
                      Default: ./vosk-model-small-en-us-0.15
                      Download: https://alphacephei.com/vosk/models
                        → vosk-model-small-en-us-0.15.zip (~40 MB)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import queue
import threading
import time
from typing import Callable, List, Optional

import numpy as np

from shared.bus import event_bus
from shared.events import Event, Priority

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vosk-model-small-en-us-0.15",
)
_SAMPLE_RATE = 16000          # Hz — Vosk requires 16kHz mono
_BLOCK_SIZE = 8000            # samples per sounddevice callback (~0.5s at 16kHz)
_WAKE_WORDS = {"holding", "hold"}   # trigger on any of these words in transcript

# How many pixels of IOU overlap needed to "match" a hand bbox to a YOLO box.
_OVERLAP_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _iou(boxA: tuple, boxB: tuple) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(aA + aB - inter)


def _hand_bbox_from_frame(
    frame: np.ndarray,
    hands,
) -> Optional[tuple]:
    """Run MediaPipe Hands on a BGR frame; return (x1,y1,x2,y2) or None."""
    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None

    h, w = frame.shape[:2]
    # Use the first detected hand.
    lm = result.multi_hand_landmarks[0].landmark
    xs = [p.x * w for p in lm]
    ys = [p.y * h for p in lm]
    pad = 20  # px padding around the tight hand bbox
    return (
        max(0, int(min(xs)) - pad),
        max(0, int(min(ys)) - pad),
        min(w, int(max(xs)) + pad),
        min(h, int(max(ys)) + pad),
    )


def _match_hand_to_detection(
    hand_box: tuple,
    detections: list,
) -> Optional[str]:
    """Return label of the YOLO detection that best overlaps the hand box."""
    best_iou = _OVERLAP_THRESHOLD
    best_label = None
    for det in detections:
        score = _iou(hand_box, det["bbox"])
        if score > best_iou:
            best_iou = score
            best_label = det["label"]
    return best_label


# ---------------------------------------------------------------------------
# Core thread
# ---------------------------------------------------------------------------
def _voice_trigger_thread(
    get_frame: Callable[[], Optional[np.ndarray]],
    get_detections: Optional[Callable[[], Optional[List[dict]]]],
) -> None:
    model_path = os.environ.get("VOSK_MODEL_PATH", _DEFAULT_MODEL_PATH)
    if not os.path.isdir(model_path):
        print(
            f"[voice_trigger] Vosk model not found at: {model_path}\n"
            f"  Download vosk-model-small-en-us-0.15.zip from:\n"
            f"  https://alphacephei.com/vosk/models\n"
            f"  Unzip into the repo root so the path above exists.\n"
            f"  Thread will idle until model is present."
        )
        while not os.path.isdir(model_path):
            time.sleep(5.0)

    try:
        import vosk
        import sounddevice as sd
        import mediapipe as mp
    except ImportError as e:
        print(f"[voice_trigger] Missing dependency: {e} — thread exiting.")
        return

    print("[voice_trigger] Loading Vosk model...")
    vosk.SetLogLevel(-1)  # suppress Vosk's verbose output
    model = vosk.Model(model_path)
    recognizer = vosk.KaldiRecognizer(model, _SAMPLE_RATE)
    print("[voice_trigger] Vosk ready. Say 'what am I holding' to trigger.")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,   # we process individual frames, not video
        max_num_hands=1,
        min_detection_confidence=0.5,
    )

    audio_q: queue.Queue = queue.Queue()

    def _audio_callback(indata, frames, time_info, status):
        audio_q.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=_SAMPLE_RATE,
        blocksize=_BLOCK_SIZE,
        dtype="int16",
        channels=1,
        callback=_audio_callback,
    ):
        while True:
            data = audio_q.get()
            if not recognizer.AcceptWaveform(data):
                # Partial result — check for wake words mid-stream.
                partial = json.loads(recognizer.PartialResult()).get("partial", "")
            else:
                partial = json.loads(recognizer.Result()).get("text", "")

            if not any(w in partial.lower() for w in _WAKE_WORDS):
                continue

            # Wake word detected — process frame.
            print(f"[voice_trigger] Wake word detected: {partial!r}")
            frame = get_frame()
            if frame is None:
                event_bus.publish(Event(
                    message="Camera not ready yet.",
                    priority=Priority.LOW,
                    type="held_object",
                    source="voice",
                ))
                continue

            hand_box = _hand_bbox_from_frame(frame, hands)
            if hand_box is None:
                event_bus.publish(Event(
                    message="I cannot see your hand. Hold the object up to the camera.",
                    priority=Priority.LOW,
                    type="held_object",
                    source="voice",
                ))
                continue

            # Try to match hand to a YOLO detection if Tier 1 is live.
            label = None
            if get_detections is not None:
                detections = get_detections()
                if detections:
                    label = _match_hand_to_detection(hand_box, detections)

            if label:
                message = f"You are holding a {label}."
            else:
                message = "I can see your hand but cannot identify the object. Try holding it steadier."

            print(f"[voice_trigger] Answer: {message!r}")
            event_bus.publish(Event(
                message=message,
                priority=Priority.LOW,
                type="held_object",
                source="voice",
            ))


def start_voice_trigger_thread(
    get_frame: Callable[[], Optional[np.ndarray]],
    get_detections: Optional[Callable[[], Optional[List[dict]]]] = None,
) -> threading.Thread:
    """Start the voice-trigger daemon thread and return it.

    Args:
        get_frame:       Latest BGR frame callable (same as scene_caption).
        get_detections:  Optional callable returning a list of Tier 1 detections
                         [{"label": str, "bbox": (x1,y1,x2,y2)}, ...].
                         Pass None until dev1/detection merges; the thread will
                         still answer "I can see your hand" without a label.
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
    """Standalone test: say 'what am I holding' while holding something to the camera.
    Uses webcam. Requires the Vosk model to be present.

    Usage:
        env_aakha/bin/python audio/voice_trigger.py
        VOSK_MODEL_PATH=/path/to/model env_aakha/bin/python audio/voice_trigger.py
    """
    import cv2
    from audio.consumer import start_consumer

    model_path = os.environ.get("VOSK_MODEL_PATH", _DEFAULT_MODEL_PATH)
    if not os.path.isdir(model_path):
        print(f"[test] Vosk model not found at: {model_path}")
        print("[test] Download vosk-model-small-en-us-0.15.zip from:")
        print("[test]   https://alphacephei.com/vosk/models")
        print("[test] Unzip into the repo root and re-run.")
        sys.exit(1)

    print("[test] Starting TTS consumer...")
    start_consumer()

    # Static frame from webcam.
    cap = cv2.VideoCapture(0)
    _latest_frame = None
    _frame_lock = threading.Lock()

    def _update_frame():
        global _latest_frame
        while True:
            ret, frame = cap.read()
            if ret:
                with _frame_lock:
                    _latest_frame = frame
            time.sleep(0.033)  # ~30fps

    def _get_frame():
        with _frame_lock:
            return _latest_frame

    cam_thread = threading.Thread(target=_update_frame, daemon=True)
    cam_thread.start()
    time.sleep(0.5)  # let webcam warm up

    print("[test] Starting voice trigger thread...")
    start_voice_trigger_thread(_get_frame, get_detections=None)

    print("[test] Say 'what am I holding' to test. Running for 60s...")
    time.sleep(60)
    cap.release()
    print("[test] Standalone test done.")
