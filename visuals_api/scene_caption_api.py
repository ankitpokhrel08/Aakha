"""Tier 2 — Scene captioning via the Moondream API (api.moondream.ai).

Drop-in replacement for visuals/scene_caption.py that calls the cloud API
instead of running the model locally.  Same public surface:

    start_scene_caption_thread(get_frame) -> threading.Thread

Requires MOONDREAM_API_KEY in the .env file at the project root.
No heavy model weights — startup is near-instant.
"""
from __future__ import annotations

import base64
import os
import sys
import threading
import time
from typing import Callable, Optional

import numpy as np
import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.bus import event_bus
from shared.events import Event, Priority

# Load .env from the project root (two levels up from this file).
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

_API_KEY = os.getenv("MOONDREAM_API_KEY", "")
_API_URL = "https://api.moondream.ai/v1/caption"
_MODEL   = "moondream3.1-9B-A2B"

_SCENE_DIFF_THRESHOLD = 15.0
_CAPTION_COOLDOWN_S   = 4.0
_LOOP_SLEEP_S         = 0.2
_THUMB_SIZE           = (128, 128)
_REQUEST_TIMEOUT_S    = 10

# Set by voice_trigger to bypass cooldown/diff gate.
_last_caption_requested: bool = False


def _scene_changed(
    current: np.ndarray,
    reference: Optional[np.ndarray],
) -> bool:
    if reference is None:
        return True
    import cv2
    cur_gray = cv2.resize(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), _THUMB_SIZE)
    ref_gray = cv2.resize(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), _THUMB_SIZE)
    diff = float(np.mean(np.abs(cur_gray.astype(np.float32) - ref_gray.astype(np.float32))))
    return diff > _SCENE_DIFF_THRESHOLD


def _frame_to_data_uri(bgr_frame: np.ndarray) -> str:
    """Encode a BGR numpy frame as a JPEG base64 data URI."""
    import cv2
    _, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _call_api(bgr_frame: np.ndarray) -> str:
    """POST the frame to the Moondream caption endpoint and return the caption."""
    if not _API_KEY:
        return "Moondream API key not set."

    payload = {
        "model":     _MODEL,
        "image_url": _frame_to_data_uri(bgr_frame),
        "length":    "short",
        "stream":    False,
    }
    headers = {
        "X-Moondream-Auth": _API_KEY,
        "Content-Type":     "application/json",
    }

    resp = requests.post(_API_URL, json=payload, headers=headers, timeout=_REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    caption = resp.json().get("caption", "").strip()
    if caption and not caption.endswith("."):
        caption += "."
    return caption or "Scene ahead unclear."


def _caption_thread(get_frame: Callable[[], Optional[np.ndarray]]) -> None:
    global _last_caption_requested

    if not _API_KEY:
        print("[scene_caption_api] ERROR: MOONDREAM_API_KEY not found in .env — thread will idle.")
        while True:
            time.sleep(60.0)

    print(f"[scene_caption_api] Ready — using {_MODEL} via API.")

    last_captioned_frame: Optional[np.ndarray] = None
    last_caption_time: float = 0.0

    while True:
        time.sleep(_LOOP_SLEEP_S)

        frame = get_frame()
        if frame is None:
            continue

        on_demand = _last_caption_requested
        if on_demand:
            _last_caption_requested = False

        now = time.time()
        cooldown_elapsed = (now - last_caption_time) >= _CAPTION_COOLDOWN_S

        if not on_demand:
            if not cooldown_elapsed:
                continue
            if not _scene_changed(frame, last_captioned_frame):
                continue

        t0 = time.time()
        try:
            caption = _call_api(frame)
        except Exception as exc:
            print(f"[scene_caption_api] API error: {exc}")
            continue

        elapsed = time.time() - t0
        print(f"[scene_caption_api] caption ({elapsed:.2f}s): {caption!r}")

        event_bus.publish(Event(
            message=caption,
            priority=Priority.LOW,
            type="caption",
            source="scene",
        ))

        last_captioned_frame = frame.copy()
        last_caption_time = time.time()


def start_scene_caption_thread(
    get_frame: Callable[[], Optional[np.ndarray]],
) -> threading.Thread:
    """Start the API-backed scene-captioning daemon thread and return it."""
    t = threading.Thread(
        target=_caption_thread,
        args=(get_frame,),
        name="_scene_caption_api",
        daemon=True,
    )
    t.start()
    return t


if __name__ == "__main__":
    """Standalone test: caption one frame (webcam or TEST_IMAGE env var)."""
    import cv2
    from audio.consumer import start_consumer

    test_image_path = os.environ.get("TEST_IMAGE")
    if test_image_path and os.path.exists(test_image_path):
        frame = cv2.imread(test_image_path)
        print(f"[test] Using image: {test_image_path}")
    else:
        print("[test] No TEST_IMAGE set — grabbing one frame from webcam (index 0)...")
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print("[test] Could not read webcam frame. Set TEST_IMAGE=/path/to/image.jpg")
            sys.exit(1)
        print("[test] Webcam frame captured.")

    print("[test] Starting TTS consumer...")
    start_consumer()

    # Bypass cooldown so the caption fires immediately.
    import visuals_api.scene_caption_api as sc_api
    sc_api._CAPTION_COOLDOWN_S = 0.0

    print("[test] Starting API scene caption thread...")
    start_scene_caption_thread(lambda: frame)

    print("[test] Waiting up to 30s for the API response...")
    time.sleep(30)
    print("[test] Standalone test done.")
