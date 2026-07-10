"""Tier 2 — Moondream2 scene captioning on a background thread.

Fires on scene-change (not every frame) and publishes LOW priority captions
to the event bus. Never blocks the Tier 1 vision loop.

Public API:
    start_scene_caption_thread(get_frame) -> threading.Thread
        get_frame: Callable[[], np.ndarray | None]
            Returns the latest BGR frame from the vision thread, or None if
            no frame is available yet. main.py provides this once Tier 1 lands;
            the standalone test below passes a static image lambda.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
import time
from typing import Callable, Optional

import numpy as np

from shared.bus import event_bus
from shared.events import Event, Priority

# --- tuning constants -------------------------------------------------------
_SCENE_DIFF_THRESHOLD = 15.0   # mean-absolute-diff below which we skip captioning
_CAPTION_COOLDOWN_S = 4.0      # minimum seconds between captions
_LOOP_SLEEP_S = 0.2            # how long the thread sleeps between frame checks
_THUMB_SIZE = (128, 128)       # size used for the cheap diff computation
# ---------------------------------------------------------------------------

# Sentinel set by voice_trigger when the user says "describe the scene".
# The caption thread checks and clears it to bypass the cooldown/diff gate.
_last_caption_requested: bool = False


def _scene_changed(
    current: np.ndarray,
    reference: Optional[np.ndarray],
) -> bool:
    """Return True when the current frame differs enough from the reference frame."""
    if reference is None:
        return True
    import cv2  # imported lazily so the module is importable without cv2 at top-level
    cur_gray = cv2.resize(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), _THUMB_SIZE)
    ref_gray = cv2.resize(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), _THUMB_SIZE)
    diff = float(np.mean(np.abs(cur_gray.astype(np.float32) - ref_gray.astype(np.float32))))
    return diff > _SCENE_DIFF_THRESHOLD


def _load_moondream():
    """Load Moondream2 model. Called once inside the thread."""
    from transformers import AutoModelForCausalLM

    print("[scene_caption] Loading Moondream2 — this may take a moment...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-06-21",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[scene_caption] Moondream2 loaded in {time.time() - t0:.1f}s")
    return model


def _run_inference(model, bgr_frame: np.ndarray) -> str:
    """Run Moondream2 on a BGR numpy frame; return a short spoken caption."""
    from PIL import Image

    rgb = bgr_frame[:, :, ::-1]
    pil_img = Image.fromarray(rgb.astype(np.uint8))

    caption = model.caption(pil_img, length="short")["caption"].strip()
    if caption and not caption.endswith("."):
        caption += "."
    return caption or "Scene ahead unclear."


def _caption_thread(get_frame: Callable[[], Optional[np.ndarray]]) -> None:
    """Background loop: detect scene changes, caption, publish to bus."""
    global _last_caption_requested
    try:
        model = _load_moondream()
    except Exception as exc:
        print(
            f"[scene_caption] Model failed to load: {exc}\n"
            f"  If you see 'all_tied_weights_keys', run:\n"
            f"    env_aakha/bin/pip install 'transformers==4.52.4'\n"
            f"  Thread will idle until restarted with a compatible transformers."
        )
        while True:
            time.sleep(60.0)

    last_captioned_frame: Optional[np.ndarray] = None
    last_caption_time: float = 0.0

    while True:
        time.sleep(_LOOP_SLEEP_S)

        frame = get_frame()
        if frame is None:
            continue

        # On-demand request from voice_trigger ("describe the scene") bypasses
        # both the cooldown and the scene-diff gate.
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

        # Run captioning.
        t0 = time.time()
        caption = _run_inference(model, frame)
        elapsed = time.time() - t0
        print(f"[scene_caption] caption ({elapsed:.2f}s): {caption!r}")

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
    """Start the scene-captioning daemon thread and return it.

    Args:
        get_frame: Callable that returns the latest BGR frame or None.
                   main.py provides this; tests pass a lambda with a static image.
    """
    t = threading.Thread(
        target=_caption_thread,
        args=(get_frame,),
        name="_scene_caption",
        daemon=True,
    )
    t.start()
    return t


if __name__ == "__main__":
    """Standalone test: caption a single test image (or first webcam frame).
    Also starts the TTS consumer so the caption is actually spoken.
    """
    import cv2
    from audio.consumer import start_consumer

    # --- get a test frame ---------------------------------------------------
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

    # Start TTS consumer so we hear the result.
    print("[test] Starting TTS consumer...")
    start_consumer()

    # Pass a lambda that always returns the same frame (simulates a static scene).
    # Override the cooldown so the first caption fires immediately.
    import visuals.scene_caption as sc
    sc._CAPTION_COOLDOWN_S = 0.0

    print("[test] Starting scene caption thread...")
    start_scene_caption_thread(lambda: frame)

    print("[test] Waiting up to 120s for Moondream2 to load and caption the frame...")
    time.sleep(120)
    print("[test] Standalone test done.")
