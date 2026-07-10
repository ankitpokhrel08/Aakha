"""Tier 3 — On-demand OCR via Tesseract, triggered by a keypress.

A background thread listens on stdin. When the user presses Enter (or any
configured key), it grabs the latest frame, runs pytesseract, and publishes
the extracted text as a LOW priority event.

Public API:
    start_ocr_thread(get_frame) -> threading.Thread
        get_frame: Callable[[], np.ndarray | None]
            Same contract as scene_caption — returns the latest BGR frame or
            None. Provided by main.py via get_latest_frame().
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np
import pytesseract

from shared.bus import event_bus
from shared.events import Event, Priority

# PSM 11: sparse text — no layout analysis, finds text anywhere in the frame.
# Correct for camera frames with signs/labels rather than full-page documents.
_TESS_CONFIG = "--oem 1 --psm 11"


def _preprocess(frame: np.ndarray) -> np.ndarray:
    """Grayscale + Otsu threshold — boosts OCR accuracy on natural images.

    Caps width at 1000px before processing; beyond that Tesseract slows down
    without meaningful accuracy gains on camera-sourced text.
    """
    h, w = frame.shape[:2]
    if w > 800:
        scale = 1000 / w
        frame = cv2.resize(frame, (800, int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def _run_ocr(frame: np.ndarray) -> str:
    """Run Tesseract on a BGR frame; return stripped text or empty string."""
    processed = _preprocess(frame)
    text = pytesseract.image_to_string(processed, config=_TESS_CONFIG)
    return text.strip()


def _ocr_thread(get_frame: Callable[[], Optional[np.ndarray]]) -> None:
    """Block on stdin; on each Enter press, OCR the latest frame and publish."""
    if not sys.stdin.isatty():
        # Non-interactive context (smoke test, pipe) — idle silently.
        # Keypress triggering only makes sense when a human is at the terminal.
        while True:
            time.sleep(60.0)
        return

    print("[ocr] Ready — press Enter to read text in the current frame.")
    while True:
        try:
            line = sys.stdin.readline()
        except (EOFError, OSError):
            time.sleep(1.0)
            continue

        if not line:
            # EOF on stdin — stop spinning, just idle.
            time.sleep(1.0)
            continue

        frame = get_frame()
        if frame is None:
            print("[ocr] No frame available yet — try again once the camera is live.")
            continue

        t0 = time.time()
        text = _run_ocr(frame)
        elapsed = time.time() - t0
        print(f"[ocr] ({elapsed*1000:.0f}ms): {text!r}")

        if not text:
            message = "No text found in frame."
        else:
            message = f"Text reads: {text}"

        event_bus.publish(Event(
            message=message,
            priority=Priority.LOW,
            type="ocr",
            source="ocr",
        ))


def start_ocr_thread(
    get_frame: Callable[[], Optional[np.ndarray]],
) -> threading.Thread:
    """Start the OCR daemon thread and return it.

    Args:
        get_frame: Callable returning the latest BGR frame or None.
                   main.py provides get_latest_frame(); tests pass a lambda.
    """
    t = threading.Thread(
        target=_ocr_thread,
        args=(get_frame,),
        name="_ocr",
        daemon=True,
    )
    t.start()
    return t


if __name__ == "__main__":
    """Standalone test: OCR a provided image file and confirm timing <500ms.

    Usage:
        TEST_IMAGE=/path/to/image.jpg env_aakha/bin/python visuals/ocr.py
    """
    from audio.consumer import start_consumer

    test_image_path = os.environ.get("TEST_IMAGE")
    if not test_image_path or not os.path.exists(test_image_path):
        print("Usage: TEST_IMAGE=/path/to/image.jpg python visuals/ocr.py")
        print("Image should contain some readable text for a meaningful test.")
        sys.exit(1)

    frame = cv2.imread(test_image_path)
    if frame is None:
        print(f"Could not read image: {test_image_path}")
        sys.exit(1)

    print(f"[test] Loaded image: {test_image_path} {frame.shape}")
    print("[test] Starting TTS consumer...")
    start_consumer()

    # Run OCR directly (bypassing keypress) and time it.
    print("[test] Running OCR...")
    t0 = time.time()
    text = _run_ocr(frame)
    elapsed = (time.time() - t0) * 1000
    print(f"[test] OCR completed in {elapsed:.0f}ms")
    print(f"[test] Extracted text: {text!r}")

    if elapsed > 500:
        print(f"[test] WARNING: OCR took {elapsed:.0f}ms — over 500ms target")
    else:
        print(f"[test] PASS: OCR within 500ms target")

    # Publish and let TTS speak it.
    message = f"Text reads: {text}" if text else "No text found in frame."
    event_bus.publish(Event(
        message=message,
        priority=Priority.LOW,
        type="ocr",
        source="ocr",
    ))
    print("[test] Event published. Waiting 10s for TTS...")
    time.sleep(10)
    print("[test] Standalone test done.")
