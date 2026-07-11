"""Tier 3 — On-demand OCR via Apple's Vision framework (ocrmac), by keypress/voice.

Migrated from Tesseract, then from PaddleOCR. Tesseract produced garbled output on
real-world camera photos (dark backgrounds, coloured text, signs, badges) that TTS
would read literally ("Ri q ricl I P A N T…"). PaddleOCR PP-OCRv4 read cleanly but
`paddlepaddle==2.6.2` deadlocks in its C++ predictor on Apple Silicon (inference
never returns) — see ocr_migration.md.

Apple's Vision framework is native to Apple Silicon (the laptop backend always runs
on the Mac; the phone only streams frames), needs no model download, and reads such
images cleanly at ~40 ms warm. We reach it through the tiny `ocrmac` wrapper, whose
only non-stdlib dep is pyobjc-framework-Vision (already installed).

A background thread listens on stdin. When the user presses Enter it grabs the
latest frame, runs OCR, and publishes the extracted text as a LOW priority event.

Public API (unchanged, so main.py / voice_trigger need no edits):
    start_ocr_thread(get_frame) -> threading.Thread
        get_frame: Callable[[], np.ndarray | None] — latest BGR frame or None.
    _run_ocr(frame) -> str — read text from a BGR frame (used by voice "read this").
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
from PIL import Image

from shared.bus import event_bus
from shared.events import Event, Priority

# The "accurate" Vision backend. "vision" returns clean line-level results with real
# per-line confidence; "livetext" is slower and fragments text into single tokens.
_FRAMEWORK = "vision"
# Vision's predictor isn't guaranteed thread-safe and OCR calls are rare +
# user-triggered, so serialise inference with a cheap lock.
_infer_lock = threading.Lock()
# Confidence floor: drop low-confidence logo/background fragments while keeping
# real text. Vision reports 0..1 per line on the "vision" backend.
_MIN_CONF = 0.4

# Resolved lazily: the ocrmac.OCR callable, or None if ocrmac/Vision is unavailable
# (e.g. non-macOS). None means "no OCR" rather than a crashed thread.
_ocr_cls = None
_ocr_lock = threading.Lock()
_ocr_probed = False


def _get_ocr():
    """Return ocrmac's OCR class, or None if Vision/ocrmac isn't available.

    Resolves lazily on first call. Returns None (with a hint) instead of raising, so
    a machine without ocrmac/Vision simply gets no OCR rather than a crashed thread.
    """
    global _ocr_cls, _ocr_probed
    if _ocr_probed:
        return _ocr_cls
    with _ocr_lock:
        if not _ocr_probed:
            try:
                from ocrmac import ocrmac
                _ocr_cls = ocrmac.OCR
            except Exception as exc:
                print(f"[ocr] Apple Vision OCR unavailable ({exc}); OCR disabled. "
                      f"On macOS install with: pip install ocrmac")
                _ocr_cls = None
            _ocr_probed = True
    return _ocr_cls


def _run_ocr(frame: np.ndarray) -> str:
    """Run Apple Vision OCR on a BGR frame; return the read text (stripped) or "".

    Passes the frame straight through as a PIL image (no temp file needed). Lines are
    returned top-to-bottom, left-to-right so the text reads in natural order.
    """
    ocr_cls = _get_ocr()
    if ocr_cls is None or frame is None:
        return ""
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        with _infer_lock:
            annotations = ocr_cls(pil, framework=_FRAMEWORK).recognize()
    except Exception as exc:
        print(f"[ocr] inference failed: {exc}")
        return ""
    if not annotations:
        return ""
    # Each annotation is (text, confidence, [x, y, w, h]) with normalised, bottom-left
    # origin coords — higher y is higher up the image. Sort top-to-bottom then
    # left-to-right so multi-line signs read in the order a person would read them.
    kept = [(t, box) for (t, conf, box) in annotations if conf >= _MIN_CONF and t.strip()]
    kept.sort(key=lambda tb: (-round(tb[1][1], 2), tb[1][0]))
    return " ".join(t.strip() for (t, _) in kept).strip()


def _ocr_thread(get_frame: Callable[[], Optional[np.ndarray]]) -> None:
    """Block on stdin; on each Enter press, OCR the latest frame and publish."""
    # Warm the engine at thread start so the first "read this" isn't a cold start.
    _get_ocr()

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
    """Standalone test: OCR a provided image file and confirm warm timing <500ms.

    Usage:
        TEST_IMAGE=/path/to/image.jpg .venv/bin/python visuals/ocr.py
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

    # Warm the engine first so we time WARM inference (not the one-time load).
    print("[test] Loading Apple Vision OCR (one-time)...")
    _get_ocr()
    _run_ocr(frame)

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
