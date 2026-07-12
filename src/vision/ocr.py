"""On-demand OCR via Apple's Vision framework (ocrmac), triggered by keypress/voice.

Vision is native to Apple Silicon (the backend runs on the Mac), needs no model
download, and reads camera photos cleanly at ~40 ms warm; earlier Tesseract/
PaddleOCR attempts were garbled or deadlocked. A background thread reads Enter,
OCRs the latest frame, and publishes the text as a LOW event. _run_ocr() is also
called by the voice "read this" command. Degrades to no-op off macOS.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

from src.core.bus import event_bus
from src.core.events import Event, Priority

# "vision" backend: clean line-level results with per-line confidence.
_FRAMEWORK = "vision"
_infer_lock = threading.Lock()   # Vision's predictor isn't guaranteed thread-safe
_MIN_CONF = 0.4                   # drop low-confidence logo/background fragments

# ocrmac.OCR, resolved lazily; None if ocrmac/Vision is unavailable (non-macOS).
_ocr_cls = None
_ocr_lock = threading.Lock()
_ocr_probed = False


def _get_ocr():
    """Return ocrmac's OCR class (resolved lazily), or None if unavailable —
    no OCR rather than a crashed thread."""
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
    """Run Apple Vision OCR on a BGR frame; return the read text (stripped) or ""."""
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
    # annotations are (text, conf, [x, y, w, h]) with bottom-left origin; sort
    # top-to-bottom then left-to-right so multi-line signs read in natural order.
    kept = [(t, box) for (t, conf, box) in annotations if conf >= _MIN_CONF and t.strip()]
    kept.sort(key=lambda tb: (-round(tb[1][1], 2), tb[1][0]))
    return " ".join(t.strip() for (t, _) in kept).strip()


def _ocr_thread(get_frame: Callable[[], Optional[np.ndarray]]) -> None:
    """Block on stdin; on each Enter press, OCR the latest frame and publish."""
    _get_ocr()   # warm the engine so the first read isn't a cold start

    if not sys.stdin.isatty():
        while True:          # non-interactive (smoke test, pipe) — idle
            time.sleep(60.0)
        return

    print("[ocr] Ready — press Enter to read text in the current frame.")
    while True:
        try:
            line = sys.stdin.readline()
        except (EOFError, OSError):
            time.sleep(1.0)
            continue

        if not line:          # EOF on stdin — idle
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
    """Start the OCR daemon thread and return it. get_frame returns the latest
    BGR frame or None (main provides get_latest_frame)."""
    t = threading.Thread(
        target=_ocr_thread,
        args=(get_frame,),
        name="_ocr",
        daemon=True,
    )
    t.start()
    return t


if __name__ == "__main__":
    # Standalone: OCR an image (TEST_IMAGE=...) and check warm timing < 500 ms.
    from src.audio.consumer import start_consumer

    test_image_path = os.environ.get("TEST_IMAGE")
    if not test_image_path or not os.path.exists(test_image_path):
        print("Usage: TEST_IMAGE=/path/to/image.jpg python -m src.vision.ocr")
        sys.exit(1)

    frame = cv2.imread(test_image_path)
    if frame is None:
        print(f"Could not read image: {test_image_path}")
        sys.exit(1)

    print(f"[test] Loaded image: {test_image_path} {frame.shape}")
    start_consumer()
    _get_ocr()        # warm so we time warm inference, not the one-time load
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
