"""Free-space — monocular-depth wall / dead-end detector (background thread).

YOLO can't see walls (not a COCO class), so an empty corridor is wrongly called
"path is clear". Depth Anything V2-small (ONNX) estimates relative depth on a
background thread at ~5 fps. Verdict = near_fraction(): how much of the "a few
steps ahead" band is as near as the ground at your feet (~0 open, 0.2-0.5 wall),
smoothed with EMA + hysteresis. Glass is a known weak spot for all passive
vision, so treat it as best-effort.

Degrades gracefully: without onnxruntime or the model, `available` is False and
`blocked` stays False, so the pipeline runs exactly as before.
"""
from __future__ import annotations

import math
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Depth Anything V2-small ONNX; gitignored, downloaded per machine on first run.
MODEL_ONNX = "depth_anything_v2_vits.onnx"
MODEL_URL = ("https://huggingface.co/onnx-community/depth-anything-v2-small/"
             "resolve/main/onnx/model.onnx")
INPUT_SIZE = 378                 # multiple of 14; ~5 fps on CPU, enough structure
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet (DPT preprocessing)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

# near-fraction verdict thresholds (see module docstring / calibration)
ON_THRESHOLD = 0.15              # EMA >= this -> blocked
OFF_THRESHOLD = 0.07             # EMA <= this -> clear again (hysteresis)
TARGET_FPS = 5.0                 # background inference cadence
# Time-constant EMA (alpha = 1 - exp(-dt/TAU)): rate-independent smoothing so the
# verdict fires at the same moment whether depth samples at 2, 3 or 5 Hz.
TAU = 0.46
EMA_ALPHA = 0.35                 # legacy; used by the per-frame overlay tool


def ensure_depth_model(path: str = MODEL_ONNX, url: str = MODEL_URL) -> str:
    """Return the depth ONNX path, downloading it once (~94 MB) if absent.
    Raises on a failed download so the caller can degrade gracefully."""
    if not Path(path).exists():
        print(f"[depth] downloading {path} (~94 MB, one-time)...")
        urllib.request.urlretrieve(url, path)
        print(f"[depth] saved {path}")
    return path


class DepthEstimator:
    """onnxruntime wrapper around Depth Anything V2-small. Raises at construction
    if onnxruntime or the model is missing, so the caller can disable the monitor."""

    def __init__(self, model_path: str = MODEL_ONNX, size: int = INPUT_SIZE) -> None:
        import onnxruntime as ort            # optional dependency

        ensure_depth_model(model_path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2          # be polite to the YOLO session
        # CPU beats CoreML here (the graph fragments into ~100 partitions, ~5x
        # slower). Measured on M2.
        self.session = ort.InferenceSession(
            model_path, so, providers=["CPUExecutionProvider"])
        self.size = size
        self._name = self.session.get_inputs()[0].name

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        """Return the relative-depth map (size x size, float32, larger = nearer)."""
        rgb = cv2.cvtColor(cv2.resize(bgr, (self.size, self.size)),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose((rgb - _MEAN) / _STD, (2, 0, 1))[None].astype(np.float32)
        return self.session.run(None, {self._name: x})[0][0]


def near_fraction(depth: np.ndarray) -> float:
    """Blockage score in [0, 1]: how much of the corridor 'ahead' band is as near
    as the ground at your feet. ~0 = open path receding away; high = wall ahead."""
    h, w = depth.shape
    col = depth[:, int(w * 0.30):int(w * 0.70)]      # central walking corridor
    near = col[int(h * 0.70):]                        # at your feet (should be near)
    ahead = col[int(h * 0.30):int(h * 0.60)]          # a few steps ahead
    thr = np.percentile(near, 40)                     # "as close as the near floor"
    return float((ahead >= thr).mean())


class FreeSpaceMonitor:
    """Background wall/dead-end monitor. `submit(frame)` each Tier-1 tick (cheap);
    read `blocked` for the smoothed verdict. All inference is on its own thread.

    If depth is unavailable (no onnxruntime / no model), `available` is False and
    `blocked` stays False forever — the pipeline behaves exactly as before.
    """

    def __init__(self, model_path: str = MODEL_ONNX, *, enabled: bool = True,
                 on_threshold: float = ON_THRESHOLD,
                 off_threshold: float = OFF_THRESHOLD,
                 tau: float = TAU, target_fps: float = TARGET_FPS) -> None:
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.tau = tau
        self._period = 1.0 / max(1e-3, target_fps)
        self._last_sample_t: Optional[float] = None
        self.blocked = False
        self.score = 0.0                     # smoothed near-fraction (EMA)
        self.available = False
        self._est: Optional[DepthEstimator] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if enabled:
            try:
                self._est = DepthEstimator(model_path)
                self.available = True
            except Exception as exc:         # missing dep/model — degrade silently
                print(f"[depth] free-space monitor disabled ({exc}); "
                      f"walls won't be detected")

    def start(self) -> None:
        if not self.available or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="freespace",
                                        daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """Hand the latest frame to the depth thread (keeps only the newest)."""
        if self.available:
            with self._lock:
                self._frame = frame

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            with self._lock:
                frame = self._frame
            if frame is None:
                time.sleep(0.02)
                continue
            try:
                nf = near_fraction(self._est.infer(frame))   # type: ignore[union-attr]
            except Exception as exc:
                print(f"[depth] inference error ({exc}); disabling")
                self.available = False
                return
            # rate-independent smoothing: alpha from the actual elapsed dt
            now = time.time()
            dt = (now - self._last_sample_t) if self._last_sample_t else self._period
            self._last_sample_t = now
            alpha = 1.0 - math.exp(-dt / self.tau)
            self.score += alpha * (nf - self.score)
            if not self.blocked and self.score >= self.on_threshold:
                self.blocked = True
            elif self.blocked and self.score <= self.off_threshold:
                self.blocked = False
            dt = time.time() - t0
            if dt < self._period:
                time.sleep(self._period - dt)
