"""Render an annotated + narrated demo video from a source clip.

Runs the real Tier 1 vision pipeline over a video, draws the overlay (boxes,
tracks, crosswalk, traffic-light, collision highlights), records every event it
publishes at that frame's timestamp, synthesizes speech for each with macOS
``say``, lays the clips down on a timeline (CRITICAL collision warnings placed
first so they're never dropped, then directional alerts fill the gaps without
overlapping), and muxes the audio onto the video with ffmpeg.

The result is a single MP4 you can watch to *see and hear* what the pipeline is
doing — a validation / demo artifact.

Usage (from the repo root):
    .venv/bin/python tools/render_narrated.py --source assets/cars.mp4 --open
    .venv/bin/python tools/render_narrated.py --source clip.mp4 --output out.mp4

Requirements: macOS ``say`` and ``ffmpeg`` on PATH (both checked at startup).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

# repo root on path so `vision` / `shared` import regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from shared.bus import event_bus  # noqa: E402
from vision import detect  # noqa: E402

SR = 22050          # narration sample rate (mono 16-bit)
SAY_RATE = 190      # words per minute for `say`
CLIP_GAP = 0.12     # min silence between spoken clips (s)
PRIORITY_RANK = {"CRITICAL": 0, "NORMAL": 1, "LOW": 2}

# On-track heartbeat beep: vision publishes empty-message type="heartbeat" events
# (~1 Hz while the path is clear). We bake the actual Purr.mp3 into the narration
# track at those timestamps so the demo video *sounds* like the live app. Callers
# tag such events with BEEP_MARKER as the "message" so _synthesize returns the beep
# clip instead of speech.
BEEP_MARKER = "\x00on-track-beep"
_BEEP_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "sound_asset", "Purr.mp3")


def _load_beep() -> np.ndarray:
    """Decode the on-track beep (sound_asset/Purr.mp3) to SR mono int16, cached
    (reuses _SYNTH_CACHE keyed by BEEP_MARKER). Silent 1-sample clip on failure."""
    if BEEP_MARKER in _SYNTH_CACHE:
        return _SYNTH_CACHE[BEEP_MARKER]
    data = np.zeros(1, dtype=np.int16)
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            wavp = tf.name
        subprocess.run(
            ["ffmpeg", "-y", "-i", _BEEP_SRC, "-ac", "1", "-ar", str(SR),
             "-sample_fmt", "s16", wavp],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with wave.open(wavp, "rb") as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            if w.getnchannels() > 1:
                data = data.reshape(-1, w.getnchannels())[:, 0].copy()
        os.unlink(wavp)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        print(f"[render] on-track beep load failed ({exc}); beeps will be silent")
    _SYNTH_CACHE[BEEP_MARKER] = data
    return data


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        sys.exit(f"error: '{tool}' not found on PATH "
                 f"({'brew install ffmpeg' if tool == 'ffmpeg' else 'macOS only'})")


_SYNTH_CACHE: dict[str, np.ndarray] = {}


def _synthesize(message: str, path: str) -> np.ndarray:
    """Render `message` to 22.05kHz mono int16 via `say`, cached per phrase.

    The guidance vocabulary is tiny and highly repetitive ("path is clear",
    "path blocked", ...), so each distinct phrase is synthesized once and reused
    — instead of one `say` per event (hundreds of calls). `say` is also bounded
    by a timeout so a single wedged call can't hang the whole render.
    """
    if message == BEEP_MARKER:            # the on-track beep, not speech
        return _load_beep()
    if message in _SYNTH_CACHE:
        return _SYNTH_CACHE[message]
    try:
        subprocess.run(
            ["say", "-r", str(SAY_RATE), "--file-format=WAVE",
             "--data-format=LEI16@22050", "-o", path, message],
            check=True, timeout=20,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        with wave.open(path, "rb") as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            if w.getnchannels() > 1:
                data = data.reshape(-1, w.getnchannels())[:, 0].copy()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        print(f"[render] say failed for {message!r} ({exc}); skipping clip")
        data = np.zeros(1, dtype=np.int16)
    _SYNTH_CACHE[message] = data
    return data


def _run_vision(source: str, annotated_path: str) -> tuple[list, float, int]:
    """Run the pipeline, saving the annotated video and capturing events.

    Returns (events, fps, n_frames) where each event is
    (t_seconds, priority_name, message).
    """
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    state = {"idx": 0}
    events: list[tuple[float, str, str]] = []

    original_publish = event_bus.publish

    def capturing_publish(event) -> None:
        t = state["idx"] / fps
        # heartbeat events carry no speech; bake the on-track beep instead
        msg = BEEP_MARKER if getattr(event, "type", "") == "heartbeat" else event.message
        events.append((t, event.priority.name, msg))
        original_publish(event)

    event_bus.publish = capturing_publish  # type: ignore[assignment]
    try:
        detect.vision_loop(
            source, publish=True, show=False, save=annotated_path,
            on_frame=lambda _f: state.__setitem__("idx", state["idx"] + 1),
        )
    finally:
        event_bus.publish = original_publish  # type: ignore[assignment]
        # drain anything the (non-existent) consumer left behind
        while not event_bus.empty():
            event_bus.get()

    return events, fps, state["idx"]


def _place_clips(events: list, workdir: str) -> list[tuple[float, np.ndarray]]:
    """Synthesize + place non-overlapping clips. CRITICAL first, then NORMAL by
    time, so safety warnings are never dropped for a directional alert."""
    placed_intervals: list[tuple[float, float]] = []
    placed: list[tuple[float, np.ndarray]] = []

    def fits(start: float, end: float) -> bool:
        return all(not (start < e and s < end) for s, e in placed_intervals)

    order = sorted(events, key=lambda e: (PRIORITY_RANK.get(e[1], 1), e[0]))
    for i, (t, _prio, msg) in enumerate(order):
        wav = os.path.join(workdir, f"clip_{i:04d}.wav")
        data = _synthesize(msg, wav)
        dur = len(data) / SR
        if fits(t, t + dur + CLIP_GAP):
            placed_intervals.append((t, t + dur + CLIP_GAP))
            placed.append((t, data))
    placed.sort(key=lambda p: p[0])
    return placed


def _build_track(placed: list, total_dur: float, path: str) -> None:
    """Lay placed clips onto one silent int16 track and write a WAV."""
    tail = max((s + len(d) / SR for s, d in placed), default=0.0)
    n = int(max(total_dur, tail) * SR) + SR  # +1s tail
    buf = np.zeros(n, dtype=np.int16)
    for start, data in placed:
        off = int(start * SR)
        seg = data[: max(0, n - off)]
        buf[off: off + len(seg)] = seg  # no overlap by construction
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(buf.tobytes())


def _mux(video: str, audio: str, out: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-i", audio,
         "-c:v", "copy", "-c:a", "aac", "-shortest", out],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="assets/cars.mp4", help="input video")
    ap.add_argument("--output", default=None,
                    help="output MP4 (default: <source>_narrated.mp4)")
    ap.add_argument("--open", action="store_true",
                    help="open the result when done (macOS `open`)")
    args = ap.parse_args()

    _require("say")
    _require("ffmpeg")
    if not os.path.exists(args.source):
        sys.exit(f"error: source not found: {args.source}")

    out = args.output or str(Path(args.source).with_name(
        Path(args.source).stem + "_narrated.mp4"))

    with tempfile.TemporaryDirectory() as tmp:
        annotated = os.path.join(tmp, "annotated.mp4")
        narration = os.path.join(tmp, "narration.wav")

        print(f"[render] processing {args.source} ...")
        events, fps, n_frames = _run_vision(args.source, annotated)
        total_dur = n_frames / fps
        print(f"[render] {n_frames} frames @ {fps:.0f}fps "
              f"({total_dur:.1f}s), {len(events)} events published")

        print(f"[render] synthesizing narration ({len(events)} candidates) ...")
        placed = _place_clips(events, tmp)
        print(f"[render] placed {len(placed)} clips "
              f"(dropped {len(events) - len(placed)} that would overlap)")

        _build_track(placed, total_dur, narration)
        print(f"[render] muxing -> {out}")
        _mux(annotated, narration, out)

    print(f"[render] done: {out}")
    if args.open:
        subprocess.run(["open", out], check=False)


if __name__ == "__main__":
    main()
