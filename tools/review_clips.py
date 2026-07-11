"""B3 — batch-infer the real (Nepal) footage and report what the pipeline does.

Runs the FULL Tier-1 pipeline (YOLO detect+track, collision, corridor + arbiter,
crosswalk, traffic-light, and the monocular-depth wall check) over every clip in a
folder, and for each clip writes:

  * <clip>_annotated.mp4 — the annotated video, re-encoded to H.264 so it plays
    in VSCode/QuickTime: YOLO boxes + track ids, the walking corridor, the
    WALL/free depth readout, crosswalk/traffic-light overlays, and the bottom
    banner showing what the guidance actually SPOKE at that moment.
  * an entry in report.md — duration, the detected COCO classes with counts, and
    the full timeline of spoken guidance (time, priority, message).

This is a validation/review artifact: watch the videos + skim report.md to see how
the model behaves on real footage (what it detects, what it says, where it's wrong).

Source defaults to assets/real_video, output to assets/real_video_inference.

Usage (from the repo root):
    .venv/bin/python tools/review_clips.py
    .venv/bin/python tools/review_clips.py --src assets/real_video \
        --out assets/real_video_inference --open
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import glob
import os
import subprocess
import sys
import tempfile

# repo root + tools dir on path so `vision`/`shared`/`render_narrated` import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

import render_narrated as narr  # noqa: E402  (say synth + clip placement + mux)
from shared.bus import event_bus  # noqa: E402
from vision import detect  # noqa: E402

VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")


def _mux_h264(video: str, audio: str | None, dst: str) -> bool:
    """Re-encode `video` to H.264 (VSCode/QuickTime-playable) and, if given, mux
    the narration WAV as AAC audio. OpenCV writes mp4v/mpeg4 that some players
    won't decode, so we always re-encode. Returns True on success.
    """
    cmd = ["ffmpeg", "-y", "-i", video]
    if audio is not None:
        cmd += ["-i", audio]
    cmd += ["-c:v", "libx264", "-crf", "23", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    cmd += (["-c:a", "aac", "-shortest"] if audio is not None else ["-an"])
    cmd += [dst]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"    ! H.264 mux failed ({exc}); keeping raw mp4v")
        return False


def _narrate(events: list[tuple[float, str, str, str]], dur: float,
             workdir: str) -> str | None:
    """Synthesize + place the spoken guidance onto a narration WAV (macOS `say`),
    reusing render_narrated's timeline logic. Returns the WAV path, or None."""
    if not events:
        return None
    # drop 'type', but turn heartbeat events into the on-track beep clip so the
    # annotated video sounds like the live app (1 Hz Purr while the path is clear)
    clips = [(t, prio, narr.BEEP_MARKER if typ == "heartbeat" else msg)
             for (t, prio, typ, msg) in events]
    placed = narr._place_clips(clips, workdir)
    wav = os.path.join(workdir, "narration.wav")
    narr._build_track(placed, dur, wav)
    return wav


def review_clip(src: str, out_dir: str) -> dict:
    """Run the pipeline on one clip; write the annotated H.264 video; return a
    report dict {name, dur, frames, fps, classes, events}."""
    fps = cv2.VideoCapture(src).get(cv2.CAP_PROP_FPS) or 30.0
    name = os.path.splitext(os.path.basename(src))[0]

    state = {"idx": 0}
    classes: collections.Counter = collections.Counter()
    events: list[tuple[float, str, str, str]] = []   # (t, priority, type, message)

    def on_frame(_f):
        state["idx"] += 1

    def on_detections(dets):
        for d in dets:
            classes[d["name"]] += 1

    original_publish = event_bus.publish

    def capturing_publish(event) -> None:
        events.append((state["idx"] / fps, event.priority.name, event.type,
                       event.message))

    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "annot.mp4")
        event_bus.publish = capturing_publish            # type: ignore[assignment]
        try:
            # silence the pipeline's per-frame prints; we print our own progress
            with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
                detect.vision_loop(src, publish=True, show=False, save=raw,
                                   on_frame=on_frame, on_detections=on_detections)
        finally:
            event_bus.publish = original_publish         # type: ignore[assignment]
            while not event_bus.empty():
                event_bus.get()

        final = os.path.join(out_dir, f"{name}_annotated.mp4")
        narration = _narrate(events, state["idx"] / fps, tmp)
        if not (os.path.exists(raw) and _mux_h264(raw, narration, final)):
            # fall back to the raw mp4v if mux/re-encode failed / no frames
            if os.path.exists(raw):
                os.replace(raw, final)

    return {"name": name, "dur": state["idx"] / fps, "frames": state["idx"],
            "fps": fps, "classes": classes, "events": events}


def _write_report(reports: list[dict], path: str, src_dir: str) -> None:
    lines = [f"# Real-footage inference report\n",
             f"Source: `{src_dir}` — {len(reports)} clip(s). "
             f"Full Tier-1 pipeline (detection, collision, corridor+arbiter, "
             f"crosswalk, traffic-light, depth wall-check).\n",
             "## Overview\n",
             "| clip | dur | frames | classes seen | spoken cues |",
             "|------|----:|-------:|--------------|------------:|"]
    for r in reports:
        cls = ", ".join(f"{c}×{n}" for c, n in r["classes"].most_common()) or "—"
        lines.append(f"| {r['name']} | {r['dur']:.1f}s | {r['frames']} | "
                     f"{cls} | {len(r['events'])} |")
    lines.append("")

    for r in reports:
        lines.append(f"## {r['name']}  ({r['dur']:.1f}s, {r['frames']} frames "
                     f"@ {r['fps']:.0f}fps)\n")
        if r["classes"]:
            lines.append("**Detected (instances):** "
                         + ", ".join(f"{c} ×{n}"
                                     for c, n in r["classes"].most_common()) + "\n")
        else:
            lines.append("**Detected:** (nothing)\n")
        spoken = collections.Counter(m for _, _, _, m in r["events"])
        lines.append(f"**Spoken guidance ({len(r['events'])} cues; unique: "
                     + (", ".join(f"{m}×{n}" for m, n in spoken.most_common())
                        or "—") + "):**\n")
        if r["events"]:
            lines.append("| t (s) | priority | message |")
            lines.append("|------:|----------|---------|")
            for t, prio, _typ, msg in r["events"]:
                lines.append(f"| {t:.2f} | {prio} | {msg} |")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="assets/real_video",
                    help="folder of source clips (default assets/real_video)")
    ap.add_argument("--out", default="assets/real_video_inference",
                    help="output folder (default assets/real_video_inference)")
    ap.add_argument("--open", action="store_true",
                    help="open the output folder when done (macOS)")
    args = ap.parse_args()

    narr._require("say")        # narration synthesis (macOS)
    narr._require("ffmpeg")     # H.264 re-encode + audio mux
    if not os.path.isdir(args.src):
        sys.exit(f"source folder not found: {args.src}")
    sources = sorted(f for f in glob.glob(os.path.join(args.src, "*"))
                     if os.path.splitext(f)[1].lower() in VIDEO_EXTS)
    if not sources:
        sys.exit(f"no video clips in {args.src}")

    os.makedirs(args.out, exist_ok=True)
    print(f"[review] {len(sources)} clip(s) from {args.src} -> {args.out}")
    reports = []
    for i, src in enumerate(sources, 1):
        print(f"[review] ({i}/{len(sources)}) {os.path.basename(src)} ...", flush=True)
        r = review_clip(src, args.out)
        top = ", ".join(f"{c}×{n}" for c, n in r["classes"].most_common(4)) or "—"
        print(f"    {r['frames']} frames, {len(r['events'])} cues; top: {top}")
        reports.append(r)

    report_path = os.path.join(args.out, "report.md")
    _write_report(reports, report_path, args.src)
    print(f"[review] done -> {args.out}/  (videos + report.md)")
    if args.open:
        subprocess.run(["open", args.out], check=False)


if __name__ == "__main__":
    main()
