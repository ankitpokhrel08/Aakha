# Aakha — real-time assistive navigation for blind / low-vision users

A camera feed is analyzed on-device (classic CV + small models, **no LLM in the
safety loop**) and turned into calm spoken guidance — "car on your left",
"closing on person", a soft on-track beep while the path ahead is clear.

The phone is the camera + speaker; a laptop runs the pipeline. The phone reaches
it over a Cloudflare HTTPS tunnel (needed for camera/mic access in the browser).

---

## 1. Prerequisites (one-time, system)

macOS with [Homebrew](https://brew.sh):

```bash
brew install tesseract ffmpeg cloudflared
```

- **tesseract** — OCR ("read this")
- **ffmpeg** — audio muxing for the offline demo videos
- **cloudflared** — public HTTPS tunnel so the phone can use the camera/mic
- `afplay` and `say` are built into macOS (beep playback + narration)

## 2. Setup (one-time, project)

```bash
# from the repo root
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Models** — YOLO11n (`yolo11n.onnx`) and depth (`depth_anything_v2_vits.onnx`)
are already in the repo. Download the Vosk speech model (~40 MB) for voice
commands:

```bash
curl -L -o /tmp/vosk.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip -q /tmp/vosk.zip -d .    # -> ./vosk-model-small-en-us-0.15/
```

> Moondream2 (Tier 2 scene captioning) downloads itself on first run. It is
> optional — the core safety loop runs without it, and it idles gracefully if it
> can't load.

## 3. Run the whole app (phone + laptop)

**Terminal 1 — the pipeline + web server** (all worker threads: vision, audio,
voice, OCR):

```bash
.venv/bin/python server.py
```

Serves on `http://0.0.0.0:8000`.

**Terminal 2 — the public HTTPS tunnel** for your phone:

```bash
cloudflared tunnel --url http://localhost:8000
```

It prints a URL like `https://<random>.trycloudflare.com` — **open that on your
phone** (Safari on iOS, Chrome on Android). The URL is new on every restart.

**On the phone:** tap the big toggle button once (starts navigation *and*
unlocks audio — iOS needs that first tap for the beep/TTS), then grant camera +
mic. Point it around: clear path → soft beep; object to the side → spoken
direction; something ahead/looming → obstacle/collision cue. Say "read this" to
hear OCR of text in view.

## 4. Handy extras

| What | Command / URL |
|---|---|
| Live detection overlay (laptop) | open `http://localhost:8000/monitor` while the phone streams |
| Event / thread dashboard | open `http://localhost:8000/dashboard` |
| Run on the laptop webcam (no phone) | `.venv/bin/python main.py` |
| Run against a video file | `VISION_SOURCE=path/to/clip.mp4 .venv/bin/python main.py` |
| Smoke test (merge gate) | `.venv/bin/python shared/smoke_test.py` |

## 5. Stopping

`Ctrl-C` in each terminal. If a server is orphaned on the port:

```bash
lsof -ti:8000 | xargs kill -9
pkill -f "cloudflared tunnel"
```
