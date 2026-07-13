# Aakha

Real-time assistive navigation for blind / low-vision users. No LLM in the safety
loop: classic CV plus small models turn a camera feed into spoken guidance ("car on
your left", "closing on person") and a soft beep while the path ahead is clear.

Phone is the camera and speaker. Laptop runs the pipeline. Phone connects over a
Cloudflare HTTPS tunnel (camera/mic need HTTPS). The installable app is Android.

macOS on Apple Silicon, Python 3.12.

## Setup

System deps (Homebrew):

```bash
brew install ffmpeg cloudflared qrencode
```

`afplay` and `say` are built into macOS (beep + TTS). OCR uses Apple's Vision
framework through the `ocrmac` pip package, so no OCR binary or model download.

Project:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Models: `yolo11n.onnx` and `depth_anything_v2_vits.onnx` auto-download / export on
first run (gitignored, per machine). The Vosk speech model (~40 MB, voice commands)
is fetched once:

```bash
curl -L -o /tmp/vosk.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip -q /tmp/vosk.zip -d .   # -> ./vosk-model-small-en-us-0.15/
```

## Run

One command:

```bash
./run.sh
```

It starts the pipeline + web server, opens the Cloudflare tunnel, and prints a QR
code of the tunnel URL. Scan it in the Aakha app. Logs go to `logs/`, `Ctrl-C`
stops server + tunnel.

The tunnel URL is random on every run, so you re-scan each session (a few seconds).
To pin it, use a named cloudflared tunnel (needs a domain); see CONTRIBUTION.md.

Manual equivalent, if you skip `run.sh`:

```bash
.venv/bin/python -m src.server.server            # http://0.0.0.0:8000
cloudflared tunnel --url http://localhost:8000    # prints the https URL
```

## Android app

`Aakha.apk` in the repo root is a debug build, sideload it:

1. Copy `Aakha.apk` to the phone (USB / Drive / Files), tap it, allow install from
   unknown sources.
2. Open Aakha. First screen is a QR scanner: point it at the QR from `run.sh` and it
   connects. Tap "Enter URL manually" to type the host instead. Grant camera + mic.
3. The gear button (top-right) re-opens the scanner when the tunnel URL changes.

No-install fallback: open the tunnel URL in Chrome on the phone. Same UI as a PWA.

Building the app from source lives in CONTRIBUTION.md.

## Phone UI

The whole screen is one button.

1. Tap to start. First tap starts navigation and unlocks audio. Grant camera + mic.
   Says "Navigation", then guides: clear path beeps, side object gives a direction,
   something ahead or looming gives an obstacle / collision cue.
2. Hold to speak a command. Says "Recording started", you speak, release. Says
   "Processing", then speaks the answer.
3. Stays paused after a command so the answer isn't cut off. Camera keeps streaming.
4. Tap to resume navigation. Tapping again while navigating pauses it.

## Voice commands

Hold, speak, release. Matching tolerates phrasing and Vosk mis-hearings, so the
examples and close variants all work.

| Command | Action | Examples |
|---|---|---|
| Read | OCR the current frame aloud | "read this", "read the text", "what does this say" |
| Identify | Names the most prominent object in view | "what am I holding", "what is this object", "what's in my hand" |
| Repeat | Re-speaks the last line | "repeat that", "say again", "one more time" |

Heard nothing: "I didn't catch that". Heard speech but no known command: reads back
the command list. Either way it returns to paused; tap to resume.

## Extras

| What | How |
|---|---|
| Live detection overlay | `http://localhost:8000/monitor` while the phone streams |
| Event / thread dashboard | `http://localhost:8000/dashboard` |
| Laptop webcam, no phone | `.venv/bin/python main.py` |
| Video file instead of camera | `VISION_SOURCE=path/to/clip.mp4 .venv/bin/python main.py` |
| Smoke test (merge gate) | `.venv/bin/python -m src.core.smoke_test` |

## Stop

`Ctrl-C` in the `run.sh` terminal stops server + tunnel. If the port is stuck:

```bash
lsof -ti:8000 | xargs kill -9
pkill -f "cloudflared tunnel"
```

## Contributing

See CONTRIBUTION.md for architecture, dev setup, the mobile build, and conventions.
