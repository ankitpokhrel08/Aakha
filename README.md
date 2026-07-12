# Aakha

Real-time assistive navigation for blind / low-vision users. No LLM in the safety
loop: classic CV plus small models turn a camera feed into spoken guidance ("car on
your left", "closing on person") and a soft beep while the path ahead is clear.

Phone is the camera and speaker. Laptop runs the pipeline. Phone connects over a
Cloudflare HTTPS tunnel (browsers require HTTPS for camera/mic).

macOS on Apple Silicon, Python 3.12.

## Setup

System deps (Homebrew):

```bash
brew install ffmpeg cloudflared
```

`afplay` and `say` are built into macOS (beep + TTS). OCR uses Apple's Vision
framework through the `ocrmac` pip package, so no OCR binary or model download.

Project:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Models: `yolo11n.onnx` and `depth_anything_v2_vits.onnx` are checked in. The Vosk
speech model (~40 MB, for voice commands) is not:

```bash
curl -L -o /tmp/vosk.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip -q /tmp/vosk.zip -d .   # -> ./vosk-model-small-en-us-0.15/
```

## Run

Two terminals. Pipeline + web server:

```bash
.venv/bin/python server.py     # http://0.0.0.0:8000
```

HTTPS tunnel for the phone:

```bash
cloudflared tunnel --url http://localhost:8000
```

This prints a `https://<random>.trycloudflare.com` URL (new on every restart). Open
it on the phone: Safari on iOS, Chrome on Android.

## Phone UI

The whole screen is one button.

1. Tap to start. First tap starts navigation and unlocks audio (iOS needs a tap
   before it will play beep/TTS). Grant camera + mic. Says "Navigation", then
   guides: clear path beeps, side object gives a direction, something ahead or
   looming gives an obstacle/collision cue.
2. Hold to speak a command. Says "Recording started", you speak (see below),
   release. Says "Processing", then speaks the answer.
3. Stays paused after a command so the answer isn't cut off. Camera keeps
   streaming.
4. Tap to resume navigation. Tapping again while navigating pauses it.

## Voice commands

Hold, speak, release. Matching tolerates phrasing and Vosk mis-hearings, so any of
the examples (and close variants) work.

| Command | Action | Examples |
|---|---|---|
| Read | OCR the current frame aloud | "read this", "read the text", "what does this say" |
| What am I holding | Names the object held up to the camera | "what am I holding", "what's in my hand", "what is this object" |
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
| Smoke test (merge gate) | `.venv/bin/python shared/smoke_test.py` |

## Stop

`Ctrl-C` in each terminal. If the port is stuck:

```bash
lsof -ti:8000 | xargs kill -9
pkill -f "cloudflared tunnel"
```
