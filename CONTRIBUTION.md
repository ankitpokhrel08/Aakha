# Contributing to Aakha

Aakha turns a phone camera feed into spoken walking guidance for blind / low-vision
users. No LLM in the safety loop: YOLO + depth + classic CV on the laptop, a single
speech channel on the phone. Read this before touching the pipeline.

Target platform: macOS on Apple Silicon, Python 3.12. The app is Android (Capacitor).

## Repo layout

```
main.py                 entry point + shared lock-guarded slots wiring threads together
config.py               one mutable `config` object; hot-path readers read attrs each frame
run.sh                  one-shot launcher: server + cloudflare tunnel + QR of the URL
Aakha.apk               prebuilt debug APK (sideload target)

src/core/               FROZEN event contract + merge gate
  events.py             Event, Priority (IntEnum: CRITICAL=0, NORMAL=1, LOW=2)   [frozen]
  bus.py                event_bus: one PriorityQueue with a monotonic tie-breaker  [frozen]
  smoke_test.py         the merge gate (see Testing)
src/vision/             Tier 1, the real-time loop
  detect.py             vision_loop: YOLO11n (ONNX) + ByteTrack, builds Candidates
  guidance.py           Corridor (trapezoid ground-contact filter) + GuidanceArbiter
  collision.py          EMA bbox-area growth -> CRITICAL "closing on X"
  depth.py              Depth Anything V2-small (ONNX), background thread, walls only
  crosswalk.py          zebra stripes (Canny + Hough)
  traffic_light.py      red / amber / green (HSV on YOLO's traffic-light box)
  path.py               clearest-path free space + off-path drift
  ocr.py                Apple Vision via ocrmac ("read this")
src/audio/              the voice side
  consumer.py           the ONLY code that touches TTS (pyttsx3). Drains the bus.
  voice_trigger.py      Vosk offline ASR + regex intent match (hold-to-talk)
src/server/             web layer
  server.py             FastAPI + Uvicorn: WS /camera /control /audio /status, PWA, dashboard
  web_assets.py         static PWA bytes (manifest, service worker, mic worklet, icon)

mobile/                 Capacitor Android wrapper (thin client; laptop does the work)
  www/                  the app bundle: index.html (client), jsQR.js, pcm-worklet.js
  android/              generated native project (icons live in app/src/main/res/mipmap-*)
  build.sh              rebuild -> mobile/dist/Aakha-debug.apk
  icon/image.svg        app-icon source
tools/                  offline validation over clips (see Offline tools)
sounds/                 Purr.mp3 (on-track beep), Submarine.mp3
```

## Architecture

Everything is threads plus one shared priority queue. The only async is FastAPI's
web layer; the pipeline itself has no async orchestration.

Event bus (`src/core/`) is a frozen v0 contract. Do not change the shape of `Event`
or `Priority`, or the bus API, without team agreement. `Event.type` is a free-form
string on purpose so new producers add types without editing the frozen files.
Lower `Priority` value dequeues first; equal priority stays FIFO via an
`itertools.count()` tie-breaker so Events never get compared.

Many producers publish, exactly one consumer speaks. `src/audio/consumer.py` is the
only TTS caller. Never start a second consumer. It drains the bus by priority.
`type="heartbeat"` renders as the on-track beep, not speech. CRITICAL is dequeued
first but does NOT preempt a sentence already playing (worst case ~1-2 s), which is
why cues are short. This is a documented limitation, not a bug.

Worker threads (`main.start_workers()`, tracked in `main.WORKER_THREADS`):
`_vision_producer`, `_audio_consumer`, `_ocr`, `_voice_trigger`. A `freespace`
depth thread starts inside `vision_loop`; a `config-watcher` thread starts when
`config.start_watching()` is called.

Tier 1 (`detect.py::vision_loop`) per frame: YOLO track -> compute signals ->
`_build_candidates` proposes `Candidate`s -> `GuidanceArbiter` emits at most one per
beat (ranked priority then urgency, per-key cooldowns). Only objects whose ground
contact falls inside the `Corridor` matter, so a bus across the road is ignored.

Mobile app: `mobile/www/index.html` is the phone client extracted from
`web_assets.py`, made server-configurable. It streams JPEG to `/camera`, mic PCM16
to `/audio`, and speaks bus events from `/status`. The Android System WebView has no
working Web Speech engine, so speech is the laptop's `/tts` synth played as HTML5
audio (same channel as the beep).

## Dev setup

```bash
brew install ffmpeg cloudflared qrencode
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# Vosk model (voice commands), one time:
curl -L -o /tmp/vosk.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip -q /tmp/vosk.zip -d .
```

Run the whole thing: `./run.sh`. Config (`config.json`) and model weights are
gitignored and per machine; do not commit them.

## Platform support

The backend (laptop) is macOS on Apple Silicon. The app is Android. That split is
deliberate, but it means a few backend pieces call native macOS tools:

| Piece | Where | macOS tool | Off-macOS today |
|---|---|---|---|
| OCR ("read this") | `src/vision/ocr.py` | `ocrmac` (Apple Vision) | pip skips it (marker); OCR self-disables |
| Phone speech | `src/server/server.py::_synth_tts` (`/tts`) | `say` | no speech to the phone |
| Laptop-local beep | `src/audio/consumer.py` | `afplay` | beep fails silently (phone beep is fine, it is served as `/beep.mp3`) |
| Offline demo tools | `tools/*` | `say` | narration step fails |

`pip install -r requirements.txt` succeeds on Linux/Windows (the `ocrmac` marker
skips it) and the code degrades instead of crashing, so you can install, import, and
work on vision / guidance logic anywhere. The full spoken pipeline still needs the
native tools above. `pyttsx3` (the laptop-local consumer voice) is cross-platform
already (SAPI5 on Windows, espeak on Linux), just different voices.

Porting the backend to Linux/Windows is welcome (see the open "porting" issues).
Keep the degrade-never-crash contract: detect the platform or missing tool and fall
back, never hard-crash, and never regress macOS.

## Testing

`src/core/smoke_test.py` is the merge gate, and the primary test (there is no unit
suite / linter / build step). It starts `main.run()` in a thread, waits 8 s, and
asserts no crash plus all `WORKER_THREADS` alive. Run it after every change to the
pipeline:

```bash
.venv/bin/python -m src.core.smoke_test    # exits non-zero on failure
```

Guidance logic (`guidance.py`) has no bus / detection imports, so it stays trivially
unit-testable in a REPL. Prefer small `.venv/bin/python -c "..."` checks for pure
functions (`display_name`, `_build_candidates`, corridor math).

## Running subsystems standalone

Each vision module has a `__main__`:

```bash
.venv/bin/python -m src.vision.detect --source clip.mp4
#   flags: --no-track --no-crosswalk --no-traffic-light --no-freespace
#          --all-classes --no-bus --save out.mp4
.venv/bin/python -m src.audio.consumer     # priority-ordering demo (LOW/NORMAL/CRITICAL)
```

## Offline tools

Validate the pipeline on real footage without a phone (each writes annotated video
and prints what was spoken):

```bash
.venv/bin/python tools/review_clips.py --src assets/real_video --open
.venv/bin/python tools/validate_wall.py --source assets/clean_road.mp4 --open
.venv/bin/python tools/depth_overlay.py --source assets/clean_road.mp4 --open
.venv/bin/python tools/render_narrated.py --source assets/cars.mp4 --open
```

These need macOS `say` + `ffmpeg`.

## Mobile app

The app is a Capacitor 7 wrapper. The web code in `mobile/www/` is the whole app;
the laptop does all inference. Toolchain (Homebrew, no sudo):

```bash
brew install openjdk@21
brew install --cask android-commandlinetools
export JAVA_HOME="/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
export ANDROID_HOME="/opt/homebrew/share/android-commandlinetools"
yes | sdkmanager --sdk_root="$ANDROID_HOME" --licenses
sdkmanager --sdk_root="$ANDROID_HOME" "platform-tools" "platforms;android-35" "build-tools;35.0.0"
```

Build (exports JAVA_HOME / ANDROID_HOME for you):

```bash
cd mobile && ./build.sh        # -> mobile/dist/Aakha-debug.apk
cp mobile/dist/Aakha-debug.apk Aakha.apk   # refresh the root sideload copy
```

Editing the client: change `mobile/www/index.html`, then `npx cap sync android`
before building (build.sh does the sync). Keep `mobile/www/index.html` and
`web_assets.py::_INDEX_HTML` behaviourally in step so the PWA and the app match.

App icon: replace `mobile/icon/image.svg`, then regenerate the
`mipmap-*/ic_launcher*.png` set (render -> trim -> composite densities). The
adaptive background colour is `ic_launcher_background.xml`.

Constant tunnel URL (avoids re-scanning): needs a domain on a free Cloudflare
account.

```bash
cloudflared login
cloudflared tunnel create aakha
cloudflared tunnel route dns aakha aakha.yourdomain.com
cloudflared tunnel run --url http://localhost:8000 aakha
```

## Conventions and gotchas

- Frozen files: `src/core/events.py`, `src/core/bus.py`. Treat as contract.
- Degrade, never crash. Every optional subsystem (depth, voice, camera, OCR)
  catches its own failure and idles, so the smoke test and other machines keep
  working. Wrap the optional dep in try/except and keep the thread alive.
- Cut-list order (`config.py::CUT_ORDER`): disable in the order
  traffic_light -> crosswalk -> voice_trigger. The collision warning is never a
  toggle and must never be cut.
- Live config: `config.py` is one shared mutable object; hot-path readers read
  attributes each frame. `config.json` edits reload within ~2 s when watching is on.
  Do not rebind `config`, mutate in place.
- Push vs local: `vision_loop(frames=...)` is push mode (server feeds phone frames);
  `source=...` opens a local `cv2.VideoCapture`. Any GUI (`show=True`) must run on
  the main thread on macOS, so threaded use passes `show=False`.
- Comments stay terse. This codebase was scrubbed of AI-style essay comments; match
  that density. Keep only the non-obvious "why".
- Docs: no em-dashes. Write plainly and technically.

## How to add things

New producer / event type: publish `Event(message=..., priority=..., type="foo",
source=...)` on `event_bus`. The consumer speaks any event with a message and a
priority, so it works on the laptop immediately. To make it speak on the phone, add
`"foo"` to the `SPEAK` set in BOTH `mobile/www/index.html` and
`web_assets.py::_INDEX_HTML` (the phone only speaks a whitelist; the laptop speaks
everything). This is the usual "works on laptop, silent on phone" bug.

New named object class: add it to `SPEAK_BY_NAME` in `guidance.py` (else it speaks
as generic "object"). Ambient naming of a non-vehicle class also needs its
confidence above `AMBIENT_MIN_CONF` in `detect.py`; vehicles / person in
`PRIORITY_CLASSES` use the `CONF` floor. The voice "identify" scan is unaffected.

New voice command: add a `(key, [regex...])` entry to `_COMMAND_PATTERNS` in
`voice_trigger.py` and handle `cmd == "key"` in `dispatch_command`, publishing the
result as an Event. Terminal replies (that end the voice session) go in the
`REPLY_TERMINAL` sets.

## Before you open a PR

1. `.venv/bin/python -m src.core.smoke_test` passes.
2. If you touched the phone client, rebuild the APK and refresh `Aakha.apk`.
3. Do not commit `config.json`, model weights (`*.onnx`, `*.pt`), the Vosk model,
   `mobile/node_modules`, `mobile/dist`, or `logs/` (all gitignored).
4. Keep commit messages short and factual (two lines is plenty).
