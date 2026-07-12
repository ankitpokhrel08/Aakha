"""VisionAid web layer — phone camera client + dashboard over WebSockets.

Dev 3 / mobile-client. This is the "Option 1" rig from the plan: the phone runs
a lightweight browser client that streams its camera to this laptop backend over
WiFi; inference happens here; the sighted dashboard watches the event bus.

Endpoints
---------
GET  /            fullscreen single-button end-user page (camera + toggle)
GET  /dashboard   sighted-teammate view: connection light + scrolling event log
WS   /camera      browser -> server: JPEG frames, handed to on_frame() callback
WS   /control     browser -> server: {"action": "nav_on"|"nav_off"|"nav_pause"|
                  "voice_start"|"voice_end"} etc.
WS   /status      server -> browser: event-bus activity + per-thread heartbeats

Import-safety
-------------
The whole team's ``shared/smoke_test.py`` imports ``main`` which may import this
module. Nothing here blocks or starts a server/thread at import time — the app
object is built, callbacks default to no-ops, and the only side effects (bus
tap, worker start, heartbeat loop) happen in the FastAPI ``startup`` handler,
i.e. only when uvicorn actually runs. ``import server`` is always safe.

Bus contract
------------
``shared/bus.py`` is a single-consumer priority queue (the one audio/TTS
thread drains it). The dashboard must *observe* events without *stealing* them,
so instead of calling ``event_bus.get()`` here we install a non-destructive tap:
we wrap the singleton's ``publish`` at runtime to mirror a copy of every event
to status subscribers. shared/bus.py itself is never modified (frozen contract).
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, Callable, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse

import web_assets
from config import config
from shared.bus import event_bus
from shared.events import Event, Priority

logger = logging.getLogger("visionaid.server")

app = FastAPI(title="VisionAid mobile-client")

# Shared UI/runtime state. `active` reflects the toggle button; the vision
# pipeline can read it later to decide whether to emit guidance.
STATE: dict[str, Any] = {"active": False, "frames_received": 0}

# When True, the FastAPI startup handler also boots main.run(block=False) so a
# plain `python server.py` brings the whole system up (workers + web). Set to
# False if you want to run the web layer against externally-started workers.
START_WORKERS = True

HEARTBEAT_INTERVAL_S = 2.0


# --------------------------------------------------------------------------- #
# Frame callback — wired to the vision pipeline later.
# --------------------------------------------------------------------------- #
def _default_on_frame(frame: bytes) -> None:
    """Placeholder. dev1/vision replaces this via set_frame_callback() with the
    real decode + YOLO handoff. Kept a no-op so /camera works before vision
    exists."""
    # Intentionally cheap: just count so the dashboard/logs show frames arriving.
    STATE["frames_received"] += 1


# Module-level reference the vision team wires up. Call set_frame_callback(fn).
on_frame: Callable[[bytes], None] = _default_on_frame


def set_frame_callback(fn: Callable[[bytes], None]) -> None:
    """Point /camera at the real vision handler. `fn` takes raw JPEG bytes and
    must return quickly (it runs on the websocket receive path) — hand heavy
    work to the vision thread rather than blocking here."""
    global on_frame
    on_frame = fn


# --------------------------------------------------------------------------- #
# Status fan-out hub — one asyncio.Queue per connected /status client.
# --------------------------------------------------------------------------- #
class StatusHub:
    """Broadcasts JSON payloads to every connected /status websocket.

    Each client gets its own bounded queue; a slow/stalled client drops its
    oldest messages instead of blocking producers or other clients.
    """

    def __init__(self, maxsize: int = 200) -> None:
        self._queues: set[asyncio.Queue] = set()
        self._maxsize = maxsize

    def register(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        self._queues.add(q)
        return q

    def unregister(self, q: asyncio.Queue) -> None:
        self._queues.discard(q)

    @property
    def client_count(self) -> int:
        return len(self._queues)

    def dispatch(self, payload: dict) -> None:
        """Fan a payload out to all clients. MUST run on the event loop thread
        (call via loop.call_soon_threadsafe from other threads)."""
        for q in list(self._queues):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # Drop oldest to make room — dashboards care about recent events.
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except Exception:
                    pass


status_hub = StatusHub()

# Captured on startup so the (threaded) bus tap can schedule work on the loop.
_loop: Optional[asyncio.AbstractEventLoop] = None
_heartbeat_task: Optional[asyncio.Task] = None


# --------------------------------------------------------------------------- #
# Non-destructive bus tap.
# --------------------------------------------------------------------------- #
def _event_to_payload(event: Event) -> dict:
    try:
        priority_val = int(event.priority)
    except Exception:
        priority_val = int(Priority.NORMAL)
    try:
        priority_name = Priority(priority_val).name
    except Exception:
        priority_name = str(getattr(event, "priority", "NORMAL"))
    return {
        "kind": "event",
        "type": getattr(event, "type", "generic"),
        "priority": priority_val,
        "priority_name": priority_name,
        "message": getattr(event, "message", ""),
        "source": getattr(event, "source", ""),
        "timestamp": getattr(event, "timestamp", None),
    }


def _install_bus_tap() -> None:
    """Wrap event_bus.publish so every published Event is mirrored to /status
    subscribers, without consuming it from the queue. Idempotent."""
    if getattr(event_bus, "_status_tapped", False):
        return
    original_publish = event_bus.publish

    def tapped_publish(event: Event) -> None:
        original_publish(event)  # normal enqueue for the audio consumer
        loop = _loop
        if loop is None:
            return
        try:
            payload = _event_to_payload(event)
            loop.call_soon_threadsafe(status_hub.dispatch, payload)
        except Exception:  # never let observation break publishing
            logger.debug("status tap dispatch failed", exc_info=True)

    event_bus.publish = tapped_publish  # type: ignore[method-assign]
    event_bus._status_tapped = True  # type: ignore[attr-defined]
    logger.info("event_bus.publish tapped for /status broadcast")


def _known_thread_heartbeats() -> list[dict]:
    """One heartbeat payload per known worker thread name (+ the web layer)."""
    now = _now()
    beats: list[dict] = [
        {"kind": "heartbeat", "thread": "web-server", "alive": True, "timestamp": now}
    ]
    try:
        import main  # lazy — main may not be importable in every context

        for t in getattr(main, "WORKER_THREADS", []):
            beats.append(
                {
                    "kind": "heartbeat",
                    "thread": t.name,
                    "alive": t.is_alive(),
                    "timestamp": now,
                }
            )
    except Exception:
        logger.debug("could not read main.WORKER_THREADS", exc_info=True)
    return beats


def _now() -> float:
    import time

    return time.time()


async def _heartbeat_loop() -> None:
    """Broadcast a per-thread heartbeat every HEARTBEAT_INTERVAL_S seconds."""
    try:
        while True:
            for beat in _known_thread_heartbeats():
                status_hub.dispatch(beat)
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("heartbeat loop crashed")


# --------------------------------------------------------------------------- #
# Lifecycle.
# --------------------------------------------------------------------------- #
@app.on_event("startup")
async def _on_startup() -> None:
    global _loop, _heartbeat_task
    _loop = asyncio.get_running_loop()
    _install_bus_tap()

    # Live config: reload config.json when it changes on disk, and let the
    # /control "set" action flip toggles mid-run. Watcher is a daemon thread.
    config.start_watching()

    if START_WORKERS:
        try:
            import main

            # Wire the phone camera into the pipeline: /camera frames -> vision.
            # main.push_frame decodes each JPEG; the vision thread consumes them
            # (push mode) instead of opening a local webcam. Must be set BEFORE
            # main.run() so the vision thread picks push mode at startup.
            main.FRAME_SOURCE = main.get_incoming_frame
            set_frame_callback(main.push_frame)

            # Start PAUSED: the app comes up asking the user to tap to start
            # navigation. Guidance stays muted until the first nav_on. (main
            # defaults nav_active True for the standalone `python main.py` path.)
            main.set_nav_active(False)

            if not getattr(main, "WORKER_THREADS", []):
                main.run(block=False)  # start workers, don't block the loop
                logger.info("started main workers: %s",
                            [t.name for t in main.WORKER_THREADS])
        except Exception:
            logger.exception("could not start main workers (continuing web-only)")

    _heartbeat_task = asyncio.create_task(_heartbeat_loop())


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    if _heartbeat_task is not None:
        _heartbeat_task.cancel()
        try:
            await _heartbeat_task
        except asyncio.CancelledError:
            pass


# --------------------------------------------------------------------------- #
# WebSocket endpoints.
# --------------------------------------------------------------------------- #
@app.websocket("/camera")
async def camera_ws(ws: WebSocket) -> None:
    """Receive JPEG frames (binary) from the browser and hand each to on_frame."""
    await ws.accept()
    logger.info("camera client connected")
    try:
        while True:
            frame = await ws.receive_bytes()
            # Always feed frames while the client is streaming (which it does the
            # whole time the app is "started", navigating or paused). The pipeline
            # keeps the latest frame + detections fresh so a voice command ("read
            # this" / "what am I holding") can see live camera output even while
            # navigation is paused. Guidance itself is gated separately by the
            # nav-active flag, not by dropping frames.
            try:
                on_frame(frame)
            except Exception:
                logger.exception("on_frame callback raised")
    except WebSocketDisconnect:
        logger.info("camera client disconnected")
    except Exception:
        logger.exception("camera ws error")


@app.websocket("/control")
async def control_ws(ws: WebSocket) -> None:
    """Receive control actions from the big button.

    Navigation model (camera streams continuously once the app is started):
      nav_on    — tap to (re)enter navigation. Guidance resumes; spoken "Navigation".
      nav_off   — tap to stop navigating. Guidance muted; spoken "Navigation off".
      nav_pause — go to paused SILENTLY (used after a voice command completes, so
                  navigation never auto-resumes — the user taps to resume).
      voice_start/voice_end — hold-to-talk boundaries (mute guidance + beep).
    """
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            action = (msg or {}).get("action")
            if action in ("nav_on", "nav_off", "nav_pause"):
                on = action == "nav_on"
                STATE["active"] = on
                try:
                    import main
                    main.set_nav_active(on)
                    if on:
                        main.set_voice_active(False)   # tapping into nav ends any cmd
                except Exception:
                    logger.debug("could not set nav_active", exc_info=True)
                # nav_pause is silent (post-command); nav_on/nav_off are spoken.
                if action != "nav_pause":
                    event_bus.publish(
                        Event(
                            message="Navigation" if on else "Navigation off",
                            priority=Priority.LOW,
                            type="control",
                            source="control",
                            data={"active": on},
                        )
                    )
                await ws.send_json({"ok": True, "active": on})
            elif action in ("voice_start", "voice_end"):
                # Phone hold-to-talk boundaries. While a voice session is live the
                # Tier 1 loop hushes the on-track heartbeat beep so it doesn't tick
                # over the spoken command / its reply (V1).
                try:
                    import main
                    main.set_voice_active(action == "voice_start")
                except Exception:
                    logger.debug("could not set voice_active", exc_info=True)
                await ws.send_json({"ok": True})
            elif action == "get":
                # Return current settings so a dashboard can render controls.
                await ws.send_json({"ok": True, "config": config.to_dict()})
            elif action == "set":
                # Flip a config value at runtime, e.g. cut traffic-light:
                #   {"action":"set","key":"traffic_light_detection","value":false}
                key = (msg or {}).get("key")
                value = (msg or {}).get("value")
                allowed = set(config.to_dict().keys())
                if key not in allowed:
                    await ws.send_json({
                        "ok": False,
                        "error": f"unknown config key: {key!r}",
                        "keys": sorted(allowed),
                    })
                else:
                    changed = config.update(**{key: value})  # persists to config.json
                    current = config.to_dict()
                    # Mirror onto the bus so the change shows on the dashboard
                    # (and is spoken as a low-priority confirmation).
                    event_bus.publish(
                        Event(
                            message=f"{key} set to {current[key]}",
                            priority=Priority.LOW,
                            type="config",
                            source="control",
                            data={"key": key, "value": current[key]},
                        )
                    )
                    await ws.send_json({"ok": True, "changed": changed, "config": current})
            else:
                await ws.send_json({"ok": False, "error": f"unknown action: {action}"})
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("control ws error")


# --------------------------------------------------------------------------- #
# Voice-command transcript echo (A4).
#
# The voice_trigger thread transcribes each clip and dispatches the command, but
# it's frozen this round and never puts the *raw transcript* on the bus — only
# the command result. So we can't tell whether recognition happened, or what it
# heard. To echo it, the server transcribes the same clip with its own cached
# Vosk model and publishes a `voice_heard` event ("Heard: ...") that shows on the
# dashboard, speaks on the phone, and logs to the server console. The extra
# transcription is cheap (voice commands are user-initiated and rare).
# --------------------------------------------------------------------------- #
_voice_model = None
_voice_model_lock = threading.Lock()
_voice_model_tried = False


def _get_voice_model():
    """Load the Vosk model once (lazily) for the transcript echo, or None."""
    global _voice_model, _voice_model_tried
    with _voice_model_lock:
        if _voice_model is None and not _voice_model_tried:
            _voice_model_tried = True
            try:
                from audio.voice_trigger import _load_vosk_model
                _voice_model = _load_vosk_model()  # None if model dir absent
            except Exception:
                logger.exception("could not load Vosk model for transcript echo")
        return _voice_model


def _transcribe_for_echo(clip: bytes) -> Optional[str]:
    """Transcribe a clip for the echo (runs in an executor thread). Returns the
    transcript, "" if nothing recognised, or None if no model is available."""
    model = _get_voice_model()
    if model is None:
        return None
    try:
        from audio.voice_trigger import transcribe_clip
        return transcribe_clip(clip, model=model)
    except Exception:
        logger.exception("echo transcription failed")
        return None


async def _echo_transcript(clip: bytes) -> None:
    """Transcribe off the event loop and publish a `voice_heard` echo event."""
    loop = asyncio.get_running_loop()
    transcript = await loop.run_in_executor(None, _transcribe_for_echo, clip)
    if transcript is None:
        return  # no model — voice_trigger already logs the download hint
    heard = transcript if transcript else "(nothing recognised)"
    logger.info("heard: %s", heard)
    event_bus.publish(Event(
        message=f"Heard: {heard}",
        priority=Priority.LOW,
        type="voice_heard",
        source="voice",
    ))


@app.websocket("/audio")
async def audio_ws(ws: WebSocket) -> None:
    """Receive short voice-command clips (raw 16 kHz mono PCM16) from the phone
    and hand each to the voice-trigger queue for Vosk transcription + dispatch.

    This is the backend half of the press-to-record trigger: the phone records
    a clip on the button press and sends the bytes here.
    """
    await ws.accept()
    logger.info("audio client connected")
    try:
        from audio.voice_trigger import clip_queue
    except Exception:
        logger.exception("voice_trigger unavailable; /audio will drop clips")
        clip_queue = None
    try:
        while True:
            clip = await ws.receive_bytes()
            # A3: confirm capture — log size + duration (16 kHz mono PCM16 => 2 bytes/sample).
            n = len(clip)
            secs = n / 2 / 16000
            if n == 0:
                logger.warning("audio clip EMPTY (0 bytes) — capture produced nothing")
            else:
                logger.info("audio clip received: %d bytes (~%.2fs @16kHz mono)", n, secs)
            if clip_queue is not None:
                clip_queue.put(clip)
                # A voice command is now executing — keep the on-track beep hushed
                # until the consumer finishes speaking the reply (it clears the
                # flag then). Covers the execution window even if voice_start was
                # late/missed; the consumer, not the phone, decides when it ends.
                if n:
                    try:
                        import main
                        main.set_voice_active(True)
                    except Exception:
                        logger.debug("could not set voice_active", exc_info=True)
                await ws.send_json({"ok": True, "bytes": n})
                # A4: echo what was recognised back to phone/dashboard + console.
                if n:
                    asyncio.create_task(_echo_transcript(clip))
    except WebSocketDisconnect:
        logger.info("audio client disconnected")
    except Exception:
        logger.exception("audio ws error")


@app.websocket("/status")
async def status_ws(ws: WebSocket) -> None:
    """Stream event-bus activity + heartbeats to dashboard/observer clients."""
    await ws.accept()
    q = status_hub.register()
    # Prime the new client with a current thread snapshot so it renders instantly.
    for beat in _known_thread_heartbeats():
        status_hub.dispatch(beat)
    try:
        while True:
            payload = await q.get()
            await ws.send_json(payload)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.debug("status ws send failed (client gone)", exc_info=True)
    finally:
        status_hub.unregister(q)


# --------------------------------------------------------------------------- #
# Static pages (inlined so there's no static-dir/path coupling to break import).
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _INDEX_HTML


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    return _DASHBOARD_HTML


# --------------------------------------------------------------------------- #
# PWA assets (manifest, service worker, audio worklet, icons).
# --------------------------------------------------------------------------- #
@app.get("/manifest.json")
async def manifest() -> Response:
    return Response(web_assets.MANIFEST, media_type="application/manifest+json")


@app.get("/sw.js")
async def service_worker() -> Response:
    return Response(web_assets.SERVICE_WORKER_JS, media_type="application/javascript")


@app.get("/pcm-worklet.js")
async def pcm_worklet() -> Response:
    return Response(web_assets.PCM_WORKLET_JS, media_type="application/javascript")


@app.get("/icon-192.png")
async def icon_192() -> Response:
    return Response(web_assets.icon_png(192), media_type="image/png")


@app.get("/icon-512.png")
async def icon_512() -> Response:
    return Response(web_assets.icon_png(512), media_type="image/png")


# On-track heartbeat beep, played in the browser on type="heartbeat" events while
# navigation is ON (the phone is the speaker) — mirrors audio/consumer's laptop
# afplay beep. Same Purr.mp3 the offline demo videos use (sound_asset/).
_BEEP_BYTES: Optional[bytes] = None


@app.get("/beep.mp3")
async def beep_mp3() -> Response:
    global _BEEP_BYTES
    if _BEEP_BYTES is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "sound_asset", "Purr.mp3")
        try:
            with open(path, "rb") as f:
                _BEEP_BYTES = f.read()
        except OSError:
            _BEEP_BYTES = b""
    return Response(_BEEP_BYTES, media_type="audio/mpeg",
                    headers={"Cache-Control": "public, max-age=86400"})


# Server-side speech for iOS. iOS Safari blocks the Web Speech API (especially as
# an installed PWA), so the phone can't speak guidance itself — but it plays HTML5
# <audio> fine (the beep proves it). So we synthesize each phrase here with macOS
# `say` (same engine the offline demo videos use) and the iOS client fetches +
# plays it through the same audio channel as the beep. Android keeps using its
# native (working) SpeechSynthesis, so it never hits this route.
_TTS_CACHE: dict[str, bytes] = {}
_TTS_MAX = 256                      # cap the cache (ambient vocab is small; OCR varies)


def _synth_tts(text: str) -> bytes:
    """`say` -> WAV bytes for one phrase, cached by text. Blocking; call in an
    executor so it never stalls the event loop."""
    key = (text or "").strip()
    if not key:
        return b""
    cached = _TTS_CACHE.get(key)
    if cached is not None:
        return cached
    import subprocess
    import tempfile
    path = None
    data = b""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            path = tf.name
        subprocess.run(
            ["say", "-r", "190", "--file-format=WAVE",
             "--data-format=LEI16@22050", "-o", path, key],
            check=True, timeout=10,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(path, "rb") as f:
            data = f.read()
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("tts synth failed for %r: %s", key, exc)
    finally:
        if path:
            try:
                os.remove(path)
            except OSError:
                pass
    if len(_TTS_CACHE) < _TTS_MAX:
        _TTS_CACHE[key] = data
    return data


@app.get("/tts")
async def tts(text: str = "") -> Response:
    data = await asyncio.get_event_loop().run_in_executor(None, _synth_tts, text)
    return Response(data, media_type="audio/wav",
                    headers={"Cache-Control": "public, max-age=86400"})


# --------------------------------------------------------------------------- #
# Live monitor — watch the phone's stream WITH detection overlay on the laptop.
# --------------------------------------------------------------------------- #
@app.get("/monitor", response_class=HTMLResponse)
async def monitor() -> str:
    return _MONITOR_HTML


@app.get("/stream.mjpg")
async def stream_mjpg() -> StreamingResponse:
    """MJPEG stream of the latest annotated (overlaid) frame from the vision
    pipeline. Open /monitor in a laptop browser to watch detections live."""
    import cv2

    async def gen():
        import main
        while True:
            frame = main.get_annotated_frame()
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + buf.tobytes() + b"\r\n")
            await asyncio.sleep(1 / 15)

    return StreamingResponse(
        gen(), media_type="multipart/x-mixed-replace; boundary=frame")


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />
<title>Aakha</title>
<meta name="theme-color" content="#101418" />
<meta name="mobile-web-app-capable" content="yes" />
<meta name="apple-mobile-web-app-capable" content="yes" />
<meta name="apple-mobile-web-app-status-bar-style" content="black" />
<meta name="apple-mobile-web-app-title" content="Aakha" />
<link rel="manifest" href="/manifest.json" />
<link rel="apple-touch-icon" href="/icon-192.png" />
<style>
  html, body { margin: 0; height: 100%; background: #000; overflow: hidden; }
  #tap {
    position: fixed; inset: 0; width: 100vw; height: 100vh;
    border: none; color: #fff; font: 700 8vw/1.2 system-ui, sans-serif;
    background: #101418; display: flex; align-items: center; justify-content: center;
    text-align: center; white-space: pre-line; -webkit-tap-highlight-color: transparent;
    touch-action: none; user-select: none;
  }
  #tap.active { background: #0b3d2e; }
  #tap.rec { background: #3d1f1f; }
  #tap:focus-visible { outline: 6px solid #4c8dff; outline-offset: -12px; }
  #hint { position: fixed; left: 0; right: 0; bottom: env(safe-area-inset-bottom, 12px);
    color: #9aa4ad; font: 500 3.5vw system-ui, sans-serif; text-align: center; pointer-events: none; }
  video, canvas { display: none; }
</style>
</head>
<body>
  <button id="tap" aria-label="Touch to start or resume navigation. Press and hold to speak a command.">Touch to start
navigation</button>
  <div id="hint" aria-hidden="true">camera: connecting…</div>
  <video id="v" playsinline muted autoplay></video>
  <canvas id="c"></canvas>
<script>
(function () {
  const OPEN = 1; // WebSocket.OPEN
  const wsBase = (location.protocol === "https:" ? "wss" : "ws") + "://" + location.host;
  const btn = document.getElementById("tap");
  const hint = document.getElementById("hint");
  const video = document.getElementById("v");
  const canvas = document.getElementById("c");
  const ctx = canvas.getContext("2d");
  const FPS = 12, JPEG_Q = 0.6, MAX_W = 640;

  // --- resilient websocket helper (returns a getter for the live socket) ---
  function connect(path, onmessage) {
    let ws;
    const open = () => {
      ws = new WebSocket(wsBase + path);
      ws.binaryType = "arraybuffer";
      ws.onmessage = (e) => onmessage && onmessage(e);
      ws.onclose = () => setTimeout(open, 1000);
      ws.onerror = () => { try { ws.close(); } catch (_) {} };
    };
    open();
    return () => ws;
  }

  const getControl = connect("/control", (e) => {
    try { const r = JSON.parse(e.data); if (typeof r.active === "boolean") setActive(r.active); } catch (_) {}
  });
  const getCamera = connect("/camera");
  const getAudio = connect("/audio");

  // --- on-device TTS: speak spoken-type bus events (phone is the voice) ---
  const SPEAK = new Set(["obstacle","collision","crosswalk","traffic_light","path",
    "path_drift","ocr","held_object","voice_no_match","voice_heard","control",
    "voice_error"]);
  // Events that speak even when navigation is OFF: on/off confirmations and
  // user-requested voice responses (the command is explicit, so echo/answer it).
  // voice_error = command failed.
  const ALWAYS_SPEAK = new Set(["control","voice_heard","voice_no_match","ocr",
    "held_object","voice_error"]);
  // Replies to an explicit voice command. During a voice session these are the
  // only things spoken; ambient guidance is paused so the answer is heard clean.
  const REPLY = new Set(["ocr","held_object","voice_no_match","voice_error",
    "voice_heard"]);
  // A reply that ENDS the voice session once it finishes speaking (guidance then
  // resumes). voice_heard is spoken but keeps the session open (it's an ack —
  // the real answer is still coming).
  const REPLY_TERMINAL = new Set(["ocr","held_object","voice_no_match",
    "voice_error"]);
  // On-track heartbeat beep — the phone is the speaker (mirrors the laptop's
  // afplay beep). Unlocked on the first tap below (iOS needs a user gesture).
  const beep = new Audio("/beep.mp3"); beep.preload = "auto";
  let ttsUnlocked = false;   // iOS: SpeechSynthesis primed on first tap (M3)
  function playBeep() {
    try { beep.currentTime = 0; beep.play().catch(() => {}); } catch (_) {}
  }
  // iOS Safari blocks the Web Speech API (silent, esp. as an installed PWA), but
  // plays HTML5 <audio> — so on iOS we speak guidance via the server's /tts synth
  // through this element (blessed on first tap, like the beep). Android/desktop
  // keep their native SpeechSynthesis. iPadOS reports as Mac, so also check touch.
  const isIOS = /iP(hone|od|ad)/.test(navigator.userAgent)
    || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  const SILENT_WAV = "data:audio/wav;base64,UklGRiwAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQgAAACAgICAgICAgA==";
  const voice = new Audio(); voice.preload = "auto";
  let voiceBusy = false, voicePending = null;   // 1-deep freshest-cue queue (iOS)
  // Speak `text` via the server synth and, when it ends, immediately play the
  // freshest cue that arrived while it was busy (voicePending) — this gives
  // laptop-like BACK-TO-BACK continuity instead of gaps, while only ever holding
  // ONE queued cue (the newest) so speech can't fall behind reality. interrupt
  // (CRITICAL / a reply) cuts whatever's playing. onEnd fires on a terminal reply.
  function speakAudio(text, interrupt, onEnd) {
    if (interrupt) { try { voice.pause(); } catch (_) {} }
    voiceBusy = true;
    const done = () => {
      voiceBusy = false;
      if (onEnd) onEnd();
      if (voicePending) {                 // drain the queued cue back-to-back
        const nx = voicePending; voicePending = null;
        speakAudio(nx.text, false, nx.onEnd);
      }
    };
    voice.onended = done; voice.onerror = done;
    voice.src = "/tts?text=" + encodeURIComponent(text);
    voice.play().catch(done);
  }
  connect("/status", (e) => {
    let p; try { p = JSON.parse(e.data); } catch (_) { return; }
    if (p.kind !== "event") return;
    // heartbeat: non-verbal "you're on track" cue — beep (only while navigating),
    // never spoken. Handled before the SPEAK guard (it has no message).
    if (p.type === "heartbeat") { if (active) playBeep(); return; }
    if (!p.message || !SPEAK.has(p.type)) return;
    const isReply = REPLY.has(p.type) || p.type === "control";
    // Voice session active: pause ambient guidance so the reply is heard
    // cleanly (this is what was talking over the OCR answer). CRITICAL collision
    // still speaks — safety overrides a voice query.
    if (voiceActive && !isReply && p.priority_name !== "CRITICAL") return;
    // Outside a voice session: when navigation is OFF, silence autonomous
    // guidance but still speak control confirmations and voice responses.
    if (!voiceActive && !active && !ALWAYS_SPEAK.has(p.type)) return;
    // Mobile TTS is a single ~real-time channel; the pipeline can offer cues
    // faster than they can actually be spoken. Keep the voice CURRENT without
    // dropping everything (the phone's "missed many sounds" symptom):
    //   - CRITICAL (looming danger) always interrupts whatever's speaking.
    //   - Any other ambient cue is SKIPPED while something is already being
    //     spoken — NOT queued (which lags behind reality) and NOT cut off
    //     mid-word (that swallowed cues). The arbiter re-offers the next relevant
    //     cue on its next beat, so nothing important stays lost for long.
    //   - Replies (OCR answer, etc.) are never skipped; their channel was
    //     already cleared when the voice session started.
    const critical = p.priority_name === "CRITICAL";
    const terminal = voiceActive && REPLY_TERMINAL.has(p.type);
    // A terminal reply, once it starts, clears the safety timeout so guidance
    // can't resume mid-answer; its end resumes guidance.
    const onEnd = terminal ? endVoiceMode : null;
    try {
      if (isIOS) {
        // iOS: speak through the server synth (Web Speech API is silent here).
        // CRITICAL / replies interrupt now; an ordinary cue that lands mid-clip is
        // held as the freshest-pending one (overwriting any older wait) and played
        // back-to-back when the current clip ends — continuous, but never stale.
        if (terminal && voiceTimer) { clearTimeout(voiceTimer); voiceTimer = null; }
        if (!critical && !isReply && voiceBusy) { voicePending = { text: p.message, onEnd }; return; }
        speakAudio(p.message, critical || isReply, onEnd);
      } else {
        if (critical) speechSynthesis.cancel();
        else if (!isReply && (speechSynthesis.speaking || speechSynthesis.pending)) return;
        speechSynthesis.resume();   // Android Chrome sometimes suspends the queue
        const u = new SpeechSynthesisUtterance(p.message);
        u.lang = "en-US";    // Android Chrome silently drops utterances with no lang
        u.rate = critical ? 1.2 : 1.05;
        if (terminal) {
          u.onstart = () => { if (voiceTimer) { clearTimeout(voiceTimer); voiceTimer = null; } };
          u.onend = endVoiceMode;      // resume guidance once the answer is spoken
          u.onerror = endVoiceMode;
        }
        speechSynthesis.speak(u);
      }
    } catch (_) { if (terminal) endVoiceMode(); }
  });

  let active = false;       // navigating (guidance on)? — driven by /control acks
  let started = false;      // app started (camera streaming)?
  function setActive(on) {
    if (!on) { try { speechSynthesis.cancel(); } catch (_) {} } // kill queued/in-flight speech on stop
    active = on; render();
  }
  function render() {
    btn.classList.toggle("rec", recording);
    btn.classList.toggle("active", active);
    btn.textContent = recording ? "Recording…\\n(release to send)"
      : !started ? "Touch to start\\nnavigation"
      : (active ? "Navigation\\nhold to speak"
                : "Paused\\ntap for navigation · hold to speak");
  }
  // Speak an immediate gesture cue ("Recording started" / "Processing") on the
  // phone, using the same backend as guidance (iOS: server /tts audio element;
  // Android/desktop: Web Speech). interrupt=true so it takes the channel now.
  function clientSpeak(text, onDone) {
    try {
      if (isIOS) { speakAudio(text, true, onDone || null); }
      else {
        speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        u.lang = "en-US"; u.rate = 1.05;
        if (onDone) { u.onend = onDone; u.onerror = onDone; }
        speechSynthesis.speak(u);
      }
    } catch (_) { if (onDone) onDone(); }
  }

  // --- voice session: hold-to-talk pauses guidance until the reply is spoken --
  // While voiceActive, the /status handler drops ambient guidance (obstacle /
  // path / etc.) so it can't talk over the answer. The session ends when the
  // reply utterance finishes (onend) or armVoiceTimeout fires as a safety net.
  let voiceActive = false, voiceTimer = null;
  function sendControl(action) {
    // Best-effort control message to the backend. UI never blocks on it.
    const c = getControl();
    if (c && c.readyState === OPEN) { try { c.send(JSON.stringify({ action })); } catch (_) {} }
  }
  function enterVoiceMode() {
    voiceActive = true;
    sendControl("voice_start");
    try { speechSynthesis.cancel(); } catch (_) {} // silence any in-flight guidance
  }
  // End the command interaction and go to PAUSED. Navigation NEVER auto-resumes
  // after a command — the user taps to go back to navigation. nav_pause is silent.
  function endVoiceMode() {
    if (voiceTimer) { clearTimeout(voiceTimer); voiceTimer = null; }
    if (!voiceActive) return;
    voiceActive = false;
    sendControl("voice_end");
    sendControl("nav_pause");
    hint.textContent = "tap for navigation · hold to speak";
    render();
  }
  function armVoiceTimeout(ms) {
    if (voiceTimer) clearTimeout(voiceTimer);
    voiceTimer = setTimeout(endVoiceMode, ms); // go paused even if no reply arrives
  }

  // --- camera stream ---
  let cameraStarted = false, cameraLive = false;
  async function startCamera() {
    if (cameraStarted) return;
    cameraStarted = true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } }, audio: false });
      video.srcObject = stream; await video.play();
      cameraLive = true; started = true; render();
      hint.textContent = "camera: streaming"; streamLoop();
    } catch (err) {
      cameraStarted = false;
      hint.textContent = "camera blocked — needs HTTPS on phones (" + err.name + ")";
    }
  }
  function streamLoop() {
    setInterval(() => {
      // Stream continuously once the camera is live — navigating OR paused — so a
      // voice command always sees fresh frames (no freeze). Guidance is gated by
      // nav-active on the server, not by withholding frames here.
      if (!cameraLive || !video.videoWidth) return;
      const cam = getCamera();
      if (!cam || cam.readyState !== OPEN) return;
      const scale = Math.min(1, MAX_W / video.videoWidth);
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((b) => { if (b && cam.readyState === OPEN) cam.send(b); }, "image/jpeg", JPEG_Q);
    }, Math.round(1000 / FPS));
  }

  // --- push-to-talk: mic -> 16kHz PCM16 (AudioWorklet) -> /audio ---
  let audioCtx, micNode, worklet, recording = false, chunks = [];
  async function setupAudio() {
    if (audioCtx) return;
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    await audioCtx.audioWorklet.addModule("/pcm-worklet.js");
    const mic = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
    micNode = audioCtx.createMediaStreamSource(mic);
    worklet = new AudioWorkletNode(audioCtx, "pcm16-downsampler");
    worklet.port.onmessage = (e) => { if (recording) chunks.push(new Int16Array(e.data)); };
    micNode.connect(worklet);
    // The worklet must reach the destination or the graph won't pull it (no
    // process() calls = no audio captured). It writes NO audio to its output,
    // so routing it to the destination is silent — no feedback.
    worklet.connect(audioCtx.destination);
  }
  async function startRecording() {
    try { await setupAudio(); } catch (err) { hint.textContent = "mic blocked (" + err.name + ")"; return; }
    if (audioCtx.state === "suspended") await audioCtx.resume();
    chunks = []; recording = true; render();
    if (navigator.vibrate) navigator.vibrate(30);
    hint.textContent = "recording… (release to send)";
  }
  function stopRecording() {
    if (!recording) return false;
    recording = false; render();
    let len = 0; chunks.forEach((c) => len += c.length);
    const pcm = new Int16Array(len); let off = 0;
    chunks.forEach((c) => { pcm.set(c, off); off += c.length; });
    const a = getAudio();
    if (a && a.readyState === OPEN && pcm.length) {
      a.send(pcm.buffer); hint.textContent = "command sent"; return true;
    }
    hint.textContent = "nothing captured"; return false;
  }

  // --- single fullscreen target: tap = toggle nav, hold = talk ---
  let holdTimer = null, held = false;
  btn.addEventListener("pointerdown", (ev) => {
    ev.preventDefault(); held = false; startCamera();
    // iOS blesses each <audio> element individually: an element may only be
    // play()'d later from a non-gesture callback if it was first play()'d inside
    // a user gesture. So bless BOTH the beep and the server-TTS voice element on
    // this first tap by playing a near-silent clip through each.
    try { beep.play().then(() => { beep.pause(); beep.currentTime = 0; }).catch(() => {}); } catch (_) {}
    if (!ttsUnlocked) {
      try {
        voice.src = SILENT_WAV;
        voice.play().then(() => { voice.pause(); voice.currentTime = 0; }).catch(() => {});
        ttsUnlocked = true;
      } catch (_) {}
    }
    // Arm hold-to-talk only once the camera is live. On the FIRST press the
    // camera-permission dialog is still up; without this guard the timer fires
    // during the dialog and steals the first tap. First tap just starts nav.
    // After a 1s hold: vibrate to cue "speak now", enter voice mode (guidance
    // pauses), and begin recording. startRecording()'s own buzz is the cue.
    if (cameraLive) holdTimer = setTimeout(() => {
      held = true;
      enterVoiceMode();
      clientSpeak("Recording started");   // announce, then capture
      startRecording();
    }, 1000);
  });
  function endPress() {
    if (holdTimer) { clearTimeout(holdTimer); holdTimer = null; }
    if (held) {
      held = false;
      const sent = stopRecording();
      // Say "Processing"; the result is spoken when it arrives, then we go PAUSED
      // (never auto-resume). 10 s safety net -> paused if no reply ever comes.
      if (sent) { clientSpeak("Processing"); hint.textContent = "processing…"; armVoiceTimeout(10000); }
      else endVoiceMode();                // nothing captured -> straight to paused
      return;
    }
    // Tap toggles navigation: nav_on resumes ("Navigation"), nav_off pauses it.
    if (navigator.vibrate) navigator.vibrate(20);
    sendControl(active ? "nav_off" : "nav_on");
  }
  btn.addEventListener("pointerup", endPress);
  btn.addEventListener("pointercancel", () => { if (held) endPress(); });

  render();
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
})();
</script>
</body>
</html>
"""


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>VisionAid — dashboard</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #0b0e11; color: #e6edf3; font: 14px/1.5 system-ui, sans-serif; }
  header { display: flex; align-items: center; gap: 12px; padding: 12px 16px; border-bottom: 1px solid #222; }
  #dot { width: 14px; height: 14px; border-radius: 50%; background: #f1c40f; box-shadow: 0 0 8px currentColor; color: #f1c40f; }
  #dot.up { background: #2ecc71; color: #2ecc71; }
  #dot.down { background: #e74c3c; color: #e74c3c; }
  h1 { font-size: 15px; margin: 0; font-weight: 600; }
  #conn { color: #9aa4ad; }
  main { display: grid; grid-template-columns: 240px 1fr; gap: 16px; padding: 16px; }
  .panel { background: #11161b; border: 1px solid #222; border-radius: 10px; padding: 12px; }
  .panel h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: #7d8790; margin: 0 0 8px; }
  .thread { display: flex; align-items: center; gap: 8px; padding: 4px 0; }
  .tdot { width: 9px; height: 9px; border-radius: 50%; background: #e74c3c; }
  .tdot.alive { background: #2ecc71; }
  .tname { font-family: ui-monospace, monospace; }
  ul#log { list-style: none; margin: 0; padding: 0; max-height: 78vh; overflow: auto; }
  #log li { padding: 8px 10px; border-left: 3px solid #444; margin-bottom: 6px; background: #0d1217; border-radius: 4px; }
  #log li .meta { color: #7d8790; font: 11px ui-monospace, monospace; }
  #log li.CRITICAL { border-left-color: #e74c3c; }
  #log li.NORMAL   { border-left-color: #4c8dff; }
  #log li.LOW      { border-left-color: #7d8790; }
  .tag { display: inline-block; font: 10px ui-monospace, monospace; padding: 1px 6px; border-radius: 999px; background: #1c2530; color: #9fb3c8; margin-right: 6px; }
</style>
</head>
<body>
  <header>
    <span id="dot" title="connection"></span>
    <h1>VisionAid dashboard</h1>
    <span id="conn">connecting…</span>
  </header>
  <main>
    <section class="panel">
      <h2>Threads</h2>
      <div id="threads"></div>
    </section>
    <section class="panel">
      <h2>Events (last 20)</h2>
      <ul id="log"></ul>
    </section>
  </main>
<script>
(function () {
  const wsBase = (location.protocol === "https:" ? "wss" : "ws") + "://" + location.host;
  const dot = document.getElementById("dot");
  const conn = document.getElementById("conn");
  const threadsEl = document.getElementById("threads");
  const log = document.getElementById("log");
  const threads = new Map();
  const MAX = 20;

  function setConn(state) {
    dot.className = state === "up" ? "up" : state === "down" ? "down" : "";
    conn.textContent = state === "up" ? "connected" : state === "down" ? "disconnected — retrying" : "connecting…";
  }

  function renderThreads() {
    threadsEl.innerHTML = "";
    for (const [name, t] of threads) {
      const row = document.createElement("div");
      row.className = "thread";
      const d = document.createElement("span");
      d.className = "tdot" + (t.alive ? " alive" : "");
      const n = document.createElement("span");
      n.className = "tname";
      n.textContent = name;
      row.append(d, n);
      threadsEl.append(row);
    }
  }

  function addEvent(ev) {
    const li = document.createElement("li");
    li.className = ev.priority_name || "NORMAL";
    const t = ev.timestamp ? new Date(ev.timestamp * 1000).toLocaleTimeString() : "";
    li.innerHTML = '<span class="tag">' + (ev.type || "generic") + "</span>" +
      '<span class="msg"></span><div class="meta"></div>';
    li.querySelector(".msg").textContent = ev.message || "";
    li.querySelector(".meta").textContent = (ev.priority_name || "") + " · " + (ev.source || "?") + " · " + t;
    log.prepend(li);
    while (log.children.length > MAX) log.removeChild(log.lastChild);
  }

  function connect() {
    setConn("connecting");
    const ws = new WebSocket(wsBase + "/status");
    ws.onopen = () => setConn("up");
    ws.onclose = () => { setConn("down"); setTimeout(connect, 1000); };
    ws.onerror = () => { try { ws.close(); } catch (_) {} };
    ws.onmessage = (e) => {
      let p; try { p = JSON.parse(e.data); } catch (_) { return; }
      if (p.kind === "heartbeat") { threads.set(p.thread, { alive: p.alive }); renderThreads(); }
      // skip the on-track audio-beep events (type "heartbeat") — they'd flood the
      // log at 1/s and carry no message; they're an audio cue, not a log entry.
      else if (p.kind === "event" && p.type !== "heartbeat") { addEvent(p); }
    };
  }
  connect();
})();
</script>
</body>
</html>
"""


_MONITOR_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Aakha — live detection</title>
<style>
  body { margin: 0; background: #0b0e11; color: #e6edf3;
    font: 14px/1.5 system-ui, sans-serif; text-align: center; }
  header { padding: 10px 16px; border-bottom: 1px solid #222; font-weight: 600; }
  #wrap { padding: 12px; }
  img { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #222; }
  .note { color: #7d8790; padding: 8px; }
</style>
</head>
<body>
  <header>Aakha — live detection (phone camera)</header>
  <div id="wrap">
    <img src="/stream.mjpg" alt="live annotated stream" />
    <p class="note">If blank: press <b>Tap to start</b> on the phone so frames stream in.
      Boxes + corridor + spoken banner are drawn here in real time.</p>
  </div>
</body>
</html>
"""


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Launch uvicorn. host=0.0.0.0 so a phone on the same WiFi can reach it."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_server()
