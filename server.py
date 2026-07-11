"""VisionAid web layer — phone camera client + dashboard over WebSockets.

Dev 3 / mobile-client. This is the "Option 1" rig from the plan: the phone runs
a lightweight browser client that streams its camera to this laptop backend over
WiFi; inference happens here; the sighted dashboard watches the event bus.

Endpoints
---------
GET  /            fullscreen single-button end-user page (camera + toggle)
GET  /dashboard   sighted-teammate view: connection light + scrolling event log
WS   /camera      browser -> server: JPEG frames, handed to on_frame() callback
WS   /control     browser -> server: {"action": "toggle"} etc.
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
            # Guard: while navigation is OFF, drain the socket but drop the frame
            # so the pipeline stays idle even if an old client keeps streaming (A1).
            if not STATE["active"]:
                continue
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
    """Receive control actions, e.g. {"action": "toggle"} from the big button."""
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            action = (msg or {}).get("action")
            if action == "toggle":
                STATE["active"] = not STATE["active"]
                on = STATE["active"]
                if not on:
                    # Stop the pipeline at the source: empty the pushed-frame
                    # slot so the vision loop idles instead of re-processing the
                    # last frame forever (A1). The client also stops streaming.
                    try:
                        import main
                        main.clear_incoming_frame()
                    except Exception:
                        logger.debug("could not clear incoming frame", exc_info=True)
                # Publish to the bus so it's both spoken (TTS) and shown (dashboard).
                event_bus.publish(
                    Event(
                        message="Navigation on" if on else "Navigation off",
                        priority=Priority.LOW,
                        type="control",
                        source="control",
                        data={"active": on},
                    )
                )
                await ws.send_json({"ok": True, "active": on})
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
  <button id="tap" aria-label="Tap to start or stop navigation. Press and hold to speak a command.">Tap to start</button>
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
    "path_drift","ocr","caption","held_object","voice_no_match","voice_heard","control"]);
  // Events that speak even when navigation is OFF: on/off confirmations and
  // user-requested voice responses (the command is explicit, so echo/answer it).
  const ALWAYS_SPEAK = new Set(["control","voice_heard","voice_no_match","ocr","held_object"]);
  connect("/status", (e) => {
    let p; try { p = JSON.parse(e.data); } catch (_) { return; }
    if (p.kind !== "event" || !p.message || !SPEAK.has(p.type)) return;
    // A2/A4: when navigation is OFF, silence autonomous guidance but still speak
    // control confirmations and user-requested voice responses.
    if (!active && !ALWAYS_SPEAK.has(p.type)) return;
    try {
      if (p.priority_name === "CRITICAL") speechSynthesis.cancel(); // danger interrupts
      const u = new SpeechSynthesisUtterance(p.message);
      u.rate = p.priority_name === "CRITICAL" ? 1.2 : 1.05;
      speechSynthesis.speak(u);
    } catch (_) {}
  });

  let active = false;
  function setActive(on) {
    if (!on) { try { speechSynthesis.cancel(); } catch (_) {} } // A2: kill queued/in-flight speech on stop
    active = on; btn.classList.toggle("active", on); render();
  }
  function render() {
    btn.classList.toggle("rec", recording);
    btn.textContent = recording ? "Listening…\\n(release to send)"
      : (active ? "Navigation ON\\ntap to stop · hold to speak"
                : "Tap to start\\nhold to speak");
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
      cameraLive = true;
      hint.textContent = "camera: streaming"; streamLoop();
    } catch (err) {
      cameraStarted = false;
      hint.textContent = "camera blocked — needs HTTPS on phones (" + err.name + ")";
    }
  }
  function streamLoop() {
    setInterval(() => {
      if (!active) return;              // navigation OFF -> stop streaming (A1)
      if (!video.videoWidth) return;
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
    hint.textContent = "listening… (release to send)";
  }
  function stopRecording() {
    if (!recording) return;
    recording = false; render();
    let len = 0; chunks.forEach((c) => len += c.length);
    const pcm = new Int16Array(len); let off = 0;
    chunks.forEach((c) => { pcm.set(c, off); off += c.length; });
    const a = getAudio();
    if (a && a.readyState === OPEN && pcm.length) { a.send(pcm.buffer); hint.textContent = "command sent"; }
    else hint.textContent = "nothing captured";
  }

  // --- single fullscreen target: tap = toggle nav, hold = talk ---
  let holdTimer = null, held = false;
  btn.addEventListener("pointerdown", (ev) => {
    ev.preventDefault(); held = false; startCamera();
    // Arm hold-to-talk only once the camera is live. On the FIRST press the
    // camera-permission dialog is still up; without this guard the 350ms timer
    // fires during the dialog and drops the user into the recording screen,
    // stealing the first tap. First tap should just start navigation.
    if (cameraLive) holdTimer = setTimeout(() => { held = true; startRecording(); }, 350);
  });
  function endPress() {
    if (holdTimer) { clearTimeout(holdTimer); holdTimer = null; }
    if (held) { stopRecording(); held = false; return; }
    if (navigator.vibrate) navigator.vibrate(20);
    const c = getControl();
    if (c && c.readyState === OPEN) c.send(JSON.stringify({ action: "toggle" }));
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
      else if (p.kind === "event") { addEvent(p); }
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
