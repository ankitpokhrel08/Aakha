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
from typing import Any, Callable, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

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

    if START_WORKERS:
        try:
            import main

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
            else:
                await ws.send_json({"ok": False, "error": f"unknown action: {action}"})
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("control ws error")


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


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />
<title>VisionAid</title>
<style>
  html, body { margin: 0; height: 100%; background: #000; overflow: hidden; }
  #tap {
    position: fixed; inset: 0; width: 100vw; height: 100vh;
    border: none; color: #fff; font: 700 8vw/1.2 system-ui, sans-serif;
    background: #101418; display: flex; align-items: center; justify-content: center;
    text-align: center; -webkit-tap-highlight-color: transparent; touch-action: manipulation;
  }
  #tap.active { background: #0b3d2e; }
  #tap:focus-visible { outline: 6px solid #4c8dff; outline-offset: -12px; }
  #hint { position: fixed; left: 0; right: 0; bottom: env(safe-area-inset-bottom, 12px);
    color: #9aa4ad; font: 500 3.5vw system-ui, sans-serif; text-align: center; pointer-events: none; }
  video, canvas { display: none; }
</style>
</head>
<body>
  <button id="tap" aria-label="Toggle navigation. Double tap to start or stop guidance.">Tap to start</button>
  <div id="hint" aria-hidden="true">camera: connecting…</div>
  <video id="v" playsinline muted autoplay></video>
  <canvas id="c"></canvas>
<script>
(function () {
  const wsBase = (location.protocol === "https:" ? "wss" : "ws") + "://" + location.host;
  const btn = document.getElementById("tap");
  const hint = document.getElementById("hint");
  const video = document.getElementById("v");
  const canvas = document.getElementById("c");
  const ctx = canvas.getContext("2d");
  const FPS = 12, JPEG_Q = 0.6, MAX_W = 640;

  // --- resilient websocket helper ---
  function connect(path, onopen, onmessage) {
    let ws;
    const open = () => {
      ws = new WebSocket(wsBase + path);
      ws.binaryType = "arraybuffer";
      ws.onopen = () => onopen && onopen(ws);
      ws.onmessage = (e) => onmessage && onmessage(e, ws);
      ws.onclose = () => setTimeout(open, 1000);
      ws.onerror = () => { try { ws.close(); } catch (_) {} };
    };
    open();
    return () => ws;
  }

  const getControl = connect("/control", null, (e) => {
    try { const r = JSON.parse(e.data); if (typeof r.active === "boolean") setActive(r.active); } catch (_) {}
  });
  const getCamera = connect("/camera");

  function setActive(on) {
    btn.classList.toggle("active", on);
    btn.textContent = on ? "Navigation ON\\n(tap to stop)" : "Tap to start";
  }

  let cameraStarted = false;
  async function startCamera() {
    if (cameraStarted) return;
    cameraStarted = true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } }, audio: false
      });
      video.srcObject = stream;
      await video.play();
      hint.textContent = "camera: streaming";
      streamLoop();
    } catch (err) {
      cameraStarted = false;
      hint.textContent = "camera blocked — needs HTTPS on phones (" + err.name + ")";
    }
  }

  function streamLoop() {
    setInterval(() => {
      if (!video.videoWidth) return;
      const cam = getCamera();
      if (!cam || cam.readyState !== WebSocket.OPEN) return;
      const scale = Math.min(1, MAX_W / video.videoWidth);
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => { if (blob && cam.readyState === WebSocket.OPEN) cam.send(blob); },
        "image/jpeg", JPEG_Q);
    }, Math.round(1000 / FPS));
  }

  function onTap() {
    if (navigator.vibrate) navigator.vibrate(200);   // client-side, no round-trip
    startCamera();                                    // first tap starts camera
    const ctrl = getControl();
    if (ctrl && ctrl.readyState === WebSocket.OPEN) ctrl.send(JSON.stringify({ action: "toggle" }));
  }
  btn.addEventListener("click", onTap);
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


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Launch uvicorn. host=0.0.0.0 so a phone on the same WiFi can reach it."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_server()
