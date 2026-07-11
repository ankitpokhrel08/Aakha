"""Static web assets for the PWA, served inline by server.py.

Kept as strings/bytes (not a static dir) so `import server` stays filesystem-
coupling-free and the tunnel/HTTPS setup doesn't depend on a static mount.
Contains: the web manifest, a minimal service worker, the AudioWorklet PCM
downsampler (mic -> 16 kHz mono PCM16 for Vosk), and a generated app icon.
"""
from __future__ import annotations

import io
import json

MANIFEST = json.dumps({
    "name": "Aakha — assistive navigation",
    "short_name": "Aakha",
    "description": "Real-time spoken navigation guidance for blind and low-vision users.",
    "start_url": "/",
    "scope": "/",
    "display": "standalone",
    "orientation": "portrait",
    "background_color": "#101418",
    "theme_color": "#101418",
    "icons": [
        {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png",
         "purpose": "any maskable"},
        {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png",
         "purpose": "any maskable"},
    ],
}, indent=2)


# Minimal service worker: precache the shell, network-first for everything (so
# fresh events/pages always win), fall back to cache/'/' offline. WebSocket
# traffic is never intercepted.
SERVICE_WORKER_JS = """
const CACHE = 'aakha-v1';
self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.add('/')).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((keys) =>
    Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
  ).then(() => self.clients.claim()));
});
self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.protocol === 'ws:' || url.protocol === 'wss:') return;
  e.respondWith(fetch(e.request).catch(() => caches.match(e.request).then((r) => r || caches.match('/'))));
});
"""


# AudioWorklet processor: resample the mic (context rate, e.g. 48 kHz) down to
# 16 kHz mono, convert Float32 -> Int16, and post raw PCM chunks to the main
# thread. Loaded via audioWorklet.addModule('/pcm-worklet.js').
PCM_WORKLET_JS = """
class PCM16Downsampler extends AudioWorkletProcessor {
  constructor() { super(); this.targetRate = 16000; this._pos = 0; }
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];                          // mono: first channel
    const ratio = sampleRate / this.targetRate;   // e.g. 48000/16000 = 3
    const out = [];
    let pos = this._pos;
    while (pos < ch.length) {
      const i = Math.floor(pos), frac = pos - i;
      const s0 = ch[i], s1 = (i + 1 < ch.length) ? ch[i + 1] : s0;
      let s = s0 + (s1 - s0) * frac;               // linear interpolation
      s = Math.max(-1, Math.min(1, s));
      out.push(s < 0 ? s * 0x8000 : s * 0x7fff);   // Float32 -> Int16
      pos += ratio;
    }
    this._pos = pos - ch.length;                   // carry fractional remainder
    if (out.length) {
      const pcm = Int16Array.from(out);
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}
registerProcessor('pcm16-downsampler', PCM16Downsampler);
"""


def icon_png(size: int) -> bytes:
    """Generate a simple app icon (dark bg + white 'aperture') as PNG bytes."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (size, size), (16, 20, 24))
    d = ImageDraw.Draw(img)
    c = size / 2
    r_out = size * 0.34
    d.ellipse([c - r_out, c - r_out, c + r_out, c + r_out], fill=(230, 237, 243))
    r_in = size * 0.14
    d.ellipse([c - r_in, c - r_in, c + r_in, c + r_in], fill=(16, 20, 24))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
