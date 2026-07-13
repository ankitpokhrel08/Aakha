#!/usr/bin/env bash
# Start Aakha end to end: pipeline+server, a Cloudflare tunnel, and a QR code of
# the tunnel URL to scan in the Android app. Ctrl-C stops everything.
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
PORT="${PORT:-8000}"
LOGDIR="logs"
mkdir -p "$LOGDIR"

command -v cloudflared >/dev/null || { echo "cloudflared not found (brew install cloudflared)"; exit 1; }
command -v qrencode   >/dev/null || { echo "qrencode not found (brew install qrencode)"; exit 1; }
[ -x "$PY" ] || { echo "no venv at $PY"; exit 1; }

SERVER_PID="" ; TUNNEL_PID=""
cleanup() {
  echo
  echo "stopping…"
  [ -n "$TUNNEL_PID" ] && kill "$TUNNEL_PID" 2>/dev/null || true
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "starting Aakha server on :$PORT  (logs: $LOGDIR/server.log)"
$PY -m src.server.server >"$LOGDIR/server.log" 2>&1 &
SERVER_PID=$!

# wait for the server port to accept connections (models load on startup)
echo -n "waiting for server"
for _ in $(seq 1 90); do
  if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then exec 3>&- 3<&- ; echo " ok"; break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo; echo "server exited early:"; tail -20 "$LOGDIR/server.log"; exit 1; }
  echo -n "." ; sleep 1
done

echo "starting cloudflare tunnel  (logs: $LOGDIR/tunnel.log)"
cloudflared tunnel --url "http://localhost:$PORT" >"$LOGDIR/tunnel.log" 2>&1 &
TUNNEL_PID=$!

# pull the random https URL out of cloudflared's log once it appears
URL=""
echo -n "waiting for tunnel URL"
for _ in $(seq 1 60); do
  URL="$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOGDIR/tunnel.log" | head -1 || true)"
  [ -n "$URL" ] && break
  kill -0 "$TUNNEL_PID" 2>/dev/null || { echo; echo "tunnel exited early:"; tail -20 "$LOGDIR/tunnel.log"; exit 1; }
  echo -n "." ; sleep 1
done
[ -n "$URL" ] || { echo; echo "could not detect tunnel URL. See $LOGDIR/tunnel.log"; exit 1; }
echo " ok"

echo
echo "======================================================================"
echo "  Aakha is live:  $URL"
echo "======================================================================"
echo "  Scan this in the Aakha app (first screen), or type the URL manually:"
echo
qrencode -t ANSIUTF8 -m 2 "$URL"
echo
echo "  server + tunnel running · Ctrl-C to stop"
echo

# stay in the foreground; if either child dies, fall through to cleanup
wait -n "$SERVER_PID" "$TUNNEL_PID" 2>/dev/null || wait
