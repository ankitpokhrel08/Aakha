"""Compatibility gate. Run after every merge to main:

    python shared/smoke_test.py

Exits non-zero if main won't import, run() crashes, or a worker thread dies.
main is imported lazily so a half-built branch fails loudly HERE rather than
at import time somewhere deep in the app.
"""
from __future__ import annotations

import os
import sys
import threading
import time

# Allow `python shared/smoke_test.py` from the repo root: put the repo root on
# the path so `import main` and `from shared...` resolve regardless of cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.bus import event_bus


def run() -> int:
    print("Starting main.run() in a background thread...")
    try:
        import main
    except Exception as exc:  # import error in someone's branch
        print(f"FAIL: could not import main: {exc!r}")
        return 1

    crash: list[BaseException] = []

    def _target() -> None:
        try:
            main.run()
        except BaseException as exc:  # noqa: BLE001 — surface anything run() throws
            crash.append(exc)

    t = threading.Thread(target=_target, name="main.run", daemon=True)
    t.start()
    time.sleep(8)  # let threads spin up

    if crash:
        print(f"FAIL: main.run() raised: {crash[0]!r}")
        return 1

    workers = getattr(main, "WORKER_THREADS", [])
    if not workers:
        print("FAIL: main.run() started no worker threads")
        return 1

    dead = [w.name for w in workers if not w.is_alive()]
    if dead:
        print(f"FAIL: worker thread(s) died: {dead}")
        return 1

    print(f"Queue size after 8s: {event_bus.qsize()}")
    print(f"Worker threads alive: {[w.name for w in workers]}")
    print("Smoke test passed — no crash, threads alive.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
