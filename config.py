"""Runtime settings for VisionAid.

One shared, mutable ``config`` object every module reads live. Thresholds and
feature toggles can be flipped mid-run *without restarting the app* two ways:

  1. Edit ``config.json`` on disk -> the file watcher reloads it within
     ~``watch_interval`` seconds (call ``start_watching()`` once at boot).
  2. Call ``config.update(traffic_light_detection=False)`` from code (e.g. a
     server control action) -> applies live and persists to ``config.json``.

Because the single ``config`` instance is mutated *in place* (never rebound),
threads that captured a reference at startup see every change immediately — a
detection loop just reads ``config.crosswalk_detection`` each frame.

Cut-list rule (from CLAUDE.md / the plan): if something is unstable, cut in
order traffic_light_detection -> crosswalk_detection -> voice_trigger. NEVER
disable collision warning — it isn't even a toggle here, on purpose.

Import-safe: loading is a guarded one-shot file read; a missing/broken
config.json falls back to defaults instead of raising. The watcher thread is
opt-in (``start_watching()``), never started at import.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, fields
from typing import Any

logger = logging.getLogger("visionaid.config")

# config.json lives next to this file so cwd doesn't matter.
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

# Guards multi-field writes and the watcher. Reads of a single attribute are
# atomic under the GIL, so hot-path readers don't need to take this.
_lock = threading.RLock()


@dataclass
class Config:
    """Live tunables. Only these keys are honored from config.json; unknown
    keys are ignored (logged) so a typo can't inject arbitrary attributes."""

    # --- thresholds ---
    collision_distance_threshold_m: float = 1.5   # "very close" trigger distance
    target_fps: int = 12                          # camera stream / Tier 1 budget

    # --- feature toggles (all on by default; flip off per the cut-list) ---
    crosswalk_detection: bool = True
    traffic_light_detection: bool = True
    voice_trigger: bool = True

    # ----- runtime state (not persisted) -----
    _watcher: "threading.Thread | None" = None
    _watch_stop: "threading.Event | None" = None

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Persistable fields only (drops private/runtime state)."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if not f.name.startswith("_")
        }

    def _coerce(self, name: str, value: Any) -> Any:
        """Coerce an incoming value to the field's declared type (json numbers
        vs bools vs strings from an editor can be sloppy)."""
        declared = {f.name: f.type for f in fields(self)}.get(name)
        try:
            if declared in ("bool", bool):
                if isinstance(value, str):
                    return value.strip().lower() in ("1", "true", "yes", "on")
                return bool(value)
            if declared in ("int", int):
                return int(value)
            if declared in ("float", float):
                return float(value)
        except (TypeError, ValueError):
            logger.warning("config: bad value for %s=%r, ignoring", name, value)
            return getattr(self, name)  # keep current
        return value

    def _apply(self, data: dict[str, Any]) -> list[str]:
        """Apply a dict of settings in place. Returns the names that changed."""
        allowed = {f.name for f in fields(self) if not f.name.startswith("_")}
        changed: list[str] = []
        with _lock:
            for key, raw in (data or {}).items():
                if key not in allowed:
                    logger.warning("config: ignoring unknown key %r", key)
                    continue
                new = self._coerce(key, raw)
                if getattr(self, key) != new:
                    setattr(self, key, new)
                    changed.append(key)
        if changed:
            logger.info("config: updated %s", {k: getattr(self, k) for k in changed})
        return changed

    # ------------------------------------------------------------------ #
    def reload(self, path: str = CONFIG_PATH) -> list[str]:
        """Re-read config.json and apply it. Missing file -> keep current
        values (defaults on first load). Malformed file -> logged, ignored."""
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("config: could not read %s: %s (keeping current)", path, exc)
            return []
        if not isinstance(data, dict):
            logger.warning("config: %s is not a JSON object, ignoring", path)
            return []
        return self._apply(data)

    def update(self, path: str = CONFIG_PATH, **kwargs: Any) -> list[str]:
        """Apply settings from code AND persist to config.json so the change
        survives and stays consistent with the file the watcher reads."""
        changed = self._apply(kwargs)
        self.save(path)
        return changed

    def save(self, path: str = CONFIG_PATH) -> None:
        """Write current persistable settings to config.json (atomic-ish)."""
        with _lock:
            payload = self.to_dict()
        tmp = f"{path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
                fh.write("\n")
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("config: could not write %s: %s", path, exc)

    def is_enabled(self, feature: str) -> bool:
        """Convenience for producers: ``if config.is_enabled('crosswalk_detection')``."""
        return bool(getattr(self, feature, False))

    # ---- cut-list helper: disable the next feature per the agreed order ---
    CUT_ORDER = ("traffic_light_detection", "crosswalk_detection", "voice_trigger")

    def cut_next(self) -> str | None:
        """Turn off the next still-enabled feature in cut priority order.
        Returns the feature cut, or None if all optional features are already
        off. Collision warning is never in this list — it can't be cut."""
        for feature in self.CUT_ORDER:
            if getattr(self, feature):
                self.update(**{feature: False})
                logger.warning("config: CUT %s (cut-list rule)", feature)
                return feature
        return None

    # ------------------------------------------------------------------ #
    def start_watching(self, interval: float = 2.0, path: str = CONFIG_PATH) -> threading.Thread:
        """Start a daemon thread that reloads config.json when its mtime changes.
        Idempotent — repeated calls return the existing watcher. NOT started at
        import; call this once from run()/server startup to enable live edits."""
        with _lock:
            if self._watcher and self._watcher.is_alive():
                return self._watcher
            self._watch_stop = threading.Event()
            stop = self._watch_stop

            def _watch() -> None:
                last = _mtime(path)
                while not stop.wait(interval):
                    m = _mtime(path)
                    if m != last:
                        last = m
                        self.reload(path)

            t = threading.Thread(target=_watch, name="config-watcher", daemon=True)
            self._watcher = t
            t.start()
            logger.info("config: watching %s every %.1fs", path, interval)
            return t

    def stop_watching(self) -> None:
        with _lock:
            if self._watch_stop:
                self._watch_stop.set()
            self._watcher = None


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return -1.0


# The single shared instance every module imports: `from config import config`.
config = Config()
# Load config.json over the defaults at import (guarded; safe if absent/broken).
config.reload()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Loaded config:")
    print(json.dumps(config.to_dict(), indent=2))
    print(f"(from {CONFIG_PATH} if present, else defaults)")
