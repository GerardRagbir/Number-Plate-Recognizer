"""Per-connection heartbeat: one sample a second of how each source is actually behaving.

Pipelines and sources only bump cumulative counters; a single sampler thread (owned by the
manager) turns them into rates. Sampling is independent of the pipelines on purpose — a stalled
connection has to show up as a flat line, not as missing samples.
"""
from __future__ import annotations

import threading
import time
from collections import deque

WINDOW_S = 900  # 15 minutes at 1 Hz
STATES = ("idle", "starting", "running", "static", "reconnecting", "error", "ended", "stopped")
FIELDS = ("t", "state", "recv_fps", "proc_fps", "gap_ms", "kbps", "infer_ms", "reconnects")


class Counters:
    """Cumulative, written by the ingest/pipeline threads. Plain attribute bumps: no lock needed."""

    __slots__ = ("received", "processed", "bytes", "reconnects", "last_frame", "max_gap", "connected_at")

    def __init__(self):
        self.received = 0      # frames that arrived from the source
        self.processed = 0     # frames that went through the models
        self.bytes = 0         # compressed bytes off the wire (network streams only)
        self.reconnects = 0
        self.last_frame = 0.0  # perf_counter of the newest arrival
        self.max_gap = 0.0     # longest wait between two arrivals since the last sample
        self.connected_at = 0.0

    def arrived(self, n_bytes: int = 0) -> None:
        now = time.perf_counter()
        if self.last_frame:
            gap = now - self.last_frame
            if gap > self.max_gap:
                self.max_gap = gap
        self.last_frame = now
        self.received += 1
        self.bytes += n_bytes


class Heartbeat:
    def __init__(self):
        self.counters = Counters()
        self.samples: deque = deque(maxlen=WINDOW_S)
        self._prev = (time.perf_counter(), 0, 0, 0)

    def sample(self, state: str, infer_ms: float, live: bool) -> None:
        c = self.counters
        now = time.perf_counter()
        t0, r0, p0, b0 = self._prev
        dt = max(now - t0, 1e-3)
        r, p, b = c.received, c.processed, c.bytes
        self._prev = (now, r, p, b)
        gap, c.max_gap = c.max_gap, 0.0
        if live and c.last_frame:  # a silence still in progress counts as a gap
            gap = max(gap, now - c.last_frame)
        active = state in ("running", "static")
        self.samples.append((
            round(time.time(), 2), STATES.index(state) if state in STATES else 0,
            round(max(r - r0, 0) / dt, 1), round(max(p - p0, 0) / dt, 1),
            round(gap * 1000) if active else None,
            round(max(b - b0, 0) * 8 / 1000 / dt) if b else None,
            round(infer_ms, 1) if active else None, c.reconnects,
        ))

    def series(self, window_s: float) -> list:
        since = time.time() - window_s
        return [s for s in self.samples if s[0] >= since]

    def summary(self) -> dict:
        recent = [s for s in self.samples if s[0] >= time.time() - 60]
        up = [s for s in recent if s[1] in (2, 3)]
        c = self.counters
        return {
            "uptime_s": round(time.perf_counter() - c.connected_at) if c.connected_at else None,
            "reconnects": c.reconnects,
            "availability_60s": round(len(up) / len(recent), 3) if recent else None,
            "last_frame_age_s": round(time.perf_counter() - c.last_frame, 1) if c.last_frame else None,
        }


class Sampler(threading.Thread):
    def __init__(self, tick):
        super().__init__(name="heartbeat", daemon=True)
        self._tick, self._halt = tick, threading.Event()

    def run(self) -> None:
        nxt = time.perf_counter()
        while not self._halt.is_set():
            nxt += 1.0
            try:
                self._tick()
            except Exception:
                pass
            self._halt.wait(max(0.05, nxt - time.perf_counter()))

    def stop(self) -> None:
        self._halt.set()
