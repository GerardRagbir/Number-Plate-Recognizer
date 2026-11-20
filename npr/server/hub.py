"""Bridges pipeline threads to asyncio websockets. Slow viewers get frames dropped, never queued."""
from __future__ import annotations

import asyncio
from collections import defaultdict, deque


class Hub:
    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self._tiles: dict[int, set[asyncio.Queue]] = defaultdict(set)
        self._events: set[asyncio.Queue] = set()
        self._last: dict[int, bytes] = {}
        self.recent: deque = deque(maxlen=200)  # replayed to dashboards that connect late

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    # ---- called from pipeline threads -------------------------------------------
    def has_viewers(self, sid: int) -> bool:
        return bool(self._tiles.get(sid))

    def frame(self, sid: int, packet: bytes) -> None:
        if self.loop is not None and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._frame, sid, packet)

    def event(self, ev: dict) -> None:
        if self.loop is not None and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._event, ev)

    # ---- loop side -------------------------------------------------------------------
    @staticmethod
    def _offer(q: asyncio.Queue, item) -> None:
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        q.put_nowait(item)

    def _frame(self, sid: int, packet: bytes) -> None:
        self._last[sid] = packet
        for q in self._tiles.get(sid, ()):
            self._offer(q, packet)

    def _event(self, ev: dict) -> None:
        if ev.get("type") in ("plate", "match"):
            self.recent.append(ev)
        for q in self._events:
            self._offer(q, ev)

    def watch_tile(self, sid: int) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._tiles[sid].add(q)
        if sid in self._last:
            q.put_nowait(self._last[sid])
        return q

    def unwatch_tile(self, sid: int, q: asyncio.Queue) -> None:
        self._tiles[sid].discard(q)

    def watch_events(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._events.add(q)
        return q

    def unwatch_events(self, q: asyncio.Queue) -> None:
        self._events.discard(q)

    def forget(self, sid: int) -> None:
        self._last.pop(sid, None)
