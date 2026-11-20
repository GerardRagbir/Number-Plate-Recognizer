"""Common interface for every ingest mode."""
from __future__ import annotations

import threading
from typing import Protocol

from ..native.frames import Frame


class Source(Protocol):
    live: bool  # True: frames arrive in real time and stale ones are dropped
    info: dict

    def open(self) -> None: ...
    def read(self) -> Frame | None: ...  # blocks; None = end of stream
    def close(self) -> None: ...


class LatestFrame:
    """Hand-off slot for push-style sources (capture callbacks): keeps only the newest frame."""

    def __init__(self):
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._closed = False
        self.error: str | None = None

    def put(self, frame: Frame) -> None:
        with self._cond:
            self._frame = frame
            self._cond.notify()

    def close(self, error: str | None = None) -> None:
        with self._cond:
            self._closed = True
            self.error = error or self.error
            self._cond.notify_all()

    def get(self, timeout: float = 10.0) -> Frame | None:
        with self._cond:
            if self._frame is None and not self._closed:
                self._cond.wait(timeout)
            if self._frame is None:
                if not self._closed:
                    self.error = self.error or f"no frames for {timeout:.0f}s"
                return None
            f, self._frame = self._frame, None
            return f


def wait_for(fn, timeout: float = 10.0):
    """Run an Objective-C API that takes a completion handler and block for its result."""
    done = threading.Event()
    box: list = []

    def handler(*args):
        box.append(args)
        done.set()

    fn(handler)
    if not done.wait(timeout):
        raise TimeoutError("native call timed out (is a permission prompt waiting?)")
    return box[0]
