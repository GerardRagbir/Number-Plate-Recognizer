"""Network video streams (RTSP / HTTP / HLS / RTMP) via FFmpeg, decoding with VideoToolbox."""
from __future__ import annotations

import threading
import time

import av

from ..native.frames import Frame
from .base import LatestFrame


class StreamSource:
    live = True
    counts_arrivals = True  # bumps the heartbeat counters itself, with byte counts

    def __init__(self, url: str):
        self.url = url
        self.info: dict = {}
        self._slot = LatestFrame()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._container = None
        self.counters = None  # heartbeat.Counters, set by the pipeline: counts arrivals and bytes

    # Deliberately no "fflags=nobuffer" / "flags=low_delay": the VideoToolbox decoder rejects packets
    # with either set. Latency is handled by LatestFrame, which drops everything but the newest frame.
    OPTIONS = {"rtsp_transport": "tcp", "stimeout": "8000000", "rw_timeout": "8000000"}

    def _open(self, hardware: bool) -> None:
        kw = {}
        if hardware:
            from av.codec.hwaccel import HWAccel

            kw["hwaccel"] = HWAccel(device_type="videotoolbox", allow_software_fallback=True)
        self._container = av.open(self.url, options=self.OPTIONS, timeout=10, **kw)
        self.info["decoder"] = "videotoolbox" if hardware else "software"
        vs = self._container.streams.video[0]
        vs.thread_type = "AUTO"
        self.info.update(width=vs.codec_context.width, height=vs.codec_context.height,
                         fps=float(vs.average_rate or 0) or 25.0, duration=0.0)

    def open(self) -> None:
        try:
            self._open(hardware=True)
        except Exception:
            self._open(hardware=False)
        self._thread = threading.Thread(target=self._run, name="stream-decode", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        i = 0
        t0 = time.time()
        try:
            while True:
                try:
                    stream = self._container.streams.video[0]
                    for packet in self._container.demux(stream):
                        if self._stop.is_set():
                            break
                        size = packet.size
                        for frame in packet.decode():
                            i += 1
                            now = time.time()
                            if self.counters is not None:
                                self.counters.arrived(size)
                                size = 0
                            # hand over the decoded frame as-is: the costly colour conversion only
                            # happens for frames the pipeline actually takes (see read)
                            self._slot.put((frame, i, now - t0, now))
                    break
                except av.FFmpegError:
                    if i or self.info.get("decoder") != "videotoolbox" or self._stop.is_set():
                        raise
                    self._container.close()  # hardware decoder rejected this stream: retry in software
                    self._open(hardware=False)
            self._slot.close("stream ended")
        except Exception as e:  # network errors surface on the reading thread
            self._slot.close(f"{type(e).__name__}: {e}")

    def read(self) -> Frame | None:
        got = self._slot.get(timeout=15)
        if got is None:
            if self._slot.error and self._slot.error != "stream ended":
                raise ConnectionError(self._slot.error)
            return None
        frame, i, ts, wall = got
        return Frame.from_bgra(frame.to_ndarray(format="bgra"), i, ts, wall)

    def close(self) -> None:
        self._stop.set()
        self._slot.close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
