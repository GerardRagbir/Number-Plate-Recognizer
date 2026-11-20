"""Frames and the CoreVideo / CoreImage glue.

A :class:`Frame` carries a ``CVPixelBuffer`` (what AVFoundation, ScreenCaptureKit, Vision and
Core ML all speak natively) and/or a BGRA numpy array, converting lazily between the two so a
frame decoded by VideoToolbox can reach Core ML and the JPEG encoder without touching numpy.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
import Quartz
from Foundation import NSDictionary

_BGRA = Quartz.kCVPixelFormatType_32BGRA
_PB_ATTRS = NSDictionary.dictionaryWithDictionary_(
    {
        Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
        Quartz.kCVPixelBufferMetalCompatibilityKey: True,
        Quartz.kCVPixelBufferCGImageCompatibilityKey: True,
    }
)


def pixel_buffer_to_bgra(pb) -> np.ndarray:
    """Copy a BGRA CVPixelBuffer into an (h, w, 4) uint8 array."""
    Quartz.CVPixelBufferLockBaseAddress(pb, Quartz.kCVPixelBufferLock_ReadOnly)
    try:
        h = Quartz.CVPixelBufferGetHeight(pb)
        w = Quartz.CVPixelBufferGetWidth(pb)
        stride = Quartz.CVPixelBufferGetBytesPerRow(pb)
        base = Quartz.CVPixelBufferGetBaseAddress(pb)
        buf = np.frombuffer(base.as_buffer(stride * h), dtype=np.uint8).reshape(h, stride)
        return np.ascontiguousarray(buf[:, : w * 4]).reshape(h, w, 4)
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pb, Quartz.kCVPixelBufferLock_ReadOnly)


def bgra_to_pixel_buffer(arr: np.ndarray):
    """Copy an (h, w, 4) BGRA array into a new IOSurface-backed CVPixelBuffer."""
    h, w = arr.shape[:2]
    err, pb = Quartz.CVPixelBufferCreate(None, w, h, _BGRA, _PB_ATTRS, None)
    if err or pb is None:
        raise RuntimeError(f"CVPixelBufferCreate failed ({err})")
    Quartz.CVPixelBufferLockBaseAddress(pb, 0)
    try:
        stride = Quartz.CVPixelBufferGetBytesPerRow(pb)
        base = Quartz.CVPixelBufferGetBaseAddress(pb)
        dst = np.frombuffer(base.as_buffer(stride * h), dtype=np.uint8).reshape(h, stride)
        dst[:, : w * 4] = arr.reshape(h, w * 4)
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pb, 0)
    return pb


@dataclass(slots=True)
class Frame:
    index: int
    ts: float  # seconds on the source's own clock
    wall: float  # unix time this frame represents (lets recorded files line up with each other)
    _pb: object | None = None
    _bgra: np.ndarray | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_pixel_buffer(cls, pb, index: int, ts: float, wall: float) -> "Frame":
        fmt = Quartz.CVPixelBufferGetPixelFormatType(pb)
        if fmt != _BGRA:
            raise ValueError(f"expected BGRA pixel buffer, got {fmt:#x}")
        return cls(index, ts, wall, _pb=pb)

    @classmethod
    def from_bgra(cls, arr: np.ndarray, index: int, ts: float, wall: float) -> "Frame":
        return cls(index, ts, wall, _bgra=np.ascontiguousarray(arr))

    @property
    def pixel_buffer(self):
        with self._lock:
            if self._pb is None:
                self._pb = bgra_to_pixel_buffer(self._bgra)
            return self._pb

    @property
    def bgra(self) -> np.ndarray:
        with self._lock:
            if self._bgra is None:
                self._bgra = pixel_buffer_to_bgra(self._pb)
            return self._bgra

    @property
    def size(self) -> tuple[int, int]:
        """(width, height)"""
        if self._bgra is not None:
            return self._bgra.shape[1], self._bgra.shape[0]
        return Quartz.CVPixelBufferGetWidth(self._pb), Quartz.CVPixelBufferGetHeight(self._pb)


class JPEGEncoder:
    """Scale + JPEG-encode pixel buffers on the GPU through a Metal-backed CIContext."""

    def __init__(self, max_width: int = 960, quality: float = 0.6):
        import Metal

        self.max_width = max_width
        self.quality = quality
        self.device = Metal.MTLCreateSystemDefaultDevice()
        opts = {Quartz.kCIContextCacheIntermediates: False}
        self.ctx = (
            Quartz.CIContext.contextWithMTLDevice_options_(self.device, opts)
            if self.device is not None
            else Quartz.CIContext.contextWithOptions_(opts)
        )
        self.cs = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceSRGB)
        self._opts = {Quartz.kCGImageDestinationLossyCompressionQuality: quality}

    def encode(self, pb, max_width: int | None = None, crop: tuple[float, float, float, float] | None = None) -> bytes:
        """Encode ``pb``; ``crop`` is (x1, y1, x2, y2) in top-left pixel coordinates."""
        img = Quartz.CIImage.imageWithCVPixelBuffer_(pb)
        h = Quartz.CVPixelBufferGetHeight(pb)
        w = Quartz.CVPixelBufferGetWidth(pb)
        if crop is not None:
            x1, y1, x2, y2 = crop
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            w, h0 = max(1.0, x2 - x1), max(1.0, y2 - y1)
            rect = Quartz.CGRectMake(x1, h - y2, w, h0)  # CoreImage origin is bottom-left
            img = img.imageByCroppingToRect_(rect).imageByApplyingTransform_(
                Quartz.CGAffineTransformMakeTranslation(-x1, -(h - y2))
            )
        mw = max_width or self.max_width
        if w > mw:
            s = mw / w
            img = img.imageByApplyingTransform_(Quartz.CGAffineTransformMakeScale(s, s))
        data = self.ctx.JPEGRepresentationOfImage_colorSpace_options_(img, self.cs, self._opts)
        return bytes(data)


def metal_device_name() -> str:
    try:
        import Metal

        dev = Metal.MTLCreateSystemDefaultDevice()
        return str(dev.name()) if dev is not None else "none"
    except Exception:  # pragma: no cover - informational only
        return "unknown"
