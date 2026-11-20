"""Live capture straight from the OS: displays/windows (ScreenCaptureKit) and cameras (AVFoundation)."""
from __future__ import annotations

import time

import AVFoundation as AVF
import CoreMedia
import objc
import Quartz
import ScreenCaptureKit as SCK
from Foundation import NSObject

from ..native.frames import Frame
from .base import LatestFrame, wait_for

try:
    from libdispatch import dispatch_queue_create
except ImportError:  # older pyobjc layouts
    from dispatch import dispatch_queue_create  # type: ignore

_BGRA = Quartz.kCVPixelFormatType_32BGRA
_SYSTEM_APPS = {"WindowManager", "Dock", "Control Centre", "Control Center", "Notification Centre"}


def _frame_from_sample(sb, index: int, t0: float) -> Frame | None:
    pb = CoreMedia.CMSampleBufferGetImageBuffer(sb)
    if pb is None or Quartz.CVPixelBufferGetPixelFormatType(pb) != _BGRA:
        return None
    now = time.time()
    return Frame.from_pixel_buffer(pb, index, now - t0, now)


# ---- screen ------------------------------------------------------------------------------

def shareable_content():
    content, err = wait_for(SCK.SCShareableContent.getShareableContentWithCompletionHandler_, 8)
    if content is None:
        raise PermissionError(
            "Screen capture unavailable — grant Screen Recording permission to the app running NPR "
            f"(System Settings › Privacy & Security). {err.localizedDescription() if err else ''}".strip())
    return content


def list_screens() -> dict:
    content = shareable_content()
    displays = [{"id": int(d.displayID()), "width": int(d.width()), "height": int(d.height()),
                 "name": f"Display {i + 1} ({int(d.width())}×{int(d.height())})"}
                for i, d in enumerate(content.displays())]
    windows = []
    for w in content.windows():
        f = w.frame()
        app = w.owningApplication()
        if not w.isOnScreen() or not w.title() or f.size.width < 200 or f.size.height < 150 or app is None:
            continue
        if w.windowLayer() != 0 or str(app.applicationName()) in _SYSTEM_APPS:
            continue
        windows.append({"id": int(w.windowID()), "app": str(app.applicationName()), "title": str(w.title()),
                        "width": int(f.size.width), "height": int(f.size.height)})
    return {"displays": displays, "windows": windows}


class _StreamOutput(NSObject):
    def initWithSlot_(self, slot):
        self = objc.super(_StreamOutput, self).init()
        if self is not None:
            self.slot, self.n, self.t0 = slot, 0, time.time()
        return self

    # SCStreamOutput
    def stream_didOutputSampleBuffer_ofType_(self, stream, sb, kind):
        if kind != SCK.SCStreamOutputTypeScreen:
            return
        self.n += 1
        f = _frame_from_sample(sb, self.n, self.t0)
        if f is not None:
            self.slot.put(f)

    # SCStreamDelegate
    def stream_didStopWithError_(self, stream, error):
        self.slot.close(str(error.localizedDescription()) if error else "capture stopped")

    # AVCaptureVideoDataOutputSampleBufferDelegate
    def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sb, conn):
        self.n += 1
        f = _frame_from_sample(sb, self.n, self.t0)
        if f is not None:
            self.slot.put(f)


class ScreenSource:
    """Capture a display (``display:<id>``, optionally cropped) or a single window (``window:<id>``)."""

    live = True

    def __init__(self, target: str = "display", fps: float = 15, max_width: int = 1920,
                 rect: list[float] | None = None):
        self.target, self.fps, self.max_width, self.rect = target or "display", fps, max_width, rect
        self.info: dict = {}
        self._slot = LatestFrame()
        self._stream = None

    def open(self) -> None:
        content = shareable_content()
        kind, _, ident = self.target.partition(":")
        if kind == "window":
            win = next((w for w in content.windows() if int(w.windowID()) == int(ident)), None)
            if win is None:
                raise ValueError(f"window {ident} is no longer open")
            flt = SCK.SCContentFilter.alloc().initWithDesktopIndependentWindow_(win)
            w, h, scale = win.frame().size.width, win.frame().size.height, 2.0
        else:
            displays = list(content.displays())
            if not displays:
                raise RuntimeError("no displays available for capture (is the screen locked or asleep?)")
            disp = next((d for d in displays if ident and int(d.displayID()) == int(ident)), displays[0])
            flt = SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(disp, [])
            w, h, scale = disp.width(), disp.height(), 2.0
        cfg = SCK.SCStreamConfiguration.alloc().init()
        if self.rect and kind != "window":
            x, y, rw, rh = self.rect
            cfg.setSourceRect_(Quartz.CGRectMake(x, y, rw, rh))
            w, h = rw, rh
        s = min(scale, self.max_width / w)
        cfg.setWidth_(int(w * s) // 2 * 2)
        cfg.setHeight_(int(h * s) // 2 * 2)
        cfg.setPixelFormat_(_BGRA)
        cfg.setShowsCursor_(False)
        cfg.setQueueDepth_(3)
        cfg.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, int(self.fps)))
        self.info = {"width": int(cfg.width()), "height": int(cfg.height()), "fps": float(self.fps), "duration": 0.0}

        self._delegate = _StreamOutput.alloc().initWithSlot_(self._slot)
        self._queue = dispatch_queue_create(b"npr.screen", None)
        self._stream = SCK.SCStream.alloc().initWithFilter_configuration_delegate_(flt, cfg, self._delegate)
        ok, err = self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
            self._delegate, SCK.SCStreamOutputTypeScreen, self._queue, None)
        if not ok:
            raise RuntimeError(f"SCStream output: {err}")
        (err,) = wait_for(self._stream.startCaptureWithCompletionHandler_, 10)
        if err is not None:
            raise PermissionError(f"screen capture failed to start: {err.localizedDescription()}")

    def read(self) -> Frame | None:
        f = self._slot.get(timeout=30)  # a static screen legitimately produces no frames for a while
        while f is None and self._slot.error and self._slot.error.startswith("no frames"):
            self._slot.error = None
            f = self._slot.get(timeout=30)
        if f is None and self._slot.error:
            raise RuntimeError(self._slot.error)
        return f

    def close(self) -> None:
        if self._stream is not None:
            try:
                wait_for(self._stream.stopCaptureWithCompletionHandler_, 3)
            except TimeoutError:
                pass
            self._stream = None
        self._slot.close()


# ---- cameras -------------------------------------------------------------------------------

def _devices():
    types = [AVF.AVCaptureDeviceTypeBuiltInWideAngleCamera]
    for name in ("AVCaptureDeviceTypeExternal", "AVCaptureDeviceTypeExternalUnknown",
                 "AVCaptureDeviceTypeContinuityCamera"):
        if hasattr(AVF, name):
            types.append(getattr(AVF, name))
    session = AVF.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        types, AVF.AVMediaTypeVideo, AVF.AVCaptureDevicePositionUnspecified)
    return list(session.devices())


def list_cameras() -> list[dict]:
    return [{"id": str(d.uniqueID()), "name": str(d.localizedName()),
             "hfov": float(d.activeFormat().videoFieldOfView()) or None} for d in _devices()]


class CameraSource:
    """A camera attached to this Mac (built-in, USB, Continuity). The lens FOV comes from the device."""

    live = True

    def __init__(self, device_id: str = ""):
        self.device_id = device_id
        self.info: dict = {}
        self._slot = LatestFrame()
        self._session = None

    def open(self) -> None:
        status = AVF.AVCaptureDevice.authorizationStatusForMediaType_(AVF.AVMediaTypeVideo)
        if status == AVF.AVAuthorizationStatusNotDetermined:
            wait_for(lambda h: AVF.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                AVF.AVMediaTypeVideo, h), 60)
            status = AVF.AVCaptureDevice.authorizationStatusForMediaType_(AVF.AVMediaTypeVideo)
        if status != AVF.AVAuthorizationStatusAuthorized:
            raise PermissionError("Camera access denied — enable it in System Settings › Privacy & Security › Camera")
        devs = _devices()
        dev = next((d for d in devs if str(d.uniqueID()) == self.device_id), devs[0] if devs else None)
        if dev is None:
            raise RuntimeError("no camera found")
        inp, err = AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(dev, None)
        if inp is None:
            raise RuntimeError(f"camera input: {err}")
        session = AVF.AVCaptureSession.alloc().init()
        session.addInput_(inp)
        out = AVF.AVCaptureVideoDataOutput.alloc().init()
        out.setVideoSettings_({Quartz.kCVPixelBufferPixelFormatTypeKey: _BGRA})
        out.setAlwaysDiscardsLateVideoFrames_(True)
        self._delegate = _StreamOutput.alloc().initWithSlot_(self._slot)
        self._queue = dispatch_queue_create(b"npr.camera", None)
        out.setSampleBufferDelegate_queue_(self._delegate, self._queue)
        session.addOutput_(out)
        session.startRunning()
        dims = CoreMedia.CMVideoFormatDescriptionGetDimensions(dev.activeFormat().formatDescription())
        self.info = {"width": int(dims.width), "height": int(dims.height), "fps": 30.0, "duration": 0.0,
                     "hfov": float(dev.activeFormat().videoFieldOfView()) or None}
        self._session = session

    def read(self) -> Frame | None:
        f = self._slot.get(timeout=10)
        if f is None and self._slot.error:
            raise RuntimeError(self._slot.error)
        return f

    def close(self) -> None:
        if self._session is not None:
            self._session.stopRunning()
            self._session = None
        self._slot.close()
