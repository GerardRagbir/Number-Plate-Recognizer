"""Video files through AVFoundation (VideoToolbox hardware decode) and stills through ImageIO."""
from __future__ import annotations

import os
import re
import time

import AVFoundation as AVF
import CoreMedia
import numpy as np
import Quartz
from Foundation import NSURL

from ..analysis.geo import hfov_from_focal_35mm
from ..native.frames import Frame

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
_ISO6709 = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXT


class VideoFileSource:
    live = False

    def __init__(self, path: str, start_time: float | None = None, loop: bool = False):
        self.path, self.loop = path, loop
        self.start_time = start_time
        self.info: dict = {}
        self._reader = self._out = None
        self._index = 0

    def open(self) -> None:
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        self._asset = AVF.AVURLAsset.URLAssetWithURL_options_(NSURL.fileURLWithPath_(self.path), None)
        tracks = self._asset.tracksWithMediaType_(AVF.AVMediaTypeVideo)
        if not tracks:
            raise ValueError(f"no video track in {self.path}")
        track = tracks[0]
        size, tf = track.naturalSize(), track.preferredTransform()
        rotated = abs(tf.b) > 0.5 or abs(tf.c) > 0.5 or tf.a < 0
        self.info = {
            "width": int(size.height if abs(tf.b) > 0.5 else size.width),
            "height": int(size.width if abs(tf.b) > 0.5 else size.height),
            "fps": float(track.nominalFrameRate()) or 30.0,
            "duration": float(CoreMedia.CMTimeGetSeconds(self._asset.duration())),
        }
        if self.start_time is None:
            self.start_time = time.time()
        self._track, self._rotated = track, rotated
        self._start_reader()

    def _start_reader(self) -> None:
        reader, err = AVF.AVAssetReader.assetReaderWithAsset_error_(self._asset, None)
        if reader is None:
            raise RuntimeError(f"AVAssetReader: {err}")
        px = {Quartz.kCVPixelBufferPixelFormatTypeKey: Quartz.kCVPixelFormatType_32BGRA,
              Quartz.kCVPixelBufferIOSurfacePropertiesKey: {}}
        if self._rotated:  # let AVFoundation apply the track's rotation for us
            out = AVF.AVAssetReaderVideoCompositionOutput.assetReaderVideoCompositionOutputWithVideoTracks_videoSettings_(
                [self._track], px)
            out.setVideoComposition_(AVF.AVVideoComposition.videoCompositionWithPropertiesOfAsset_(self._asset))
        else:
            out = AVF.AVAssetReaderTrackOutput.assetReaderTrackOutputWithTrack_outputSettings_(self._track, px)
        out.setAlwaysCopiesSampleData_(False)
        reader.addOutput_(out)
        if not reader.startReading():
            raise RuntimeError(f"AVAssetReader failed to start: {reader.error()}")
        self._reader, self._out = reader, out

    def read(self) -> Frame | None:
        while True:
            sb = self._out.copyNextSampleBuffer()
            if sb is None:
                if self._reader.status() == AVF.AVAssetReaderStatusFailed:
                    raise RuntimeError(f"decode failed: {self._reader.error()}")
                if self.loop:
                    self.start_time += self.info["duration"]
                    self._start_reader()
                    continue
                return None
            pb = CoreMedia.CMSampleBufferGetImageBuffer(sb)
            if pb is None:
                continue
            ts = float(CoreMedia.CMTimeGetSeconds(CoreMedia.CMSampleBufferGetPresentationTimeStamp(sb)))
            self._index += 1
            return Frame.from_pixel_buffer(pb, self._index, ts, self.start_time + ts)

    def close(self) -> None:
        if self._reader is not None:
            self._reader.cancelReading()
        self._reader = self._out = None


class ImageSource:
    """A single still: yields exactly one frame."""

    live = False

    def __init__(self, path: str, start_time: float | None = None):
        self.path, self.start_time = path, start_time
        self.info: dict = {}
        self._done = False

    def open(self) -> None:
        src = Quartz.CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(self.path), None)
        if src is None:
            raise ValueError(f"cannot read image {self.path}")
        # a "thumbnail" at full size is ImageIO's way of decoding with EXIF orientation applied
        img = Quartz.CGImageSourceCreateThumbnailAtIndex(src, 0, {
            Quartz.kCGImageSourceCreateThumbnailFromImageAlways: True,
            Quartz.kCGImageSourceCreateThumbnailWithTransform: True,
            Quartz.kCGImageSourceThumbnailMaxPixelSize: 4096,
        })
        if img is None:
            raise ValueError(f"cannot decode image {self.path}")
        w, h = Quartz.CGImageGetWidth(img), Quartz.CGImageGetHeight(img)
        buf = bytearray(w * h * 4)
        ctx = Quartz.CGBitmapContextCreate(
            buf, w, h, 8, w * 4, Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceSRGB),
            Quartz.kCGImageAlphaPremultipliedFirst | Quartz.kCGBitmapByteOrder32Little)  # = BGRA
        Quartz.CGContextDrawImage(ctx, Quartz.CGRectMake(0, 0, w, h), img)
        del ctx
        self._bgra = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        self.info = {"width": w, "height": h, "fps": 0.0, "duration": 0.0}
        if self.start_time is None:
            self.start_time = time.time()

    def read(self) -> Frame | None:
        if self._done:
            return None
        self._done = True
        return Frame.from_bgra(self._bgra, 1, 0.0, self.start_time)

    def close(self) -> None:
        pass


# ---- metadata: where was this shot, which way was it facing, how wide is the lens? -------------

def _parse_exif_date(s: str) -> float | None:
    try:
        return time.mktime(time.strptime(str(s)[:19], "%Y:%m:%d %H:%M:%S"))
    except ValueError:
        return None


def image_metadata(path: str) -> dict:
    out: dict = {}
    src = Quartz.CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(path), None)
    props = Quartz.CGImageSourceCopyPropertiesAtIndex(src, 0, None) if src else None
    if not props:
        return out
    gps = props.get(Quartz.kCGImagePropertyGPSDictionary) or {}
    exif = props.get(Quartz.kCGImagePropertyExifDictionary) or {}
    if "Latitude" in gps and "Longitude" in gps:
        out["lat"] = float(gps["Latitude"]) * (-1 if gps.get("LatitudeRef") == "S" else 1)
        out["lon"] = float(gps["Longitude"]) * (-1 if gps.get("LongitudeRef") == "W" else 1)
    if "ImgDirection" in gps:
        out["heading"] = float(gps["ImgDirection"])
    f35 = exif.get("FocalLenIn35mmFilm")
    if f35:
        hfov = hfov_from_focal_35mm(float(f35))
        w, h = props.get(Quartz.kCGImagePropertyPixelWidth, 3), props.get(Quartz.kCGImagePropertyPixelHeight, 2)
        if int(props.get(Quartz.kCGImagePropertyOrientation, 1)) >= 5 or h > w:
            hfov = hfov * 2 / 3  # portrait: the long sensor edge is vertical
        out["hfov"], out["fov_source"] = hfov, "exif"
    t = _parse_exif_date(exif.get("DateTimeOriginal", ""))
    if t:
        out["start_time"] = t
    return out


def video_metadata(path: str) -> dict:
    out: dict = {}
    asset = AVF.AVURLAsset.URLAssetWithURL_options_(NSURL.fileURLWithPath_(path), None)
    for item in list(asset.metadata() or []):
        ident, val = str(item.identifier() or ""), item.value()
        if "location.ISO6709" in ident or ident.endswith("/\xa9xyz"):
            m = _ISO6709.match(str(val))
            if m:
                out["lat"], out["lon"] = float(m.group(1)), float(m.group(2))
        elif "focal_length.35mm_equivalent" in ident:
            try:
                out["hfov"], out["fov_source"] = hfov_from_focal_35mm(float(val)), "metadata"
            except (TypeError, ValueError, ZeroDivisionError):
                pass
    created = asset.creationDate()
    if created is not None and created.dateValue() is not None:
        out["start_time"] = float(created.dateValue().timeIntervalSince1970())
    return out


def file_metadata(path: str) -> dict:
    try:
        return image_metadata(path) if is_image(path) else video_metadata(path)
    except Exception:  # metadata is a nicety; never block ingest on it
        return {}
