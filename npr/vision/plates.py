"""Number plates: locate the plate on a vehicle crop, read it with Apple Vision's text recogniser."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import Vision

_CLEAN = re.compile(r"[^A-Z0-9]")


@dataclass(slots=True)
class PlateRead:
    text: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in source pixels
    det_conf: float


class PlateDetector:
    """ONNX plate detector (open-image-models) running on Core ML via ONNX Runtime."""

    def __init__(self, model: str, conf: float = 0.45):
        import logging

        import onnxruntime as ort
        from open_image_models import create_detector

        from ..config import settings

        for name in list(logging.root.manager.loggerDict):
            if name.startswith("open_image_models"):
                logging.getLogger(name).setLevel(logging.WARNING)
        ort.set_default_logger_severity(3)
        providers: list = ["CPUExecutionProvider"]
        if "CoreMLExecutionProvider" in ort.get_available_providers():
            # Static shapes only: the model's built-in NMS has data-dependent shapes (zero boxes when
            # there is no plate), which Core ML rejects; those few nodes stay on the CPU.
            cache = settings.models_dir / "ort-coreml-cache"  # skip the slow Neural Engine compile next launch
            cache.mkdir(parents=True, exist_ok=True)
            providers.insert(0, ("CoreMLExecutionProvider",
                                 {"RequireStaticInputShapes": "1", "ModelCacheDirectory": str(cache)}))
        self.providers = [p if isinstance(p, str) else p[0] for p in providers]
        self._det = create_detector(model, conf_thresh=conf, providers=providers)

    def detect(self, bgr: np.ndarray) -> list[tuple[tuple[float, float, float, float], float]]:
        out = []
        for d in self._det.predict(bgr):
            bb = d.bounding_box
            out.append(((float(bb.x1), float(bb.y1), float(bb.x2), float(bb.y2)), float(d.confidence)))
        return out


def roi(box, width: int, height: int, pad: float = 0.0):
    """Source-pixel box → Vision's normalised, bottom-left-origin region of interest."""
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0.0, x1 - px), max(0.0, y1 - py)
    x2, y2 = min(float(width), x2 + px), min(float(height), y2 + py)
    return ((x1 / width, 1.0 - y2 / height), ((x2 - x1) / width, (y2 - y1) / height))


def normalise_plate(text: str) -> str:
    return _CLEAN.sub("", text.upper())


def read_text(handler, box, width: int, height: int, min_chars: int = 4, max_chars: int = 10) -> tuple[str, float] | None:
    """OCR one plate region. Multi-line plates are joined top to bottom; small print is ignored."""
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(False)
    req.setRegionOfInterest_(roi(box, width, height, pad=0.08))
    ok, _ = handler.performRequests_error_([req], None)
    if not ok:
        return None
    lines = []
    for obs in req.results() or []:
        cands = obs.topCandidates_(1)
        if not cands:
            continue
        bb = obs.boundingBox()
        lines.append((bb.size.height, bb.origin.y, normalise_plate(str(cands[0].string())), float(cands[0].confidence())))
    lines = [l for l in lines if l[2]]
    if not lines:
        return None
    tallest = max(l[0] for l in lines)
    main = sorted((l for l in lines if l[0] >= 0.6 * tallest), key=lambda l: -l[1])  # top line first
    text = "".join(l[2] for l in main)
    if not (min_chars <= len(text) <= max_chars):
        return None
    return text, min(l[3] for l in main)


def feature_print(handler, box, width: int, height: int) -> np.ndarray | None:
    """Apple Vision image feature-print of a region: an L2-normalised appearance embedding."""
    req = Vision.VNGenerateImageFeaturePrintRequest.alloc().init()
    req.setRegionOfInterest_(roi(box, width, height))
    ok, _ = handler.performRequests_error_([req], None)
    res = req.results() if ok else None
    if not res:
        return None
    fp = res[0]
    dtype = np.float64 if fp.elementType() == Vision.VNElementTypeDouble else np.float32
    v = np.frombuffer(bytes(fp.data()), dtype=dtype, count=int(fp.elementCount())).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else None


def colour_histogram(bgra: np.ndarray, box, mask: np.ndarray | None) -> np.ndarray | None:
    """40-bin colour signature of a vehicle, counting only pixels under its segmentation mask."""
    import cv2

    x1, y1, x2, y2 = (int(v) for v in box)
    crop = bgra[max(y1, 0):y2, max(x1, 0):x2, :3]
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    small = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    if mask is not None and mask.size:
        m = cv2.resize(mask.astype(np.uint8), (48, 48), interpolation=cv2.INTER_NEAREST).reshape(-1) > 0
        if m.sum() >= 64:
            hsv = hsv[m]
    h, s, v = hsv[:, 0].astype(int), hsv[:, 1].astype(int), hsv[:, 2].astype(int)
    chroma = (s > 60) & (v > 50)
    hist = np.zeros(40, dtype=np.float32)
    np.add.at(hist, (h[chroma] * 12 // 180) * 3 + np.minimum((s[chroma] - 60) // 65, 2), 1)  # 12 hues × 3 sat
    np.add.at(hist, 36 + np.minimum(v[~chroma] // 64, 3), 1)  # 4 grey levels: black … white
    total = hist.sum()
    return hist / total if total else None
