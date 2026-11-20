"""Models shared by every pipeline, loaded once."""
from __future__ import annotations

import threading

from ..config import settings
from ..native.frames import metal_device_name


class ModelsMissing(RuntimeError):
    pass


class Engine:
    def __init__(self):
        self._lock = threading.Lock()
        self.vehicles = None
        self.plates = None
        self.error: str | None = None

    def vehicle_model_path(self):
        return settings.models_dir / f"{settings.vehicle_model}.mlpackage"

    def load(self) -> "Engine":
        with self._lock:
            if self.vehicles is not None:
                return self
            from .plates import PlateDetector
            from .vehicles import VehicleSegmenter

            path = self.vehicle_model_path()
            if not path.exists():
                self.error = (f"vehicle model {path.name} not found — run "
                              "`uv run python -m npr setup` once to build it")
                raise ModelsMissing(self.error)
            vehicles = VehicleSegmenter(path, settings.vehicle_classes)
            self.plates = PlateDetector(settings.plate_model, settings.plate_conf)
            self._warm_up(vehicles)
            self.vehicles = vehicles
            self.error = None
            return self

    def _warm_up(self, vehicles) -> None:
        """First use of each model pays a one-off compile/load cost; pay it before video arrives."""
        import numpy as np
        import Vision

        from ..native.frames import bgra_to_pixel_buffer
        from .plates import feature_print, read_text

        blank = np.full((384, 640, 4), 127, dtype=np.uint8)
        handler = Vision.VNImageRequestHandler.alloc().initWithCVPixelBuffer_options_(
            bgra_to_pixel_buffer(blank), None)
        vehicles.detect(vehicles.request(), handler, 640, 384)
        read_text(handler, (100, 100, 300, 160), 640, 384)
        feature_print(handler, (0, 0, 640, 384), 640, 384)
        self.plates.detect(np.ascontiguousarray(blank[..., :3]))

    def describe(self) -> dict:
        return {
            "metal_device": metal_device_name(),
            "vehicle_model": settings.vehicle_model,
            "plate_model": settings.plate_model,
            "plate_providers": getattr(self.plates, "providers", None),
            "ocr": "Apple Vision (VNRecognizeTextRequest)",
            "embedding": "Apple Vision (VNGenerateImageFeaturePrintRequest)",
            "loaded": self.vehicles is not None,
            "error": self.error,
        }
