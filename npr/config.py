"""Runtime configuration. Everything is overridable with NPR_* environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("NPR_HOME", Path(__file__).resolve().parent.parent))


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(slots=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    models_dir: Path = field(default_factory=lambda: ROOT / "models")

    # --- models -----------------------------------------------------------
    vehicle_model: str = os.environ.get("NPR_VEHICLE_MODEL", "yolo26s-seg")
    plate_model: str = os.environ.get("NPR_PLATE_MODEL", "yolo-v9-t-384-license-plate-end2end")
    # COCO class ids kept by the vehicle model.
    vehicle_classes: dict[int, str] = field(
        default_factory=lambda: {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    )
    vehicle_conf: float = _f("NPR_VEHICLE_CONF", 0.35)
    plate_conf: float = _f("NPR_PLATE_CONF", 0.45)
    ocr_conf: float = _f("NPR_OCR_CONF", 0.30)
    plate_min_chars: int = 4
    plate_max_chars: int = 10

    # --- per-frame work budget ---------------------------------------------
    plate_attempts_per_frame: int = 4
    embeds_per_frame: int = 3
    min_vehicle_px: int = 48  # ignore vehicles narrower than this for plates/embeddings

    # --- output -------------------------------------------------------------
    tile_width: int = int(_f("NPR_TILE_WIDTH", 960))
    jpeg_quality: float = _f("NPR_JPEG_QUALITY", 0.6)
    max_fps: float = _f("NPR_MAX_FPS", 30)

    # --- cross-camera re-identification ---------------------------------------
    plate_match: float = _f("NPR_PLATE_MATCH", 0.80)  # normalised edit similarity
    appearance_match: float = _f("NPR_APPEARANCE_MATCH", 0.80)
    reid_window_s: float = _f("NPR_REID_WINDOW", 6 * 3600)
    max_speed_kmh: float = _f("NPR_MAX_SPEED", 180)

    def ensure(self) -> None:
        for d in (self.data_dir, self.models_dir, self.data_dir / "uploads", self.data_dir / "thumbs",
                  self.data_dir / "snaps", self.data_dir / "obs"):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
