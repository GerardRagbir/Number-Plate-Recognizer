"""The local camera profile database (``npr/data/cameras.json``).

A profile turns "I have a <brand> <model>" into what the app needs: how to build the stream
address, which paths to try, the lens angle and reach for the map cone, and the camera type.
The JSON file is the whole database; add brands and models there.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "cameras.json"


def horizontal_fov(fov: float, basis: str, aspect: float = 16 / 9) -> float:
    """Horizontal angle for the map cone.

    Makers quote one headline angle, normally the diagonal. For a 16:9 picture the horizontal
    angle is narrower; this converts with the rectilinear relation, which slightly understates
    very wide (fisheye-like) lenses. A figure already given as horizontal is used as it is.
    """
    if basis == "horizontal":
        return float(fov)
    half = math.radians(min(fov, 175.0) / 2)
    return round(2 * math.degrees(math.atan(math.tan(half) * aspect / math.hypot(aspect, 1.0))), 1)


@lru_cache(maxsize=1)
def _load() -> dict:
    return json.loads(DB_PATH.read_text())


def catalogue() -> dict:
    """The database as the dashboard wants it: each model with its cone angle worked out."""
    db = _load()
    brands = []
    for b in db["brands"]:
        models = []
        for m in b["models"]:
            m = dict(m)
            lens = m.get("lens")
            if lens and lens.get("fov"):
                m["hfov"] = horizontal_fov(lens["fov"], lens.get("basis", "unspecified"))
            models.append(m)
        brands.append({**b, "models": models})
    return {"version": db["version"], "updated": db.get("updated"), "brands": brands}


def find(model_id: str | None) -> tuple[dict, dict] | None:
    if not model_id:
        return None
    for b in catalogue()["brands"]:
        for m in b["models"]:
            if m["id"] == model_id:
                return b, m
    return None


def candidate_paths(model_id: str | None) -> list[str]:
    """Stream paths worth trying for a model, likeliest first; a generic set when it is unknown."""
    hit = find(model_id)
    if hit is None:
        return [f"/live{i}" for i in range(6)]
    brand, model = hit
    c = brand["connect"]
    first = (model.get("rtsp") or {}).get("channels") or []
    order = list(dict.fromkeys([*first, *c.get("channels", [])]))
    return [c["path"].replace("{channel}", str(ch)) for ch in order]
