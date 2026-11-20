"""Reviewing one recording: its timeline of objects and events, and what is known about one object.

Built from *observations*: while a recording is analysed, each tracked vehicle is sampled a few times
a second (where it was in the picture, how large) and a handful of those samples keep a preview image.
"""
from __future__ import annotations

import math

from .geo import destination


def describe_motion(obs: list[dict]) -> dict:
    """Plain-language motion of one object from its observations (positions are 0–1 of the picture)."""
    if len(obs) < 2:
        return {"summary": "Seen in a single frame", "moving": False, "angle": None, "across": None, "depth": None,
                "duration_s": 0.0, "travel": 0.0}
    a, b = obs[0], obs[-1]
    dx, dy = b["x"] - a["x"], b["y"] - a["y"]
    growth = (b["w"] * b["h"]) / max(a["w"] * a["h"], 1e-9)
    travel = sum(math.hypot(q["x"] - p["x"], q["y"] - p["y"]) for p, q in zip(obs, obs[1:]))
    across = "left to right" if dx > 0.08 else "right to left" if dx < -0.08 else None
    depth = "towards the camera" if growth > 1.3 else "away from the camera" if growth < 0.77 else None
    moving = bool(across or depth or travel > 0.12)
    parts = [p for p in (across, depth) if p]
    summary = ("Moving " + ", ".join(parts)) if parts else ("Moving within the view" if moving else "Stationary")
    return {"summary": summary, "moving": moving, "across": across, "depth": depth,
            "angle": round(math.degrees(math.atan2(dy, dx)), 1) if moving else None,
            "duration_s": round(b["ts"] - a["ts"], 2), "travel": round(travel, 3), "growth": round(growth, 2)}


def ground_path(obs: list[dict], source: dict) -> list[list[float]] | None:
    """Rough positions on the map for an object seen by a placed camera.

    The bearing comes from where the object sits across the picture (within the lens angle). The
    distance has no real measurement behind it: lower in the picture is taken to be nearer, scaled
    to the camera's range. Good enough to show direction of travel, not for measuring.
    """
    if None in (source.get("lat"), source.get("lon"), source.get("heading")) or not obs:
        return None
    hfov, rng = source.get("hfov") or 70.0, source.get("range_m") or 60.0
    out = []
    for o in obs:
        bearing = source["heading"] + (o["x"] - 0.5) * hfov
        foot = min(max(o["y"] + o["h"] / 2, 0.0), 1.0)          # bottom edge of the box: where it meets the ground
        dist = rng * min(max(1.0 - foot, 0.06), 1.0)
        lat, lon = destination(source["lat"], source["lon"], bearing, dist)
        out.append([round(lat, 7), round(lon, 7)])
    return out
