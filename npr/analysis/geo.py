"""Small geodesy helpers shared by re-identification and the map."""
from __future__ import annotations

import math

EARTH_R = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def destination(lat: float, lon: float, bearing_deg: float, dist_m: float) -> tuple[float, float]:
    b, d = math.radians(bearing_deg), dist_m / EARTH_R
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180


def fov_cone(lat: float, lon: float, heading: float, hfov: float, range_m: float, steps: int = 16) -> list[tuple[float, float]]:
    """Polygon (lat, lon) for a camera's field-of-view sector."""
    pts = [(lat, lon)]
    for i in range(steps + 1):
        pts.append(destination(lat, lon, heading - hfov / 2 + hfov * i / steps, range_m))
    return pts


def hfov_from_focal_35mm(f35: float) -> float:
    """Horizontal FOV (degrees) from a 35mm-equivalent focal length (36 mm wide frame)."""
    return math.degrees(2 * math.atan(36.0 / (2 * f35)))
