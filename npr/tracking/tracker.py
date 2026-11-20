"""ByteTrack-style multi-object tracker (constant-velocity Kalman + two-stage IoU association).

Kept dependency-light on purpose: numpy for the filter, scipy for the assignment problem.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(slots=True)
class Detection:
    box: np.ndarray  # x1, y1, x2, y2 in source pixels
    conf: float
    cls: int
    polygon: np.ndarray | None = None  # (n, 2) mask outline in source pixels
    mask: np.ndarray | None = None  # bool mask cropped to the box (model resolution)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


class Kalman:
    """State [cx, cy, w, h, vx, vy, vw, vh]; noise scales with box size like SORT/ByteTrack."""

    F = np.eye(8)
    F[:4, 4:] = np.eye(4)
    H = np.eye(4, 8)

    def __init__(self, z: np.ndarray):
        self.x = np.concatenate([z, np.zeros(4)])
        s = max(z[2], z[3])
        self.P = np.diag([(s * 0.1) ** 2] * 4 + [(s * 0.2) ** 2] * 4)

    def predict(self) -> None:
        s = max(self.x[2], self.x[3], 1.0)
        q = np.diag([(s * 0.05) ** 2] * 4 + [(s * 0.02) ** 2] * 4)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + q

    def update(self, z: np.ndarray) -> None:
        s = max(z[2], z[3], 1.0)
        r = np.diag([(s * 0.05) ** 2] * 4)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + r
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        w, h = max(w, 1.0), max(h, 1.0)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def _xyxy_to_z(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]], dtype=float)


@dataclass(slots=True)
class Track:
    id: int
    kf: Kalman
    cls: int
    conf: float
    first_ts: float
    last_ts: float
    first_wall: float
    last_wall: float
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    polygon: np.ndarray | None = None
    mask: np.ndarray | None = None
    cls_votes: dict[int, float] = field(default_factory=dict)
    trail: deque = field(default_factory=lambda: deque(maxlen=90))  # (ts, x, y) ground-contact points
    data: dict = field(default_factory=dict)  # owned by the pipeline (plates, embeddings, identity…)

    @property
    def box(self) -> np.ndarray:
        return self.kf.box

    @property
    def visible(self) -> bool:
        return self.misses == 0


class Tracker:
    def __init__(self, high: float = 0.5, low: float = 0.15, match_iou: float = 0.2,
                 max_misses: int = 30, min_hits: int = 3):
        self.high, self.low, self.match_iou = high, low, match_iou
        self.max_misses, self.min_hits = max_misses, min_hits
        self.tracks: list[Track] = []
        self._next = 1

    def _associate(self, tracks: list[Track], dets: list[Detection], thresh: float):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        iou = iou_matrix(np.array([t.box for t in tracks]), np.array([d.box for d in dets]))
        rows, cols = linear_sum_assignment(-iou)
        pairs = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= thresh]
        mr, mc = {r for r, _ in pairs}, {c for _, c in pairs}
        return pairs, [i for i in range(len(tracks)) if i not in mr], [j for j in range(len(dets)) if j not in mc]

    def _hit(self, t: Track, d: Detection, ts: float, wall: float) -> None:
        t.kf.update(_xyxy_to_z(d.box))
        t.hits += 1
        t.misses = 0
        t.conf = d.conf
        t.polygon = d.polygon
        t.last_ts, t.last_wall = ts, wall
        t.cls_votes[d.cls] = t.cls_votes.get(d.cls, 0.0) + d.conf
        t.cls = max(t.cls_votes, key=t.cls_votes.get)
        t.mask = d.mask
        b = t.box
        t.trail.append((ts, float((b[0] + b[2]) / 2), float(b[3])))
        if t.hits >= self.min_hits:
            t.confirmed = True

    def update(self, dets: list[Detection], ts: float, wall: float) -> tuple[list[Track], list[Track]]:
        """Advance one frame. Returns (live tracks, tracks that just ended)."""
        for t in self.tracks:
            t.kf.predict()
        high = [d for d in dets if d.conf >= self.high]
        low = [d for d in dets if self.low <= d.conf < self.high]

        pairs, un_t, un_d = self._associate(self.tracks, high, self.match_iou)
        for r, c in pairs:
            self._hit(self.tracks[r], high[c], ts, wall)
        # second pass: weak detections rescue tracks that were visible last frame
        rest = [self.tracks[i] for i in un_t]
        pairs2, un_t2, _ = self._associate(rest, low, 0.4)
        for r, c in pairs2:
            self._hit(rest[r], low[c], ts, wall)
        for i in un_t2:
            rest[i].misses += 1
            rest[i].polygon = rest[i].mask = None

        for j in un_d:
            d = high[j]
            t = Track(self._next, Kalman(_xyxy_to_z(d.box)), d.cls, d.conf, ts, ts, wall, wall,
                      polygon=d.polygon, cls_votes={d.cls: d.conf})
            t.mask = d.mask
            b = d.box
            t.trail.append((ts, float((b[0] + b[2]) / 2), float(b[3])))
            t.confirmed = self.min_hits <= 1
            self._next += 1
            self.tracks.append(t)

        ended, alive = [], []
        for t in self.tracks:
            limit = self.max_misses if t.confirmed else 2
            (ended if t.misses > limit else alive).append(t)
        self.tracks = alive
        return alive, [t for t in ended if t.confirmed]

    def flush(self) -> list[Track]:
        out = [t for t in self.tracks if t.confirmed]
        self.tracks = []
        return out
