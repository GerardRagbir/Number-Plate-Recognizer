"""Cross-camera re-identification.

Every confirmed track becomes a *sighting*. Sightings are grouped into global *identities* using,
in order of trust:

1. the number plate (fuzzy, so one mis-read character does not split an identity),
2. appearance — an Apple Vision feature-print of the vehicle plus the colour histogram taken
   under its segmentation mask,

and both are gated by physics: the same vehicle cannot be on two non-adjacent cameras at once,
nor travel between two geo-located cameras faster than ``max_speed_kmh``.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from ..config import settings
from ..store import Store
from .geo import haversine_m


def plate_similarity(a: str, b: str) -> float:
    """1 - normalised Levenshtein distance."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or a.shape != b.shape:
        return None
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / n if n else None


def hist_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(np.minimum(a, b).sum())  # histograms are L1-normalised


@dataclass(slots=True)
class Match:
    identity_id: int
    kind: str  # "plate" | "appearance" | "new"
    score: float


class ReID:
    def __init__(self, store: Store):
        self.store = store
        self._lock = threading.Lock()

    # -- gating -----------------------------------------------------------------
    def _feasible(self, src: dict, t_first: float, t_last: float, other: dict, cams: dict[int, dict]) -> bool:
        """Could a vehicle seen as ``other`` (a sighting row) also be this one?"""
        o_src = cams.get(other["source_id"])
        same_cam = other["source_id"] == src["id"]
        overlap = min(t_last, other["t_last"]) - max(t_first, other["t_first"])
        if same_cam:
            # two simultaneous tracks on one camera are two vehicles; sequential ones may be the same
            return overlap < 0
        if o_src is None or None in (src.get("lat"), src.get("lon"), o_src.get("lat"), o_src.get("lon")):
            return True  # no geometry to argue with
        dist = haversine_m(src["lat"], src["lon"], o_src["lat"], o_src["lon"])
        reach = (src.get("range_m") or 0) + (o_src.get("range_m") or 0)
        gap = max(0.0, -overlap)
        travel = max(0.0, dist - reach)  # views that overlap can see one vehicle simultaneously
        return travel <= (settings.max_speed_kmh / 3.6) * gap + 1e-6

    # -- scoring ------------------------------------------------------------------
    def _score(self, s: dict, ident: dict) -> tuple[str, float] | None:
        sp, ip = s.get("plate"), ident.get("plate")
        if sp and ip:
            sim = plate_similarity(sp, ip)
            if sim >= settings.plate_match:
                return "plate", sim
            if min(s.get("plate_conf") or 0, ident.get("plate_conf") or 0) >= 0.5:
                return None  # two confident, different plates: different vehicles
        if s.get("cls") and ident.get("cls") and s["cls"] != ident["cls"]:
            if {s["cls"], ident["cls"]} != {"car", "truck"}:  # pickups/SUVs flip between these
                return None
        emb = cosine(s.get("embedding"), ident.get("embedding"))
        if emb is None:
            return None
        col = hist_similarity(s.get("color"), ident.get("color"))
        score = emb if col is None else 0.75 * emb + 0.25 * col
        return ("appearance", score) if score >= settings.appearance_match else None

    def assign(self, sighting_id: int, s: dict, src: dict) -> Match:
        """(Re)assign sighting ``s`` to the best identity, creating one if nothing fits.

        ``s`` holds: source_id, t_first, t_last, cls, plate, plate_conf, embedding, color and the
        current identity_id (or None).
        """
        with self._lock:
            cams = {c["id"]: c for c in self.store.sources()}
            best: tuple[float, str, dict] | None = None
            for ident in self.store.identities_since(s["t_first"] - settings.reid_window_s):
                scored = self._score(s, ident)
                if scored is None:
                    continue
                kind, score = scored
                others = [o for o in self.store.sightings_of(ident["id"]) if o["id"] != sighting_id]
                if not others:
                    continue  # empty, or only this sighting: nothing to match against
                if not all(self._feasible(src, s["t_first"], s["t_last"], o, cams) for o in others):
                    continue
                rank = score + (1.0 if kind == "plate" else 0.0)
                if best is None or rank > best[0]:
                    best = (rank, kind, ident)

            prev = s.get("identity_id")
            if best is None:
                if prev is not None and len(self.store.sightings_of(prev)) == 1:
                    iid, kind, score = prev, "new", 1.0  # keep our own singleton identity
                else:
                    iid = self.store.add_identity(first_seen=s["t_first"], last_seen=s["t_last"], n=0)
                    kind, score = "new", 1.0
            else:
                rank, kind, ident = best
                iid, score = ident["id"], rank - (1.0 if kind == "plate" else 0.0)

            self.store.update_sighting(sighting_id, identity_id=iid, match_kind=kind, match_score=score)
            if prev is not None and prev != iid:
                self.store.delete_identity_if_empty(prev)
            self._refresh(iid)
            return Match(iid, kind, score)

    def _refresh(self, iid: int) -> None:
        """Recompute an identity's summary (best plate, mean appearance) from its sightings."""
        rows = self.store.sightings_of(iid, arrays=True)
        if not rows:
            return
        plates: dict[str, float] = {}
        for r in rows:
            if r["plate"]:
                plates[r["plate"]] = plates.get(r["plate"], 0.0) + (r["plate_conf"] or 0.0)
        plate = max(plates, key=plates.get) if plates else None
        embs = [r["embedding"] for r in rows if r["embedding"] is not None]
        cols = [r["color"] for r in rows if r["color"] is not None]
        emb = None
        if embs and all(e.shape == embs[0].shape for e in embs):
            emb = np.mean(embs, axis=0)
            emb = emb / (np.linalg.norm(emb) or 1.0)
        classes: dict[str, int] = {}
        for r in rows:
            if r["cls"]:
                classes[r["cls"]] = classes.get(r["cls"], 0) + 1
        self.store.update_identity(
            iid, plate=plate,
            plate_conf=max((r["plate_conf"] or 0) for r in rows if r["plate"] == plate) if plate else 0,
            cls=max(classes, key=classes.get) if classes else None,
            embedding=emb, color=np.mean(cols, axis=0) if cols else None, n=len(rows),
            first_seen=min(r["t_first"] for r in rows), last_seen=max(r["t_last"] for r in rows),
        )
