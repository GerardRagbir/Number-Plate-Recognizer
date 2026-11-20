"""Movement patterns derived from sightings: camera-to-camera flows, routes, repeat visitors."""
from __future__ import annotations

import statistics
import time
from collections import Counter, defaultdict

from ..store import Store
from .geo import haversine_m

VISIT_GAP_S = 600  # sightings further apart than this on one camera count as separate visits


def identity_route(store: Store, iid: int) -> list[dict]:
    """Chronological camera visits for one identity (consecutive sightings on a camera merged)."""
    route: list[dict] = []
    for s in store.sightings_of(iid):
        if route and route[-1]["source_id"] == s["source_id"] and s["t_first"] - route[-1]["t_last"] < VISIT_GAP_S:
            route[-1]["t_last"] = max(route[-1]["t_last"], s["t_last"])
            route[-1]["sightings"].append(s["id"])
            continue
        route.append({"source_id": s["source_id"], "t_first": s["t_first"], "t_last": s["t_last"],
                      "sightings": [s["id"]], "match_kind": s["match_kind"], "match_score": s["match_score"]})
    return route


def patterns(store: Store, since: float | None = None) -> dict:
    since = since if since is not None else 0.0
    cams = {c["id"]: c for c in store.sources()}
    by_identity: dict[int, list[dict]] = defaultdict(list)
    per_cam_hours: dict[int, Counter] = defaultdict(Counter)
    dwell: dict[int, list[float]] = defaultdict(list)
    for s in store.all_sightings(since):
        by_identity[s["identity_id"]].append(s)
        per_cam_hours[s["source_id"]][time.localtime(s["t_first"]).tm_hour] += 1
        dwell[s["source_id"]].append(s["t_last"] - s["t_first"])

    edges: dict[tuple[int, int], list[float]] = defaultdict(list)
    routes: Counter = Counter()
    repeat: list[dict] = []
    for iid, rows in by_identity.items():
        visits: list[dict] = []
        for s in rows:  # already time-ordered
            if visits and visits[-1]["source_id"] == s["source_id"] and s["t_first"] - visits[-1]["t_last"] < VISIT_GAP_S:
                visits[-1]["t_last"] = max(visits[-1]["t_last"], s["t_last"])
            else:
                visits.append({"source_id": s["source_id"], "t_first": s["t_first"], "t_last": s["t_last"]})
        for a, b in zip(visits, visits[1:]):
            if a["source_id"] != b["source_id"]:
                edges[(a["source_id"], b["source_id"])].append(max(0.0, b["t_first"] - a["t_last"]))
        seq = tuple(v["source_id"] for v in visits)
        if len(set(seq)) > 1:
            routes[seq] += 1
        if len(visits) > 1:
            repeat.append({"identity_id": iid, "visits": len(visits), "cameras": len(set(seq)),
                           "first": visits[0]["t_first"], "last": visits[-1]["t_last"]})

    flows = []
    for (a, b), gaps in edges.items():
        med = statistics.median(gaps)
        f = {"from": a, "to": b, "count": len(gaps), "median_s": med, "min_s": min(gaps), "max_s": max(gaps)}
        ca, cb = cams.get(a), cams.get(b)
        if ca and cb and None not in (ca["lat"], ca["lon"], cb["lat"], cb["lon"]):
            f["distance_m"] = haversine_m(ca["lat"], ca["lon"], cb["lat"], cb["lon"])
            if med > 0:
                f["median_kmh"] = f["distance_m"] / med * 3.6
        flows.append(f)
    flows.sort(key=lambda f: -f["count"])
    repeat.sort(key=lambda r: (-r["visits"], -r["last"]))

    return {
        "flows": flows,
        "routes": [{"cameras": list(seq), "count": n} for seq, n in routes.most_common(20)],
        "repeat": repeat[:50],
        "cameras": [
            {"source_id": sid, "sightings": sum(h.values()), "hours": [h.get(i, 0) for i in range(24)],
             "median_dwell_s": statistics.median(dwell[sid]) if dwell[sid] else 0}
            for sid, h in per_cam_hours.items()
        ],
        "identities": len(by_identity),
    }
