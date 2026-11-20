"""Logic that needs no models or Apple frameworks: tracking, re-identification, patterns, geometry."""
import numpy as np
import pytest

from npr.analysis import patterns as pat
from npr.analysis.geo import fov_cone, haversine_m, hfov_from_focal_35mm
from npr.analysis.reid import ReID, plate_similarity
from npr.store import Store
from npr.tracking.tracker import Detection, Tracker


def det(x, y, w=100, h=60, conf=0.9, cls=2):
    return Detection(np.array([x, y, x + w, y + h], dtype=float), conf, cls)


def test_tracker_keeps_identity_through_motion_and_occlusion():
    tr = Tracker(min_hits=3, max_misses=10)
    ids = []
    for i in range(30):
        dets = [det(10 + i * 8, 100), det(600 - i * 8, 300)]
        if 12 <= i < 17:
            dets = dets[1:]  # first vehicle hidden for five frames
        live, _ = tr.update(dets, i / 30, i / 30)
        ids.append(sorted(t.id for t in live if t.confirmed and t.visible))
    assert ids[-1] == [1, 2]
    assert ids[14] == [2]
    assert {i for frame in ids for i in frame} == {1, 2}  # no identity switches, no extra tracks


def test_tracker_reports_ended_tracks_once():
    tr = Tracker(min_hits=2, max_misses=3)
    for i in range(5):
        tr.update([det(10 + i * 5, 10)], i, i)
    ended = []
    for i in range(5, 12):
        ended += tr.update([], i, i)[1]
    assert [t.id for t in ended] == [1]


def test_plate_similarity():
    assert plate_similarity("PDX1234", "PDX1234") == 1
    assert plate_similarity("PDX1234", "PDX1Z34") == pytest.approx(6 / 7)
    assert plate_similarity("PDX1234", "TAB9087") < 0.3
    assert plate_similarity("", "ABC") == 0


def test_geometry():
    assert haversine_m(10.65, -61.5, 10.65, -61.5) == 0
    assert haversine_m(0, 0, 0, 1) == pytest.approx(111_195, rel=1e-3)
    assert hfov_from_focal_35mm(26) == pytest.approx(69.4, abs=0.1)  # a typical phone main camera
    cone = fov_cone(10.0, -61.0, 90, 60, 100)
    assert cone[0] == (10.0, -61.0)
    assert all(lon > -61.0 for _, lon in cone[1:])  # a camera facing east sees east
    assert all(haversine_m(10.0, -61.0, la, lo) == pytest.approx(100, abs=0.5) for la, lo in cone[1:])


@pytest.fixture
def world(tmp_path):
    store = Store(tmp_path / "t.sqlite3")
    a = store.add_source(name="Gate", kind="file", uri="a", lat=10.0, lon=-61.0, range_m=30)
    b = store.add_source(name="Junction", kind="file", uri="b", lat=10.0, lon=-60.99, range_m=30)  # ~1.1 km east
    return store, ReID(store), a, b


def emb(seed):
    v = np.random.default_rng(seed).normal(size=64).astype(np.float32)
    return v / np.linalg.norm(v)


def sight(store, reid, src, t0, t1, plate=None, e=None, conf=0.9, cls="car"):
    fields = dict(t_first=t0, t_last=t1, cls=cls, plate=plate, plate_conf=conf if plate else 0, embedding=e, color=None)
    sid = store.add_sighting(source_id=src["id"], track_id=sid_counter(), **fields)
    m = reid.assign(sid, {**fields, "source_id": src["id"], "identity_id": None}, src)
    return sid, m


_n = [0]


def sid_counter():
    _n[0] += 1
    return _n[0]


def test_reid_links_by_plate_across_cameras_and_tolerates_one_misread(world):
    store, reid, a, b = world
    _, m1 = sight(store, reid, a, 1000, 1010, "PDX1234", emb(1))
    _, m2 = sight(store, reid, b, 1100, 1110, "PDX1Z34", emb(2))  # different look, one char off
    assert m1.kind == "new"
    assert (m2.kind, m2.identity_id) == ("plate", m1.identity_id)
    assert store.identity(m1.identity_id)["n"] == 2


def test_reid_links_by_appearance_when_no_plate(world):
    store, reid, a, b = world
    e = emb(3)
    _, m1 = sight(store, reid, a, 1000, 1010, None, e)
    _, m2 = sight(store, reid, b, 1100, 1110, None, e + 0.02 * emb(4))
    _, m3 = sight(store, reid, b, 1200, 1210, None, emb(5))
    assert (m2.kind, m2.identity_id) == ("appearance", m1.identity_id)
    assert m3.kind == "new" and m3.identity_id != m1.identity_id


def test_reid_refuses_impossible_travel_and_conflicting_plates(world):
    store, reid, a, b = world
    e = emb(6)
    _, m1 = sight(store, reid, a, 1000, 1010, None, e)
    _, m2 = sight(store, reid, b, 1012, 1020, None, e)  # 1.1 km in 2 s
    assert m2.identity_id != m1.identity_id
    _, m3 = sight(store, reid, a, 5000, 5010, "AAA111", e)
    _, m4 = sight(store, reid, b, 5200, 5210, "ZZZ999", e)  # looks the same, confidently different plate
    assert m4.identity_id != m3.identity_id


def test_reid_same_camera_simultaneous_tracks_are_distinct(world):
    store, reid, a, _ = world
    e = emb(7)
    _, m1 = sight(store, reid, a, 1000, 1010, None, e)
    _, m2 = sight(store, reid, a, 1005, 1015, None, e)
    assert m1.identity_id != m2.identity_id


def test_patterns_flows_and_routes(world):
    store, reid, a, b = world
    for i, plate in enumerate(["AAA111", "BBB222", "CCC333"]):
        t = 1000 + i * 500
        sight(store, reid, a, t, t + 10, plate, emb(10 + i))
        sight(store, reid, b, t + 90, t + 100, plate, emb(20 + i))
    p = pat.patterns(store)
    assert p["identities"] == 3
    (flow,) = p["flows"]
    assert (flow["from"], flow["to"], flow["count"]) == (a["id"], b["id"], 3)
    assert flow["median_s"] == 80
    assert flow["median_kmh"] == pytest.approx(flow["distance_m"] / 80 * 3.6)
    assert p["routes"][0] == {"cameras": [a["id"], b["id"]], "count": 3}
    route = pat.identity_route(store, 1)
    assert [r["source_id"] for r in route] == [a["id"], b["id"]]


def test_motion_gate_skips_static_frames_and_wakes_on_change():
    import numpy as np

    from npr.native.frames import Frame
    from npr.pipeline import Pipeline

    p = Pipeline({"id": 1, "kind": "file", "uri": "", "opts": {}}, None, None, None, None)
    p.conf.update(motion=True, motion_fraction=0.002, hold_s=1.0, keepalive_s=10.0)
    p._still, p._ref, p._hold_until, p._last_proc = False, None, None, -1e9
    base = np.full((360, 640, 4), 90, dtype=np.uint8)
    moved = base.copy()
    moved[100:160, 200:300] = 200  # a car-sized patch changes

    def static(img, ts):
        idle = p._static(Frame.from_bgra(img, 0, ts, ts))
        if not idle:
            p._last_proc = ts
        return idle

    assert static(base, 0.0) is False            # first frame is always analysed
    assert static(base, 2.0) is False            # start-up hold (3 s)
    assert static(base, 4.0) is True             # nothing changed: skip
    assert static(base, 9.0) is True
    assert static(moved, 9.5) is False           # change wakes it
    assert static(moved, 10.0) is False          # hold after motion
    assert static(moved, 11.0) is True           # settled on the new picture
    assert static(moved, 21.0) is False          # keep-alive re-check
    noisy = moved.copy()
    noisy[::7, ::11] += 3                        # sensor noise must not wake it
    assert static(noisy, 22.5) is True


def test_heartbeat_rates():
    from npr.heartbeat import FIELDS, Heartbeat

    hb = Heartbeat()
    hb.sample("starting", 0.0, True)
    for _ in range(30):
        hb.counters.arrived(1000)
    hb.counters.processed += 10
    hb.sample("running", 12.0, True)
    row = dict(zip(FIELDS, hb.samples[-1]))
    assert row["recv_fps"] > row["proc_fps"] > 0 and row["kbps"] > 0 and row["infer_ms"] == 12.0
