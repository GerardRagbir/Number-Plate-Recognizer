"""HTTP API without starting any pipeline (no models needed)."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    from npr import config

    monkeypatch.setattr(config.settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(config.settings, "models_dir", tmp_path / "models")
    from npr.server.app import create_app

    with TestClient(create_app()) as c:
        yield c


def test_source_lifecycle_and_cone(client):
    r = client.post("/api/sources", json={"kind": "stream", "uri": "rtsp://example.invalid/x", "name": "Gate", "start": False})
    assert r.status_code == 200, r.text
    src = r.json()
    assert src["status"]["state"] == "idle" and src["fov_source"] == "default" and "cone" not in src

    r = client.patch(f"/api/sources/{src['id']}", json={"lat": 10.65, "lon": -61.5, "heading": 45, "hfov": 90})
    row = r.json()
    assert row["fov_source"] == "manual"
    assert len(row["cone"]) == 18 and row["cone"][0] == [10.65, -61.5]

    assert [s["name"] for s in client.get("/api/sources").json()] == ["Gate"]
    assert client.delete(f"/api/sources/{src['id']}").json() == {"ok": True}
    assert client.get("/api/sources").json() == []


def test_bad_sources_are_rejected(client):
    assert client.post("/api/sources", json={"kind": "file", "uri": "/nope/missing.mp4"}).status_code == 404
    assert client.post("/api/sources", json={"kind": "stream", "uri": ""}).status_code == 400
    assert client.post("/api/sources", json={"kind": "bogus"}).status_code == 400


def test_empty_analysis_endpoints(client):
    assert client.get("/api/plates").json() == []
    assert client.get("/api/identities").json() == []
    assert client.get("/api/patterns").json()["flows"] == []
    assert client.get("/api/identities/1").status_code == 404
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_passwords_never_leave_the_server(client):
    r = client.post("/api/sources", json={"kind": "stream", "uri": "rtsp://admin:hunter2@cam.invalid/live", "start": False})
    src = r.json()
    assert src["uri"] == "rtsp://admin:***@cam.invalid/live" and "hunter2" not in r.text
    assert "hunter2" not in client.get("/api/sources").text
    # editing the masked address keeps the stored password; a new one replaces it
    client.patch(f"/api/sources/{src['id']}", json={"uri": "rtsp://admin:***@cam.invalid/live2"})
    assert client.app.state.store.source(src["id"])["uri"] == "rtsp://admin:hunter2@cam.invalid/live2"
    client.patch(f"/api/sources/{src['id']}", json={"uri": "rtsp://admin:new@cam.invalid/live2"})
    assert client.app.state.store.source(src["id"])["uri"] == "rtsp://admin:new@cam.invalid/live2"


def test_global_and_camera_config(client):
    meta = client.get("/api/config").json()
    assert meta["config"]["motion"]["enabled"] is True and "car" in meta["object_types"]
    cfg = meta["config"]
    cfg["objects"]["bicycle"] = False
    cfg["motion"]["sensitivity"] = "nonsense"
    saved = client.put("/api/config", json=cfg).json()["config"]
    assert saved["objects"]["bicycle"] is False and saved["motion"]["sensitivity"] == "medium"
    assert client.get("/api/config").json()["config"]["objects"]["bicycle"] is False  # persisted

    src = client.post("/api/sources", json={"kind": "stream", "uri": "rtsp://x.invalid/a", "start": False,
                                            "config": {"camera_type": "ptz"}}).json()
    assert src["config"]["camera_type"] == "ptz" and src["config"]["objects"] == {}
    row = client.patch(f"/api/sources/{src['id']}", json={
        "lat": 1.0, "lon": 2.0, "config": {"objects": {"bicycle": "allow", "truck": "block", "bogus": "allow"}, "motion": "off"}}).json()
    assert row["config"]["objects"] == {"bicycle": "allow", "truck": "block"} and row["config"]["camera_type"] == "ptz"
    mgr = client.app.state.manager
    eff = mgr.conf_for(client.app.state.store.source(src["id"]))
    assert eff["allowed"] == {"bicycle", "car", "motorcycle", "bus"} and eff["motion"] is False


def test_heartbeat_and_nettest_endpoints(client):
    hb = client.get("/api/heartbeat").json()
    assert hb["sources"] == {} and hb["fields"][0] == "t"
    r = client.post("/api/nettest", json={"uri": "rtsp://user:pw@127.0.0.1:9/none"})
    import json

    steps = [json.loads(line) for line in r.text.splitlines()]
    assert [s["id"] for s in steps] == ["address", "dns", "tcp", "handshake", "open", "frames"]
    assert [s["status"] for s in steps] == ["ok", "ok", "fail", "skip", "skip", "skip"]
    assert "pw" not in r.text.replace("pw@", "") and steps[2]["hint"]
    assert client.post("/api/nettest", json={"uri": "banana"}).text.splitlines()[0].count('"fail"') == 1


def test_camera_profiles_and_probe(client):
    cat = client.get("/api/camera-profiles").json()
    eufy = next(b for b in cat["brands"] if b["id"] == "eufy")
    assert eufy["connect"]["path"] == "/live{channel}" and eufy["connect"]["port"] == 554
    cam = next(m for m in eufy["models"] if m["id"] == "eufy-indoor-cam-2k")
    assert 115 < cam["hfov"] < 121 and cam["lens"]["fov"] == 125      # horizontal worked out from the headline angle
    doorbell = next(m for m in eufy["models"] if m["id"] == "eufy-doorbell-dual")
    assert doorbell["hfov"] == 160                                     # already horizontal: used as is
    for m in eufy["models"]:
        assert m["id"] and m["name"] and "rtsp" in m

    from npr import profiles

    assert profiles.candidate_paths("eufycam-s330")[0] == "/live3"
    assert profiles.candidate_paths(None) == [f"/live{i}" for i in range(6)]

    src = client.post("/api/sources", json={"kind": "stream", "uri": "rtsp://u:p@cam.invalid/live0", "start": False,
                                            "hfov": 118, "fov_source": "profile",
                                            "config": {"profile": "eufy-indoor-cam-2k", "make_model": "eufy Indoor Cam 2K"}}).json()
    assert src["fov_source"] == "profile" and src["config"]["profile"] == "eufy-indoor-cam-2k"
    assert client.post("/api/nettest/probe", json={"uri": "http://x.invalid/a"}).status_code == 400
    r = client.post("/api/nettest/probe", json={"uri": "rtsp://u:secret@127.0.0.1:9/live1", "profile": "eufycam-2"}).json()
    assert r["found"] is None and r["paths"][0]["path"] == "/live0" and "secret" not in str(r)


def test_groups_layer_between_global_and_camera(client):
    g = client.post("/api/groups", json={"name": "Perimeter", "objects": {"truck": "block", "bogus": "x"}, "motion": "off"}).json()
    assert g["id"] == 1 and g["objects"] == {"truck": "block"} and g["motion"] == "off"
    a = client.post("/api/sources", json={"kind": "stream", "uri": "rtsp://a.invalid/x", "start": False, "config": {"group": g["id"]}}).json()
    b = client.post("/api/sources", json={"kind": "stream", "uri": "rtsp://b.invalid/x", "start": False,
                                          "config": {"group": g["id"], "objects": {"truck": "allow"}, "motion": "on"}}).json()
    assert a["config"]["group"] == 1
    mgr, store = client.app.state.manager, client.app.state.store
    ea, eb = mgr.conf_for(store.source(a["id"])), mgr.conf_for(store.source(b["id"]))
    assert "truck" not in ea["allowed"] and ea["motion"] is False        # group rule applies
    assert "truck" in eb["allowed"] and eb["motion"] is True             # the camera's own rule wins
    client.patch(f"/api/groups/{g['id']}", json={"name": "Gates", "objects": {}})
    assert client.get("/api/groups").json()[0]["name"] == "Gates"
    assert "truck" in mgr.conf_for(store.source(a["id"]))["allowed"]
    assert client.delete(f"/api/groups/{g['id']}").json() == {"ok": True}
    assert client.get("/api/groups").json() == []
    assert client.get("/api/sources").json()[0]["config"]["group"] is None  # cameras survive, ungrouped
    assert client.get(f"/api/sources/{a['id']}/snapshot.jpg").status_code == 404
    assert client.get("/api/sources").json()[0]["snapshot_t"] is None


def test_activity_feed_filters_by_colour_type_time_and_text(client):
    import numpy as np

    from npr.search import colour_family

    def sig(hue_bin=None, grey=None):
        h = np.zeros(40, dtype=np.float32)
        h[36] = 0.2                                   # every vehicle has some dark pixels
        if hue_bin is not None:
            h[hue_bin * 3 + 1] = 0.8
        else:
            h[36 + grey] += 0.8
        return h

    assert [colour_family(sig(0)), colour_family(sig(11)), colour_family(sig(7)), colour_family(sig(grey=3)),
            colour_family(sig(grey=0)), colour_family(sig(grey=2)), colour_family(None)] == \
        ["red", "red", "blue", "white", "black", "grey", None]

    store = client.app.state.store
    def vehicle(plate, cls, colour, t):
        iid = store.add_identity(plate=plate, plate_conf=0.9, cls=cls, n=1, first_seen=t, last_seen=t + 5)
        sid = store.add_sighting(identity_id=iid, source_id=1, track_id=iid, t_first=t, t_last=t + 5, cls=cls, plate=plate, color=colour)
        if plate:
            store.add_plate_read(source_id=1, track_id=iid, sighting_id=sid, identity_id=iid, plate=plate, conf=0.9, t=t + 1)
        return iid

    red = vehicle("PDE1234", "car", sig(0), 1000)
    blue = vehicle("TCX77", "truck", sig(7), 2000)
    vehicle(None, "car", sig(grey=3), 3000)

    feed = client.get("/api/activity").json()
    assert feed["counts"] == {"plates": 2, "vehicles": 3} and feed["items"][0]["t"] > feed["items"][-1]["t"]
    assert {i["kind"] for i in feed["items"]} == {"plate", "vehicle"}

    only_red = client.get("/api/activity?colour=red").json()["items"]
    assert {i["identity_id"] for i in only_red} == {red} and all(i["colour"] == "red" for i in only_red)
    assert {i["identity_id"] for i in client.get("/api/activity?colour=red,blue&kinds=vehicles").json()["items"]} == {red, blue}
    assert client.get("/api/activity?colour=red&cls=truck").json()["items"] == []            # facets are AND-ed
    assert [i["plate"] for i in client.get("/api/activity?q=tcx-77&kinds=plates").json()["items"]] == ["TCX77"]
    assert {i["identity_id"] for i in client.get("/api/activity?since=1500&until=2500").json()["items"]} == {blue}
    asc = client.get("/api/activity?order=asc&kinds=vehicles").json()["items"]
    assert [i["t"] for i in asc] == sorted(i["t"] for i in asc)

    facets = {f["key"]: f for f in client.get("/api/search/facets").json()}
    assert "maroon" in next(v for v in facets["colour"]["values"] if v["id"] == "red")["words"]
    assert facets["type"]["param"] == "cls"


def test_recording_review_timeline_and_object_graph(client, tmp_path):
    from npr.analysis.review import describe_motion

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a video")
    src = client.post("/api/sources", json={"kind": "file", "uri": str(clip), "start": False, "start_time": 5000,
                                            "lat": 10.0, "lon": -61.0, "heading": 90, "opts": {"mode": "immediate", "speed": 0}}).json()
    store = client.app.state.store
    iid = store.add_identity(plate="PDE1234", plate_conf=0.9, cls="car", n=1, first_seen=5002, last_seen=5006)
    sid = store.add_sighting(identity_id=iid, source_id=src["id"], track_id=1, t_first=5002, t_last=5006, cls="car", plate="PDE1234")
    for k in range(5):   # drives left to right, getting larger
        store.add_observation(source_id=src["id"], sighting_id=sid, ts=2 + k, x=0.1 + 0.2 * k, y=0.5, w=0.1 + 0.03 * k, h=0.1 + 0.03 * k, thumb=int(k % 2 == 0))
    store.add_plate_read(source_id=src["id"], track_id=1, sighting_id=sid, identity_id=iid, plate="PDE1234", conf=0.9, t=5003.5)

    rv = client.get(f"/api/sources/{src['id']}/review").json()
    assert rv["mode"] == "immediate" and rv["located"] is True and rv["state"] == "idle"
    (t,) = rv["tracks"]
    assert (t["t0"], t["t1"], t["cls"], t["plate"]) == (2.0, 6.0, "car", "PDE1234")
    assert len(t["path"]) == 5 and [x["ts"] for x in t["thumbs"]] == [2.0, 4.0, 6.0]
    assert rv["events"] == [{"type": "plate", "ts": 3.5, "plate": "PDE1234", "sighting": sid, "conf": 0.9}]

    g = client.get(f"/api/sightings/{sid}/graph").json()
    assert g["motion"]["across"] == "left to right" and g["motion"]["depth"] == "towards the camera"
    assert len(g["frames"]) == 3 and len(g["ground"]) == 5 and g["ground"][0] != g["ground"][-1]
    assert g["sighting"]["t0"] == 2 and g["identity"]["plate"] == "PDE1234" and g["others"] == []
    assert describe_motion([{"ts": 0, "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}] * 3)["summary"] == "Stationary"

    assert client.get(f"/api/sources/{src['id']}/media").status_code == 200
    assert client.get("/api/obs/999999.jpg").status_code == 404
    stream = client.post("/api/sources", json={"kind": "stream", "uri": "rtsp://x.invalid/a", "start": False}).json()
    assert client.get(f"/api/sources/{stream['id']}/review").status_code == 400
    assert client.delete(f"/api/sources/{src['id']}").json() == {"ok": True}      # observations go with the recording
    assert store.observations(sid) == []


def test_wall_layout_settings_are_validated_and_kept(client):
    cfg = client.get("/api/config").json()["config"]
    assert cfg["ui"]["wall"] == {"mode": "auto", "per_page": 4, "pages": []}
    cfg["ui"]["wall"] = {"mode": "custom", "per_page": 5, "pages": [
        {"name": "Gates", "size": "3", "tiles": [1, "x", 2, 9, 9]}, {"name": "", "size": "bogus", "tiles": []}]}
    wall = client.put("/api/config", json=cfg).json()["config"]["ui"]["wall"]
    assert wall["mode"] == "custom" and wall["per_page"] == 4
    assert wall["pages"][0] == {"name": "Gates", "size": "3", "tiles": [1, None, 2]}
    assert wall["pages"][1] == {"name": "Page 2", "size": "4", "tiles": [None] * 4}
    # saving other settings must not wipe the layout
    cfg2 = client.get("/api/config").json()["config"]
    cfg2["motion"]["enabled"] = False
    assert client.put("/api/config", json=cfg2).json()["config"]["ui"]["wall"]["mode"] == "custom"
