"""HTTP + WebSocket API and the dashboard."""
from __future__ import annotations

import asyncio
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import appconfig, heartbeat, nettest, profiles, search
from ..analysis.geo import fov_cone
from ..analysis.patterns import identity_route, patterns
from ..analysis.review import describe_motion, ground_path
from ..config import settings
from ..native.preload import preload
from ..store import Store
from .hub import Hub
from .manager import Manager

WEB = Path(__file__).resolve().parent.parent / "web"


class SourceIn(BaseModel):
    kind: str
    uri: str = ""
    name: str | None = None
    lat: float | None = None
    lon: float | None = None
    heading: float | None = None
    hfov: float | None = None
    range_m: float | None = None
    start_time: float | None = None
    opts: dict | None = None
    config: dict | None = None
    fov_source: str | None = None
    start: bool = True


class SourcePatch(BaseModel):
    name: str | None = None
    lat: float | None = None
    lon: float | None = None
    heading: float | None = None
    hfov: float | None = None
    range_m: float | None = None
    start_time: float | None = None
    opts: dict | None = None
    uri: str | None = None
    config: dict | None = None


class GroupIn(BaseModel):
    name: str | None = None
    objects: dict | None = None
    motion: str | None = None


class ProbeIn(BaseModel):
    source_id: int | None = None
    uri: str | None = None
    profile: str | None = None


class NetTestIn(BaseModel):
    source_id: int | None = None
    uri: str | None = None
    seconds: float = 4.0


def create_app() -> FastAPI:
    settings.ensure()
    preload()
    store = Store(settings.data_dir / "npr.sqlite3")
    hub = Hub()
    mgr = Manager(store, hub)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.attach(asyncio.get_running_loop())
        await asyncio.to_thread(mgr.boot)
        yield
        await asyncio.to_thread(mgr.shutdown)

    app = FastAPI(title="Iris", version="1.0.0", lifespan=lifespan)
    app.state.manager, app.state.store, app.state.hub = mgr, store, hub

    def with_cone(row: dict) -> dict:
        if None not in (row.get("lat"), row.get("lon"), row.get("heading")):
            row["cone"] = fov_cone(row["lat"], row["lon"], row["heading"], row.get("hfov") or 70.0,
                                   row.get("range_m") or 60.0)
        return row

    @app.get("/api/system")
    def system():
        return {**mgr.engine.describe(), "time": time.time(),
                "model_ready": mgr.engine.vehicle_model_path().exists()}

    # ---- sources -------------------------------------------------------------------------
    @app.get("/api/sources")
    def sources():
        return [with_cone(r) for r in mgr.list()]

    @app.post("/api/sources")
    def add_source(body: SourceIn):
        try:
            row = mgr.create(body.kind, body.uri, body.name, lat=body.lat, lon=body.lon, heading=body.heading,
                             hfov=body.hfov, range_m=body.range_m, start_time=body.start_time, opts=body.opts,
                             config=body.config, fov_source=body.fov_source)
        except FileNotFoundError as e:
            raise HTTPException(404, f"file not found: {e}")
        except (ValueError, PermissionError) as e:
            raise HTTPException(400, str(e))
        if body.start:
            row = mgr.start(row["id"])
        hub.event({"type": "sources"})
        return with_cone(row)

    @app.post("/api/upload")
    async def upload(files: list[UploadFile], start: bool = True):
        out = []
        for f in files:
            name = re.sub(r"[^\w.\- ]", "_", Path(f.filename or "upload").name)
            dest = settings.data_dir / "uploads" / f"{int(time.time() * 1000)}_{name}"
            with dest.open("wb") as fh:
                while chunk := await f.read(1 << 20):
                    fh.write(chunk)
            row = await asyncio.to_thread(mgr.create, "file", str(dest), name)
            if start:
                row = mgr.start(row["id"])
            out.append(with_cone(row))
        hub.event({"type": "sources"})
        return out

    @app.patch("/api/sources/{sid}")
    async def patch_source(sid: int, body: SourcePatch):
        old = store.source(sid)
        if old is None:
            raise HTTPException(404, "no such source")
        fields = body.model_dump(exclude_unset=True)
        if "hfov" in fields:
            fields["fov_source"] = "manual"
        restart = False
        if "uri" in fields:
            if old["kind"] != "stream" or not (fields["uri"] or "").strip():
                fields.pop("uri")
            else:  # the dashboard only ever sees the masked password: keep the stored one
                fields["uri"] = appconfig.unmask_uri(fields["uri"].strip(), old["uri"])
                restart = fields["uri"] != old["uri"]
        if "opts" in fields or "config" in fields:
            opts = dict(fields.pop("opts", None) or old["opts"])
            opts["config"] = appconfig.clean_camera({**(old["opts"].get("config") or {}), **(fields.pop("config", None) or {})})
            fields["opts"] = opts
        row = store.update_source(sid, **fields)
        mgr.source_changed(row)
        p = mgr.pipelines.get(sid)
        if restart and p is not None and p.is_alive():
            await asyncio.to_thread(mgr.stop, sid)
            mgr.start(sid)
        hub.event({"type": "sources"})
        return with_cone(mgr.describe(row))

    @app.delete("/api/sources/{sid}")
    async def delete_source(sid: int):
        await asyncio.to_thread(mgr.delete, sid)
        hub.event({"type": "sources"})
        return {"ok": True}

    @app.post("/api/sources/{sid}/start")
    def start_source(sid: int):
        try:
            return with_cone(mgr.start(sid))
        except KeyError:
            raise HTTPException(404, "no such source")

    @app.post("/api/sources/{sid}/stop")
    async def stop_source(sid: int):
        await asyncio.to_thread(mgr.stop, sid)
        return {"ok": True}

    @app.get("/api/devices")
    def devices():
        """Displays, windows and cameras this Mac can capture from."""
        out: dict = {"displays": [], "windows": [], "cameras": [], "errors": {}}
        try:
            from ..ingest.capture import list_screens

            out.update(list_screens())
        except Exception as e:
            out["errors"]["screen"] = str(e)
        try:
            from ..ingest.capture import list_cameras

            out["cameras"] = list_cameras()
        except Exception as e:
            out["errors"]["camera"] = str(e)
        return out

    # ---- configuration, heartbeat, connection test -----------------------------------------
    @app.get("/api/config")
    def get_config():
        return {"config": mgr.config, "object_types": appconfig.object_types(),
                "camera_types": [t for t in appconfig.CAMERA_TYPES if t], "camera_defaults": appconfig.camera_defaults()}

    @app.put("/api/config")
    def put_config(body: dict):
        cfg = mgr.set_config(body)
        hub.event({"type": "config"})
        return {"config": cfg}

    @app.get("/api/heartbeat")
    def get_heartbeat(window: float = 300):
        return {"fields": heartbeat.FIELDS, "states": heartbeat.STATES, "now": time.time(),
                "window": window, "sources": mgr.heartbeat(min(max(window, 10), heartbeat.WINDOW_S))}

    # ---- groups & snapshots -----------------------------------------------------------------
    @app.get("/api/groups")
    def groups():
        return mgr.groups

    @app.post("/api/groups")
    def add_group(body: GroupIn):
        g = mgr.save_group(body.model_dump(exclude_unset=True))
        hub.event({"type": "groups"})
        return g

    @app.patch("/api/groups/{gid}")
    def patch_group(gid: int, body: GroupIn):
        try:
            g = mgr.save_group(body.model_dump(exclude_unset=True), gid)
        except KeyError:
            raise HTTPException(404, "no such group")
        hub.event({"type": "groups"})
        return g

    @app.delete("/api/groups/{gid}")
    def delete_group(gid: int):
        mgr.delete_group(gid)
        hub.event({"type": "groups"})
        hub.event({"type": "sources"})
        return {"ok": True}

    @app.get("/api/sources/{sid}/snapshot.jpg")
    def snapshot(sid: int):
        jpg = mgr.snapshot(sid)
        if jpg is None:
            raise HTTPException(404, "no picture yet")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=120"})

    # ---- reviewing a recording ----------------------------------------------------------------
    durations: dict[int, float] = {}

    def duration_of(row: dict) -> float:
        p = mgr.pipelines.get(row["id"])
        d = (p.info.get("duration") if p is not None else None) or durations.get(row["id"])
        if not d:  # not analysed in this run: ask the file
            try:
                import cv2

                cap = cv2.VideoCapture(row["uri"])
                fps, n = cap.get(cv2.CAP_PROP_FPS) or 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                cap.release()
                d = n / fps if fps else 0.0
            except Exception:
                d = 0.0
        durations[row["id"]] = d or 0.0
        return d or 0.0

    def file_source(sid: int) -> dict:
        row = store.source(sid)
        if row is None:
            raise HTTPException(404, "no such source")
        if row["kind"] != "file":
            raise HTTPException(400, "only recordings can be reviewed")
        return row

    @app.get("/api/sources/{sid}/review")
    def review(sid: int):
        """The recording's timeline: every object with its path and previews, plus events."""
        row = file_source(sid)
        data = store.review(sid, row.get("start_time") or 0.0)
        p = mgr.pipelines.get(sid)
        st = p.status() if p is not None else {"state": "idle", "progress": None}
        dur = duration_of(row) or max([t["t1"] for t in data["tracks"]], default=0.0)
        return {"id": sid, "name": row["name"], "duration": round(dur, 2), "state": st["state"], "error": st.get("error"),
                "progress": st.get("progress"), "fps": st.get("fps"), "mode": row["opts"].get("mode"),
                "located": None not in (row.get("lat"), row.get("lon"), row.get("heading")), **data}

    @app.get("/api/sources/{sid}/media")
    def media(sid: int):
        """The original file, for the review player (supports seeking)."""
        row = file_source(sid)
        if not Path(row["uri"]).is_file():
            raise HTTPException(404, "the file is no longer where it was")
        return FileResponse(row["uri"])

    @app.get("/api/obs/{oid}.jpg")
    def obs_thumb(oid: int):
        f = settings.data_dir / "obs" / f"{oid}.jpg"
        if not f.exists():
            raise HTTPException(404, "no preview")
        return FileResponse(f, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/sightings/{sid}/graph")
    def sighting_graph(sid: int):
        """One object and what connects to it: the frames it was seen in, its motion, and other sightings."""
        s = store.sighting(sid)
        if s is None:
            raise HTTPException(404, "no such sighting")
        src = store.source(s["source_id"]) or {}
        start = src.get("start_time") or 0.0
        obs = store.observations(sid)
        ident = store.identity(s["identity_id"]) if s["identity_id"] else None
        names = {r["id"]: (r["name"], r["kind"]) for r in store.sources()}
        others = [{"id": o["id"], "source_id": o["source_id"], "source": names.get(o["source_id"], ("?", ""))[0],
                   "kind": names.get(o["source_id"], ("?", ""))[1], "t_first": o["t_first"], "t_last": o["t_last"],
                   "plate": o["plate"], "by": o["match_kind"], "score": o["match_score"]}
                  for o in (store.sightings_of(s["identity_id"]) if s["identity_id"] else []) if o["id"] != sid]
        return {
            "sighting": {"id": sid, "source_id": s["source_id"], "source": src.get("name"), "cls": s["cls"],
                         "plate": s["plate"], "plate_conf": s["plate_conf"], "colour": s.get("colour_name"),
                         "identity_id": s["identity_id"], "t0": s["t_first"] - start, "t1": s["t_last"] - start,
                         "t_first": s["t_first"], "by": s["match_kind"], "score": s["match_score"]},
            "identity": {"id": ident["id"], "plate": ident["plate"], "cls": ident["cls"]} if ident else None,
            "frames": [{"id": o["id"], "ts": o["ts"], "x": o["x"], "y": o["y"], "w": o["w"], "h": o["h"]} for o in obs if o["thumb"]],
            "path": [[round(o["ts"], 2), round(o["x"], 4), round(o["y"], 4)] for o in obs],
            "motion": describe_motion(obs),
            "ground": ground_path(obs, src),
            "camera": {k: src.get(k) for k in ("lat", "lon", "heading", "hfov", "range_m", "name")},
            "others": others,
        }

    @app.get("/api/camera-profiles")
    def camera_profiles():
        """The local camera database: brands, how to connect, and per-model lens and stream details."""
        return profiles.catalogue()

    @app.post("/api/nettest/probe")
    def probe(body: ProbeIn):
        """Find which stream path a camera answers on, using its profile's likely paths."""
        url, profile = body.uri, body.profile
        if body.source_id is not None:
            row = store.source(body.source_id)
            if row is None:
                raise HTTPException(404, "no such source")
            url = row["uri"]
            profile = profile or ((row["opts"].get("config") or {}).get("profile"))
        if not url:
            raise HTTPException(400, "nothing to probe")
        try:
            rows = nettest.probe_paths(url, profiles.candidate_paths(profile))
        except ValueError as e:
            raise HTTPException(400, str(e))
        found = next((r["path"] for r in rows if r["ok"]), None)
        fixed = None
        if found:  # hand back the address with the working path; the password stays masked
            from urllib.parse import urlsplit

            fixed = appconfig.mask_uri(urlsplit(url)._replace(path=found, query="").geturl())
        hint = None
        if not found and rows and all(r["code"] == 404 for r in rows):
            hint = ("The camera answers and accepts the login, but has no video on any slot right now. Cameras that "
                    "stream only while recording an event look exactly like this between events: in the camera's app, "
                    "set NAS (RTSP) to Continuous, or trigger motion and try again.")
        elif rows and rows[-1]["code"] == 401:
            hint = "The camera rejected the username or password."
        return {"paths": rows, "found": found, "uri": fixed, "hint": hint}

    @app.post("/api/nettest")
    def run_nettest(body: NetTestIn):
        """Runs the connection checks, streaming one JSON line per finished step."""
        url = body.uri
        if body.source_id is not None:
            row = store.source(body.source_id)
            if row is None:
                raise HTTPException(404, "no such source")
            if row["kind"] != "stream":
                raise HTTPException(400, "only network streams have a connection to test")
            url = row["uri"]
        if not url:
            raise HTTPException(400, "nothing to test")
        import json

        def lines():
            for step in nettest.run(url, min(max(body.seconds, 1.0), 15.0)):
                yield json.dumps(step) + "\n"

        return StreamingResponse(lines(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})

    # ---- analysis ----------------------------------------------------------------------------
    @app.get("/api/plates")
    def plates(limit: int = 100, q: str | None = None):
        return store.plate_reads(min(limit, 1000), q)

    @app.get("/api/identities")
    def identities(limit: int = 200, q: str | None = None):
        return store.identity_rows(min(limit, 1000), q)

    @app.get("/api/identities/{iid}")
    def identity(iid: int):
        ident = store.identity(iid)
        if ident is None:
            raise HTTPException(404, "no such identity")
        ident.pop("embedding", None), ident.pop("color", None)
        return {**ident, "sightings": store.sightings_of(iid), "route": identity_route(store, iid)}

    @app.get("/api/patterns")
    def get_patterns(hours: float | None = None, since: float | None = None):
        return patterns(store, since if since is not None else (time.time() - hours * 3600 if hours else None))

    @app.get("/api/search/facets")
    def search_facets():
        """What the search bar can recognise: each facet, its values, and the words that mean them."""
        return search.facets()

    @app.get("/api/activity")
    def activity(kinds: str = "plates,vehicles", q: str | None = None, colour: str | None = None, cls: str | None = None,
                 since: float | None = None, until: float | None = None, order: str = "desc", limit: int = 200):
        split = lambda v: tuple(x for x in (v or "").split(",") if x)  # noqa: E731
        text = re.sub(r"[^A-Za-z0-9]", "", q or "").upper() or None
        return store.activity(kinds=split(kinds), q=text, colours=split(colour), classes=split(cls),
                              since=since, until=until, newest_first=order != "asc", limit=min(max(limit, 1), 1000))

    @app.get("/api/events/recent")
    def recent():
        return list(hub.recent)

    @app.get("/api/thumbs/{sighting}.jpg")
    def thumb(sighting: int):
        p = settings.data_dir / "thumbs" / f"{sighting}.jpg"
        if not p.exists():
            raise HTTPException(404, "no thumbnail")
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})

    @app.post("/api/reset")
    def reset():
        """Forget all sightings, identities and plate reads (sources are kept)."""
        store.clear_analysis()
        for f in (settings.data_dir / "thumbs").glob("*.jpg"):
            f.unlink(missing_ok=True)
        hub.recent.clear()
        hub.event({"type": "reset"})
        return {"ok": True}

    # ---- live ------------------------------------------------------------------------------------
    @app.websocket("/ws/tile/{sid}")
    async def tile(ws: WebSocket, sid: int):
        await ws.accept()
        q = hub.watch_tile(sid)
        try:
            while True:
                await ws.send_bytes(await q.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.unwatch_tile(sid, q)

    @app.websocket("/ws/events")
    async def events(ws: WebSocket):
        await ws.accept()
        q = hub.watch_events()
        try:
            while True:
                await ws.send_json(await q.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.unwatch_events(q)

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})

    @app.middleware("http")
    async def no_stale_assets(request: Request, call_next):
        resp: Response = await call_next(request)
        if request.url.path.startswith("/static/") and "/vendor/" not in request.url.path:
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app
