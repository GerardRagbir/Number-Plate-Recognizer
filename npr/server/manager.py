"""Owns the running pipelines and the rules for creating sources."""
from __future__ import annotations

import os
import time

from .. import appconfig
from ..analysis.reid import ReID
from ..config import settings
from ..heartbeat import Heartbeat, Sampler
from ..ingest.files import file_metadata
from ..pipeline import Pipeline
from ..store import Store
from ..vision.engine import Engine
from .hub import Hub

LIVE_KINDS = {"stream", "screen", "camera"}


class Manager:
    def __init__(self, store: Store, hub: Hub):
        self.store, self.hub = store, hub
        self.engine = Engine()
        self.reid = ReID(store)
        self.pipelines: dict[int, Pipeline] = {}
        self.heartbeats: dict[int, Heartbeat] = {}
        self.config = appconfig.clean_global(store.get_kv("config"))
        self.groups: list[dict] = [appconfig.clean_group(g) for g in (store.get_kv("groups") or [])]
        self._sampler = Sampler(self._heartbeat_tick)

    # ---- configuration -------------------------------------------------------------
    def conf_for(self, row: dict) -> dict:
        cam = (row.get("opts") or {}).get("config") or {}
        return appconfig.effective(self.config, cam, self.group(cam.get("group")))

    def _reapply(self) -> None:
        for p in self.pipelines.values():
            p.conf = self.conf_for(p.row)

    # ---- groups ----------------------------------------------------------------------
    def group(self, gid) -> dict | None:
        return next((g for g in self.groups if g["id"] == gid), None) if gid else None

    def save_group(self, doc: dict, gid: int | None = None) -> dict:
        if gid is None:
            g = appconfig.clean_group({**doc, "id": max([x["id"] for x in self.groups], default=0) + 1})
            self.groups.append(g)
        else:
            old = self.group(gid)
            if old is None:
                raise KeyError(gid)
            g = appconfig.clean_group({**old, **doc, "id": gid})
            self.groups[self.groups.index(old)] = g
        self.store.set_kv("groups", self.groups)
        self._reapply()
        return g

    def delete_group(self, gid: int) -> None:
        """Remove a group; its cameras stay and simply become ungrouped."""
        self.groups = [g for g in self.groups if g["id"] != gid]
        self.store.set_kv("groups", self.groups)
        for row in self.store.sources():
            cfg = (row["opts"].get("config") or {})
            if cfg.get("group") == gid:
                row["opts"]["config"] = {**cfg, "group": None}
                self.source_changed(self.store.update_source(row["id"], opts=row["opts"]))
        self._reapply()

    # ---- snapshots ---------------------------------------------------------------------
    def snapshot(self, sid: int) -> bytes | None:
        """The newest picture of a source: live if it is running, otherwise the last one kept on disk."""
        p = self.pipelines.get(sid)
        last = getattr(p, "_last", None) if p is not None else None
        if last is not None:
            return last[4]
        f = settings.data_dir / "snaps" / f"{sid}.jpg"
        return f.read_bytes() if f.exists() else None

    def set_config(self, doc: dict) -> dict:
        self.config = appconfig.clean_global(doc)
        self.store.set_kv("config", self.config)
        self._reapply()
        return self.config

    def source_changed(self, row: dict) -> None:
        """Push an edited source row into its running pipeline."""
        p = self.pipelines.get(row["id"])
        if p is not None:
            p.row.update({k: row[k] for k in ("lat", "lon", "heading", "hfov", "range_m", "name", "opts")})
            p.conf = self.conf_for(row)

    # ---- heartbeat -------------------------------------------------------------------
    def _heartbeat_tick(self) -> None:
        for sid, hb in list(self.heartbeats.items()):
            p = self.pipelines.get(sid)
            if p is None:
                hb.sample("idle", 0.0, False)
            else:
                state = p.hb_state() if p.is_alive() else (p.state if p.state in ("error", "ended") else "stopped")
                hb.sample(state, p.stats["infer_ms"], p.row["kind"] in LIVE_KINDS and state in ("running", "static"))

    def heartbeat(self, window_s: float) -> dict:
        return {sid: {"samples": hb.series(window_s), **hb.summary()} for sid, hb in self.heartbeats.items()}

    # ---- sources -------------------------------------------------------------------
    def create(self, kind: str, uri: str = "", name: str | None = None, **extra) -> dict:
        if kind not in ("file", "stream", "screen", "camera"):
            raise ValueError(f"unknown source kind {kind!r}")
        d = self.config["defaults"]
        fields: dict = {"hfov": d["hfov"], "fov_source": "default", "range_m": d["range_m"]}
        opts = dict(extra.pop("opts", None) or {})
        if extra.get("config"):
            opts["config"] = appconfig.clean_camera(extra["config"])
        extra.pop("config", None)
        fields["opts"] = opts
        if kind == "file":
            uri = os.path.abspath(os.path.expanduser(uri))
            if not os.path.isfile(uri):
                raise FileNotFoundError(uri)
            fields.update(file_metadata(uri))
            if "start_time" not in fields:
                fields["start_time"] = os.path.getmtime(uri)
            name = name or os.path.basename(uri)
        elif kind == "camera":
            from ..ingest.capture import list_cameras

            cam = next((c for c in list_cameras() if c["id"] == uri), None)
            if cam:
                name = name or cam["name"]
                if cam["hfov"]:
                    fields.update(hfov=cam["hfov"], fov_source="device")
        elif kind == "stream" and not uri:
            raise ValueError("a stream needs a URL")
        fields.update({k: v for k, v in extra.items() if v is not None})
        if extra.get("hfov") is not None:  # a lens angle from a camera profile is labelled as such
            fields["fov_source"] = "profile" if extra.get("fov_source") == "profile" else "manual"
        return self.describe(self.store.add_source(kind=kind, uri=uri, name=name or f"{kind} source", **fields))

    def describe(self, row: dict) -> dict:
        p = self.pipelines.get(row["id"])
        row = dict(row)  # never hand the caller's row back with the password masked
        row["status"] = p.status() if p else {"state": "idle", "error": None, "info": {}}
        row["uri"] = appconfig.mask_uri(row["uri"])
        row["config"] = appconfig.clean_camera((row.get("opts") or {}).get("config"))
        last = getattr(p, "_last", None) if p is not None else None
        snap = settings.data_dir / "snaps" / f"{row['id']}.jpg"
        row["snapshot_t"] = last[5] if last is not None else (snap.stat().st_mtime if snap.exists() else None)
        return row

    def list(self) -> list[dict]:
        return [self.describe(r) for r in self.store.sources()]

    def start(self, sid: int) -> dict:
        row = self.store.source(sid)
        if row is None:
            raise KeyError(sid)
        old = self.pipelines.get(sid)
        if old is not None and old.is_alive():
            return self.describe(row)
        if row["kind"] == "file":  # re-analysing a recording replaces its earlier results
            self.clear_source_results(sid)
        self.store.update_source(sid, autostart=1)
        hb = self.heartbeats.setdefault(sid, Heartbeat())
        p = Pipeline(row, self.engine, self.store, self.reid, self.hub, hb, self.conf_for(row))
        self.pipelines[sid] = p
        p.start()
        return self.describe(row)

    def stop(self, sid: int, wait: float = 3.0) -> None:
        p = self.pipelines.get(sid)
        if p is not None:
            p.stop()
            p.join(wait)
        if self.store.source(sid):
            self.store.update_source(sid, autostart=0)

    def delete(self, sid: int) -> None:
        self.stop(sid)
        self.pipelines.pop(sid, None)
        self.heartbeats.pop(sid, None)
        (settings.data_dir / "snaps" / f"{sid}.jpg").unlink(missing_ok=True)
        self.clear_source_results(sid)
        self.store.delete_source(sid)
        self.hub.forget(sid)

    def clear_source_results(self, sid: int) -> None:
        with self.store._lock:
            idents = [r["identity_id"] for r in self.store._q(
                "SELECT DISTINCT identity_id FROM sightings WHERE source_id=? AND identity_id IS NOT NULL", (sid,))]
            for r in self.store._q("SELECT id FROM sightings WHERE source_id=?", (sid,)):
                (settings.data_dir / "thumbs" / f"{r['id']}.jpg").unlink(missing_ok=True)
            for r in self.store._q("SELECT id FROM observations WHERE source_id=? AND thumb=1", (sid,)):
                (settings.data_dir / "obs" / f"{r['id']}.jpg").unlink(missing_ok=True)
            self.store._x("DELETE FROM observations WHERE source_id=?", (sid,))
            self.store._x("DELETE FROM sightings WHERE source_id=?", (sid,))
            self.store._x("DELETE FROM plate_reads WHERE source_id=?", (sid,))
            for iid in idents:
                self.store.delete_identity_if_empty(iid)
                self.reid._refresh(iid)

    def boot(self) -> None:
        self.store.close_stale_sightings()
        self._sampler.start()
        for row in self.store.sources():
            if row["autostart"] and row["kind"] in LIVE_KINDS:
                self.start(row["id"])

    def shutdown(self) -> None:
        self._sampler.stop()
        for p in self.pipelines.values():
            p.stop()
        deadline = time.time() + 4
        for p in self.pipelines.values():
            p.join(max(0.0, deadline - time.time()))
