"""SQLite persistence for sources (cameras/files), sightings, identities and plate reads."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL, uri TEXT NOT NULL DEFAULT '',
    lat REAL, lon REAL, heading REAL, hfov REAL, range_m REAL DEFAULT 60,
    fov_source TEXT DEFAULT 'default', start_time REAL, opts TEXT NOT NULL DEFAULT '{}',
    autostart INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS identities (
    id INTEGER PRIMARY KEY, plate TEXT, plate_conf REAL DEFAULT 0, cls TEXT,
    embedding BLOB, color BLOB, n INTEGER NOT NULL DEFAULT 0,
    first_seen REAL, last_seen REAL
);
CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY, identity_id INTEGER REFERENCES identities(id),
    source_id INTEGER NOT NULL, track_id INTEGER NOT NULL,
    t_first REAL NOT NULL, t_last REAL NOT NULL, cls TEXT,
    plate TEXT, plate_conf REAL DEFAULT 0, embedding BLOB, color BLOB,
    entry_x REAL, entry_y REAL, exit_x REAL, exit_y REAL,
    match_kind TEXT, match_score REAL, open INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS sightings_identity ON sightings(identity_id, t_first);
CREATE INDEX IF NOT EXISTS sightings_source ON sightings(source_id, t_first);
CREATE TABLE IF NOT EXISTS plate_reads (
    id INTEGER PRIMARY KEY, source_id INTEGER NOT NULL, track_id INTEGER NOT NULL,
    sighting_id INTEGER, identity_id INTEGER, plate TEXT NOT NULL, conf REAL, t REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS plate_reads_t ON plate_reads(t);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY, source_id INTEGER NOT NULL, sighting_id INTEGER NOT NULL, ts REAL NOT NULL,
    x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL, thumb INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS observations_sighting ON observations(sighting_id, ts);
CREATE INDEX IF NOT EXISTS observations_source ON observations(source_id, ts);
"""


def _blob(a: np.ndarray | None) -> bytes | None:
    return None if a is None else np.asarray(a, dtype=np.float32).tobytes()


def _arr(b: bytes | None) -> np.ndarray | None:
    return None if b is None else np.frombuffer(b, dtype=np.float32)


class Store:
    def __init__(self, path: Path | str):
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._lock = threading.RLock()
        self._migrate()

    def _migrate(self) -> None:
        """Bring an older database up to date (columns added since it was created)."""
        cols = {r["name"] for r in self._db.execute("PRAGMA table_info(sightings)")}
        if "colour_name" not in cols:
            self._db.execute("ALTER TABLE sightings ADD COLUMN colour_name TEXT")
        self._db.execute("CREATE INDEX IF NOT EXISTS sightings_colour ON sightings(colour_name)")
        from .search import colour_family

        for r in self._db.execute("SELECT id, color FROM sightings WHERE colour_name IS NULL AND color IS NOT NULL").fetchall():
            name = colour_family(_arr(r["color"]))
            if name:
                self._db.execute("UPDATE sightings SET colour_name=? WHERE id=?", (name, r["id"]))

    def _q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def _x(self, sql: str, args: tuple = ()) -> int:
        with self._lock:
            return self._db.execute(sql, args).lastrowid

    # ---- settings documents ---------------------------------------------------
    def get_kv(self, key: str):
        rows = self._q("SELECT value FROM kv WHERE key=?", (key,))
        return json.loads(rows[0]["value"]) if rows else None

    def set_kv(self, key: str, value) -> None:
        self._x("INSERT INTO kv (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)))

    # ---- sources -------------------------------------------------------------
    SOURCE_FIELDS = ("name", "kind", "uri", "lat", "lon", "heading", "hfov", "range_m",
                     "fov_source", "start_time", "opts", "autostart")

    @staticmethod
    def _source(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["opts"] = json.loads(d["opts"] or "{}")
        d["autostart"] = bool(d["autostart"])
        return d

    def sources(self) -> list[dict]:
        return [self._source(r) for r in self._q("SELECT * FROM sources ORDER BY id")]

    def source(self, sid: int) -> dict | None:
        rows = self._q("SELECT * FROM sources WHERE id=?", (sid,))
        return self._source(rows[0]) if rows else None

    def add_source(self, **kw) -> dict:
        kw = {k: v for k, v in kw.items() if k in self.SOURCE_FIELDS}
        kw["opts"] = json.dumps(kw.get("opts") or {})
        kw["created"] = time.time()
        cols = ",".join(kw)
        sid = self._x(f"INSERT INTO sources ({cols}) VALUES ({','.join('?' * len(kw))})", tuple(kw.values()))
        return self.source(sid)

    def update_source(self, sid: int, **kw) -> dict | None:
        kw = {k: v for k, v in kw.items() if k in self.SOURCE_FIELDS}
        if "opts" in kw:
            kw["opts"] = json.dumps(kw["opts"] or {})
        if kw:
            self._x(f"UPDATE sources SET {','.join(f'{k}=?' for k in kw)} WHERE id=?", (*kw.values(), sid))
        return self.source(sid)

    def delete_source(self, sid: int) -> None:
        self._x("DELETE FROM sources WHERE id=?", (sid,))

    # ---- identities ------------------------------------------------------------
    @staticmethod
    def _identity(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["embedding"], d["color"] = _arr(d["embedding"]), _arr(d["color"])
        return d

    def identities_since(self, t: float) -> list[dict]:
        return [self._identity(r) for r in self._q("SELECT * FROM identities WHERE last_seen>=?", (t,))]

    def identity(self, iid: int) -> dict | None:
        rows = self._q("SELECT * FROM identities WHERE id=?", (iid,))
        return self._identity(rows[0]) if rows else None

    def add_identity(self, **kw) -> int:
        kw["embedding"], kw["color"] = _blob(kw.get("embedding")), _blob(kw.get("color"))
        return self._x(f"INSERT INTO identities ({','.join(kw)}) VALUES ({','.join('?' * len(kw))})",
                       tuple(kw.values()))

    def update_identity(self, iid: int, **kw) -> None:
        for k in ("embedding", "color"):
            if k in kw:
                kw[k] = _blob(kw[k])
        self._x(f"UPDATE identities SET {','.join(f'{k}=?' for k in kw)} WHERE id=?", (*kw.values(), iid))

    def delete_identity_if_empty(self, iid: int) -> None:
        with self._lock:
            if not self._q("SELECT 1 FROM sightings WHERE identity_id=? LIMIT 1", (iid,)):
                self._x("DELETE FROM identities WHERE id=?", (iid,))

    # ---- sightings ---------------------------------------------------------------
    @staticmethod
    def _sighting(r: sqlite3.Row, arrays: bool = False) -> dict:
        d = dict(r)
        if arrays:
            d["embedding"], d["color"] = _arr(d["embedding"]), _arr(d["color"])
        else:
            d.pop("embedding", None), d.pop("color", None)
        return d

    @staticmethod
    def _name_colour(kw: dict) -> None:
        """Keep the searchable colour family in step with the colour signature."""
        if kw.get("color") is not None:
            from .search import colour_family

            kw["colour_name"] = colour_family(kw["color"])

    def add_sighting(self, **kw) -> int:
        self._name_colour(kw)
        kw["embedding"], kw["color"] = _blob(kw.get("embedding")), _blob(kw.get("color"))
        return self._x(f"INSERT INTO sightings ({','.join(kw)}) VALUES ({','.join('?' * len(kw))})",
                       tuple(kw.values()))

    def update_sighting(self, sid: int, **kw) -> None:
        self._name_colour(kw)
        for k in ("embedding", "color"):
            if k in kw:
                kw[k] = _blob(kw[k])
        self._x(f"UPDATE sightings SET {','.join(f'{k}=?' for k in kw)} WHERE id=?", (*kw.values(), sid))

    def sightings_of(self, iid: int, arrays: bool = False) -> list[dict]:
        return [self._sighting(r, arrays) for r in
                self._q("SELECT * FROM sightings WHERE identity_id=? ORDER BY t_first", (iid,))]

    def all_sightings(self, since: float = 0) -> list[dict]:
        return [self._sighting(r) for r in
                self._q("SELECT * FROM sightings WHERE t_last>=? AND identity_id IS NOT NULL ORDER BY t_first", (since,))]

    def close_stale_sightings(self) -> None:
        self._x("UPDATE sightings SET open=0 WHERE open=1")

    # ---- plate reads -----------------------------------------------------------------
    def add_plate_read(self, **kw) -> int:
        return self._x(f"INSERT INTO plate_reads ({','.join(kw)}) VALUES ({','.join('?' * len(kw))})",
                       tuple(kw.values()))

    def plate_reads(self, limit: int = 100, q: str | None = None) -> list[dict]:
        if q:
            rows = self._q("SELECT * FROM plate_reads WHERE plate LIKE ? ORDER BY t DESC LIMIT ?",
                           (f"%{q.upper()}%", limit))
        else:
            rows = self._q("SELECT * FROM plate_reads ORDER BY t DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def identity_rows(self, limit: int = 200, q: str | None = None) -> list[dict]:
        sql = """SELECT i.id, i.plate, i.plate_conf, i.cls, i.first_seen, i.last_seen,
                        COUNT(s.id) AS sightings, COUNT(DISTINCT s.source_id) AS cameras,
                        MAX(s.id) AS last_sighting
                 FROM identities i JOIN sightings s ON s.identity_id = i.id {where}
                 GROUP BY i.id ORDER BY i.last_seen DESC LIMIT ?"""
        if q:
            return [dict(r) for r in self._q(sql.format(where="WHERE i.plate LIKE ?"), (f"%{q.upper()}%", limit))]
        return [dict(r) for r in self._q(sql.format(where=""), (limit,))]

    # ---- observations: where each object was, sampled through a recording --------------
    def add_observation(self, **kw) -> int:
        return self._x(f"INSERT INTO observations ({','.join(kw)}) VALUES ({','.join('?' * len(kw))})",
                       tuple(kw.values()))

    def observations(self, sighting_id: int) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM observations WHERE sighting_id=? ORDER BY ts", (sighting_id,))]

    def sighting(self, sid: int) -> dict | None:
        rows = self._q("SELECT * FROM sightings WHERE id=?", (sid,))
        return self._sighting(rows[0]) if rows else None

    def review(self, source_id: int, start_time: float) -> dict:
        """Everything found in one recording, on the recording's own clock (seconds from its start)."""
        tracks: dict[int, dict] = {}
        for r in self._q("""SELECT id, track_id, identity_id, cls, plate, plate_conf, colour_name, t_first, t_last,
                                   match_kind, match_score, open FROM sightings WHERE source_id=? ORDER BY t_first""",
                         (source_id,)):
            tracks[r["id"]] = {"id": r["id"], "track_id": r["track_id"], "identity_id": r["identity_id"],
                               "cls": r["cls"] or "vehicle", "plate": r["plate"], "plate_conf": r["plate_conf"],
                               "colour": r["colour_name"], "t0": r["t_first"] - start_time, "t1": r["t_last"] - start_time,
                               "match": r["match_kind"], "open": bool(r["open"]), "path": [], "thumbs": []}
        for o in self._q("SELECT id, sighting_id, ts, x, y, w, h, thumb FROM observations WHERE source_id=? ORDER BY ts",
                         (source_id,)):
            t = tracks.get(o["sighting_id"])
            if t is None:
                continue
            t["path"].append([round(o["ts"], 2), round(o["x"], 4), round(o["y"], 4), round(o["w"], 4), round(o["h"], 4)])
            if o["thumb"]:
                t["thumbs"].append({"id": o["id"], "ts": round(o["ts"], 2)})
        for t in tracks.values():  # the sampled path is the better clock when it exists
            if t["path"]:
                t["t0"], t["t1"] = min(t["t0"], t["path"][0][0]), max(t["t1"], t["path"][-1][0])
            t["t0"], t["t1"] = round(max(t["t0"], 0.0), 2), round(max(t["t1"], 0.0), 2)
        events = [{"type": "plate", "ts": round(r["t"] - start_time, 2), "plate": r["plate"], "sighting": r["sighting_id"],
                   "conf": r["conf"]}
                  for r in self._q("SELECT t, plate, sighting_id, conf FROM plate_reads WHERE source_id=? ORDER BY t",
                                   (source_id,))]
        events += [{"type": "link", "ts": t["t0"], "sighting": t["id"], "by": t["match"]}
                   for t in tracks.values() if t["match"] and t["match"] != "new"]
        return {"tracks": list(tracks.values()), "events": sorted(events, key=lambda e: e["ts"])}

    # ---- activity feed -----------------------------------------------------------------
    def activity(self, kinds=("plates", "vehicles"), q: str | None = None, colours=(), classes=(),
                 since: float | None = None, until: float | None = None, newest_first: bool = True,
                 limit: int = 200) -> dict:
        """Plate reads and vehicles as one time-ordered list, filtered by text, facets and time.

        Text matches the plate. Facet values are OR-ed within a facet and AND-ed across facets.
        A vehicle's colour is the family seen most often across its sightings.
        """
        order = "DESC" if newest_first else "ASC"
        marks = lambda xs: ",".join("?" * len(xs))  # noqa: E731
        items: list[dict] = []
        counts = {"plates": 0, "vehicles": 0}

        if "plates" in kinds:
            where, args = ["1=1"], []
            if q:
                where.append("p.plate LIKE ?"); args.append(f"%{q}%")
            if colours:
                where.append(f"s.colour_name IN ({marks(colours)})"); args += list(colours)
            if classes:
                where.append(f"s.cls IN ({marks(classes)})"); args += list(classes)
            if since is not None:
                where.append("p.t >= ?"); args.append(since)
            if until is not None:
                where.append("p.t <= ?"); args.append(until)
            base = f"FROM plate_reads p LEFT JOIN sightings s ON s.id = p.sighting_id WHERE {' AND '.join(where)}"
            counts["plates"] = self._q(f"SELECT COUNT(*) AS n {base}", tuple(args))[0]["n"]
            for r in self._q(f"""SELECT p.id, p.t, p.plate, p.conf, p.source_id, p.sighting_id, p.identity_id,
                                        s.cls, s.colour_name AS colour {base} ORDER BY p.t {order} LIMIT ?""",
                             (*args, limit)):
                items.append({"kind": "plate", **dict(r)})

        if "vehicles" in kinds:
            colour_sql = """(SELECT c.colour_name FROM sightings c WHERE c.identity_id = i.id AND c.colour_name IS NOT NULL
                             GROUP BY c.colour_name ORDER BY COUNT(*) DESC LIMIT 1)"""
            where, args = ["1=1"], []
            if q:
                where.append("i.plate LIKE ?"); args.append(f"%{q}%")
            if classes:
                where.append(f"i.cls IN ({marks(classes)})"); args += list(classes)
            if since is not None:
                where.append("i.last_seen >= ?"); args.append(since)
            if until is not None:
                where.append("i.first_seen <= ?"); args.append(until)
            having, hargs = "", []
            if colours:
                having, hargs = f"HAVING colour IN ({marks(colours)})", list(colours)
            sql = f"""SELECT i.id, i.plate, i.plate_conf AS conf, i.cls, i.first_seen, i.last_seen AS t,
                             COUNT(s.id) AS sightings, COUNT(DISTINCT s.source_id) AS cameras,
                             MAX(s.id) AS sighting_id, {colour_sql} AS colour
                      FROM identities i JOIN sightings s ON s.identity_id = i.id
                      WHERE {' AND '.join(where)} GROUP BY i.id {having}"""
            counts["vehicles"] = self._q(f"SELECT COUNT(*) AS n FROM ({sql})", (*args, *hargs))[0]["n"]
            for r in self._q(f"{sql} ORDER BY i.last_seen {order} LIMIT ?", (*args, *hargs, limit)):
                d = dict(r)
                items.append({"kind": "vehicle", "identity_id": d["id"], **d})

        items.sort(key=lambda x: x["t"] or 0, reverse=newest_first)
        return {"items": items[:limit], "counts": counts,
                "totals": {"plates": self._q("SELECT COUNT(*) AS n FROM plate_reads")[0]["n"],
                           "vehicles": self._q("SELECT COUNT(*) AS n FROM identities")[0]["n"]}}

    def clear_analysis(self) -> None:
        with self._lock:
            for t in ("plate_reads", "sightings", "identities", "observations"):
                self._db.execute(f"DELETE FROM {t}")
