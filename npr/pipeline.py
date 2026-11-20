"""One worker thread per source: ingest → detect/segment → track → plates → appearance → publish."""
from __future__ import annotations

import json
import re
import struct
import threading
import time
import traceback

import cv2
import numpy as np
import Vision

from .analysis.reid import ReID
from .appconfig import mask_uri
from .heartbeat import Heartbeat
from .config import settings
from .ingest.files import ImageSource, VideoFileSource, is_image
from .native.frames import Frame, JPEGEncoder
from .store import Store
from .tracking.tracker import Track, Tracker
from .vision.engine import Engine, ModelsMissing
from .vision.plates import colour_histogram, feature_print, read_text

PLATE_CONFIRM = 1.0  # summed OCR confidence needed before a plate is believed (≈ two agreeing reads)


def _brief(e: Exception) -> str:
    """One readable line for an ingest failure: no errno, no URL (it may carry a password)."""
    msg = re.sub(r"\[Errno -?\d+\]\s*", "", str(e))
    msg = re.sub(r":?\s*'[a-z]+://[^']*'", "", msg).strip() or type(e).__name__
    return mask_uri(msg if isinstance(e, (ConnectionError, OSError, TimeoutError)) else f"{type(e).__name__}: {msg}")


def make_source(row: dict):
    kind, uri, opts = row["kind"], row["uri"], row["opts"]
    if kind == "file":
        if is_image(uri):
            return ImageSource(uri, row.get("start_time"))
        return VideoFileSource(uri, row.get("start_time"), loop=bool(opts.get("loop")))
    if kind == "stream":
        from .ingest.stream import StreamSource

        return StreamSource(uri)
    if kind == "screen":
        from .ingest.capture import ScreenSource

        return ScreenSource(uri, fps=float(opts.get("fps", 15)), rect=opts.get("rect"))
    if kind == "camera":
        from .ingest.capture import CameraSource

        return CameraSource(uri)
    raise ValueError(f"unknown source kind {kind!r}")


class Pipeline(threading.Thread):
    def __init__(self, row: dict, engine: Engine, store: Store, reid: ReID, hub,
                 heartbeat: Heartbeat | None = None, conf: dict | None = None):
        super().__init__(name=f"pipeline-{row['id']}", daemon=True)
        self.row, self.engine, self.store, self.reid, self.hub = row, engine, store, reid, hub
        self.sid = row["id"]
        self.hb = heartbeat or Heartbeat()
        # effective configuration (appconfig.effective); the manager swaps it when settings change
        self.conf = conf or {"allowed": set(settings.vehicle_classes.values()), "motion": False,
                             "motion_fraction": 0.002, "keepalive_s": 10.0, "hold_s": 2.0, "reconnect": False}
        self.state, self.error = "starting", None
        self.stats = {"fps": 0.0, "frames": 0, "vehicles": 0, "infer_ms": 0.0, "progress": None,
                      "idle": False, "skipped": 0}
        self.tracker = None
        self.info: dict = {}
        self._halt = threading.Event()
        self._encoder = JPEGEncoder(settings.tile_width, settings.jpeg_quality)
        self._thumb_encoder = JPEGEncoder(320, 0.7)

    # ---- lifecycle ---------------------------------------------------------------------
    def stop(self) -> None:
        self._halt.set()

    def status(self) -> dict:
        return {"state": self.state, "error": self.error, "info": self.info, **self.stats}

    def _set_state(self, state: str, error: str | None = None) -> None:
        self.state, self.error = state, mask_uri(error)
        self.hub.event({"type": "source", "id": self.sid, "status": self.status()})

    def hb_state(self) -> str:
        return "static" if self.state == "running" and self.stats["idle"] else self.state

    def run(self) -> None:
        try:
            self.engine.load()
        except Exception as e:
            if not isinstance(e, ModelsMissing):
                traceback.print_exc()
            self._set_state("error", str(e) if isinstance(e, ModelsMissing) else f"{type(e).__name__}: {e}")
            return
        attempt = 0
        while not self._halt.is_set():
            source, started = None, time.perf_counter()
            try:
                source = make_source(self.row)
                source.counters = self.hb.counters
                source.open()
                self._connected(source)
                try:
                    self._loop(source)
                finally:
                    for t in self.tracker.flush():
                        self._finish(t)
                    self.stats["vehicles"] = 0
                self._publish_meta_only()
                err = "stream ended"
            except Exception as e:
                err = _brief(e)
                if not isinstance(e, (ConnectionError, OSError, TimeoutError)):
                    print(mask_uri(traceback.format_exc()))
            finally:
                if source is not None:
                    try:
                        source.close()
                    except Exception:
                        pass
            if self._halt.is_set():
                break
            if not (self.row["kind"] == "stream" and self.conf.get("reconnect")):
                self._set_state("ended" if err == "stream ended" else "error", None if err == "stream ended" else err)
                return
            # a network stream that drops is retried forever, backing off to once every 30 s
            attempt = 1 if time.perf_counter() - started > 30 else attempt + 1  # only a stable run resets the backoff
            delay = min(30, 2 ** attempt)
            self.hb.counters.reconnects += 1
            self.hb.counters.last_frame = 0.0
            self._set_state("reconnecting", f"{err} · retry {attempt} in {delay}s")
            self._halt.wait(delay)
        self._set_state("stopped")

    def _connected(self, source) -> None:
        self.info = dict(source.info)
        self._still = isinstance(source, ImageSource)
        self._observe = self.row["kind"] == "file"  # recordings keep a sampled path and previews per object
        fps = self.info.get("fps") or 25.0
        next_id = self.tracker._next if self.tracker is not None else 1  # ids stay unique across reconnects
        self.tracker = Tracker(min_hits=1 if self._still else 3, max_misses=int(max(15, fps * 1.2)),
                               high=settings.vehicle_conf)
        self.tracker._next = next_id
        self._req = self.engine.vehicles.request()
        self._counts = not getattr(source, "counts_arrivals", False)
        self._ref = None           # downscaled picture of the last frame that was analysed
        self._hold_until = None    # source-clock time until which every frame is analysed
        self._last_proc = self._idle_tick = -1e9
        self.stats["idle"] = False
        self.hb.counters.connected_at = time.perf_counter()
        self._set_state("running")

    def _loop(self, source) -> None:
        speed = float(self.row["opts"].get("speed", 1.0))
        min_dt = 1.0 / settings.max_fps
        t0 = ts0 = None
        last = 0.0
        fps_t, fps_n = time.perf_counter(), 0
        c = self.hb.counters
        while not self._halt.is_set():
            if source.live and self.stats["idle"]:
                # nothing is moving: look at the scene ten times a second instead of at full rate
                self._halt.wait(max(0.0, last + 0.1 - time.perf_counter()))
            frame = source.read()
            if frame is None:
                break
            if self._counts:
                c.arrived()
            now = time.perf_counter()
            if source.live:
                if now - last < min_dt:
                    continue
            elif speed > 0:
                if t0 is None:
                    t0, ts0 = now, frame.ts
                wait = t0 + (frame.ts - ts0) / speed - now
                if wait > 0 and self._halt.wait(wait):
                    break
                if wait < -1.0:  # can't keep up: re-anchor instead of racing to catch up
                    t0, ts0 = time.perf_counter(), frame.ts
            last = now
            if (dur := self.info.get("duration")):
                self.stats["progress"] = min(1.0, frame.ts / dur)
            if self._static(frame):
                self._idle(frame)
            else:
                self.stats["idle"] = False
                self._last_proc = frame.ts
                self._process(frame)
                fps_n += 1
                self.stats["frames"] += 1
                c.processed += 1
            el = time.perf_counter() - fps_t
            if el >= 1.0:
                self.stats["fps"] = fps_n / el
                fps_t, fps_n = time.perf_counter(), 0

    # ---- motion gate ------------------------------------------------------------------------
    def _static(self, frame: Frame) -> bool:
        """True when this frame shows nothing new, so the models can be skipped.

        The frame is compared with the last *analysed* frame, not the previous one, so slow drift
        (light, shadows) adds up and eventually triggers a fresh look. After any change every frame
        is analysed for ``hold_s`` so the tracker and plate reader can settle.
        """
        conf = self.conf
        if not conf["motion"] or self._still or self.row["opts"].get("mode") == "immediate":
            self._ref = None  # a recording being reviewed is analysed frame by frame, nothing skipped
            return False
        bgra = frame.bgra
        step = max(1, bgra.shape[1] // 640)
        small = cv2.resize(bgra[::step, ::step, 1], (320, 180), interpolation=cv2.INTER_AREA)
        if self._hold_until is None:
            self._hold_until = frame.ts + max(conf["hold_s"], 3.0)
        ref = self._ref
        moving = ref is None or np.count_nonzero(cv2.absdiff(small, ref) > 18) >= conf["motion_fraction"] * small.size
        if moving:
            self._hold_until = max(self._hold_until, frame.ts + conf["hold_s"])
        keepalive = conf["keepalive_s"]
        if (moving or frame.ts < self._hold_until or frame.ts < self._last_proc  # (clock restarted: looping file)
                or (keepalive and frame.ts - self._last_proc >= keepalive)):
            self._ref = small
            return False
        return True

    def _idle(self, frame: Frame) -> None:
        """A skipped frame: the vehicles in view are still there, so keep them alive at 1 Hz."""
        self.stats["idle"] = True
        self.stats["skipped"] += 1
        if abs(frame.ts - self._idle_tick) < 1.0:
            return
        self._idle_tick = frame.ts
        for t in self.tracker.tracks:
            if t.confirmed and t.misses == 0:
                t.last_ts, t.last_wall = frame.ts, frame.wall
                d = t.data
                if "sighting" in d and frame.wall - d.get("dirty_at", 0.0) > 5.0:
                    d["dirty_at"] = frame.wall
                    self.store.update_sighting(d["sighting"], t_last=frame.wall)
        last = getattr(self, "_last", None)
        if last is not None and self.hub.has_viewers(self.sid):  # same picture, fresh status: no re-encode
            w, h, jpg = last[2], last[3], last[4]
            meta = json.dumps(self._meta(frame.ts, frame.wall, self.tracker.tracks, w, h, -99),
                              separators=(",", ":")).encode()
            self.hub.frame(self.sid, struct.pack(">I", len(meta)) + meta + jpg)

    # ---- per frame ------------------------------------------------------------------------
    def _process(self, frame: Frame) -> None:
        w, h = frame.size
        t = time.perf_counter()
        handler = Vision.VNImageRequestHandler.alloc().initWithCVPixelBuffer_options_(frame.pixel_buffer, None)
        dets = self.engine.vehicles.detect(self._req, handler, w, h)
        allowed, names = self.conf["allowed"], self.engine.vehicles.classes
        dets = [d for d in dets if names.get(d.cls) in allowed]  # per-camera / global object filter
        self.stats["infer_ms"] = 0.9 * self.stats["infer_ms"] + 0.1 * (time.perf_counter() - t) * 1000
        tracks, ended = self.tracker.update(dets, frame.ts, frame.wall)
        visible = [t for t in tracks if t.confirmed and t.visible]
        self.stats["vehicles"] = len(visible)

        self._read_plates(frame, handler, visible, w, h)
        self._embed(frame, handler, visible, w, h)
        for t in visible:
            self._sight(t, frame, w, h)
        for t in ended:
            self._finish(t)
        self._publish(frame, tracks, w, h)

    def _read_plates(self, frame: Frame, handler, tracks: list[Track], w: int, h: int) -> None:
        due = []
        for t in tracks:
            b = t.box
            if b[2] - b[0] < settings.min_vehicle_px or self.engine.vehicles.classes.get(t.cls) == "bicycle":
                continue
            d = t.data
            every = 1 if self._still else (20 if d.get("plate_ok") else 2)
            if frame.index - d.get("plate_try", -999) >= every:
                due.append(((b[2] - b[0]) * (b[3] - b[1]) * (0.25 if d.get("plate_ok") else 1.0), t))
        due.sort(key=lambda x: -x[0])
        budget = len(due) if self._still else settings.plate_attempts_per_frame
        bgra = frame.bgra if due else None
        for _, t in due[:budget]:
            d = t.data
            d["plate_try"] = frame.index
            x1, y1 = max(int(t.box[0]), 0), max(int(t.box[1]), 0)
            x2, y2 = min(int(t.box[2]), w), min(int(t.box[3]), h)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            found = self.engine.plates.detect(np.ascontiguousarray(bgra[y1:y2, x1:x2, :3]))
            if not found:
                continue
            (px1, py1, px2, py2), det_conf = max(found, key=lambda f: f[1])
            box = (x1 + px1, y1 + py1, x1 + px2, y1 + py2)
            d["plate_box"], d["plate_box_frame"] = box, frame.index
            if px2 - px1 < 24:  # too few pixels across the plate to read
                continue
            got = read_text(handler, box, w, h, settings.plate_min_chars, settings.plate_max_chars)
            if got is None or got[1] < settings.ocr_conf:
                continue
            text, conf = got
            votes = d.setdefault("plate_votes", {})
            votes[text] = votes.get(text, 0.0) + conf * det_conf
            d["plate_reads"] = d.get("plate_reads", 0) + 1
            best = max(votes, key=votes.get)
            total = sum(votes.values())
            need = 0.25 if self._still else PLATE_CONFIRM
            if votes[best] >= need:
                d["plate_ok"] = True
                share = votes[best] / total
                if best != d.get("plate"):
                    d["plate"], d["plate_dirty"] = best, True
                d["plate_conf"] = round(min(0.99, share * (1 - 0.5 ** (votes[best] / need))), 3)

    def _embed(self, frame: Frame, handler, tracks: list[Track], w: int, h: int) -> None:
        due = []
        for t in tracks:
            b, d = t.box, t.data
            if b[2] - b[0] < settings.min_vehicle_px or d.get("embeds", 0) >= 6:
                continue
            if frame.index - d.get("embed_at", -999) < 8:
                continue
            inside = b[0] > 2 and b[1] > 2 and b[2] < w - 2 and b[3] < h - 2  # whole vehicle in view
            due.append((float((b[2] - b[0]) * (b[3] - b[1])) * (1.0 if inside else 0.2), t))
        due.sort(key=lambda x: -x[0])
        for _, t in due[: len(due) if self._still else settings.embeds_per_frame]:
            d = t.data
            d["embed_at"] = frame.index
            box = tuple(np.clip(t.box, 0, [w, h, w, h]))
            emb = feature_print(handler, box, w, h)
            if emb is None:
                continue
            n = d.get("embeds", 0)
            d["embedding"] = emb if n == 0 else (d["embedding"] * n + emb) / (n + 1)
            col = colour_histogram(frame.bgra, box, t.mask)
            if col is not None:
                d["color"] = col if d.get("color") is None else (d["color"] * n + col) / (n + 1)
            d["embeds"] = n + 1
            d["embed_dirty"] = n in (0, 2, 5)

    # ---- sightings & identities ---------------------------------------------------------------
    def _sighting_fields(self, t: Track) -> dict:
        d = t.data
        emb = d.get("embedding")
        if emb is not None:
            emb = emb / (np.linalg.norm(emb) or 1.0)
        return {"t_first": t.first_wall, "t_last": t.last_wall, "cls": self.engine.vehicles.classes.get(t.cls),
                "plate": d.get("plate"), "plate_conf": d.get("plate_conf", 0.0),
                "embedding": emb, "color": d.get("color")}

    def _sight(self, t: Track, frame: Frame, w: int, h: int) -> None:
        d = t.data
        if "sighting" not in d:
            if not (self._still or t.hits >= 8) or (d.get("embedding") is None and not d.get("plate")):
                return
            f = self._sighting_fields(t)
            ex, ey = (t.trail[0][1] / w, t.trail[0][2] / h) if t.trail else (None, None)
            d["sighting"] = self.store.add_sighting(source_id=self.sid, track_id=t.id, entry_x=ex, entry_y=ey, **f)
            d["dirty_at"] = 0.0
            self._assign(t, f)
        elif d.get("plate_dirty") or d.get("embed_dirty"):
            f = self._sighting_fields(t)
            self.store.update_sighting(d["sighting"], **f)
            self._assign(t, f)
        elif frame.wall - d.get("dirty_at", 0.0) > 2.0:
            d["dirty_at"] = frame.wall
            self.store.update_sighting(d["sighting"], t_last=t.last_wall)
        if d.pop("plate_dirty", False) and d.get("plate"):
            self._plate_event(t)
        d.pop("embed_dirty", None)
        b = t.box
        if self._observe and "sighting" in d and (self._still or abs(frame.ts - d.get("obs_ts", -1e9)) >= 0.4):
            d["obs_ts"] = frame.ts
            bw, bh = float(b[2] - b[0]), float(b[3] - b[1])
            inside = b[0] > 1 and b[1] > 1 and b[2] < w - 1 and b[3] < h - 1
            keep = d.get("obs_thumbs", 0) < 8 and (inside or d.get("obs_thumbs", 0) == 0) \
                and abs(frame.ts - d.get("obs_thumb_ts", -1e9)) >= 1.0 and bw >= 24
            oid = self.store.add_observation(source_id=self.sid, sighting_id=d["sighting"], ts=frame.ts,
                                             x=float(b[0] + b[2]) / 2 / w, y=float(b[1] + b[3]) / 2 / h,
                                             w=bw / w, h=bh / h, thumb=int(keep))
            if keep:
                d["obs_thumbs"], d["obs_thumb_ts"] = d.get("obs_thumbs", 0) + 1, frame.ts
                try:
                    box = tuple(np.clip(b, 0, [w, h, w, h]))
                    (settings.data_dir / "obs" / f"{oid}.jpg").write_bytes(
                        self._thumb_encoder.encode(frame.pixel_buffer, crop=box))
                except Exception:
                    pass
        # keep the largest view of the vehicle as its thumbnail
        area = float((b[2] - b[0]) * (b[3] - b[1]))
        if "sighting" in d and area > d.get("thumb_area", 0.0) * 1.3 and b[0] > 0 and b[2] < w:
            d["thumb_area"] = area
            try:
                jpg = self._thumb_encoder.encode(frame.pixel_buffer, crop=tuple(b))
                (settings.data_dir / "thumbs" / f"{d['sighting']}.jpg").write_bytes(jpg)
            except Exception:
                pass

    def _assign(self, t: Track, fields: dict) -> None:
        d = t.data
        prev = d.get("identity")
        m = self.reid.assign(d["sighting"], {**fields, "source_id": self.sid, "identity_id": prev}, self.row)
        d["identity"], d["match_kind"], d["match_score"] = m.identity_id, m.kind, m.score
        if m.identity_id != prev and m.kind != "new":
            others = [s for s in self.store.sightings_of(m.identity_id) if s["id"] != d["sighting"]]
            if others:
                o = max(others, key=lambda s: s["t_last"])
                ident = self.store.identity(m.identity_id) or {}
                self.hub.event({
                    "type": "match", "identity": m.identity_id, "plate": ident.get("plate"),
                    "cls": fields["cls"], "kind": m.kind, "score": round(m.score, 3),
                    "from_source": o["source_id"], "to_source": self.sid, "sighting": d["sighting"],
                    "from_sighting": o["id"], "t": t.first_wall, "gap_s": t.first_wall - o["t_last"],
                    "cross_camera": o["source_id"] != self.sid,
                })

    def _plate_event(self, t: Track) -> None:
        d = t.data
        rid = self.store.add_plate_read(source_id=self.sid, track_id=t.id, sighting_id=d.get("sighting"),
                                        identity_id=d.get("identity"), plate=d["plate"],
                                        conf=d.get("plate_conf", 0.0), t=t.last_wall)
        self.hub.event({"type": "plate", "id": rid, "source_id": self.sid, "track_id": t.id,
                        "sighting_id": d.get("sighting"), "identity_id": d.get("identity"),
                        "plate": d["plate"], "conf": d.get("plate_conf", 0.0), "t": t.last_wall,
                        "cls": self.engine.vehicles.classes.get(t.cls)})

    def _finish(self, t: Track) -> None:
        d = t.data
        if "sighting" not in d:
            return
        f = self._sighting_fields(t)
        w, h = self.info.get("width") or 1, self.info.get("height") or 1
        ex, ey = (t.trail[-1][1] / w, t.trail[-1][2] / h) if t.trail else (None, None)
        self.store.update_sighting(d["sighting"], exit_x=ex, exit_y=ey, open=0, **f)
        self._assign(t, f)

    # ---- output ------------------------------------------------------------------------------------
    def _meta(self, frame_ts: float, wall: float, tracks: list[Track], w: int, h: int, index: int) -> dict:
        out = []
        for t in tracks:
            if not t.confirmed or t.misses > 8:
                continue
            d = t.data
            item = {
                "id": t.id, "gid": d.get("identity"), "cls": self.engine.vehicles.classes.get(t.cls, "vehicle"),
                "conf": round(t.conf, 2), "box": [round(float(v), 1) for v in t.box], "lost": t.misses > 0,
                "plate": d.get("plate"), "plate_conf": d.get("plate_conf"), "match": d.get("match_kind"),
                "trail": [[round(x, 1), round(y, 1)] for _, x, y in list(t.trail)[-60::2]],
            }
            if t.polygon is not None and t.misses == 0:
                item["poly"] = np.round(t.polygon, 1).tolist()
            if index >= 0 and d.get("plate_box_frame", -99) >= index - 2:
                item["plate_box"] = [round(float(v), 1) for v in d["plate_box"]]
            out.append(item)
        return {"t": frame_ts, "wall": wall, "w": w, "h": h, "tracks": out,
                "fps": round(self.stats["fps"], 1), "infer_ms": round(self.stats["infer_ms"], 1),
                "progress": self.stats["progress"], "state": self.state, "idle": self.stats["idle"]}

    def _publish(self, frame: Frame, tracks: list[Track], w: int, h: int) -> None:
        if not (self.hub.has_viewers(self.sid) or self._still or frame.index % 30 == 1 or not hasattr(self, "_last")):
            return
        meta = json.dumps(self._meta(frame.ts, frame.wall, tracks, w, h, frame.index),
                          separators=(",", ":")).encode()
        jpg = self._encoder.encode(frame.pixel_buffer)
        now = time.time()
        self._last = (frame.ts, frame.wall, w, h, jpg, now)
        if now - getattr(self, "_snap_t", 0.0) > 180:  # keep a picture on disk for when the camera is off
            self._snap_t = now
            try:
                (settings.data_dir / "snaps" / f"{self.sid}.jpg").write_bytes(jpg)
            except OSError:
                pass
        self.hub.frame(self.sid, struct.pack(">I", len(meta)) + meta + jpg)

    def _publish_meta_only(self) -> None:
        """After the source ends, re-send the last picture with the final state."""
        last = getattr(self, "_last", None)
        if last is None or self._still:
            return
        ts, wall, w, h, jpg = last[:5]
        self.state = "ended"
        meta = json.dumps(self._meta(ts, wall, [], w, h, 0), separators=(",", ":")).encode()
        self.hub.frame(self.sid, struct.pack(">I", len(meta)) + meta + jpg)
