// OpenStreetMap view: one marker + field-of-view cone per camera/file, flows between them,
// and the route of the selected vehicle.
import { $, api, esc, fmtDur, state, sourceName, toast } from "./app.js";

const R = 6371000, rad = (d) => (d * Math.PI) / 180, deg = (r) => (r * 180) / Math.PI;

function destination([lat, lon], bearing, dist) {
  const b = rad(bearing), d = dist / R, p1 = rad(lat), l1 = rad(lon);
  const p2 = Math.asin(Math.sin(p1) * Math.cos(d) + Math.cos(p1) * Math.sin(d) * Math.cos(b));
  const l2 = l1 + Math.atan2(Math.sin(b) * Math.sin(d) * Math.cos(p1), Math.cos(d) - Math.sin(p1) * Math.sin(p2));
  return [deg(p2), ((deg(l2) + 540) % 360) - 180];
}
function bearing([lat1, lon1], [lat2, lon2]) {
  const p1 = rad(lat1), p2 = rad(lat2), dl = rad(lon2 - lon1);
  return (deg(Math.atan2(Math.sin(dl) * Math.cos(p2), Math.cos(p1) * Math.sin(p2) - Math.sin(p1) * Math.cos(p2) * Math.cos(dl))) + 360) % 360;
}
export function cone(s) {
  const pts = [[s.lat, s.lon]], n = 20, hfov = s.hfov || 70, range = s.range_m || 60;
  for (let i = 0; i <= n; i++) pts.push(destination([s.lat, s.lon], s.heading - hfov / 2 + (hfov * i) / n, range));
  return pts;
}
// gentle arc so A→B and B→A don't sit on top of each other
function arc(a, b) {
  const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2, dx = b[0] - a[0], dy = b[1] - a[1];
  const c = [mx - dy * 0.18, my + dx * 0.18], pts = [];
  for (let i = 0; i <= 16; i++) {
    const t = i / 16, u = 1 - t;
    pts.push([u * u * a[0] + 2 * u * t * c[0] + t * t * b[0], u * u * a[1] + 2 * u * t * c[1] + t * t * b[1]]);
  }
  return pts;
}
const FOV_SOURCE = { exif: "lens from EXIF", metadata: "lens from metadata", device: "lens from device", profile: "lens from camera profile", manual: "set by hand", default: "assumed 70°" };

export class MapView {
  constructor({ refreshSources, onConfigure }) {
    this.onConfigure = onConfigure;
    this.refreshSources = refreshSources;
    this.items = new Map();      // source id → { marker, cone }
    this.selected = null;
    this.placing = null;
    this.flows = [];
    this.showFlows = true;
    let view = null;
    try { view = JSON.parse(localStorage.getItem("npr.map")); } catch {}
    this.map = L.map("map", { zoomControl: true, worldCopyJump: true }).setView(view?.c || [20, 0], view?.z || 2);
    this.hadView = !!view;
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19, attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(this.map);
    this.coneLayer = L.layerGroup().addTo(this.map);
    this.flowLayer = L.layerGroup().addTo(this.map);
    this.routeLayer = L.layerGroup().addTo(this.map);
    this.map.on("moveend", () => {
      if (!this.shown) return;
      const c = this.map.getCenter();
      localStorage.setItem("npr.map", JSON.stringify({ c: [c.lat, c.lng], z: this.map.getZoom() }));
    });
    this.map.on("click", (e) => (this.placing != null ? this.place(e.latlng) : this.select(null)));
    $("#map-fit").onclick = () => this.fit(true);
    $("#map-unplaced").addEventListener("click", (e) => { if (e.target.dataset.place) this.beginPlace(+e.target.dataset.place); });
    $("#map-flows").onclick = (e) => { this.showFlows = !this.showFlows; e.target.classList.toggle("on", this.showFlows); this.drawFlows(); };
  }

  resize() { setTimeout(() => this.map.invalidateSize(), 50); }

  // the map is created while hidden (zero size): size it when its view opens, and frame the cameras
  // the first time unless a real saved position exists
  show() {
    this.map.invalidateSize();
    if (!this.shown) { this.shown = true; this.fit(!this.hadView || this.map.getZoom() < 4); }
  }

  fit(force = false) {
    const pts = [...state.sources.values()].filter((s) => s.lat != null).map((s) => [s.lat, s.lon]);
    if (pts.length && (force || !this.hadView)) this.map.fitBounds(L.latLngBounds(pts).pad(0.5), { maxZoom: 17 });
  }

  // ---- markers & cones ------------------------------------------------------------
  sync() {
    for (const [id, it] of this.items) {
      if (!state.sources.has(id) || state.sources.get(id).lat == null) {
        it.marker.remove(); it.cone?.remove(); this.items.delete(id);
        if (this.selected === id) this.select(null);
      }
    }
    for (const s of state.sources.values()) {
      if (s.lat == null) continue;
      let it = this.items.get(s.id);
      if (!it) {
        const marker = L.marker([s.lat, s.lon], { draggable: true, icon: this.icon(s), zIndexOffset: 500 }).addTo(this.map);
        marker.bindTooltip("", { direction: "top", offset: [0, -14] });
        it = { marker, cone: null, dragging: false };
        marker.on("click", (e) => { L.DomEvent.stopPropagation(e); this.select(s.id); });
        marker.on("dragstart", () => (it.dragging = true));
        marker.on("drag", () => { const p = marker.getLatLng(), cur = state.sources.get(s.id); cur.lat = p.lat; cur.lon = p.lng; this.drawCone(cur); this.drawHandle(); });
        marker.on("dragend", async () => {
          it.dragging = false;
          const p = marker.getLatLng();
          await this.patch(s.id, { lat: p.lat, lon: p.lng });
        });
        this.items.set(s.id, it);
      }
      if (!it.dragging) {
        it.marker.setLatLng([s.lat, s.lon]);
        it.marker.setIcon(this.icon(s));
        const st = s.status || {};
        it.marker.setTooltipContent(`${esc(s.name)}${st.state === "running" ? ` · ${st.vehicles || 0} veh` : ""}`);
        if (!this.editing) this.drawCone(s);
      }
    }
    this.drawFlows();
    // sources with no position yet: one click to start placing each
    const loose = [...state.sources.values()].filter((s) => s.lat == null), box = $("#map-unplaced");
    const html = loose.length ? `<span>Not on the map:</span>${loose.map((s) => `<button data-place="${s.id}">${esc(s.name)}</button>`).join("")}` : "";
    box.hidden = !loose.length;
    if (box.innerHTML !== html) box.innerHTML = html;
  }

  icon(s) {
    const st = s.status || {}, live = st.state === "running";
    const cls = ["cam", live && "live", live && st.vehicles > 0 && "busy", this.selected === s.id && "sel"].filter(Boolean).join(" ");
    return L.divIcon({ className: "", html: `<div class="${cls}" data-cam="${s.id}">${s.id}</div>`, iconSize: [26, 26], iconAnchor: [13, 13] });
  }

  drawCone(s) {
    const it = this.items.get(s.id);
    if (!it) return;
    if (s.heading == null) { it.cone?.remove(); it.cone = null; return; }
    const sel = this.selected === s.id;
    const style = { color: sel ? "#4cc2ff" : "#8fd3ff", weight: sel ? 1.5 : 1, fillOpacity: sel ? 0.28 : 0.14, opacity: 0.8, interactive: false };
    if (it.cone) it.cone.setLatLngs(cone(s)).setStyle(style);
    else it.cone = L.polygon(cone(s), style).addTo(this.coneLayer);
  }

  // ---- selection / inspector --------------------------------------------------------------
  select(id, pan = false) {
    const prev = this.selected;
    this.selected = id;
    for (const sid of [prev, id]) {
      const s = state.sources.get(sid), it = this.items.get(sid);
      if (s && it) { it.marker.setIcon(this.icon(s)); this.drawCone(s); }
    }
    const s = state.sources.get(id);
    if (pan && s?.lat != null) this.map.setView([s.lat, s.lon], Math.max(this.map.getZoom(), 16));
    this.drawInspector();
    this.drawHandle();
  }

  drawInspector() {
    const el = $("#inspector"), s = state.sources.get(this.selected);
    el.hidden = !s || s.lat == null;
    if (el.hidden) return;
    const heading = s.heading ?? 0, hfov = s.hfov ?? 70, range = s.range_m ?? 60;
    el.innerHTML = `<header><strong>${esc(s.name)}</strong><span class="tag" title="How the field of view was determined">${FOV_SOURCE[s.fov_source] || s.fov_source}</span>
        <button data-cfg title="All settings for this camera">Configure</button>
        <button class="ghost" data-x title="Close" style="padding:0 6px">✕</button></header>
      <label>Heading<input type="range" name="heading" min="0" max="359" step="1" value="${heading}"><output>${Math.round(heading)}°</output></label>
      <label>Lens FOV<input type="range" name="hfov" min="5" max="180" step="1" value="${hfov}"><output>${Math.round(hfov)}°</output></label>
      <label>Range<input type="range" name="range_m" min="5" max="500" step="5" value="${range}"><output>${Math.round(range)} m</output></label>`;
    el.querySelector("[data-x]").onclick = () => this.select(null);
    el.querySelector("[data-cfg]").onclick = () => this.onConfigure(s.id);
    el.oninput = (e) => {
      if (!e.target.name) return;
      const cur = state.sources.get(this.selected), v = +e.target.value;
      cur[e.target.name] = v;
      if (e.target.name === "hfov") cur.fov_source = "manual";
      e.target.nextElementSibling.textContent = e.target.name === "range_m" ? `${v} m` : `${v}°`;
      this.editing = true;
      this.drawCone(cur); this.drawHandle();
    };
    el.onchange = async (e) => { if (!e.target.name) return; await this.patch(this.selected, { [e.target.name]: +e.target.value }); this.editing = false; };
  }

  // a draggable dot at the far end of the cone: drag it to aim the camera and set its reach
  drawHandle() {
    const s = state.sources.get(this.selected);
    if (!s || s.lat == null) { this.handle?.remove(); this.handle = null; return; }
    const pos = destination([s.lat, s.lon], s.heading ?? 0, s.range_m || 60);
    if (!this.handle) {
      this.handle = L.marker(pos, { draggable: true, zIndexOffset: 900, icon: L.divIcon({ className: "", html: `<div class="handle" title="Drag to aim"></div>`, iconSize: [12, 12], iconAnchor: [6, 6] }) }).addTo(this.map);
      this.handle.on("drag", () => {
        const cur = state.sources.get(this.selected), p = this.handle.getLatLng();
        cur.heading = Math.round(bearing([cur.lat, cur.lon], [p.lat, p.lng]));
        cur.range_m = Math.max(5, Math.round(this.map.distance([cur.lat, cur.lon], p)));
        this.editing = true;
        this.drawCone(cur);
        const el = $("#inspector");
        for (const [k, unit] of [["heading", "°"], ["range_m", " m"]]) {
          const inp = el.querySelector(`[name=${k}]`);
          if (inp) { inp.value = cur[k]; inp.nextElementSibling.textContent = `${cur[k]}${unit}`; }
        }
      });
      this.handle.on("dragend", async () => {
        const cur = state.sources.get(this.selected);
        await this.patch(cur.id, { heading: cur.heading, range_m: cur.range_m });
        this.editing = false;
      });
    } else if (!this.editing) this.handle.setLatLng(pos);
  }

  async patch(id, fields) {
    try {
      const row = await api(`/api/sources/${id}`, { method: "PATCH", json: fields });
      Object.assign(state.sources.get(id) || {}, row);
      this.sync();
      if (this.selected === id) { this.drawInspector(); this.drawHandle(); }
    } catch (e) { toast(esc(e.message), { err: true }); }
  }

  // ---- placing a source that has no location yet --------------------------------------------
  beginPlace(id) {
    this.placing = id;
    const b = $("#map-banner");
    b.hidden = false;
    b.textContent = `Click the map where “${sourceName(id)}” is — Esc to cancel`;
    this.map.getContainer().style.cursor = "crosshair";
  }
  cancelPlace() {
    this.placing = null;
    $("#map-banner").hidden = true;
    this.map.getContainer().style.cursor = "";
  }
  async place(latlng) {
    const id = this.placing, s = state.sources.get(id);
    this.cancelPlace();
    await this.patch(id, { lat: latlng.lat, lon: latlng.lng, heading: s?.heading ?? 0 });
    await this.refreshSources();
    this.select(id);
  }

  // ---- flows, routes, activity ------------------------------------------------------------------
  setFlows(flows) { this.flows = flows || []; this.drawFlows(); }
  drawFlows() {
    this.flowLayer.clearLayers();
    if (!this.showFlows) return;
    for (const f of this.flows) {
      const a = state.sources.get(f.from), b = state.sources.get(f.to);
      if (!a || !b || a.lat == null || b.lat == null) continue;
      L.polyline(arc([a.lat, a.lon], [b.lat, b.lon]), { color: "#3ddc97", weight: 1.5 + Math.log2(f.count + 1) * 1.4, opacity: 0.75, className: "flow" })
        .bindTooltip(`${esc(a.name)} → ${esc(b.name)}<br>${f.count} vehicle${f.count > 1 ? "s" : ""} · median ${fmtDur(f.median_s)}${f.median_kmh ? ` · ${f.median_kmh.toFixed(0)} km/h` : ""}`, { sticky: true })
        .addTo(this.flowLayer);
    }
  }

  setRoute(detail) {
    this.routeLayer.clearLayers();
    if (!detail) return;
    const stops = detail.route.map((v) => state.sources.get(v.source_id)).filter((s) => s && s.lat != null);
    if (!stops.length) return;
    const pts = stops.map((s) => [s.lat, s.lon]);
    if (pts.length > 1) L.polyline(pts, { color: "#ffd23f", weight: 3, opacity: 0.95 }).addTo(this.routeLayer);
    stops.forEach((s, i) => L.circleMarker([s.lat, s.lon], { radius: 18, color: "#ffd23f", weight: 2, fill: false })
      .bindTooltip(`${i + 1}`, { permanent: true, direction: "right", offset: [16, 0] }).addTo(this.routeLayer));
  }

  flash(...ids) {
    for (const id of ids) {
      const el = document.querySelector(`.cam[data-cam="${id}"]`);
      if (!el) continue;
      el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
    }
  }
}
