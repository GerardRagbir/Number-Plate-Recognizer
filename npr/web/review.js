// Review: one recording, full screen.
//
//   player (loops, boxes drawn from the sampled paths)        summary
//   timeline — while analysing: one strip filling in with objects and events
//            — when finished:   a lane per object type, each object a bar with preview images at their times
//   click a preview → the object's connections as linked cards: the frames it was seen in, how it moved,
//                     where that puts it on the map, and where else the same vehicle was seen
import { $, api, esc, fmtTime, fmtDur, state, toast, plateBadge } from "./app.js";
import { icon } from "./icons.js";
import { cone } from "./map.js";

const TYPE_COLOUR = { car: "#3987e5", truck: "#d95926", bus: "#199e70", motorcycle: "#c98500", bicycle: "#d55181", vehicle: "#9085e9" };
const TYPE_ORDER = ["car", "truck", "bus", "motorcycle", "bicycle", "vehicle"];
const SWATCH = { red: "#d0463b", yellow: "#e2b33c", green: "#3f9b5a", blue: "#3f78d0", purple: "#8a5bd0", pink: "#d86aa8", white: "#eef1f4", grey: "#8d98a5", black: "#15181c" };
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : w.endsWith("s") ? "es" : "s"}`;
const clock = (s) => { s = Math.max(0, s); const m = Math.floor(s / 60); return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}${s < 10 ? "." + Math.floor((s % 1) * 10) : ""}`; };

export class Review {
  constructor({ onClose, onPlace }) {
    this.onClose = onClose; this.onPlace = onPlace;
    this.id = null; this.data = null; this.timer = null; this.graphMap = null;
    this.video = $("#rv-video"); this.overlay = $("#rv-overlay");
    $("#rv-back").onclick = () => this.close();
    $("#rv-pick").onchange = (e) => this.open(+e.target.value);
    $("#rv-loop").onclick = (e) => { this.video.loop = !this.video.loop; e.currentTarget.classList.toggle("on", this.video.loop); };
    $("#rv-boxes").onclick = (e) => { this.boxes = !this.boxes; e.currentTarget.classList.toggle("on", this.boxes); this.drawOverlay(); };
    this.boxes = true;
    const tl = $("#rv-timeline");
    tl.addEventListener("click", (e) => this.timelineClick(e));
    $("#rv-graph").addEventListener("click", (e) => this.graphClick(e));
    const tick = () => { if (this.id != null && !$("#view-review").hidden) { this.drawPlayhead(); this.drawOverlay(); } requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
    new ResizeObserver(() => { this.drawOverlay(); this.drawLinks(); }).observe($("#view-review"));
    this.video.addEventListener("error", () => { $("#rv-novideo").hidden = false; });
    this.video.addEventListener("loadeddata", () => { $("#rv-novideo").hidden = true; });
  }

  // ---- lifecycle ----------------------------------------------------------------------
  async open(id) {
    this.closeGraph();
    this.id = id; this.data = null; this.sig = "";
    this.video.src = `/api/sources/${id}/media`;
    this.video.loop = true; $("#rv-loop").classList.add("on");
    this.video.play().catch(() => {});
    const files = [...state.sources.values()].filter((s) => s.kind === "file");
    $("#rv-pick").innerHTML = files.map((s) => `<option value="${s.id}" ${s.id === id ? "selected" : ""}>${esc(s.name)}</option>`).join("");
    $("#rv-pick").hidden = files.length < 2;
    $("#rv-timeline").innerHTML = `<p class="hint pad2">Loading…</p>`;
    await this.poll();
  }
  close() { this.stop(); this.video.pause(); this.video.removeAttribute("src"); this.video.load(); this.id = null; this.onClose(); }
  stop() { clearTimeout(this.timer); this.timer = null; }
  hidden() { this.stop(); this.video.pause(); }
  shown() { if (this.id != null && !this.timer) this.poll(); }

  async poll() {
    this.stop();
    const id = this.id;
    if (id == null) return;
    try {
      const d = await api(`/api/sources/${id}/review`);
      if (id !== this.id) return;
      this.data = d;
      this.render();
      const busy = ["running", "starting"].includes(d.state);
      if (busy && !$("#view-review").hidden) this.timer = setTimeout(() => this.poll(), 1000);   // keep building while it analyses
    } catch (e) { toast(esc(e.message), { err: true }); }
  }

  // ---- header, summary, timeline -------------------------------------------------------
  render() {
    const d = this.data, busy = ["running", "starting"].includes(d.state), done = !busy && d.state !== "error";
    $("#rv-title").textContent = d.name;
    const phase = $("#rv-phase");
    phase.className = `badge ${busy ? "busy" : d.state}`;
    phase.innerHTML = busy ? `<i></i>Analysing every frame · ${Math.round((d.progress || 0) * 100)}%${d.fps ? ` · ${Math.round(d.fps)} fps` : ""}`
      : d.state === "error" ? `<i></i>${esc(d.error || "Failed")}` : d.tracks.length ? `<i></i>Finished · ${plural(d.tracks.length, "object")}` : `<i></i>${d.state === "idle" ? "Not analysed yet" : "Finished · nothing found"}`;
    const byType = new Map();
    for (const t of d.tracks) byType.set(t.cls, (byType.get(t.cls) || 0) + 1);
    const plates = [...new Map(d.tracks.filter((t) => t.plate).map((t) => [t.plate, t])).values()];
    $("#rv-info").innerHTML = `
      <h3>Found</h3>
      <div class="rv-counts">${TYPE_ORDER.filter((k) => byType.has(k)).map((k) => `<div><i style="background:${TYPE_COLOUR[k]}"></i><b>${byType.get(k)}</b><span>${plural(byType.get(k), k).replace(/^\d+ /, "")}</span></div>`).join("") || `<p class="hint">${busy ? "Nothing yet." : "No vehicles."}</p>`}</div>
      <h3>Plates</h3>
      <div class="rv-plates">${plates.map((t) => `<button data-sighting="${t.id}" data-tip="Jump to this vehicle and open its connections.">${plateBadge(t.plate)}<span>${clock(t.t0)}</span></button>`).join("") || `<p class="hint">${busy ? "None read yet." : "None read."}</p>`}</div>
      ${d.located ? "" : `<p class="hint">Not on the map. <button class="linkish" data-place>Place it</button> to plot movement.</p>`}`;
    $("#rv-info").onclick = (e) => {
      if (e.target.closest("[data-place]")) return this.onPlace(this.id);
      const b = e.target.closest("[data-sighting]");
      if (b) this.openGraph(+b.dataset.sighting);
    };
    // only rebuild the timeline when something actually changed (keeps hover and images stable)
    const sig = JSON.stringify([done, d.duration, d.tracks.map((t) => [t.id, t.t0, t.t1, t.plate, t.thumbs.length]), d.events.length, Math.round((d.progress || 0) * 200)]);
    if (sig !== this.sig) { this.sig = sig; done ? this.drawLanes() : this.drawBuilding(); }
  }

  ruler(dur) {
    const step = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600].find((s) => dur / s <= 10) || 900;
    let out = "";
    for (let t = 0; t <= dur + 1e-6; t += step) out += `<span style="left:${(t / dur) * 100}%">${clock(t).replace(/\.\d$/, "")}</span>`;
    return `<div class="tl-ruler">${out}</div>`;
  }

  // While analysing: a single strip that fills in. Objects are bars (coloured by type), plate reads are ticks.
  drawBuilding() {
    const d = this.data, dur = Math.max(d.duration, 0.1), pct = (t) => `${Math.min(Math.max(t / dur, 0), 1) * 100}%`;
    const rows = pack(d.tracks);
    const h = Math.max(rows.n, 3) * 9 + 6;
    $("#rv-timeline").innerHTML = `<div class="tl tl-building">
      <div class="tl-label"><b>Building timeline</b><span>${plural(d.tracks.length, "object")} · ${plural(d.events.filter((e) => e.type === "plate").length, "plate read")}</span>
        <span class="tl-key">${TYPE_ORDER.filter((k) => d.tracks.some((t) => t.cls === k)).map((k) => `<i style="background:${TYPE_COLOUR[k]}"></i>${cap(k)}`).join("")}<i class="ev"></i>Plate read</span></div>
      <div class="tl-track" data-seek>${this.ruler(dur)}
        <div class="tl-body" style="height:${h}px">
          <div class="tl-scanned" style="width:${pct((d.progress || 0) * dur)}"></div>
          ${d.tracks.map((t) => `<i class="bar" style="left:${pct(t.t0)};width:max(3px, calc(${pct(t.t1)} - ${pct(t.t0)}));top:${rows.row.get(t.id) * 9 + 3}px;background:${TYPE_COLOUR[t.cls] || TYPE_COLOUR.vehicle}"></i>`).join("")}
          ${d.events.filter((e) => e.type === "plate").map((e) => `<i class="tick" style="left:${pct(e.ts)}" data-label="${esc(e.plate)}" data-tip="Plate read at ${clock(e.ts)}."></i>`).join("")}
        </div><div class="tl-playhead" id="rv-playhead"></div></div></div>`;
  }

  // Finished: one lane per object type; each object is a bar carrying its preview images at the times they were taken.
  drawLanes() {
    const d = this.data, dur = Math.max(d.duration, 0.1), pct = (t) => `${Math.min(Math.max(t / dur, 0), 1) * 100}%`;
    const types = TYPE_ORDER.filter((k) => d.tracks.some((t) => t.cls === k));
    const lane = (k) => {
      const list = d.tracks.filter((t) => t.cls === k), rows = pack(list, dur * 0.045);
      return `<div class="tl-lane"><div class="tl-label"><span class="tl-type" style="--c:${TYPE_COLOUR[k]}">${icon("car")}</span><b>${cap(plural(list.length, k).replace(/^\d+ /, ""))}</b><span>${list.length}</span></div>
        <div class="tl-track" data-seek><div class="tl-body" style="height:${rows.n * 44 + 4}px">${list.map((t) => `
          <div class="obj" data-sighting="${t.id}" style="left:${pct(t.t0)};width:max(6px, calc(${pct(t.t1)} - ${pct(t.t0)}));top:${rows.row.get(t.id) * 44 + 3}px;--c:${TYPE_COLOUR[k]}"
               data-label="${esc(t.plate || `${cap(k)} #${t.identity_id ?? t.track_id}`)}" data-tip="${clock(t.t0)} to ${clock(t.t1)}. Click a picture for its connections.">
            <i class="span"></i>${t.plate ? `<span class="pl">${esc(t.plate)}</span>` : ""}</div>
          ${t.thumbs.map((th) => `<img class="pv" data-sighting="${t.id}" data-ts="${th.ts}" src="/api/obs/${th.id}.jpg" alt="" loading="lazy" style="left:${pct(th.ts)};top:${rows.row.get(t.id) * 44 + 9}px"
               data-label="${esc(t.plate || cap(k))} at ${clock(th.ts)}" data-tip="Open this object's connections.">`).join("")}`).join("")}</div></div></div>`;
    };
    const plates = d.events.filter((e) => e.type === "plate");
    $("#rv-timeline").innerHTML = `<div class="tl tl-lanes">
      <div class="tl-lane tl-head"><div class="tl-label"></div><div class="tl-track" data-seek>${this.ruler(dur)}</div></div>
      ${types.map(lane).join("") || `<p class="hint pad2">Nothing was found in this recording.</p>`}
      ${plates.length ? `<div class="tl-lane"><div class="tl-label"><span class="tl-type" style="--c:#ffd23f">${icon("plates")}</span><b>Plate reads</b><span>${plates.length}</span></div>
        <div class="tl-track" data-seek><div class="tl-body" style="height:26px">${plates.map((e) => `<button class="ev-plate" data-sighting="${e.sighting}" data-ts="${e.ts}" style="left:${pct(e.ts)}" data-label="${esc(e.plate)}" data-tip="Read at ${clock(e.ts)}, ${Math.round((e.conf || 0) * 100)}% sure.">${esc(e.plate)}</button>`).join("")}</div></div></div>` : ""}
      <div class="tl-playhead" id="rv-playhead"></div></div>`;
  }

  drawPlayhead() {
    const ph = $("#rv-playhead"), d = this.data;
    if (!ph || !d) return;
    const box = ph.parentElement, track = box.matches(".tl-track") ? box : box.querySelector(".tl-track");
    const host = box.getBoundingClientRect(), r = track.getBoundingClientRect();
    const dur = Math.max(d.duration || this.video.duration || 0.1, 0.1);
    ph.style.left = `${r.left - host.left + Math.min(this.video.currentTime / dur, 1) * r.width}px`;
  }

  timelineClick(e) {
    const s = e.target.closest("[data-sighting]"), d = this.data;
    if (s) {
      const t = d.tracks.find((x) => x.id === +s.dataset.sighting);
      this.seek(s.dataset.ts != null ? +s.dataset.ts : t?.t0 ?? 0);
      if (s.matches(".pv, .ev-plate")) this.openGraph(+s.dataset.sighting);
      return;
    }
    const track = e.target.closest("[data-seek]");
    if (!track) return;
    const r = track.getBoundingClientRect();
    this.seek(((e.clientX - r.left) / r.width) * Math.max(d.duration, 0.1));
  }
  seek(t) { try { this.video.currentTime = Math.max(0, t); } catch {} }

  // boxes on the player, interpolated between the sampled positions
  drawOverlay() {
    const cv = this.overlay, v = this.video, d = this.data;
    const r = v.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    if (!r.width) return;
    if (cv.width !== Math.round(r.width * dpr) || cv.height !== Math.round(r.height * dpr)) { cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr); }
    const c = cv.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, r.width, r.height);
    if (!d || !this.boxes || !v.videoWidth) return;
    const k = Math.min(r.width / v.videoWidth, r.height / v.videoHeight), W = v.videoWidth * k, H = v.videoHeight * k, ox = (r.width - W) / 2, oy = (r.height - H) / 2;
    const now = v.currentTime;
    c.font = "600 11px ui-monospace, Menlo, monospace"; c.textBaseline = "middle";
    for (const t of d.tracks) {
      const p = t.path;
      if (!p.length || now < p[0][0] - 0.3 || now > p[p.length - 1][0] + 0.3) continue;
      let i = p.findIndex((q) => q[0] >= now);
      if (i < 0) i = p.length - 1;
      const a = p[Math.max(i - 1, 0)], b = p[i], f = b[0] > a[0] ? Math.min(Math.max((now - a[0]) / (b[0] - a[0]), 0), 1) : 0;
      const [x, y, w, h] = [1, 2, 3, 4].map((j) => a[j] + (b[j] - a[j]) * f);
      const col = TYPE_COLOUR[t.cls] || TYPE_COLOUR.vehicle, sel = t.id === this.selected;
      c.strokeStyle = sel ? "#fff" : col; c.lineWidth = sel ? 2.5 : 1.5;
      c.strokeRect(ox + (x - w / 2) * W, oy + (y - h / 2) * H, w * W, h * H);
      const tag = t.plate || `${t.cls} #${t.identity_id ?? t.track_id}`, tw = c.measureText(tag).width + 10;
      const lx = ox + (x - w / 2) * W, ly = Math.max(oy + (y - h / 2) * H - 16, oy);
      c.fillStyle = t.plate ? "#ffd23f" : col; c.fillRect(lx, ly, tw, 16);
      c.fillStyle = "#0b0b0b"; c.fillText(tag, lx + 5, ly + 8.5);
    }
  }

  // ---- node view: one object and what connects to it -----------------------------------
  async openGraph(sid) {
    let g;
    try { g = await api(`/api/sightings/${sid}/graph`); } catch (e) { return toast(esc(e.message), { err: true }); }
    this.selected = sid;
    const s = g.sighting, el = $("#rv-graph"), m = g.motion;
    const sw = s.colour && SWATCH[s.colour] ? `<i class="sw" style="background:${SWATCH[s.colour]}"></i>${cap(s.colour)}` : "";
    const hero = g.frames.length ? g.frames.reduce((a, b) => (b.w * b.h > a.w * a.h ? b : a)) : null;
    const pts = g.path.map(([, x, y]) => `${(x * 160).toFixed(1)},${(y * 90).toFixed(1)}`).join(" ");
    const first = g.path[0], last = g.path[g.path.length - 1];
    el.innerHTML = `
      <header><h3>Connections</h3><span class="hint">Lines show what this object is linked to. Click a frame to jump there.</span>
        <button class="ghost" data-close data-label="Close" data-tip="Back to the timeline.">✕</button></header>
      <div class="graph">
        <svg class="links" aria-hidden="true"></svg>
        <div class="col col-frames"><div class="col-title">Seen in ${plural(g.frames.length, "frame")}</div>
          ${g.frames.map((f) => `<button class="node frame" data-node data-ts="${f.ts}"><img src="/api/obs/${f.id}.jpg" alt=""><span><b>${clock(f.ts)}</b>${Math.round(f.x * 100)}% across · ${Math.round(f.w * 100)}% of width</span></button>`).join("") || `<div class="node" data-node><p class="hint">No preview frames were kept.</p></div>`}</div>
        <div class="col col-centre"><div class="node object" data-centre>
          ${hero ? `<img src="/api/obs/${hero.id}.jpg" alt="">` : ""}
          <div class="obj-line">${plateBadge(s.plate)}<span class="kind">${esc(s.cls || "vehicle")}</span>${sw ? `<span class="colour">${sw}</span>` : ""}</div>
          <dl><dt>In view</dt><dd>${clock(s.t0)} to ${clock(s.t1)} (${fmtDur(Math.max(s.t1 - s.t0, 0))})</dd>
            ${g.identity ? `<dt>Vehicle</dt><dd>#${g.identity.id}${s.by && s.by !== "new" ? ` · matched by ${esc(s.by)} (${Math.round((s.score || 0) * 100)}%)` : ""}</dd>` : ""}
            ${s.plate ? `<dt>Plate</dt><dd>${Math.round((s.plate_conf || 0) * 100)}% sure</dd>` : ""}</dl></div></div>
        <div class="col col-facts"><div class="col-title">What we know</div>
          <div class="node motion" data-node><h4>Motion</h4>
            <div class="motion-row">${m.angle != null ? `<svg class="arrow" viewBox="0 0 40 40" style="transform:rotate(${m.angle}deg)"><path d="M6 20h26M24 12l9 8-9 8" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>` : `<span class="still">●</span>`}
              <div><b>${esc(m.summary)}</b><span>${m.duration_s ? `over ${fmtDur(m.duration_s)}` : ""}</span></div></div>
            ${g.path.length > 1 ? `<svg class="pathplot" viewBox="0 0 160 90" data-label="Path through the picture" data-tip="Start is the hollow dot, end is the filled one."><rect width="160" height="90" rx="3"/><polyline points="${pts}"/>
              <circle class="a" cx="${first[1] * 160}" cy="${first[2] * 90}" r="3.5"/><circle class="b" cx="${last[1] * 160}" cy="${last[2] * 90}" r="3.5"/></svg>` : ""}</div>
          <div class="node mapnode" data-node><h4>On the map</h4>${g.ground ? `<div id="rv-map" class="rv-map"></div><p class="hint">Direction is from the picture; distance is a rough guess from how low the vehicle sits in the frame.</p>`
            : `<p class="hint">This recording has no map position, so movement cannot be plotted.</p><button data-place>Place it on the map</button>`}</div>
          ${g.others.length ? `<div class="node" data-node><h4>Same vehicle seen again</h4>${g.others.map((o) => `<button class="other" ${o.kind === "file" ? `data-open="${o.source_id}"` : "disabled"}>
            <b>${esc(o.source)}</b><span>${fmtTime(o.t_first)}${o.by && o.by !== "new" ? ` · by ${esc(o.by)} ${Math.round((o.score || 0) * 100)}%` : ""}</span></button>`).join("")}</div>` : ""}</div>
      </div>`;
    el.hidden = false;
    $("#view-review").classList.add("graph-open");
    this.seek(s.t0);
    if (g.ground) this.drawGraphMap(g);
    requestAnimationFrame(() => this.drawLinks());
    el.querySelectorAll("img").forEach((img) => img.addEventListener("load", () => this.drawLinks(), { once: true }));
  }

  drawGraphMap(g) {
    const cam = g.camera, map = (this.graphMap = L.map("rv-map", { zoomControl: false, attributionControl: false }));
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19 }).addTo(map);
    L.polygon(cone(cam), { color: "#8fd3ff", weight: 1, fillOpacity: 0.12, interactive: false }).addTo(map);
    L.circleMarker([cam.lat, cam.lon], { radius: 5, color: "#4cc2ff", weight: 2, fillOpacity: 1 }).bindTooltip(esc(cam.name || "Camera")).addTo(map);
    L.polyline(g.ground, { color: "#ffd23f", weight: 3 }).addTo(map);
    L.circleMarker(g.ground[0], { radius: 4, color: "#ffd23f", weight: 2, fill: false }).addTo(map);
    L.circleMarker(g.ground[g.ground.length - 1], { radius: 5, color: "#ffd23f", weight: 2, fillColor: "#ffd23f", fillOpacity: 1 }).addTo(map);
    setTimeout(() => { map.invalidateSize(); map.fitBounds(L.latLngBounds([[cam.lat, cam.lon], ...g.ground]).pad(0.35), { maxZoom: 19 }); }, 60);
  }

  // curved connectors from the object card to every other card
  drawLinks() {
    const host = document.querySelector("#rv-graph .graph");
    if (!host || $("#rv-graph").hidden) return;
    const svg = host.querySelector(".links"), hr = host.getBoundingClientRect(), centre = host.querySelector("[data-centre]").getBoundingClientRect();
    svg.setAttribute("viewBox", `0 0 ${hr.width} ${host.scrollHeight}`);
    svg.style.height = `${host.scrollHeight}px`;
    const top = hr.top - host.scrollTop;
    let d = "";
    for (const n of host.querySelectorAll("[data-node]")) {
      const r = n.getBoundingClientRect(), left = r.right < centre.left;
      const x1 = (left ? centre.left : centre.right) - hr.left, y1 = centre.top + centre.height / 2 - top;
      const x2 = (left ? r.right : r.left) - hr.left, y2 = r.top + r.height / 2 - top, mx = (x1 + x2) / 2;
      d += `<path d="M${x1} ${y1} C${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}"/><circle cx="${x2}" cy="${y2}" r="3"/>`;
    }
    svg.innerHTML = d;
  }

  graphClick(e) {
    if (e.target.closest("[data-close]")) return this.closeGraph();
    if (e.target.closest("[data-place]")) return this.onPlace(this.id);
    const o = e.target.closest("[data-open]");
    if (o) return this.open(+o.dataset.open);
    const f = e.target.closest("[data-ts]");
    if (f) { this.seek(+f.dataset.ts); this.video.pause(); }
  }
  closeGraph() {
    try { this.graphMap?.remove(); } catch {}
    this.graphMap = null; this.selected = null;
    $("#rv-graph").hidden = true; $("#rv-graph").innerHTML = "";
    $("#view-review").classList.remove("graph-open");
  }
}

// Greedy row packing so overlapping objects sit on separate rows. `pad` keeps room for a preview image.
function pack(tracks, pad = 0) {
  const ends = [], row = new Map();
  for (const t of [...tracks].sort((a, b) => a.t0 - b.t0)) {
    let i = ends.findIndex((e) => e <= t.t0);
    if (i < 0) { i = ends.length; ends.push(0); }
    ends[i] = Math.max(t.t1, t.t0 + pad) + pad * 0.2;
    row.set(t.id, i);
  }
  return { row, n: Math.max(ends.length, 1) };
}
