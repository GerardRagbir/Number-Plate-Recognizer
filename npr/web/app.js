// Iris — dashboard. No build step: plain ES modules + Leaflet.
import { MapView } from "./map.js";
import { Monitor, runTest } from "./monitor.js";
import { Settings } from "./settings.js";
import { Home } from "./home.js";
import { addCamera, generalSetup } from "./wizard.js";
import { initIcons, icon } from "./icons.js";
import { Activity } from "./activity.js";
import { Review } from "./review.js";

export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export async function api(path, opts = {}) {
  if (opts.json !== undefined) {
    opts = { ...opts, body: JSON.stringify(opts.json), headers: { "Content-Type": "application/json" } };
  }
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.json();
}

const pad = (n) => String(n).padStart(2, "0");
export function fmtTime(t) {
  const d = new Date(t * 1000), now = new Date();
  const hms = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  if (d.toDateString() === now.toDateString()) return hms;
  return `${d.toLocaleDateString(undefined, { month: "short", day: "numeric" })} ${hms}`;
}
export function fmtDur(s) {
  if (s == null) return "—";
  s = Math.round(s);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${pad(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${pad(Math.floor((s % 3600) / 60))}m`;
}
export const colour = (key, a = 1) => `hsl(${(key * 67) % 360} 85% 62% / ${a})`;

const saved = (() => { try { return JSON.parse(localStorage.getItem("npr.ui")) || {}; } catch { return {}; } })();
export const state = {
  sources: new Map(),
  groups: [],           // camera groups: [{ id, name, objects, motion }]
  view: saved.view || "wall",
  wall: { mode: "auto", per_page: 4, pages: [] },   // Settings → Layout (saved on the server)
  pages: [],            // what the wall is showing: [{ name, ids, layout? }]
  page: saved.page || 0,
  override: null,       // a one-off page, e.g. "show this group on the wall"
  layers: { masks: true, boxes: true, trails: true, plates: true, labels: true, ...(saved.layers || {}) },
  widths: saved.widths || {},   // dragged panel widths in px: { rail, side }
  selTile: 0,
  identity: null,       // selected global identity id
  identityDetail: null,
  tab: "plates",
  q: "",
};
function persist() {
  localStorage.setItem("npr.ui", JSON.stringify({
    page: state.page, layers: state.layers,
    drawers: { rail: !document.body.classList.contains("no-rail"), side: !document.body.classList.contains("no-side") },
    view: state.view, widths: state.widths,
  }));
}
export const sourceName = (id) => state.sources.get(id)?.name ?? `source ${id}`;

export function toast(html, { err = false, ms = 6000, onclick } = {}) {
  const el = document.createElement("div");
  el.className = "toast" + (err ? " err" : "");
  el.innerHTML = html;
  el.onclick = () => { onclick?.(); el.remove(); };
  $("#toasts").append(el);
  setTimeout(() => el.remove(), ms);
}

// ---------------------------------------------------------------- tiles
const WALL_SIZES = { 1: 1, 2: 2, 3: 3, 4: 4, 6: 6, focus: 6, 9: 9, 16: 16 };   // layout → tiles
const tiles = [];

class Tile {
  constructor(i) {
    this.i = i;
    this.sourceId = null;
    this.meta = null;
    this.bitmap = null;
    this.el = document.createElement("div");
    this.el.className = "tile";
    this.el.innerHTML = `<canvas></canvas>
      <div class="hud"><span class="num">${i + 1}</span><span class="tname"></span><span class="stat"></span></div>
      <div class="empty"></div><div class="bar2"><i style="width:0"></i></div>`;
    this.canvas = $("canvas", this.el);
    this.ctx = this.canvas.getContext("2d");
    this.label = $(".tname", this.el);
    this.stat = $(".stat", this.el);
    this.empty = $(".empty", this.el);
    this.bar = $(".bar2 i", this.el);
    this.el.addEventListener("mousedown", () => selectTile(i));
    this.el.addEventListener("dblclick", (e) => { if (e.target === this.canvas) toggleMax(i); });
    this.canvas.addEventListener("click", (e) => this.pick(e));
    new ResizeObserver(() => this.draw()).observe(this.el);
    this.refreshOptions();
  }

  refreshOptions() {
    this.label.textContent = state.sources.get(this.sourceId)?.name ?? "";
    this.updateEmpty();
  }

  updateEmpty() {
    const s = state.sources.get(this.sourceId);
    let msg = "";
    if (!s) msg = state.sources.size ? ""
      : this.i === 0 ? `<div>No cameras yet.<br><button data-settings>Add one in Settings</button></div>` : "";
    else if (!this.bitmap) {
      const st = s.status?.state;
      msg = st === "error" ? `⚠ ${esc(s.status.error)}` : st === "reconnecting" ? `<div>Reconnecting…<br><small>${esc(s.status.error)}</small></div>`
        : st === "idle" || st === "stopped" ? "Stopped — press ▶ in the source list"
        : st === "starting" ? "Starting…" : "Waiting for video…";
    }
    this.empty.innerHTML = msg;
  }

  setSource(id) {
    if (id === this.sourceId) return;
    this.close();
    this.sourceId = id;
    this.meta = null;
    this.bitmap?.close?.();
    this.bitmap = null;
    this.label.textContent = state.sources.get(id)?.name ?? "";
    this.stat.textContent = "";
    this.bar.style.width = "0";
    this.draw();
    this.updateEmpty();
    if (id != null) this.open();
    renderSources();
  }

  open() {
    const id = this.sourceId;
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/tile/${id}`);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    ws.onmessage = async (ev) => {
      const view = new DataView(ev.data);
      const n = view.getUint32(0);
      const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(ev.data, 4, n)));
      const bmp = await createImageBitmap(new Blob([new Uint8Array(ev.data, 4 + n)], { type: "image/jpeg" }));
      if (this.ws !== ws) return bmp.close();
      this.bitmap?.close?.();
      this.bitmap = bmp;
      this.meta = meta;
      this.empty.innerHTML = "";
      this.stat.innerHTML = meta.state === "ended" ? "ended"
        : meta.idle ? `<b>${meta.tracks.filter((t) => !t.lost).length}</b> veh · scene static, models paused`
        : `<b>${meta.tracks.filter((t) => !t.lost).length}</b> veh · ${meta.fps} fps · ${meta.infer_ms} ms`;
      this.bar.style.width = meta.progress != null ? `${meta.progress * 100}%` : "0";
      this.draw();
    };
    ws.onclose = () => {
      if (this.ws === ws && this.sourceId === id) setTimeout(() => this.ws === ws && this.open(), 1500);
    };
  }

  close() {
    const ws = this.ws;
    this.ws = null;
    ws?.close();
  }

  fit() {
    const cw = this.canvas.width, ch = this.canvas.height;
    const bw = this.bitmap.width, bh = this.bitmap.height;
    const k = Math.min(cw / bw, ch / bh);
    return { x: (cw - bw * k) / 2, y: (ch - bh * k) / 2, w: bw * k, h: bh * k };
  }

  draw() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(this.el.clientWidth * dpr)), h = Math.max(1, Math.round(this.el.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    const c = this.ctx;
    c.clearRect(0, 0, w, h);
    if (!this.bitmap || !this.meta) return;
    const f = this.fit();
    c.drawImage(this.bitmap, f.x, f.y, f.w, f.h);
    const s = f.w / this.meta.w;
    const L = state.layers;
    c.save();
    c.beginPath(); c.rect(f.x, f.y, f.w, f.h); c.clip();
    c.translate(f.x, f.y);
    c.lineJoin = "round";
    const fs = Math.max(10, Math.min(13, f.w / 70)) * dpr;
    c.font = `600 ${fs}px ui-monospace, Menlo, monospace`;
    c.textBaseline = "middle";

    for (const t of this.meta.tracks) {
      const key = t.gid ?? t.id + 1000;
      const sel = t.gid != null && t.gid === state.identity;
      const [x1, y1, x2, y2] = t.box.map((v) => v * s);
      if (L.masks && t.poly) {
        c.beginPath();
        t.poly.forEach(([x, y], i) => (i ? c.lineTo(x * s, y * s) : c.moveTo(x * s, y * s)));
        c.closePath();
        c.fillStyle = colour(key, sel ? 0.45 : 0.28);
        c.fill();
        c.strokeStyle = colour(key, 0.95);
        c.lineWidth = 1.25 * dpr;
        c.stroke();
      }
      if (L.trails && t.trail.length > 1) {
        c.beginPath();
        t.trail.forEach(([x, y], i) => (i ? c.lineTo(x * s, y * s) : c.moveTo(x * s, y * s)));
        c.strokeStyle = colour(key, 0.9);
        c.lineWidth = 2 * dpr;
        c.stroke();
        const [lx, ly] = t.trail.at(-1);
        c.beginPath(); c.arc(lx * s, ly * s, 3 * dpr, 0, 7); c.fillStyle = colour(key); c.fill();
      }
      if (L.boxes || sel) {
        c.setLineDash(t.lost ? [5 * dpr, 4 * dpr] : []);
        c.strokeStyle = sel ? "#fff" : colour(key, t.lost ? 0.5 : 0.95);
        c.lineWidth = (sel ? 2.5 : 1.5) * dpr;
        c.strokeRect(x1, y1, x2 - x1, y2 - y1);
        c.setLineDash([]);
      }
      if (L.plates && t.plate_box && !t.lost) {
        const [a, b, d, e] = t.plate_box.map((v) => v * s);
        c.strokeStyle = "#ffd23f"; c.lineWidth = 1.5 * dpr;
        c.strokeRect(a, b, d - a, e - b);
      }
      if (L.labels && !t.lost) {
        const tag = `${t.gid != null ? "#" + t.gid : "·" + t.id} ${t.cls}${t.match && t.match !== "new" ? " ⇄" : ""}`;
        const th = fs + 5 * dpr, pw = 5 * dpr;
        let tw = c.measureText(tag).width + pw * 2;
        let ly = y1 - th; if (ly < 0) ly = y1;
        c.fillStyle = colour(key, 0.92);
        c.fillRect(x1, ly, tw, th);
        c.fillStyle = "#06090c";
        c.fillText(tag, x1 + pw, ly + th / 2 + 0.5);
        if (L.plates && t.plate) {
          const pwid = c.measureText(t.plate).width + pw * 2;
          c.fillStyle = "#ffd23f";
          c.fillRect(x1 + tw, ly, pwid, th);
          c.fillStyle = "#111";
          c.fillText(t.plate, x1 + tw + pw, ly + th / 2 + 0.5);
        }
      }
    }
    c.restore();
  }

  pick(e) {
    if (!this.bitmap || !this.meta) return;
    const dpr = window.devicePixelRatio || 1;
    const r = this.canvas.getBoundingClientRect(), f = this.fit(), s = f.w / this.meta.w;
    const x = ((e.clientX - r.left) * dpr - f.x) / s, y = ((e.clientY - r.top) * dpr - f.y) / s;
    const hit = this.meta.tracks.filter((t) => t.gid != null && x >= t.box[0] && x <= t.box[2] && y >= t.box[1] && y <= t.box[3])
      .sort((a, b) => (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]) - (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]))[0];
    if (hit) selectIdentity(hit.gid);
  }
}

function selectTile(i) {
  state.selTile = i;
  tiles.forEach((t) => t.el.classList.toggle("sel", t.i === i));
}

function toggleMax(i) {
  const grid = $("#grid");
  const on = !(grid.classList.contains("max") && tiles[i].el.classList.contains("maxed"));
  grid.classList.toggle("max", on);
  tiles.forEach((t) => t.el.classList.toggle("maxed", on && t.i === i));
  if (on) selectTile(i);
}

// ---- the wall lays itself out ----------------------------------------------------------
// Cameras are split into pages (Settings → Layout decides how); each page picks the grid that fits
// the cameras on it, so there is nothing to choose until there are more cameras than fit one page.
const autoLayout = (n) => (n <= 1 ? "1" : n <= 4 ? String(n) : n <= 6 ? "6" : n <= 9 ? "9" : "16");
function wallPages() {
  const w = state.wall, per = w.per_page || 4, all = [...state.sources.keys()];
  const chunk = (name, ids, single) => {
    const out = [];
    for (let i = 0; i < Math.max(ids.length, 1); i += per) out.push({ name: ids.length > per ? `${name} ${i / per + 1}`.trim() : single ?? name, ids: ids.slice(i, i + per) });
    return out;
  };
  if (w.mode === "custom" && w.pages.length) return w.pages.map((p) => ({ name: p.name, layout: p.size, ids: p.tiles.map((id) => (state.sources.has(id) ? id : null)) }));
  if (w.mode === "groups" && state.groups.length) {
    const known = new Set(state.groups.map((g) => g.id)), pages = [];
    for (const g of state.groups) {
      const ids = all.filter((id) => state.sources.get(id).config?.group === g.id);
      if (ids.length) pages.push(...chunk(g.name, ids));
    }
    const loose = all.filter((id) => !known.has(state.sources.get(id).config?.group));
    if (loose.length) pages.push(...chunk("Ungrouped", loose));
    if (pages.length) return pages;
  }
  return chunk("Page", all, "All cameras");
}
function renderWall() {
  const pages = wallPages();
  if (state.override) pages.push({ name: state.override.name, ids: state.override.ids.filter((id) => state.sources.has(id)), temp: true });
  state.pages = pages;
  state.page = state.override ? pages.length - 1 : Math.min(Math.max(state.page, 0), pages.length - 1);
  const pg = pages[state.page], layout = pg.layout || autoLayout(pg.ids.length), n = WALL_SIZES[layout];
  const grid = $("#grid"), key = `${state.page}|${layout}|${pg.ids.join(",")}`;
  if (grid.dataset.key !== key) {   // a different page or grid: leave maximised mode
    grid.dataset.key = key;
    grid.dataset.layout = `s${layout}`;
    grid.classList.remove("max");
    tiles.forEach((t) => t.el.classList.remove("maxed"));
  }
  while (tiles.length < n) { const t = new Tile(tiles.length); tiles.push(t); grid.append(t.el); }
  while (tiles.length > n) { const t = tiles.pop(); t.close(); t.el.remove(); }
  tiles.forEach((t, i) => { t.setSource(pg.ids[i] ?? null); t.refreshOptions(); });
  if (state.selTile >= n) state.selTile = 0;
  selectTile(state.selTile);
  drawPager();
}
function drawPager() {
  const el = $("#pager"), pages = state.pages;
  el.hidden = pages.length < 2;
  if (el.hidden) { el.innerHTML = ""; return; }
  const html = pages.length <= 6
    ? pages.map((p, i) => `<button role="tab" data-page="${i}" class="${i === state.page ? "on" : ""}" aria-selected="${i === state.page}" data-label="${esc(p.name)}" data-tip="${p.temp ? "A one-off view. Pick another page to leave it." : `${p.ids.filter((x) => x != null).length} camera${p.ids.filter((x) => x != null).length === 1 ? "" : "s"}. L or Page Down for the next page.`}">${esc(p.name)}${p.temp ? " ✕" : ""}</button>`).join("")
    : `<button data-page="${state.page - 1}" ${state.page ? "" : "disabled"} data-label="Previous page" aria-label="Previous page">‹</button>
       <button class="on" data-page="${state.page}" data-label="${esc(pages[state.page].name)}" data-tip="Page ${state.page + 1} of ${pages.length}.">${esc(pages[state.page].name)} · ${state.page + 1}/${pages.length}</button>
       <button data-page="${state.page + 1}" ${state.page < pages.length - 1 ? "" : "disabled"} data-label="Next page" aria-label="Next page">›</button>`;
  if (el.dataset.html !== html) { el.dataset.html = html; el.innerHTML = html; }
}
function setPage(i) {
  const leaving = state.override && state.pages[i]?.temp !== true;
  if (state.override && (leaving || i === state.pages.length - 1)) state.override = null;   // clicking the one-off page closes it
  state.page = Math.min(Math.max(i, 0), wallPages().length - 1);
  renderWall();
  persist();
}
// bring a camera into view: turn to the page that has it
function showSource(id) {
  let i = state.pages.findIndex((p) => p.ids.includes(id));
  if (i < 0) { state.override = null; i = wallPages().findIndex((p) => p.ids.includes(id)); }
  if (i < 0) return;
  if (!state.pages[i]?.temp) state.override = null;
  state.page = i;
  renderWall();
  const t = tiles.find((x) => x.sourceId === id);
  if (t) selectTile(t.i);
  persist();
}

// ---------------------------------------------------------------- sources rail
const KIND_LABEL = { file: "file", stream: "stream", screen: "screen", camera: "camera" };

function renderSources() {
  const box = $("#sources");
  if (box.querySelector("[contenteditable=true], [contenteditable=plaintext-only], [data-armed]")) return;   // someone is typing or confirming
  const tileOf = new Map(tiles.filter((t) => t.sourceId != null).map((t) => [t.sourceId, t.i + 1]));
  $("#source-count").textContent = $("#rail-count").textContent = state.sources.size || "";
  $("#sources-empty").hidden = state.sources.size > 0;
  const snapTick = Math.floor(Date.now() / SNAP_MS);   // the picture address changes every 3 minutes, so it is re-fetched then
  const row = (s) => {
    const st = s.status || {}, running = ["running", "starting", "reconnecting"].includes(st.state);
    const meta = st.state === "error" || st.state === "reconnecting" ? `<div class="meta ${st.state === "error" ? "err" : "warn"}">${st.state === "reconnecting" ? "reconnecting · " : ""}${esc(st.error)}</div>`
      : `<div class="meta">${st.state === "running" && st.idle ? "still, paused" : st.state || "idle"}${st.state === "running" ? ` · ${(st.fps || 0).toFixed(0)} fps · ${st.vehicles || 0} veh` : ""}</div>`;
    const prog = st.progress != null && running ? `<div class="prog"><i style="width:${(st.progress * 100).toFixed(1)}%"></i></div>` : "";
    const tile = tileOf.get(s.id);
    // trailing end: where it is on the wall, or (when it is not on the wall) its last picture
    const trail = tile ? `<span class="ontile" data-tip="Showing live in tile ${tile}.">Tile ${tile}</span>`
      : s.snapshot_t ? `<img class="snap" src="/api/sources/${s.id}/snapshot.jpg?t=${snapTick}" alt="" loading="lazy" data-label="Last picture" data-tip="Taken ${fmtTime(s.snapshot_t)}. Refreshes every 3 minutes.">`
      : `<span class="snap none" data-tip="No picture yet. One appears once the camera has delivered video.">—</span>`;
    return `<div class="src ${tile ? "shown" : ""}" data-id="${s.id}" data-state="${st.state || "idle"}" tabindex="0" draggable="true">
      <span class="avatar">${icon({ file: "film", screen: "screen" }[s.kind] || "cameras")}<i class="dot"></i></span>
      <div class="txt"><span class="name">${esc(s.name)}</span>${meta}</div>
      <span class="trail">${trail}</span>
      <span class="acts">
        <button data-act="${running ? "stop" : "start"}" data-icon="${running ? "stop" : "start"}" data-tip="${running ? "Stop analysing this camera." : "Start analysing this camera."}">${running ? "Stop" : "Start"}</button>
        <button data-act="locate" data-icon="pin" data-tip="${s.lat == null ? "Click the map to position it." : "Go to it on the map."}">${s.lat == null ? "Place on map" : "Show on map"}</button>
        <button data-act="config" data-icon="settings" data-tip="Location, type and object rules.">Settings</button>
        ${s.kind === "file" ? `<button data-act="review" data-icon="review" data-tip="Player, timelines and connections.">Review</button>` : ""}
        <button data-act="rename" data-icon="rename" data-tip="Change the name.">Rename</button>
        <button data-act="delete" data-icon="remove" data-tip="Remove it and its results. Asks first.">Remove</button>
      </span>${prog}</div>`;
  };
  const all = [...state.sources.values()], known = new Set(state.groups.map((g) => g.id));
  const section = (g, rows) => {
    const id = g ? g.id : "none", closed = collapsed.has(String(id));
    const live = rows.filter((s) => ["running", "starting"].includes(s.status?.state)).length;
    const rules = g ? [...Object.entries(g.objects).map(([t, v]) => `${t} ${v === "allow" ? "allowed" : "blocked"}`), g.motion !== "inherit" ? `skip still frames ${g.motion}` : ""].filter(Boolean).join(", ") : "";
    return `<section class="grp ${closed ? "closed" : ""}" data-group="${id}">
      <header class="grp-head"><button class="chev" data-gact="toggle" aria-expanded="${!closed}" aria-label="${closed ? "Expand" : "Collapse"} ${esc(g ? g.name : "Ungrouped")}">▾</button>
        <span class="gname" ${g ? `data-tip="${rules ? esc(`Group rules: ${rules}.`) : "No group rules: follows the general settings."}"` : ""}>${esc(g ? g.name : "Ungrouped")}</span>
        <span class="gcount">${rows.length}${live ? ` · ${live} live` : ""}</span>
        <span class="gacts">${rows.length ? `<button data-gact="wall" data-icon="live" data-tip="Fill the wall with these cameras.">Show on wall</button>` : ""}
          ${g ? `<button data-gact="config" data-icon="settings" data-tip="Name and rules for this group.">Group settings</button>` : ""}</span></header>
      <div class="grp-body">${rows.map(row).join("") || `<p class="grp-empty">Drag cameras here.</p>`}</div></section>`;
  };
  const html = state.groups.length
    ? state.groups.map((g) => section(g, all.filter((s) => s.config?.group === g.id))).join("") + section(null, all.filter((s) => !known.has(s.config?.group)))
    : all.map(row).join("");
  if (box.dataset.html !== html) { box.dataset.html = html; box.innerHTML = html; }
}

// ---- groups in the drawer: collapse, show on the wall, drag a camera onto a group ----
const SNAP_MS = 180000;
const collapsed = new Set((() => { try { return JSON.parse(localStorage.getItem("npr.collapsed")) || []; } catch { return []; } })());
setInterval(renderSources, SNAP_MS);
function showGroupOnWall(ids, name) {
  setView("wall");
  const i = state.wall.mode === "groups" ? wallPages().findIndex((p) => ids.length && ids.every((id) => p.ids.includes(id))) : -1;
  if (i >= 0) { state.override = null; state.page = i; }       // the group already has its own page
  else state.override = { name, ids: ids.slice(0, 16) };        // otherwise show it as a one-off page
  renderWall();
  persist();
}
async function groupAction(sec, act) {
  const gid = sec.dataset.group, ids = [...state.sources.values()].filter((s) => (gid === "none" ? !state.groups.some((g) => g.id === s.config?.group) : s.config?.group === +gid)).map((s) => s.id);
  if (act === "toggle") {
    collapsed.has(gid) ? collapsed.delete(gid) : collapsed.add(gid);
    localStorage.setItem("npr.collapsed", JSON.stringify([...collapsed]));
    renderSources();
  } else if (act === "wall") showGroupOnWall(ids, gid === "none" ? "Ungrouped" : state.groups.find((g) => g.id === +gid)?.name || "Group");
  else if (act === "config") { settings.selectGroup(+gid); setView("config"); }
}
$("#group-add").onclick = async () => {
  try {
    const g = await api("/api/groups", { method: "POST", json: { name: "New group" } });
    await refreshSources();
    const el = $(`.grp[data-group="${g.id}"] .gname`);   // name it straight away, in place
    if (!el) return;
    el.contentEditable = "plaintext-only";
    el.focus();
    document.getSelection().selectAllChildren(el);
    const done = async (keep) => {
      el.contentEditable = "false";
      const name = el.textContent.trim();
      if (keep && name && name !== g.name) await api(`/api/groups/${g.id}`, { method: "PATCH", json: { name } });
      refreshSources();
    };
    el.onblur = () => done(true);
    el.onkeydown = (k) => { if (k.key === "Enter") { k.preventDefault(); el.blur(); } if (k.key === "Escape") { el.onblur = null; done(false); } k.stopPropagation(); };
  } catch (e) { toast(esc(e.message), { err: true }); }
};
const railBox = $("#sources");
railBox.addEventListener("dragstart", (e) => {
  const r = e.target.closest?.(".src");
  if (!r) return;
  e.dataTransfer.setData("text/npr-source", r.dataset.id);
  e.dataTransfer.effectAllowed = "move";
});
railBox.addEventListener("dragover", (e) => {
  const sec = e.target.closest?.(".grp");
  if (!sec || !e.dataTransfer.types.includes("text/npr-source")) return;
  e.preventDefault();
  railBox.querySelectorAll(".grp.over").forEach((x) => x !== sec && x.classList.remove("over"));
  sec.classList.add("over");
});
railBox.addEventListener("dragleave", (e) => { if (!railBox.contains(e.relatedTarget)) railBox.querySelectorAll(".grp.over").forEach((x) => x.classList.remove("over")); });
railBox.addEventListener("drop", async (e) => {
  const sec = e.target.closest?.(".grp"), id = +e.dataTransfer.getData("text/npr-source");
  railBox.querySelectorAll(".grp.over").forEach((x) => x.classList.remove("over"));
  if (!sec || !id) return;
  e.preventDefault();
  const group = sec.dataset.group === "none" ? null : +sec.dataset.group;
  if ((state.sources.get(id)?.config?.group ?? null) === group) return;
  try { await api(`/api/sources/${id}`, { method: "PATCH", json: { config: { group } } }); await refreshSources(); }
  catch (err) { toast(esc(err.message), { err: true }); }
});

$("#sources").addEventListener("click", async (e) => {
  const gact = e.target.closest("[data-gact]")?.dataset.gact;
  if (gact) return groupAction(e.target.closest(".grp"), gact);
  const row = e.target.closest(".src");
  if (!row) return;
  const id = +row.dataset.id, act = e.target.closest("button")?.dataset.act, s = state.sources.get(id);
  try {
    if (!act) return pickSource(id);
    if (act === "start") { await api(`/api/sources/${id}/start`, { method: "POST" }); if (state.view === "wall") showSource(id); }
    else if (act === "stop") await api(`/api/sources/${id}/stop`, { method: "POST" });
    else if (act === "locate") locate(id);
    else if (act === "config") { settings.select(id); setView("config"); }
    else if (act === "review") { showDrawer("rail", false); return openReview(id); }
    else if (act === "rename") {
      const el = $(".name", row);
      el.contentEditable = "plaintext-only";
      el.focus();
      document.getSelection().selectAllChildren(el);
      const done = async (save) => {
        el.contentEditable = "false";
        const name = el.textContent.trim();
        if (save && name && name !== s.name) await api(`/api/sources/${id}`, { method: "PATCH", json: { name } });
        refreshSources();
      };
      el.onblur = () => done(true);
      el.onkeydown = (k) => { if (k.key === "Enter") { k.preventDefault(); el.blur(); } if (k.key === "Escape") { el.onblur = null; done(false); } k.stopPropagation(); };
      return;
    } else if (act === "delete") {
      if (!row.dataset.armed) {  // two-step confirm without a blocking dialog
        row.dataset.armed = "1";
        e.target.closest("button").textContent = "sure?";
        setTimeout(() => { delete row.dataset.armed; renderSources(); }, 2500);
        return;
      }
      await api(`/api/sources/${id}`, { method: "DELETE" });
    }
    refreshSources();
  } catch (err) { toast(esc(err.message), { err: true }); }
});
$("#sources").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.target.classList.contains("src")) pickSource(+e.target.dataset.id);
});

let refreshing = null;
export async function refreshSources() {
  if (refreshing) return refreshing;
  refreshing = (async () => {
    const [rows, groups] = await Promise.all([api("/api/sources"), api("/api/groups")]);
    state.groups = groups;
    const before = new Set(state.sources.keys());
    state.sources = new Map(rows.map((r) => [r.id, r]));
    if (booted) {
      renderWall();   // the wall re-fits itself to the cameras that exist
      const added = rows.find((r) => !before.has(r.id));
      if (added && before.size && state.view === "wall") showSource(added.id);   // turn to a newly added camera
    }
    if (!document.activeElement?.isContentEditable) renderSources();
    map.sync();
    settings.sourcesChanged();
    home.render();
  })().finally(() => { refreshing = null; });
  return refreshing;
}

// ---------------------------------------------------------------- activity (plates, vehicles, patterns)
// The drawer itself lives in activity.js; this is the glue the rest of the app uses.
export const thumb = (sid) => sid ? `<img loading="lazy" src="/api/thumbs/${sid}.jpg" alt="" onerror="this.style.visibility='hidden'">` : `<span class="ph"></span>`;
export const plateBadge = (p) => p ? `<span class="plate">${esc(p)}</span>` : `<span class="plate blank">no plate</span>`;

let patternData = null;   // unfiltered: feeds the flow lines on the map and the Guide's readiness checks
async function loadPatterns() {
  patternData = await api("/api/patterns");
  map.setFlows(patternData.flows);
}

export async function selectIdentity(id) {
  state.identity = id;
  state.identityDetail = null;
  if (id != null) {
    try { state.identityDetail = await api(`/api/identities/${id}`); } catch { state.identity = null; }
    if (state.identityDetail) showSide(true);
  }
  map.setRoute(state.identityDetail);
  activity.render();
  tiles.forEach((t) => t.draw());
}

// "plates" / "vehicles" / "patterns" narrows the drawer to that category; anything else shows everything
function setTab(tab) { activity.setModes(["plates", "vehicles", "patterns"].includes(tab) ? [tab] : []); }

// ---------------------------------------------------------------- events
let dirty = { vehicles: false, patterns: false };
function connectEvents() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/events`);
  ws.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    if (ev.type === "source") {
      const s = state.sources.get(ev.id);
      if (s) { s.status = ev.status; renderSources(); tiles.forEach((t) => t.sourceId === ev.id && t.updateEmpty()); map.sync(); }
      if (ev.status.state === "ended") dirty.patterns = dirty.vehicles = true;
    } else if (ev.type === "sources") refreshSources();
    else if (ev.type === "groups") refreshSources();
    else if (ev.type === "config") { applyUi(); if (state.view === "config" && !settings.dirty) settings.show(); }
    else if (ev.type === "plate") { activity.loadSoon(); dirty.vehicles = true; }
    else if (ev.type === "match") {
      dirty.vehicles = dirty.patterns = true;
      if (ev.cross_camera) {
        map.flash(ev.from_source, ev.to_source);
        toast(`${plateBadge(ev.plate)} <b>${esc(sourceName(ev.from_source))}</b> → <b>${esc(sourceName(ev.to_source))}</b> · ${fmtDur(Math.abs(ev.gap_s))} · by ${ev.kind}`,
          { onclick: () => selectIdentity(ev.identity) });
      }
    } else if (ev.type === "reset") { selectIdentity(null); activity.load(); loadPatterns(); }
  };
  ws.onclose = () => setTimeout(connectEvents, 1500);
}
setInterval(() => {
  if (dirty.vehicles || dirty.patterns) activity.loadSoon();
  if (dirty.patterns) loadPatterns();
  if (dirty.vehicles && state.identity != null) {
    api(`/api/identities/${state.identity}`).then((d) => { state.identityDetail = d; map.setRoute(d); activity.render(); }).catch(() => {});
  }
  dirty = { vehicles: false, patterns: false };
}, 3000);
setInterval(refreshSources, 2500);

// ---------------------------------------------------------------- top bar
$("#pager").addEventListener("click", (e) => { const b = e.target.closest("[data-page]"); if (b && !b.disabled) setPage(+b.dataset.page); });
$("#layers").addEventListener("click", (e) => {
  const k = e.target.dataset.layer;
  if (!k) return;
  state.layers[k] = !state.layers[k];
  e.target.classList.toggle("on", state.layers[k]);
  tiles.forEach((t) => t.draw());
  persist();
});
// The camera list (left) and Activity (right) are drawers that float over the feed or map and
// slide off-screen when closed, so the picture always has the full width.
function showDrawer(name, on) {
  document.body.classList.toggle(`no-${name}`, !on);
  const btn = $(`#toggle-${name}`);
  btn.classList.toggle("on", on);
  btn.setAttribute("aria-expanded", on);
  $(`#${name}`).inert = !on;   // a closed drawer is off-screen: keep it out of the tab order
  if (name === "side" && on) activity.opened();
  persist();
}
const drawerOpen = (name) => !document.body.classList.contains(`no-${name}`);
const showSide = (on) => showDrawer("side", on);
$("#toggle-side").onclick = () => showDrawer("side", !drawerOpen("side"));
$("#toggle-rail").onclick = () => showDrawer("rail", !drawerOpen("rail"));
document.addEventListener("click", (e) => { const n = e.target.closest("[data-close]")?.dataset.close; if (n) showDrawer(n, false); });

// ---------------------------------------------------------------- views
// Wall, Map, Network and Config each take over the whole centre of the screen.
const VIEWS = ["wall", "map", "network", "config", "home", "review"];
function setView(view) {
  if (!VIEWS.includes(view)) view = "wall";
  state.view = view;
  document.body.dataset.view = view;
  $("#grid").hidden = view !== "wall";
  for (const v of VIEWS) if (v !== "wall") $(`#view-${v}`).hidden = v !== view;
  $$("#views button").forEach((b) => b.classList.toggle("on", b.dataset.view === view));
  if (view === "map") map.show();
  if (view === "wall") tiles.forEach((t) => t.draw());
  if (view === "config") settings.show().catch((e) => toast(esc(e.message), { err: true }));
  if (view === "home") home.render();
  view === "review" ? review.shown() : review.hidden();
  monitor.setActive(view === "network");
  persist();
}
$("#views").addEventListener("click", (e) => e.target.dataset.view && setView(e.target.dataset.view));

function locate(id) {
  setView("map");
  state.sources.get(id)?.lat == null ? map.beginPlace(id) : map.select(id, true);
}
// clicking a source means "show me this one" in whichever view is open
function pickSource(id) {
  if (state.view === "wall") showSource(id);
  else if (state.view === "home") { setView("wall"); showSource(id); }
  else if (state.view === "map") locate(id);
  else if (state.view === "network") monitor.focus(id);
  else settings.select(id);
}
setInterval(() => { const d = new Date(); $("#clock").textContent = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`; }, 500);

document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, select, textarea, [contenteditable]") || e.metaKey || e.ctrlKey || e.altKey || $("#wiz").open) return;
  if (state.view === "wall" && /^[1-9]$/.test(e.key) && +e.key <= tiles.length) selectTile(+e.key - 1);
  else if (state.view === "wall" && (e.key === "f" || e.key === "Enter")) toggleMax(state.selTile);
  else if (e.key === "a" || e.key === "]") showDrawer("side", !drawerOpen("side"));
  else if (e.key === "[") showDrawer("rail", !drawerOpen("rail"));
  else if (e.key === "m") setView(state.view === "map" ? "wall" : "map");
  else if (e.key === "w") setView("wall");
  else if (e.key === "g") setView("home");
  else if (e.key === "n") setView("network");
  else if (e.key === "c") setView("config");
  else if (state.view === "wall" && (e.key === "l" || e.key === "PageDown")) setPage((state.page + 1) % state.pages.length);
  else if (state.view === "wall" && e.key === "PageUp") setPage((state.page - 1 + state.pages.length) % state.pages.length);
  else if (e.key === "Escape") { if (drawerOpen("rail") || drawerOpen("side")) { showDrawer("rail", false); showDrawer("side", false); return; } map.cancelPlace(); $("#grid").classList.remove("max"); tiles.forEach((t) => t.el.classList.remove("maxed")); }
});

// ---------------------------------------------------------------- guided flows & home actions
// What each wizard's closing buttons, and each Home button, actually does.
function afterAdd(go, rows) {
  const id = rows?.[0]?.id;
  if (go === "live") { setView("wall"); rows?.forEach((r) => showSource(r.id)); }
  else if (go === "map" && id != null) { setView("map"); map.select(id, true); }
  else if (go === "place" && id != null) locate(id);
  else if (go === "review" && id != null) openReview(id);
  home.render();
}
// the full-screen review of one recording; remembers where you came from
let reviewFrom = "wall";
function openReview(id) {
  if (state.view !== "review") reviewFrom = state.view;
  setView("review");
  review.open(id);
}
const startAdd = (kind = null) => addCamera({ kind, onDone: afterAdd }).catch((e) => toast(esc(e.message), { err: true }));
const startSetup = () => generalSetup({ onDone: (go) => { if (go === "settings") { settings.tab("general"); setView("config"); } else if (state.view === "config") settings.show(); home.render(); } })
  .catch((e) => toast(esc(e.message), { err: true }));

function facts() {
  const all = [...state.sources.values()], live = (s) => ["running", "starting"].includes(s.status?.state);
  let setupDone = false;
  try { setupDone = localStorage.getItem("npr.setup") === "1"; } catch {}
  return {
    total: all.length, cameras: all.filter((s) => s.kind !== "file").length, files: all.filter((s) => s.kind === "file").length,
    streams: all.filter((s) => s.kind === "stream").length, running: all.filter(live).length,
    camerasRunning: all.filter((s) => s.kind !== "file" && live(s)).length,
    down: all.filter((s) => s.kind === "stream" && ["reconnecting", "error"].includes(s.status?.state)).length,
    placed: all.filter((s) => s.lat != null).length, plates: activity.data.totals.plates,
    flows: patternData?.flows?.length || 0, linked: patternData?.routes?.reduce((a, r) => a + r.count, 0) || 0,
    setupDone, modelReady: systemInfo?.model_ready !== false,
  };
}
function homeAction(action) {
  const firstUnplaced = [...state.sources.values()].find((s) => s.lat == null);
  switch (action) {
    case "add": return startAdd();
    case "add-stream": return startAdd("stream");
    case "add-file": return startAdd("file");
    case "setup": return startSetup();
    case "place": return firstUnplaced ? locate(firstUnplaced.id) : setView("map");
    case "live": return setView("wall");
    case "gate": setView("wall"); showSide(true); return setTab("plates");
    case "results": setView("wall"); showSide(true); return setTab("vehicles");
    case "follow": setView("map"); showSide(true); setTab("vehicles"); return activity.focus();
    case "patterns": setView("map"); showSide(true); return setTab("patterns");
    case "health": return setView("network");
  }
}
$("#grid").addEventListener("click", (e) => { if (e.target.matches("[data-settings]")) setView("config"); });

// ---------------------------------------------------------------- resizable side panels
// Each panel has a floor it stays usable at and a ceiling that always leaves room for the video.
const PANELS = {
  rail: { el: "#split-rail", css: "--rail", min: 170, max: 420, sign: 1 },
  side: { el: "#split-side", css: "--side", min: 320, max: 760, sign: -1 },
};
function setPanelWidth(name, px) {
  const p = PANELS[name], other = name === "rail" ? "side" : "rail";
  const otherW = drawerOpen(other) ? $(other === "rail" ? ".rail" : "#side").getBoundingClientRect().width : 0;
  const ceiling = Math.max(p.min, Math.min(p.max, window.innerWidth - otherW - 120));   // drawers never meet or cover everything
  if (px == null) { document.documentElement.style.removeProperty(p.css); delete state.widths[name]; }
  else {
    const w = Math.round(Math.min(Math.max(px, p.min), ceiling));
    document.documentElement.style.setProperty(p.css, `${w}px`);
    state.widths[name] = w;
  }
  $(p.el).setAttribute("aria-valuenow", state.widths[name] ?? "");
  map.resize();
}
for (const [name, p] of Object.entries(PANELS)) {
  const bar = $(p.el), panel = () => $(name === "rail" ? ".rail" : "#side");
  bar.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    bar.setPointerCapture(e.pointerId);
    const x0 = e.clientX, w0 = panel().getBoundingClientRect().width;
    document.body.classList.add("resizing");
    const move = (m) => setPanelWidth(name, w0 + (m.clientX - x0) * p.sign);
    const up = () => { bar.removeEventListener("pointermove", move); document.body.classList.remove("resizing"); persist(); };
    bar.addEventListener("pointermove", move);
    bar.addEventListener("pointerup", up, { once: true });
    bar.addEventListener("pointercancel", up, { once: true });
  });
  bar.addEventListener("dblclick", () => { setPanelWidth(name, null); persist(); });
  bar.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault(); e.stopPropagation();
    setPanelWidth(name, panel().getBoundingClientRect().width + (e.key === "ArrowRight" ? 16 : -16) * p.sign);
    persist();
  });
}
window.addEventListener("resize", () => { for (const n of Object.keys(state.widths)) setPanelWidth(n, state.widths[n]); });
$("#config-add").onclick = () => startAdd();
$("#config-guided").onclick = () => startSetup();

// ---------------------------------------------------------------- boot
initIcons();

// ---------------------------------------------------------------- overlay toolbar (bottom centre)
// Always there in the live view; with auto-hide on (Settings → General → Interface) it stays off-screen
// until the pointer comes near the bottom edge, and slips away again shortly after it leaves.
const dock = $("#dock");
let dockTimer = null;
function dockPeek(ms = 1800) {
  dock.classList.add("show");
  clearTimeout(dockTimer);
  dockTimer = setTimeout(() => { if (!dock.matches(":hover, :focus-within")) dock.classList.remove("show"); else dockPeek(ms); }, ms);
}
document.querySelector(".center").addEventListener("mousemove", (e) => {
  if (!document.body.classList.contains("dock-autohide") || state.view !== "wall") return;
  const r = e.currentTarget.getBoundingClientRect();
  if (e.clientY > r.bottom - 110) dockPeek();
});
dock.addEventListener("focusin", () => dockPeek());
async function applyUi() {
  try {
    const cfg = (await api("/api/config")).config;
    document.body.classList.toggle("dock-autohide", cfg.ui?.toolbar_autohide !== false);
    const wall = cfg.ui?.wall || state.wall, changed = JSON.stringify(wall) !== JSON.stringify(state.wall);
    state.wall = wall;
    if (changed && booted) { state.override = null; renderWall(); }
  } catch {}
}
document.body.classList.add("dock-autohide");   // the default, until the saved setting arrives
applyUi().then(() => dockPeek(3500));           // show it once on arrival so people know it is there

let booted = false;
const map = new MapView({ refreshSources, onConfigure: (id) => { settings.select(id); setView("config"); } });
const monitor = new Monitor();
const review = new Review({ onClose: () => setView(reviewFrom === "review" ? "wall" : reviewFrom), onPlace: (id) => locate(id) });
const activity = new Activity();
let systemInfo = null;
const home = new Home({ facts, act: homeAction });
const settings = new Settings({
  onPlace: locate,
  onReview: (id) => openReview(id),
  onTest: (id) => { setView("network"); monitor.poll().then(() => monitor.focus(id, true)); },
});
(async function boot() {
  showDrawer("rail", saved.drawers?.rail === true);   // both drawers start closed until opened once
  showDrawer("side", saved.drawers?.side === true);
  for (const [n, w] of Object.entries(state.widths)) if (PANELS[n]) setPanelWidth(n, w);
  $$("#layers button").forEach((b) => b.classList.toggle("on", state.layers[b.dataset.layer]));
  await refreshSources().catch(() => {});
  await applyUi();
  renderWall();
  booted = true;
  renderSources();
  // deep links: #view=network, #tab=patterns, #vehicle=12
  const link = new URLSearchParams(location.hash.slice(1));
  const startView = link.get("view") || (["home", "review"].includes(state.view) ? "wall" : state.view);   // always land on a working view
  if (link.get("review") && state.sources.has(+link.get("review"))) openReview(+link.get("review")); else setView(startView === "review" ? "wall" : startView);
  connectEvents();
  activity.load().then(() => home.render()).catch(() => {});
  loadPatterns().then(() => home.render()).catch(() => {});
  if (link.get("tab")) setTab(link.get("tab"));
  if (link.get("vehicle")) selectIdentity(+link.get("vehicle"));
  try {
    const sys = systemInfo = await api("/api/system");
    home.render();
    const chip = $("#sys");
    if (!sys.model_ready) { chip.classList.add("bad"); chip.textContent = "vehicle model missing — run: uv run python -m npr setup"; chip.title = sys.error || ""; }
    else { chip.textContent = `${sys.metal_device} · ${sys.vehicle_model}`; chip.title = `Vehicles: ${sys.vehicle_model} (Core ML)\nPlates: ${sys.plate_model}\nOCR: ${sys.ocr}\nRe-ID: ${sys.embedding}`; }
  } catch {}
})();
