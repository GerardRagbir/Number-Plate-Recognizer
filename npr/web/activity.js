// Activity drawer: one search bar over one feed.
//
//   [mode ▾] [chip][chip] free text…                [filter]
//
// • mode    which kinds of result to show (plates, vehicles, patterns); none ticked = all
// • chips   keywords the search understands (a colour, a vehicle type). They come from the facet list
//           the server publishes (/api/search/facets), so new categories appear here without changes.
//           Click a chip to swap it for another value of the same facet.
// • text    whatever is left is matched against the plate
// • filter  newest/oldest first, and a time range
import { $, api, esc, fmtTime, fmtDur, state, sourceName, selectIdentity, toast, thumb, plateBadge } from "./app.js";
import { icon } from "./icons.js";

const MODES = [
  ["plates", "Plates", "plates", "Every plate that was read."],
  ["vehicles", "Vehicles", "car", "Each distinct vehicle, with its route."],
  ["patterns", "Patterns", "flows", "Flows, routes and repeat visitors."],
];
const RANGES = [
  ["any", "Any time"], ["15m", "Last 15 min", 900], ["1h", "Last hour", 3600], ["today", "Today"],
  ["24h", "Last 24 hours", 86400], ["7d", "Last 7 days", 604800], ["custom", "Custom range"],
];
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const localInput = (t) => { const d = new Date(t - new Date().getTimezoneOffset() * 60000); return d.toISOString().slice(0, 16); };

export class Activity {
  constructor() {
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem("npr.activity")) || {}; } catch {}
    this.modes = new Set((saved.modes || []).filter((m) => MODES.some((x) => x[0] === m)));   // empty = all
    this.order = saved.order === "asc" ? "asc" : "desc";
    this.range = RANGES.some((r) => r[0] === saved.range) ? saved.range : "any";
    this.from = saved.from || null; this.to = saved.to || null;   // custom range, ms
    this.chips = [];          // [{ facet, value }]
    this.text = "";
    this.facets = [];
    this.words = new Map();   // typed word → { facet, value }
    this.data = { items: [], counts: { plates: 0, vehicles: 0 }, totals: { plates: 0, vehicles: 0 } };
    this.patterns = null;
    this.stale = true;
    this.pop = null;          // which popover is open: "mode" | "filter" | chip index

    this.input = $("#search");
    this.input.addEventListener("input", () => this.typed(false));
    this.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === ",") { e.preventDefault(); this.typed(true); }
      else if (e.key === "Backspace" && !this.input.value && this.chips.length) { this.chips.pop(); this.changed(); }
      else if (e.key === "Escape") { this.closePop(); }
      e.stopPropagation();
    });
    $("#sb-field").addEventListener("click", (e) => { if (e.target.id === "sb-field" || e.target.id === "sb-chips") this.input.focus(); });
    $("#act-mode").onclick = () => this.togglePop("mode");
    $("#act-filter").onclick = () => this.togglePop("filter");
    $("#sb-chips").addEventListener("click", (e) => {
      const chip = e.target.closest("[data-chip]");
      if (!chip) return;
      const i = +chip.dataset.chip;
      if (e.target.closest(".x")) { this.chips.splice(i, 1); this.closePop(); return this.changed(); }
      this.togglePop(i);
    });
    $("#act-pop").addEventListener("click", (e) => this.popClick(e));
    $("#act-pop").addEventListener("change", (e) => this.popChange(e));
    document.addEventListener("mousedown", (e) => { if (this.pop != null && !e.target.closest("#act-pop, #searchbar")) this.closePop(); });
    $("#act-list").addEventListener("click", (e) => this.listClick(e));
    this.drawBar();
    api("/api/search/facets").then((f) => {
      this.facets = f;
      for (const facet of f) for (const v of facet.values) for (const w of v.words) this.words.set(w.toLowerCase(), { facet: facet.key, value: v.id });
      this.drawBar();
    }).catch(() => {});
  }

  // ---- state helpers ---------------------------------------------------------------
  has(mode) { return this.modes.size === 0 || this.modes.has(mode); }
  setModes(list) { this.modes = new Set(list.filter((m) => MODES.some((x) => x[0] === m))); if (this.modes.size === MODES.length) this.modes.clear(); this.changed(); }
  focus() { this.input.focus(); }
  facet(key) { return this.facets.find((f) => f.key === key); }
  value(chip) { return this.facet(chip.facet)?.values.find((v) => v.id === chip.value); }
  window() {
    const now = Date.now() / 1000, r = RANGES.find((x) => x[0] === this.range);
    if (this.range === "today") { const d = new Date(); d.setHours(0, 0, 0, 0); return [d.getTime() / 1000, null]; }
    if (this.range === "custom") return [this.from ? this.from / 1000 : null, this.to ? this.to / 1000 : null];
    return [r?.[2] ? now - r[2] : null, null];
  }
  save() { try { localStorage.setItem("npr.activity", JSON.stringify({ modes: [...this.modes], order: this.order, range: this.range, from: this.from, to: this.to })); } catch {} }

  // turn recognised words into chips; `all` also converts the word still being typed
  typed(all) {
    const raw = this.input.value, parts = raw.split(/([\s,]+)/), keep = [];
    let made = false;
    for (let i = 0; i < parts.length; i += 2) {
      const word = parts[i], finished = all || i + 1 < parts.length;
      const hit = finished && this.words.get(word.toLowerCase());
      if (hit && word) {
        if (!this.chips.some((c) => c.facet === hit.facet && c.value === hit.value)) this.chips.push({ ...hit });
        made = true;
      } else keep.push(word + (parts[i + 1] ? " " : ""));
    }
    if (made) this.input.value = keep.join("").replace(/^\s+/, "");
    this.text = this.input.value.trim();
    made ? this.changed() : this.loadSoon(250);
  }
  changed() { this.save(); this.drawBar(); this.loadSoon(0); }

  // ---- the bar ------------------------------------------------------------------------
  drawBar() {
    const sel = MODES.filter((m) => this.modes.has(m[0]));
    const mode = $("#act-mode");
    mode.innerHTML = sel.length ? sel.map((m) => icon(m[2])).join("") : icon("all");
    mode.dataset.label = sel.length ? sel.map((m) => m[1]).join(" + ") : "Everything";
    mode.dataset.tip = "Choose what to show. Pick several, or none for everything.";
    mode.classList.toggle("on", sel.length > 0);
    $("#sb-chips").innerHTML = this.chips.map((c, i) => {
      const v = this.value(c);
      return `<button type="button" class="tok" data-chip="${i}" data-label="${esc(this.facet(c.facet)?.label || c.facet)}" data-tip="Click to change it. × removes it.">
        ${v?.swatch ? `<i class="sw" style="background:${v.swatch}"></i>` : ""}<span>${esc(v?.label || c.value)}</span><span class="x" aria-label="Remove">×</span></button>`;
    }).join("");
    this.input.placeholder = this.chips.length ? "" : "Plate, colour, type…";
    const filt = $("#act-filter"), active = this.order !== "desc" || this.range !== "any";
    filt.innerHTML = icon("filter") + (active ? '<i class="badge-dot"></i>' : "");
    filt.dataset.label = "Sort and time";
    filt.dataset.tip = active ? `${this.order === "asc" ? "Oldest first" : "Newest first"} · ${RANGES.find((r) => r[0] === this.range)[1]}` : "Order, and limit to a time range.";
    filt.classList.toggle("on", active);
  }

  // ---- popovers ------------------------------------------------------------------------
  togglePop(which) { this.pop === which ? this.closePop() : this.openPop(which); }
  closePop() { this.pop = null; $("#act-pop").hidden = true; }
  openPop(which) {
    this.pop = which;
    const el = $("#act-pop");
    if (which === "mode") {
      el.innerHTML = `<div class="pop-title">Show</div>
        <button class="opt ${this.modes.size ? "" : "on"}" data-mode="">${icon("all")}<span><b>Everything</b><small>All categories together.</small></span><i class="tick">✓</i></button>
        ${MODES.map(([k, label, ic, help]) => `<button class="opt ${this.modes.has(k) ? "on" : ""}" data-mode="${k}">${icon(ic)}<span><b>${label}</b><small>${help}</small></span><i class="tick">✓</i></button>`).join("")}`;
      el.dataset.side = "left";
    } else if (which === "filter") {
      const [from, to] = [this.from, this.to];
      el.innerHTML = `<div class="pop-title">Order</div>
        <div class="seg"><button data-order="desc" class="${this.order === "desc" ? "on" : ""}">Newest first</button><button data-order="asc" class="${this.order === "asc" ? "on" : ""}">Oldest first</button></div>
        <div class="pop-title">Time</div>
        <div class="presets">${RANGES.map(([k, label]) => `<button data-range="${k}" class="${this.range === k ? "on" : ""}">${label}</button>`).join("")}</div>
        <div class="custom" ${this.range === "custom" ? "" : "hidden"}>
          <label>From<input type="datetime-local" name="from" value="${from ? localInput(from) : ""}"></label>
          <label>To<input type="datetime-local" name="to" value="${to ? localInput(to) : ""}"></label></div>
        <button class="reset" data-reset>Reset</button>`;
      el.dataset.side = "right";
    } else {
      const chip = this.chips[which], facet = this.facet(chip.facet);
      el.innerHTML = `<div class="pop-title">${esc(facet.label)}</div>
        <div class="values ${facet.values.some((v) => v.swatch) ? "swatches" : ""}">${facet.values.map((v) =>
          `<button data-value="${esc(v.id)}" class="${v.id === chip.value ? "on" : ""}" data-label="${esc(v.label)}" data-tip="${esc(`Also matches: ${v.words.slice(0, 6).join(", ")}.`)}">
            ${v.swatch ? `<i class="sw" style="background:${v.swatch}"></i>` : ""}<span>${esc(v.label)}</span></button>`).join("")}</div>
        <button class="reset" data-remove>Remove this filter</button>`;
      el.dataset.side = "left";
    }
    el.hidden = false;
  }
  popClick(e) {
    const b = e.target.closest("button");
    if (!b) return;
    const d = b.dataset;
    if ("mode" in d) {
      if (!d.mode) this.modes.clear();
      else { this.modes.has(d.mode) ? this.modes.delete(d.mode) : this.modes.add(d.mode); if (this.modes.size === MODES.length) this.modes.clear(); }
      this.changed(); this.openPop("mode");   // stays open so several can be ticked
    } else if (d.order) { this.order = d.order; this.changed(); this.openPop("filter"); }
    else if (d.range) { this.range = d.range; this.changed(); this.openPop("filter"); }
    else if ("reset" in d) { this.order = "desc"; this.range = "any"; this.from = this.to = null; this.changed(); this.openPop("filter"); }
    else if (d.value) { this.chips[this.pop].value = d.value; this.closePop(); this.changed(); }
    else if ("remove" in d) { this.chips.splice(this.pop, 1); this.closePop(); this.changed(); }
  }
  popChange(e) {
    if (e.target.name !== "from" && e.target.name !== "to") return;
    this[e.target.name] = e.target.value ? new Date(e.target.value).getTime() : null;
    this.range = "custom";
    this.save(); this.drawBar(); this.loadSoon(0);
  }

  // ---- loading --------------------------------------------------------------------------
  loadSoon(ms = 800) {
    if (document.body.classList.contains("no-side")) { this.stale = true; return; }   // closed: load when it opens
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.load(), ms);
  }
  opened() { if (this.stale) this.load(); }
  async load() {
    this.stale = false;
    const [since, until] = this.window();
    const kinds = ["plates", "vehicles"].filter((k) => this.has(k));
    const p = new URLSearchParams({ kinds: kinds.join(","), order: this.order, limit: "200" });
    if (this.text) p.set("q", this.text);
    for (const f of this.facets) {
      const vals = this.chips.filter((c) => c.facet === f.key).map((c) => c.value);
      if (vals.length) p.set(f.param, vals.join(","));
    }
    if (since != null) p.set("since", since);
    if (until != null) p.set("until", until);
    const seq = (this.seq = (this.seq || 0) + 1);
    try {
      const [data, patterns] = await Promise.all([
        api(`/api/activity?${p}`),
        this.has("patterns") ? api(`/api/patterns${since != null ? `?since=${since}` : ""}`) : null,
      ]);
      if (seq !== this.seq) return;   // a newer search overtook this one
      this.data = data; this.patterns = patterns;
      this.render();
    } catch (e) { toast(esc(e.message), { err: true }); }
  }

  // ---- results ----------------------------------------------------------------------------
  render() {
    const d = this.data, filtered = this.text || this.chips.length || this.range !== "any";
    const bits = [];
    if (this.has("plates")) bits.push(`${d.counts.plates} plate${d.counts.plates === 1 ? "" : "s"}`);
    if (this.has("vehicles")) bits.push(`${d.counts.vehicles} vehicle${d.counts.vehicles === 1 ? "" : "s"}`);
    $("#act-summary").innerHTML = `${bits.join(" · ") || "Patterns"}${filtered ? ` <span>matching</span>` : ""}${
      d.items.length < d.counts.plates + d.counts.vehicles ? ` <span>· showing ${d.items.length}</span>` : ""}`;

    const sw = (c) => { const v = this.facet("colour")?.values.find((x) => x.id === c); return v ? `<i class="sw" style="background:${v.swatch}" data-label="${esc(v.label)}" data-tip="Colour family, estimated from the picture."></i>` : ""; };
    const item = (r) => r.kind === "plate" ? `
      <div class="item ${r.identity_id === state.identity ? "sel" : ""}" data-identity="${r.identity_id ?? ""}">
        ${thumb(r.sighting_id)}
        <div class="l1"><span class="kind-ico" data-label="Plate read">${icon("plates")}</span>${plateBadge(r.plate)}${sw(r.colour)}<span class="kind">${esc(r.cls || "")}</span></div>
        <div class="when">${fmtTime(r.t)}</div>
        <div class="sub">${esc(sourceName(r.source_id))}${r.identity_id ? ` · #${r.identity_id}` : ""}</div>
        <div class="when">${Math.round((r.conf || 0) * 100)}%</div>
      </div>` : `
      <div class="item ${r.id === state.identity ? "sel" : ""}" data-identity="${r.id}">
        ${thumb(r.sighting_id)}
        <div class="l1"><span class="kind-ico" data-label="Vehicle">${icon("car")}</span>${plateBadge(r.plate)}${sw(r.colour)}<span class="kind">${esc(r.cls || "")}</span>${r.cameras > 1 ? `<span class="xcam">⇄ ${r.cameras}</span>` : ""}</div>
        <div class="when">${fmtTime(r.t)}</div>
        <div class="sub">#${r.id} · ${r.sightings} sighting${r.sightings === 1 ? "" : "s"}</div><div></div>
      </div>`;

    let html = this.detail();
    if (this.has("plates") || this.has("vehicles")) {
      html += d.items.length ? d.items.map(item).join("")
        : `<div class="none">${filtered ? "Nothing matches. Remove a chip or widen the time range." : "Plates and vehicles appear here as they are seen."}</div>`;
    }
    if (this.has("patterns")) html += this.patternsHTML();
    $("#act-list").innerHTML = html;
  }

  detail() {
    const d = state.identityDetail;
    if (!d) return "";
    const stops = d.route.map((v, i) => {
      const prev = d.route[i - 1];
      const leg = prev ? `<div class="leg">${fmtDur(v.t_first - prev.t_last)} later${v.match_kind && v.match_kind !== "new" ? ` · linked by ${v.match_kind} (${Math.round(v.match_score * 100)}%)` : ""}</div>` : "";
      return `${leg}<div class="stop">${thumb(v.sightings[0])}<div><strong>${esc(sourceName(v.source_id))}</strong>
        <div class="sub" style="color:var(--dim);font-size:11px">${v.sightings.length > 1 ? v.sightings.length + " passes · " : ""}in view ${fmtDur(v.t_last - v.t_first)}</div></div>
        <div class="t">${fmtTime(v.t_first)}</div></div>`;
    }).join("");
    return `<div class="detail"><header>${plateBadge(d.plate)}<span class="kind">${esc(d.cls || "vehicle")} · #${d.id}</span>
      <button class="ghost x" data-clear data-label="Clear selection" data-tip="Stop following this vehicle.">✕</button></header><div class="route">${stops}</div></div>`;
  }

  patternsHTML() {
    const p = this.patterns;
    if (!p || !p.identities) return `<div class="sect"><h4>Patterns</h4><p class="hint">Patterns build up as vehicles are seen. Place cameras on the map to get distances and speeds.</p></div>`;
    const flows = p.flows.length ? `<table><tr><th>From → to</th><th class="n">Vehicles</th><th class="n">Median</th><th class="n">km/h</th></tr>${
      p.flows.map((f) => `<tr><td>${esc(sourceName(f.from))}<span class="arrow">→</span>${esc(sourceName(f.to))}</td><td class="n">${f.count}</td><td class="n">${fmtDur(f.median_s)}</td><td class="n">${f.median_kmh ? f.median_kmh.toFixed(0) : "—"}</td></tr>`).join("")}</table>`
      : `<p class="hint">No vehicle has been linked across two cameras yet.</p>`;
    const routes = p.routes.length ? `<table><tr><th>Route</th><th class="n">Vehicles</th></tr>${
      p.routes.map((r) => `<tr><td>${r.cameras.map((c) => esc(sourceName(c))).join('<span class="arrow">→</span>')}</td><td class="n">${r.count}</td></tr>`).join("")}</table>` : "";
    const repeat = p.repeat.length ? `<table><tr><th>Vehicle</th><th class="n">Visits</th><th class="n">Cameras</th><th class="n">Last</th></tr>${
      p.repeat.slice(0, 15).map((r) => `<tr class="click" data-identity="${r.identity_id}"><td>#${r.identity_id}</td><td class="n">${r.visits}</td><td class="n">${r.cameras}</td><td class="n">${fmtTime(r.last)}</td></tr>`).join("")}</table>` : "";
    const cams = p.cameras.map((c) => {
      const max = Math.max(1, ...c.hours);
      return `<tr><td>${esc(sourceName(c.source_id))}</td><td class="n">${c.sightings}</td><td class="n">${fmtDur(c.median_dwell_s)}</td>
        <td style="width:38%"><div class="hours" title="Sightings by hour of day (00–23)">${c.hours.map((h) => `<i style="height:${(h / max) * 100}%"></i>`).join("")}</div></td></tr>`;
    }).join("");
    return `<div class="sect"><h4>Camera-to-camera flows</h4>${flows}</div>
      ${routes ? `<div class="sect"><h4>Common routes</h4>${routes}</div>` : ""}
      ${repeat ? `<div class="sect"><h4>Repeat vehicles</h4>${repeat}</div>` : ""}
      <div class="sect"><h4>Per camera</h4><table><tr><th>Camera</th><th class="n">Seen</th><th class="n">In view</th><th>By hour</th></tr>${cams}</table></div>
      <div class="sect"><button id="reset" class="ghost">Clear all analysis data</button></div>`;
  }

  async listClick(e) {
    if (e.target.closest("[data-clear]")) return selectIdentity(null);
    if (e.target.id === "reset") {
      if (e.target.dataset.armed) { await api("/api/reset", { method: "POST" }); return; }
      e.target.dataset.armed = "1"; e.target.textContent = "Click again to erase sightings, vehicles and plates";
      return;
    }
    const row = e.target.closest("[data-identity]");
    if (row?.dataset.identity) selectIdentity(+row.dataset.identity);
  }
}
