// Config view: global rules on the left, one camera's settings on the right.
// The same camera form is reused (blank) in the Add-source dialog.
import { $, api, esc, state, toast, refreshSources } from "./app.js";

const TYPE_LABEL = { fixed: "Fixed", ptz: "PTZ", dome: "Dome", fisheye: "Fisheye", doorbell: "Doorbell", dashcam: "Dashcam", traffic: "Traffic / ANPR", other: "Other" };
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const val = (v) => (v == null ? "" : esc(v));

let meta = null;   // { config, object_types, camera_types, camera_defaults }
export async function loadMeta() { meta = await api("/api/config"); return meta; }

// ---------------------------------------------------------------- shared camera form
export function cameraFormHTML(src, { blank = false } = {}) {
  const c = { ...meta.camera_defaults, ...(src?.config || {}) };
  const grp = state.groups.find((x) => x.id === c.group);
  // what this camera gets when it sets nothing itself: its group's rule if there is one, else the general setting
  const from = (rule) => (grp && rule ? grp.name : "General");
  const tri = (t) => {
    const cur = c.objects?.[t] || "inherit";
    const gr = grp?.objects?.[t];
    const g = gr ? (gr === "allow" ? "tracked" : "blocked") : meta.config.objects[t] ? "tracked" : "blocked";
    return `<div class="tri" role="radiogroup" aria-label="${cap(t)}"><span class="tri-name">${cap(t)}</span>
      ${[["inherit", `${esc(from(gr))} (${g})`], ["allow", "Allow"], ["block", "Block"]].map(([k, label]) =>
        `<label><input type="radio" name="obj-${t}" value="${k}" ${cur === k ? "checked" : ""}><span>${label}</span></label>`).join("")}</div>`;
  };
  return `
    ${blank ? "" : `<label class="f"><span>Name</span><input type="text" name="name" value="${val(src.name)}"></label>`}
    ${!blank && src.kind === "stream" ? `<label class="f"><span>Address</span><input type="text" name="uri" value="${val(src.uri)}" spellcheck="false" autocomplete="off">
      <small>The password is stored but never shown. Leave *** in place to keep it.</small></label>` : ""}
    <fieldset><legend>Location and view</legend>
      <div class="f2">
        <label class="f"><span>Latitude</span><input type="number" name="lat" step="any" min="-90" max="90" value="${val(src?.lat)}" placeholder="—"></label>
        <label class="f"><span>Longitude</span><input type="number" name="lon" step="any" min="-180" max="180" value="${val(src?.lon)}" placeholder="—"></label>
      </div>
      <div class="f3">
        <label class="f"><span>Heading °</span><input type="number" name="heading" min="0" max="359" step="1" value="${val(src?.heading)}" placeholder="0 = north"></label>
        <label class="f"><span>Lens FOV °</span><input type="number" name="hfov" min="5" max="180" step="1" value="${val(blank ? null : src?.hfov)}" placeholder="${meta.config.defaults.hfov}"></label>
        <label class="f"><span>Range m</span><input type="number" name="range_m" min="5" max="2000" step="1" value="${val(blank ? null : src?.range_m)}" placeholder="${meta.config.defaults.range_m}"></label>
      </div>
      <div class="f2">
        <label class="f"><span>Mount height m</span><input type="number" name="mount_height_m" min="0" max="200" step="0.1" value="${val(c.mount_height_m)}" placeholder="—"></label>
        <label class="f"><span>Downward tilt °</span><input type="number" name="tilt_deg" min="0" max="90" step="1" value="${val(c.tilt_deg)}" placeholder="—"></label>
      </div>
      ${blank ? "" : `<button type="button" data-act="place" data-icon="pin" class="keep" data-tip="Position and aim it on the map.">${src.lat == null ? "Place on map" : "Adjust on map"}</button>`}
    </fieldset>
    <fieldset><legend>Camera</legend>
      <div class="f2">
        <label class="f"><span>Type</span><select name="camera_type"><option value="">Not set</option>${meta.camera_types.map((t) => `<option value="${t}" ${c.camera_type === t ? "selected" : ""}>${TYPE_LABEL[t] || cap(t)}</option>`).join("")}</select></label>
        <label class="f"><span>Make / model</span><input type="text" name="make_model" value="${val(c.make_model)}"></label>
      </div>
      <label class="f"><span>Notes</span><input type="text" name="notes" value="${val(c.notes)}"></label>
      <label class="f"><span>Group</span><select name="group"><option value="">Ungrouped</option>${state.groups.map((x) => `<option value="${x.id}" ${x.id === c.group ? "selected" : ""}>${esc(x.name)}</option>`).join("")}</select>
        <small>A group's rules apply to its cameras unless a camera sets its own.</small></label>
    </fieldset>
    <fieldset><legend>Objects on this camera</legend>${meta.object_types.map(tri).join("")}</fieldset>
    <fieldset><legend>Skip unchanged frames</legend>
      <div class="tri" role="radiogroup" aria-label="Skip unchanged frames">
        ${[["inherit", `${esc(from(grp && grp.motion !== "inherit"))} (${(grp && grp.motion !== "inherit" ? grp.motion === "on" : meta.config.motion.enabled) ? "on" : "off"})`], ["on", "On"], ["off", "Off"]].map(([k, label]) =>
          `<label><input type="radio" name="motion" value="${k}" ${c.motion === k ? "checked" : ""}><span>${label}</span></label>`).join("")}</div>
    </fieldset>`;
}

// → { fields: columns on the source, config: the camera's config document }. Empty inputs are left out.
export function readCameraForm(el) {
  const get = (n) => el.querySelector(`[name="${n}"]`);
  const numOrNull = (n) => { const i = get(n); return !i || i.value === "" ? null : +i.value; };
  const fields = {};
  for (const n of ["lat", "lon", "heading", "hfov", "range_m"]) { const v = numOrNull(n); if (v != null) fields[n] = v; }
  if (get("name")?.value.trim()) fields.name = get("name").value.trim();
  if (get("uri")?.value.trim()) fields.uri = get("uri").value.trim();
  const objects = {};
  for (const t of meta.object_types) {
    const v = el.querySelector(`[name="obj-${t}"]:checked`)?.value;
    if (v && v !== "inherit") objects[t] = v;
  }
  const config = {
    camera_type: get("camera_type").value, make_model: get("make_model").value.trim(), notes: get("notes").value.trim(),
    mount_height_m: numOrNull("mount_height_m"), tilt_deg: numOrNull("tilt_deg"),
    objects, motion: el.querySelector('[name="motion"]:checked')?.value || "inherit",
    group: get("group")?.value ? +get("group").value : null,
  };
  const touched = Object.keys(fields).length > 0 || config.camera_type || config.make_model || config.notes
    || config.mount_height_m != null || config.tilt_deg != null || Object.keys(objects).length > 0 || config.motion !== "inherit";
  return { fields, config, touched: !!touched };
}

// ---------------------------------------------------------------- the view
export class Settings {
  constructor({ onPlace, onTest, onReview }) {
    this.onPlace = onPlace; this.onTest = onTest; this.onReview = onReview;
    this.selected = null;
    this.dirty = false;
    const g = $("#config-global"), c = $("#config-camera");
    g.addEventListener("change", () => this.saveGlobal());
    c.addEventListener("input", () => { this.dirty = true; $("#config-save", c)?.classList.add("primary"); });
    c.addEventListener("click", (e) => {
      const act = e.target.dataset.act;
      if (act === "save") this.saveCamera();
      else if (act === "place") this.onPlace(this.selected);
      else if (act === "test") this.onTest(this.selected);
    });
    $("#config-tabs").addEventListener("click", (e) => e.target.dataset.ctab && this.tab(e.target.dataset.ctab));
    $("#config-list").addEventListener("click", (e) => this.listClick(e));
    $("#config-groups").addEventListener("click", (e) => this.groupClick(e));
  }

  tab(name) {
    document.querySelectorAll("#config-tabs button").forEach((b) => b.classList.toggle("on", b.dataset.ctab === name));
    $("#config-cameras").hidden = name !== "cameras";
    $("#config-groups").hidden = name !== "groups";
    $("#config-general").hidden = name !== "general";
    $("#config-layout").hidden = name !== "layout";
    if (name === "layout") this.renderWallForm(true);
    if (name === "groups") this.renderGroups();
  }

  // ---- layout of the live wall: automatic, by group, or pages you build yourself ----
  renderWallForm(reset) {
    if (!meta) return;
    if (reset || !this.wall) this.wall = structuredClone(meta.config.ui.wall);
    const w = this.wall, el = $("#config-wall"), SIZES = [["1", "1 tile"], ["2", "2 side by side"], ["3", "1 large + 2"], ["4", "2 × 2"], ["6", "3 × 2"], ["focus", "1 large + 5"], ["9", "3 × 3"], ["16", "4 × 4"]];
    const N = { 1: 1, 2: 2, 3: 3, 4: 4, 6: 6, focus: 6, 9: 9, 16: 16 };
    const cams = [...state.sources.values()];
    const mode = (k, title, text) => `<button type="button" class="choice ${w.mode === k ? "on" : ""}" data-wmode="${k}"><strong>${title}</strong><span>${text}</span></button>`;
    el.innerHTML = `<div class="card-head"><h3>Live wall layout</h3></div>
      <p class="hint">The wall arranges itself. Choose how cameras are split into pages once there are more than fit on one.</p>
      <div class="choices three">
        ${mode("auto", "Automatic", "Cameras fill the wall in order. More than fit become pages.")}
        ${mode("groups", "By group", "Each camera group gets its own page, named after it.")}
        ${mode("custom", "Custom", "Build the pages yourself: grid, name, and which camera goes where.")}
      </div>
      <div class="f2 wall-opts" ${w.mode === "custom" ? "hidden" : ""}>
        <label class="f"><span>Cameras per page</span><select name="per_page">${[1, 2, 3, 4, 6, 9, 16].map((n) => `<option value="${n}" ${w.per_page === n ? "selected" : ""}>${n}</option>`).join("")}</select>
          <small>With fewer cameras than this, the grid shrinks to fit them.</small></label></div>
      <div class="wall-pages" ${w.mode === "custom" ? "" : "hidden"}>
        ${w.pages.map((p, i) => `<fieldset class="wpage" data-wpage="${i}"><legend>Page ${i + 1}</legend>
          <div class="f2"><label class="f"><span>Name</span><input type="text" name="wname" value="${val(p.name)}"></label>
            <label class="f"><span>Grid</span><select name="wsize">${SIZES.map(([k, t]) => `<option value="${k}" ${p.size === k ? "selected" : ""}>${t}</option>`).join("")}</select></label></div>
          <div class="wgrid" data-layout="s${p.size}">${Array.from({ length: N[p.size] }, (_, t) => `<label class="wtile"><span>${t + 1}</span><select name="wtile" data-tile="${t}"><option value="">Empty</option>${cams.map((c) => `<option value="${c.id}" ${p.tiles[t] === c.id ? "selected" : ""}>${esc(c.name)}</option>`).join("")}</select></label>`).join("")}</div>
          <button type="button" class="danger" data-wdel="${i}" data-icon="remove" data-tip="Delete this page.">Remove page</button></fieldset>`).join("") || `<p class="hint">No pages yet. Add one to start.</p>`}
        <button type="button" data-wadd data-icon="add" class="keep" data-tip="Add another page to the wall.">Add page</button>
      </div>
      <div class="actions"><button id="wall-save" data-icon="save" class="keep primary" data-tip="Apply the layout to the live wall now.">Save layout</button></div>`;
    el.onclick = (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      if (b.dataset.wmode) { w.mode = b.dataset.wmode; if (w.mode === "custom" && !w.pages.length) w.pages.push({ name: "Page 1", size: "4", tiles: cams.slice(0, 4).map((c) => c.id) }); this.renderWallForm(); }
      else if ("wadd" in b.dataset) { w.pages.push({ name: `Page ${w.pages.length + 1}`, size: "4", tiles: [] }); this.renderWallForm(); }
      else if (b.dataset.wdel != null) { w.pages.splice(+b.dataset.wdel, 1); this.renderWallForm(); }
      else if (b.id === "wall-save") this.saveWall();
    };
    el.onchange = (e) => {
      const pg = e.target.closest("[data-wpage]"), p = pg ? w.pages[+pg.dataset.wpage] : null, n = e.target.name;
      if (n === "per_page") w.per_page = +e.target.value;
      else if (p && n === "wname") p.name = e.target.value;
      else if (p && n === "wsize") { p.size = e.target.value; p.tiles = p.tiles.slice(0, N[p.size]); this.renderWallForm(); }
      else if (p && n === "wtile") p.tiles[+e.target.dataset.tile] = e.target.value ? +e.target.value : null;
    };
  }

  async saveWall() {
    try {
      const w = this.wall;
      for (const p of w.pages) p.tiles = Array.from({ length: { 1: 1, 2: 2, 3: 3, 4: 4, 6: 6, focus: 6, 9: 9, 16: 16 }[p.size] }, (_, i) => p.tiles[i] ?? null);
      meta.config = (await api("/api/config", { method: "PUT", json: { ...meta.config, ui: { ...meta.config.ui, wall: w } } })).config;
      this.renderWallForm(true);
      toast("Layout saved", { ms: 1500 });
    } catch (e) { toast(esc(e.message), { err: true }); }
  }

  // ---- groups: a list, and the selected group's name and rules ----
  selectGroup(id) { this.group = id; this.tab("groups"); }

  renderGroups() {
    if (!meta) return;
    const list = $("#config-grouplist"), form = $("#config-group");
    const count = (g) => [...state.sources.values()].filter((s) => s.config?.group === g.id).length;
    const rules = (g) => [...Object.entries(g.objects).map(([t, v]) => `${t} ${v === "allow" ? "allowed" : "blocked"}`)].join(", ") || "General";
    const html = `<table class="list"><thead><tr><th>Group</th><th>Cameras</th><th>Object rules</th><th>Skip still frames</th><th></th></tr></thead><tbody>${
      state.groups.map((g) => `<tr data-gid="${g.id}" class="${g.id === this.group ? "sel" : ""}"><td><strong>${esc(g.name)}</strong></td><td>${count(g)}</td><td>${esc(rules(g))}</td>
        <td>${{ inherit: "General", on: "On", off: "Off" }[g.motion]}</td>
        <td class="acts"><button data-grow="edit" data-icon="edit" data-tip="Name and rules for this group.">Edit</button><button data-grow="delete" class="danger" data-icon="remove" data-tip="Delete the group. Its cameras are kept.">Delete</button></td></tr>`).join("")
      || `<tr><td colspan="5" class="dim">No groups yet. A group lets several cameras share rules, and puts them together in the camera list.</td></tr>`}</tbody></table>
      <div class="list-foot"><button id="group-new" data-icon="group" class="keep" data-tip="Create an empty group.">New group</button></div>`;
    if (list.dataset.html !== html) { list.dataset.html = html; list.innerHTML = html; }
    const g = state.groups.find((x) => x.id === this.group);
    form.hidden = !g;
    if (!g || form.contains(document.activeElement)) return;
    const tri = (t) => `<div class="tri" role="radiogroup" aria-label="${cap(t)}"><span class="tri-name">${cap(t)}</span>
      ${[["inherit", `General (${meta.config.objects[t] ? "tracked" : "blocked"})`], ["allow", "Allow"], ["block", "Block"]].map(([k, label]) =>
        `<label><input type="radio" name="gobj-${t}" value="${k}" ${(g.objects[t] || "inherit") === k ? "checked" : ""}><span>${label}</span></label>`).join("")}</div>`;
    form.innerHTML = `<div class="card-head"><h3>Group: ${esc(g.name)}</h3></div>
      <p class="hint">These apply to every camera in the group, over the general settings. A camera's own rule still wins.</p>
      <div class="form"><label class="f"><span>Name</span><input type="text" name="gname" value="${val(g.name)}"></label>
        <fieldset><legend>Objects in this group</legend>${meta.object_types.map(tri).join("")}</fieldset>
        <fieldset><legend>Skip unchanged frames</legend><div class="tri" role="radiogroup" aria-label="Skip unchanged frames">
          ${[["inherit", `General (${meta.config.motion.enabled ? "on" : "off"})`], ["on", "On"], ["off", "Off"]].map(([k, label]) =>
            `<label><input type="radio" name="gmotion" value="${k}" ${g.motion === k ? "checked" : ""}><span>${label}</span></label>`).join("")}</div></fieldset></div>
      <div class="actions"><button id="group-save" data-icon="save" class="keep primary" data-tip="Apply to every camera in the group now.">Save group</button></div>`;
  }

  async groupClick(e) {
    try {
      if (e.target.id === "group-new") { const g = await api("/api/groups", { method: "POST", json: { name: "New group" } }); this.group = g.id; await refreshSources(); return this.renderGroups(); }
      if (e.target.id === "group-save") {
        const f = $("#config-group"), objects = {};
        for (const t of meta.object_types) { const v = f.querySelector(`[name="gobj-${t}"]:checked`)?.value; if (v && v !== "inherit") objects[t] = v; }
        await api(`/api/groups/${this.group}`, { method: "PATCH", json: { name: f.querySelector("[name=gname]").value.trim() || "Group", objects, motion: f.querySelector('[name="gmotion"]:checked')?.value || "inherit" } });
        await refreshSources();
        document.activeElement?.blur();
        this.renderGroups();
        return toast("Group saved", { ms: 1500 });
      }
      const tr = e.target.closest("tr[data-gid]");
      if (!tr) return;
      const id = +tr.dataset.gid;
      if (e.target.dataset.grow === "delete") {
        if (!e.target.dataset.armed) { e.target.dataset.armed = "1"; e.target.textContent = "Really delete?"; return; }
        await api(`/api/groups/${id}`, { method: "DELETE" });
        if (this.group === id) this.group = null;
        await refreshSources();
        $("#config-grouplist").dataset.html = "";
        return this.renderGroups();
      }
      this.group = id;
      $("#config-group").innerHTML = "";
      this.renderGroups();
    } catch (err) { toast(esc(err.message), { err: true }); }
  }

  // ---- the camera list: one row per source, the usual actions at the end of the row
  renderList() {
    const rows = [...state.sources.values()];
    const html = rows.length ? `<table class="list"><thead><tr><th>Name</th><th>Group</th><th>Source</th><th>Status</th><th>On map</th><th>Type</th><th>Object rules</th><th>Skip still frames</th><th></th></tr></thead><tbody>${rows.map((s) => {
      const st = s.status || {}, c = s.config || {}, running = ["running", "starting", "reconnecting"].includes(st.state);
      const rules = Object.entries(c.objects || {}).map(([t, v]) => `${t} ${v === "allow" ? "allowed" : "blocked"}`).join(", ") || "Global";
      return `<tr data-id="${s.id}" class="${s.id === this.selected ? "sel" : ""}">
        <td><strong>${esc(s.name)}</strong></td><td>${esc(state.groups.find((g) => g.id === c.group)?.name || "") || '<span class="dim">—</span>'}</td><td>${{ stream: "Network", file: "File", camera: "Mac camera", screen: "Screen" }[s.kind]}</td>
        <td><span class="pill ${st.state || "idle"}"><i></i>${st.state === "running" && st.idle ? "running · still" : st.state || "idle"}</span></td>
        <td>${s.lat == null ? '<span class="dim">Not placed</span>' : "Yes"}</td><td>${c.camera_type ? esc(TYPE_LABEL[c.camera_type] || c.camera_type) : '<span class="dim">—</span>'}</td>
        <td>${esc(rules)}</td><td>${{ inherit: "Global", on: "On", off: "Off" }[c.motion || "inherit"]}</td>
        <td class="acts"><button data-row="edit" data-icon="edit" data-tip="Open this camera's full settings.">Edit</button>${s.kind === "stream" ? '<button data-row="test" data-icon="health" data-tip="Run the connection test.">Test</button>' : ""}${s.kind === "file" ? '<button data-row="review" data-icon="review" data-tip="Player, timelines and connections.">Review</button>' : ""}
          <button data-row="${running ? "stop" : "start"}" data-icon="${running ? "stop" : "start"}" data-tip="${running ? "Stop analysing." : "Start analysing."}">${running ? "Stop" : "Start"}</button><button data-row="delete" class="danger" data-icon="remove" data-tip="Remove it and its results. Asks first.">Remove</button></td></tr>`;
    }).join("")}</tbody></table>` : `<p class="hint pad2">No cameras yet. Use “Add camera” above; it walks you through it.</p>`;
    const el = $("#config-list");
    if (el.dataset.html !== html) { el.dataset.html = html; el.innerHTML = html; }
  }

  async listClick(e) {
    const tr = e.target.closest("tr[data-id]");
    if (!tr) return;
    const id = +tr.dataset.id, act = e.target.dataset.row;
    try {
      if (!act || act === "edit") { this.select(id); $("#config-camera").scrollIntoView({ block: "nearest", behavior: "smooth" }); return; }
      if (act === "test") return this.onTest(id);
      if (act === "review") return this.onReview(id);
      if (act === "start" || act === "stop") await api(`/api/sources/${id}/${act}`, { method: "POST" });
      if (act === "delete") {
        if (!e.target.dataset.armed) { e.target.dataset.armed = "1"; e.target.textContent = "Really remove?"; return; }
        await api(`/api/sources/${id}`, { method: "DELETE" });
      }
      await refreshSources();
    } catch (err) { toast(esc(err.message), { err: true }); }
  }

  async show() { await loadMeta(); this.renderGlobal(); this.renderList(); this.renderCamera(true); }
  select(id) { this.selected = id; this.dirty = false; this.tab("cameras"); if (meta) { this.renderList(); this.renderCamera(true); } }
  // sources refresh every few seconds: never redraw a form someone is typing in
  sourcesChanged() {
    if (!meta || $("#view-config").hidden) return;
    if (!$("#config-list").querySelector("[data-armed]")) this.renderList();
    if (!$("#config-groups").hidden && !$("#config-grouplist").querySelector("[data-armed]")) this.renderGroups();
    if (!this.dirty && !$("#config-camera").contains(document.activeElement)) this.renderCamera(false);
  }

  renderGlobal() {
    const c = meta.config;
    $("#config-global").innerHTML = `<h3>General settings</h3>
      <p class="hint">Applies to every camera unless that camera overrides it. Changes save as you make them and take effect immediately. Prefer to be asked? Use “Guided setup” above.</p>
      <fieldset><legend>Objects to track</legend>
        <div class="checks">${meta.object_types.map((t) => `<label class="check"><input type="checkbox" name="obj-${t}" ${c.objects[t] ? "checked" : ""}> ${cap(t)}</label>`).join("")}</div>
        <small>Unticked types are ignored everywhere, except on cameras set to Allow.</small></fieldset>
      <fieldset><legend>Skip unchanged frames</legend>
        <label class="check"><input type="checkbox" name="motion-enabled" ${c.motion.enabled ? "checked" : ""}> Only run the models when the picture changes</label>
        <div class="f3">
          <label class="f"><span>Sensitivity</span><select name="motion-sensitivity">${["low", "medium", "high"].map((k) => `<option value="${k}" ${c.motion.sensitivity === k ? "selected" : ""}>${cap(k)}</option>`).join("")}</select></label>
          <label class="f"><span>Keep analysing for (s)</span><input type="number" name="motion-hold" min="0" max="60" step="0.5" value="${c.motion.hold_s}"></label>
          <label class="f"><span>Re-check every (s)</span><input type="number" name="motion-keepalive" min="0" max="3600" step="1" value="${c.motion.keepalive_s}"></label>
        </div>
        <small>High sensitivity reacts to small or distant movement. After a change, every frame is analysed for the hold time. A static scene is still re-checked on the interval (0 = never).</small></fieldset>
      <fieldset><legend>Network</legend>
        <label class="check"><input type="checkbox" name="net-reconnect" ${c.network.reconnect ? "checked" : ""}> Reconnect dropped streams automatically</label></fieldset>
      <fieldset><legend>Interface</legend>
        <label class="check"><input type="checkbox" name="ui-autohide" ${c.ui?.toolbar_autohide !== false ? "checked" : ""}> Auto-hide the overlay toolbar</label>
        <small>The Masks / Boxes / Paths / Plates / Labels toolbar at the bottom of the live view. When on, it slides away and returns when the pointer nears the bottom edge.</small></fieldset>
      <fieldset><legend>Defaults for new cameras</legend>
        <div class="f2">
          <label class="f"><span>Lens FOV °</span><input type="number" name="def-hfov" min="5" max="180" step="1" value="${c.defaults.hfov}"></label>
          <label class="f"><span>Range m</span><input type="number" name="def-range" min="5" max="2000" step="1" value="${c.defaults.range_m}"></label>
        </div></fieldset>`;
  }

  async saveGlobal() {
    const g = $("#config-global"), q = (n) => g.querySelector(`[name="${n}"]`);
    const doc = {
      objects: Object.fromEntries(meta.object_types.map((t) => [t, q(`obj-${t}`).checked])),
      motion: { enabled: q("motion-enabled").checked, sensitivity: q("motion-sensitivity").value, hold_s: +q("motion-hold").value, keepalive_s: +q("motion-keepalive").value },
      network: { reconnect: q("net-reconnect").checked },
      defaults: { hfov: +q("def-hfov").value, range_m: +q("def-range").value },
      ui: { ...meta.config.ui, toolbar_autohide: q("ui-autohide").checked },
    };
    try {
      meta.config = (await api("/api/config", { method: "PUT", json: doc })).config;
      toast("Global settings saved", { ms: 1500 });
      if (!this.dirty) this.renderCamera(true);   // the "Global (…)" labels follow
    } catch (e) { toast(esc(e.message), { err: true }); }
  }

  renderCamera(force) {
    const el = $("#config-camera");
    if (!state.sources.has(this.selected)) this.selected = state.sources.keys().next().value ?? null;
    const src = state.sources.get(this.selected);
    if (!force && el.dataset.for === String(this.selected) && el.dataset.rev === JSON.stringify([src?.name, src?.lat, src?.lon, src?.heading, src?.hfov, src?.range_m, src?.uri, src?.config])) return;
    el.dataset.for = String(this.selected);
    el.dataset.rev = JSON.stringify([src?.name, src?.lat, src?.lon, src?.heading, src?.hfov, src?.range_m, src?.uri, src?.config]);
    el.hidden = !src;
    if (!src) return;
    this.renderList();
    el.innerHTML = `<div class="card-head"><h3>Edit: ${esc(src.name)}</h3></div>
      <div class="form cols">${cameraFormHTML(src)}</div>
      <div class="actions"><button id="config-save" data-act="save" data-icon="save" class="keep" data-tip="Apply these settings now.">Save camera</button>
        ${src.kind === "stream" ? `<button data-act="test" data-icon="health" class="keep" data-tip="Six checks, and where it fails if it does.">Test connection</button>` : ""}</div>`;
    this.dirty = false;
  }

  async saveCamera() {
    const el = $("#config-camera"), id = this.selected;
    const { fields, config } = readCameraForm(el);
    // an emptied location box means "take it off the map"
    if (fields.hfov === state.sources.get(id)?.hfov) delete fields.hfov;   // unchanged: keep how the FOV was determined
    for (const n of ["lat", "lon", "heading"]) if (!(n in fields) && el.querySelector(`[name="${n}"]`).value === "") fields[n] = null;
    try {
      await api(`/api/sources/${id}`, { method: "PATCH", json: { ...fields, config } });
      this.dirty = false;
      await refreshSources();
      this.renderCamera(true);
      toast("Camera saved", { ms: 1500 });
    } catch (e) { toast(esc(e.message), { err: true }); }
  }
}
