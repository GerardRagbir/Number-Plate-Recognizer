// Guided flows. One dialog, one question per step, Back / Next at the bottom.
//   addCamera()   — type → connect → location → details → add
//   generalSetup() — what to track → when to analyse → connections
// Everything asked here can be changed later in Settings; the wizards only ask what most people need.
import { $, api, esc, state, refreshSources } from "./app.js";
import { cone } from "./map.js";
import { runTest } from "./monitor.js";
import { loadMeta } from "./settings.js";

const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const dlg = () => $("#wiz");

// ---------------------------------------------------------------- step runner
// step: { id, label, title, render(body, ctx), next?(ctx) → false to stay, nextLabel?, skip?: "label" }
function run({ title, steps, ctx, finish, done }) {
  const d = dlg(), body = $("#wiz-body"), err = $("#wiz-error");
  const back = $("#wiz-back"), next = $("#wiz-next"), skip = $("#wiz-skip");
  let i = 0, busy = false, finished = false;
  ctx.setNext = (label) => { next.textContent = label; };
  ctx.error = (msg) => { err.hidden = !msg; err.textContent = msg || ""; };
  ctx.go = () => next.click();

  const show = () => {
    const s = steps[i];
    $("#wiz-title").textContent = title;
    $("#wiz-steps").innerHTML = steps.map((st, k) =>
      `<li class="${k < i ? "done" : k === i ? "now" : ""}" ${k === i ? 'aria-current="step"' : ""}><i>${k < i ? "✓" : k + 1}</i>${esc(st.label)}</li>`).join("");
    body.innerHTML = `<h4 class="wiz-q">${esc(s.title)}</h4>${s.help ? `<p class="wiz-help">${esc(s.help)}</p>` : ""}<div class="wiz-step"></div>`;
    ctx.error("");
    back.hidden = i === 0;
    skip.hidden = !s.skip;
    skip.textContent = s.skip || "";
    next.hidden = !!s.noNext;
    next.textContent = s.nextLabel || (i === steps.length - 1 ? "Finish" : "Next");
    $("#wiz-foot").hidden = false;
    s.render($(".wiz-step", body), ctx);
    body.scrollTop = 0;
  };
  const advance = async (skipping) => {
    if (busy) return;
    busy = true; next.disabled = true;
    try {
      const s = steps[i];
      if (!skipping && s.next && (await s.next(ctx)) === false) return;
      if (skipping && s.onSkip) s.onSkip(ctx);
      if (i < steps.length - 1) { i++; show(); return; }
      const result = await finish(ctx);
      finished = true;
      $("#wiz-steps").innerHTML = "";
      $("#wiz-foot").hidden = true;
      done(body, ctx, result, () => d.close());
    } catch (e) { ctx.error(e.message); } finally { busy = false; next.disabled = false; }
  };
  next.onclick = () => advance(false);
  skip.onclick = () => advance(true);
  back.onclick = () => { if (i > 0 && !busy) { i--; show(); } };
  body.onkeydown = (e) => { if (e.key === "Enter" && e.target.matches("input") && !next.hidden) { e.preventDefault(); advance(false); } };
  d.onclose = () => { ctx.cleanup?.(); if (finished) ctx.afterClose?.(); };
  show();
  if (!d.open) d.showModal();
}

// ---------------------------------------------------------------- add a camera
const KINDS = [
  ["stream", "Network camera", "An IP camera, NVR or any RTSP / HTTP video address."],
  ["file", "Recorded video or photos", "Analyse files you already have. Good for reviewing an incident."],
  ["camera", "This Mac's camera", "The built-in camera, a USB webcam or a Continuity Camera."],
  ["screen", "A screen or window", "Watch whatever is on screen, e.g. another CCTV viewer."],
];
const TYPE_LABEL = { fixed: "Fixed", ptz: "PTZ", dome: "Dome", fisheye: "Fisheye", doorbell: "Doorbell", dashcam: "Dashcam", traffic: "Traffic / ANPR", other: "Other" };

export async function addCamera({ kind = null, onDone } = {}) {
  const meta = await loadMeta();
  const cat = await api("/api/camera-profiles").catch(() => ({ brands: [] }));
  const ctx = { kind, uri: "", name: "", files: [], path: "", mode: null, device: null,
    brand: "", profile: "", profileHfov: null, make_model: "", build: { ip: "", user: "", pass: "", ch: null },
    lat: null, lon: null, heading: 0, hfov: meta.config.defaults.hfov, range_m: meta.config.defaults.range_m,
    camera_type: "", objects: { ...meta.config.objects }, tested: null };

  const kindStep = {
    id: "kind", label: "Type", title: "What are you adding?", noNext: true,
    render(el) {
      el.innerHTML = `<div class="choices">${KINDS.map(([k, t, d]) =>
        `<button class="choice ${ctx.kind === k ? "on" : ""}" data-kind="${k}"><strong>${t}</strong><span>${d}</span></button>`).join("")}</div>`;
      el.onclick = (e) => { const b = e.target.closest("[data-kind]"); if (b) { if (ctx.kind !== b.dataset.kind) { ctx.device = null; ctx.tested = null; } ctx.kind = b.dataset.kind; ctx.go(); } };
    },
    next: () => !!ctx.kind,
  };

  const connectStep = {
    id: "connect", label: "Connect",
    get title() { return { stream: "Where is the camera's video?", file: "Which recordings?", camera: "Which camera?", screen: "Which screen or window?" }[ctx.kind]; },
    get help() { return ctx.kind === "stream" ? "Paste the stream address from the camera's app or manual. The connection is checked before you continue." : ""; },
    render(el) {
      if (ctx.kind === "stream") {
        const brand = cat.brands.find((b) => b.id === ctx.brand), model = brand?.models.find((m) => m.id === ctx.profile);
        const conn = brand?.connect, chans = conn ? [...new Set([...(model?.rtsp?.channels || []), ...conn.channels])] : [];
        if (conn && ctx.build.ch == null) ctx.build.ch = chans[0];
        const CONF = { official: "maker's spec", retailer: "retailer listing", "third-party": "a review", community: "community lists", unconfirmed: "not confirmed for this model" };
        const facts = !model ? "" : `<ul class="wiz-facts">
            <li>${model.hfov ? `Lens about <b>${Math.round(model.hfov)}°</b> wide <span>(${model.lens.basis === "horizontal" ? "horizontal figure" : `worked out from the ${model.lens.fov}° headline figure`}; ${CONF[model.lens.confidence] || model.lens.confidence})</span>` : "Lens angle not on file yet <span>(the default is used; adjust it on the next step)</span>"}</li>
            ${model.range_m ? `<li>Reach <b>${model.range_m} m</b> <span>(${esc(model.range_basis || "")})</span></li>` : ""}
            <li>${{ yes: "Streams over RTSP once switched on in the app", conflicting: "RTSP support is disputed for this model", unknown: "RTSP support is not confirmed for this model" }[model.rtsp?.support || "unknown"]}
              ${model.rtsp?.note ? `<span>${esc(model.rtsp.note)}</span>` : ""}${model.power === "battery" ? " <span>Battery camera: it sleeps between events, so expect drops.</span>" : ""}</li></ul>`;
        el.innerHTML = `<div class="f2">
            <label class="f"><span>Make</span><select id="wz-brand"><option value="">Other / not listed</option>${cat.brands.map((b) => `<option value="${b.id}" ${b.id === ctx.brand ? "selected" : ""}>${esc(b.name)}</option>`).join("")}</select></label>
            <label class="f"><span>Model</span><select id="wz-model" ${brand ? "" : "disabled"}><option value="">${brand ? "Choose the model…" : "—"}</option>${(brand?.models || []).map((m) => `<option value="${m.id}" ${m.id === ctx.profile ? "selected" : ""}>${esc(m.name)}</option>`).join("")}</select></label>
          </div>
          ${facts}
          ${conn ? `<details class="wiz-how"><summary>How to switch the stream on</summary><ol>${conn.enable.map((s) => `<li>${esc(s)}</li>`).join("")}</ol></details>
          <fieldset><legend>Build the address</legend>
            <div class="f3"><label class="f"><span>Camera IP</span><input type="text" id="wz-ip" placeholder="192.168.1.20" value="${esc(ctx.build.ip)}" spellcheck="false" autocomplete="off"></label>
              <label class="f"><span>Username</span><input type="text" id="wz-user" value="${esc(ctx.build.user)}" spellcheck="false" autocomplete="off"></label>
              <label class="f"><span>Password</span><input type="password" id="wz-pass" value="${esc(ctx.build.pass)}" autocomplete="off"></label></div>
            <label class="f"><span>Slot</span><select id="wz-ch" title="${esc(conn.channel_help)}">${chans.map((c) => `<option value="${c}" ${c === ctx.build.ch ? "selected" : ""}>${esc(conn.path.replace("{channel}", c))}</option>`).join("")}</select>
              <small>${esc(conn.channel_help)}</small></label></fieldset>` : ""}
          <label class="f"><span>Stream address</span><input type="text" id="wz-url" placeholder="rtsp://user:password@192.168.1.20/stream1" value="${esc(ctx.uri)}" spellcheck="false" autocomplete="off"></label>
          ${conn ? "" : `<p class="wiz-note">Usually <code>rtsp://user:password@camera-ip/path</code>. The path differs by brand; the camera's app normally shows it under RTSP or ONVIF.</p>`}
          <div id="wz-test"></div>`;
        const inp = $("#wz-url", el);
        const changed = () => { ctx.tested = null; ctx.setNext("Check connection"); $("#wz-test", el).innerHTML = ""; ctx.error(""); };
        inp.oninput = () => { ctx.uri = inp.value.trim(); changed(); };
        $("#wz-brand", el).onchange = (e) => {
          ctx.brand = e.target.value; ctx.profile = ""; ctx.profileHfov = null; ctx.make_model = ""; ctx.build.ch = null;
          connectStep.render(el, ctx);
        };
        $("#wz-model", el).onchange = (e) => {
          ctx.profile = e.target.value;
          const m = brand.models.find((x) => x.id === ctx.profile);
          ctx.build.ch = null;
          ctx.make_model = m ? `${brand.name} ${m.name}` : "";
          ctx.profileHfov = m?.hfov ? Math.round(m.hfov) : null;   // the profile sets the map cone: lens angle and reach
          ctx.hfov = ctx.profileHfov ?? meta.config.defaults.hfov;
          ctx.range_m = m?.range_m ?? meta.config.defaults.range_m;
          if (m?.camera_type) ctx.camera_type = m.camera_type;
          connectStep.render(el, ctx);
        };
        if (conn) {
          const compose = () => {
            const b = ctx.build;
            b.ip = $("#wz-ip", el).value.trim(); b.user = $("#wz-user", el).value.trim(); b.pass = $("#wz-pass", el).value; b.ch = +$("#wz-ch", el).value;
            if (!b.ip) return;
            const login = b.user ? `${encodeURIComponent(b.user)}${b.pass ? ":" + encodeURIComponent(b.pass) : ""}@` : "";
            inp.value = ctx.uri = `${conn.protocol}://${login}${b.ip}${conn.port && conn.port !== 554 ? ":" + conn.port : ""}${conn.path.replace("{channel}", b.ch)}`;
            changed();
          };
          for (const id of ["wz-ip", "wz-user", "wz-pass", "wz-ch"]) $(`#${id}`, el).oninput = compose;
        }
        ctx.setNext(ctx.tested ? "Next" : "Check connection");
        setTimeout(() => inp.focus(), 50);
      } else if (ctx.kind === "file") {
        el.innerHTML = `<label class="drop" id="wz-drop"><input type="file" id="wz-files" multiple accept="video/*,image/*" hidden>
            <strong></strong><span>Drop videos or photos here, or click to choose.</span></label>
          <div class="or">or use a file already on this Mac</div>
          <label class="f"><span>File path</span><input type="text" id="wz-path" placeholder="/path/to/recording.mp4" value="${esc(ctx.path)}" spellcheck="false"></label>
          <div id="wz-modes" ${ctx.files.length || ctx.path ? "" : "hidden"}>
            <h5 class="wiz-sub">How should it be processed?</h5>
            <div class="choices">
              <button type="button" class="choice ${ctx.mode === "immediate" ? "on" : ""}" data-mode="immediate"><strong>Process immediately</strong>
                <span>Analyse every frame as fast as possible, then open a review: a looping player, timelines of everything found, and each object's connections.</span></button>
              <button type="button" class="choice ${ctx.mode === "temporal" ? "on" : ""}" data-mode="temporal"><strong>Process temporally</strong>
                <span>Play it back in real time on the live wall, placed on the shared timeline at the moment it was recorded, like a camera that was there.</span></button>
            </div></div>`;
        const drop = $("#wz-drop", el), label = $("strong", drop);
        const sync = () => { $("#wz-modes", el).hidden = !(ctx.files.length || ctx.path); label.textContent = ctx.files.length ? `${ctx.files.length} file${ctx.files.length > 1 ? "s" : ""} selected: ${ctx.files.map((f) => f.name).slice(0, 3).join(", ")}${ctx.files.length > 3 ? "…" : ""}` : "Choose recordings"; };
        sync();
        $("#wz-files", el).onchange = (e) => { ctx.files = [...e.target.files]; sync(); };
        ["dragenter", "dragover"].forEach((n) => drop.addEventListener(n, (e) => { e.preventDefault(); drop.classList.add("over"); }));
        ["dragleave", "drop"].forEach((n) => drop.addEventListener(n, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
        drop.addEventListener("drop", (e) => { ctx.files = [...e.dataTransfer.files]; sync(); });
        $("#wz-path", el).oninput = (e) => { ctx.path = e.target.value.trim(); sync(); };
        $("#wz-modes", el).onclick = (e) => {
          const b = e.target.closest("[data-mode]");
          if (!b) return;
          ctx.mode = b.dataset.mode; ctx.error("");
          el.querySelectorAll("[data-mode]").forEach((x) => x.classList.toggle("on", x === b));
        };
      } else {
        el.innerHTML = `<p class="hint">Looking…</p>`;
        api("/api/devices").then((d) => {
          const failed = d.errors[ctx.kind];
          const items = ctx.kind === "camera" ? d.cameras.map((x) => [x.id, x.name, x.hfov ? `${x.hfov.toFixed(0)}° lens` : "", x.name])
            : [...d.displays.map((x) => [`display:${x.id}`, x.name, "whole display", `Screen · ${x.name}`]),
               ...d.windows.map((x) => [`window:${x.id}`, x.app, x.title, `Window · ${x.app}`])];
          el.innerHTML = failed ? `<p class="error">${esc(failed)}</p><p class="wiz-note">macOS asks for ${ctx.kind === "camera" ? "Camera" : "Screen Recording"} permission the first time. Allow it in System Settings → Privacy & Security, then try again.</p>`
            : items.length ? `<div class="pick">${items.map(([id, name, sub, label]) =>
              `<button data-uri="${esc(id)}" data-name="${esc(label)}" class="${ctx.device === id ? "on" : ""}">${esc(name)} <span>${esc(sub)}</span></button>`).join("")}</div>`
            : `<p class="hint">Nothing found.</p>`;
          el.onclick = (e) => {
            const b = e.target.closest("[data-uri]");
            if (!b) return;
            ctx.device = ctx.uri = b.dataset.uri;
            if (!ctx.nameTouched) ctx.name = b.dataset.name;
            el.querySelectorAll("[data-uri]").forEach((x) => x.classList.toggle("on", x === b));
          };
        }).catch((e) => { el.innerHTML = `<p class="error">${esc(e.message)}</p>`; });
      }
    },
    async next() {
      if (ctx.kind === "stream") {
        if (!ctx.uri) { ctx.error("Enter the camera's stream address."); return false; }
        if (ctx.tested) return true;
        const ok = await runTest($("#wz-test"), { uri: ctx.uri, seconds: 3 });
        ctx.tested = ok ? "ok" : "failed";
        if (ok) return true;
        ctx.setNext("Add anyway");   // let them continue: the camera may just be off right now
        ctx.error("The connection check failed. Fix the address, or continue and the app will keep retrying.");
        if (/^rtsps?:/i.test(ctx.uri) && ctx.brand) {   // the camera answered but maybe not on this slot: go and look
          const box = $("#wz-test"), btn = document.createElement("button");
          btn.type = "button"; btn.className = "wiz-probe"; btn.textContent = "Find the right slot for me";
          btn.onclick = async () => {
            btn.disabled = true; btn.textContent = "Trying each slot…";
            try {
              const r = await api("/api/nettest/probe", { method: "POST", json: { uri: ctx.uri, profile: ctx.profile || null } });
              const tried = r.paths.map((p) => `${p.path} ${p.ok ? "✓" : p.code || "no answer"}`).join(" · ");
              if (r.found) {
                ctx.uri = ctx.uri.replace(/^(rtsps?:\/\/[^/]+)(\/.*)?$/i, `$1${r.found}`);
                $("#wz-url").value = ctx.uri; ctx.tested = null; ctx.error("");
                box.innerHTML = `<p class="verdict ok" style="margin-left:0">Found video on ${esc(r.found)}. Checking it now…</p><p class="wiz-note">${esc(tried)}</p>`;
                ctx.setNext("Check connection"); ctx.go();
              } else { btn.remove(); box.insertAdjacentHTML("beforeend", `<p class="wiz-note">No slot answered with video: ${esc(tried)}.</p><p class="wiz-note" style="color:var(--text)">${esc(r.hint || "Check that the stream is switched on for this camera in its app.")}</p>`); }
            } catch (e) { btn.remove(); ctx.error(e.message); }
          };
          box.append(btn);
        }
        return false;
      }
      if (ctx.kind === "file") {
        if (!ctx.files.length && !ctx.path) { ctx.error("Choose at least one file, or enter a path."); return false; }
        if (!ctx.mode) { ctx.error("Choose how to process it: immediately or temporally."); return false; }
        return true;
      }
      if (!ctx.device) { ctx.error("Pick one from the list."); return false; }
      return true;
    },
  };

  const placeStep = {
    id: "place", label: "Location", title: "Where is it, and which way does it face?", skip: "Skip for now",
    get help() { return ctx.kind === "file" ? "Optional. Files with GPS information place themselves. Otherwise click the map where this was recorded." : "Click the map where the camera is mounted, then set the direction it looks. This is what lets vehicles be followed from camera to camera."; },
    onSkip() { ctx.lat = ctx.lon = null; },
    render(el) {
      el.innerHTML = `<div class="wiz-maptools"><button type="button" id="wz-locate">Go to my location</button>
          <span class="wiz-note">or zoom and drag the map to the right street</span></div>
        <div id="wz-map" class="wiz-map"></div>
        <div class="wiz-sliders">
          <label>Direction<input type="range" id="wz-heading" min="0" max="359" step="1" value="${ctx.heading}"><output></output></label>
          <label>Width of view<input type="range" id="wz-hfov" min="10" max="180" step="1" value="${ctx.hfov}"><output></output></label>
          <label>Reach<input type="range" id="wz-range" min="5" max="300" step="5" value="${ctx.range_m}"><output></output></label>
        </div><p class="wiz-note" id="wz-place-note"></p>`;
      const others = [...state.sources.values()].filter((s) => s.lat != null);
      let view = null;
      try { view = JSON.parse(localStorage.getItem("npr.map")); } catch {}
      const map = L.map($("#wz-map", el), { zoomControl: true, attributionControl: true });
      ctx.cleanup = () => { try { map.remove(); } catch {} ctx.cleanup = null; };
      L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "© OpenStreetMap contributors" }).addTo(map);
      for (const s of others) {
        L.circleMarker([s.lat, s.lon], { radius: 6, color: "#8492a3", weight: 2, fillOpacity: 0.4 }).bindTooltip(esc(s.name)).addTo(map);
        if (s.heading != null) L.polygon(cone(s), { color: "#8492a3", weight: 1, fillOpacity: 0.08, interactive: false }).addTo(map);
      }
      if (ctx.lat != null) map.setView([ctx.lat, ctx.lon], 18);
      else if (others.length) map.fitBounds(L.latLngBounds(others.map((s) => [s.lat, s.lon])).pad(0.6), { maxZoom: 17 });
      else map.setView(view?.c || [20, 0], view?.z || 2);
      let marker = null, shape = null;
      const note = $("#wz-place-note", el);
      const draw = () => {
        for (const [id, key, unit] of [["wz-heading", "heading", "°"], ["wz-hfov", "hfov", "°"], ["wz-range", "range_m", " m"]]) {
          const inp = $(`#${id}`, el); ctx[key] = +inp.value; inp.nextElementSibling.textContent = `${inp.value}${unit}`;
        }
        note.textContent = ctx.lat == null ? "Not placed yet: click the map." : `Placed at ${ctx.lat.toFixed(5)}, ${ctx.lon.toFixed(5)}. Drag the pin to adjust.`;
        if (ctx.lat == null) return;
        if (!marker) {
          marker = L.marker([ctx.lat, ctx.lon], { draggable: true, icon: L.divIcon({ className: "", html: `<div class="cam sel">+</div>`, iconSize: [26, 26], iconAnchor: [13, 13] }) }).addTo(map);
          marker.on("drag", () => { const p = marker.getLatLng(); ctx.lat = p.lat; ctx.lon = p.lng; draw(); });
        } else marker.setLatLng([ctx.lat, ctx.lon]);
        const pts = cone(ctx);
        shape ? shape.setLatLngs(pts) : (shape = L.polygon(pts, { color: "#4cc2ff", weight: 1.5, fillOpacity: 0.25, interactive: false }).addTo(map));
      };
      map.on("click", (e) => {
        ctx.error("");
        if (map.getZoom() < 14) {   // a click on a country-sized map is not a camera position: zoom in first
          map.setView(e.latlng, Math.min(map.getZoom() + 5, 16));
          note.textContent = "Zoomed in. Keep clicking towards the spot; the pin drops once streets are visible.";
          return;
        }
        ctx.lat = e.latlng.lat; ctx.lon = e.latlng.lng; draw();
      });
      $("#wz-locate", el).onclick = () => {
        if (!navigator.geolocation) { note.textContent = "This browser cannot report a location."; return; }
        note.textContent = "Asking the browser for your location…";
        navigator.geolocation.getCurrentPosition(
          (p) => { map.setView([p.coords.latitude, p.coords.longitude], 17); note.textContent = "Here is your area. Click the exact spot where the camera is."; },
          () => { note.textContent = "Location was not available. Zoom and drag the map instead."; },
          { timeout: 8000 });
      };
      $(".wiz-sliders", el).oninput = draw;
      setTimeout(() => { map.invalidateSize(); draw(); }, 60);
    },
    next() {
      if (ctx.lat == null) { ctx.error("Click the map to place it, or choose “Skip for now”."); return false; }
      ctx.cleanup?.();
      return true;
    },
  };

  const detailsStep = {
    id: "details", label: "Details", title: "Name it and choose what it should watch for",
    render(el) {
      const defName = ctx.name || { stream: "", file: "", camera: "", screen: "" }[ctx.kind];
      el.innerHTML = `
        ${ctx.kind === "file" && (ctx.files.length > 1 || (!ctx.files.length && !ctx.path)) ? "" : `<label class="f"><span>Name</span><input type="text" id="wz-name" value="${esc(defName)}" placeholder="${ctx.kind === "file" ? "Uses the file name if left blank" : "e.g. Front gate"}"></label>`}
        <label class="f"><span>Camera type (optional)</span><select id="wz-type"><option value="">Not set</option>${meta.camera_types.map((t) => `<option value="${t}" ${ctx.camera_type === t ? "selected" : ""}>${TYPE_LABEL[t] || cap(t)}</option>`).join("")}</select></label>
        ${state.groups.length ? `<label class="f"><span>Group (optional)</span><select id="wz-group"><option value="">Ungrouped</option>${state.groups.map((g) => `<option value="${g.id}" ${ctx.group === g.id ? "selected" : ""}>${esc(g.name)}</option>`).join("")}</select></label>` : ""}
        <fieldset><legend>Track on this camera</legend>
          <div class="checks">${meta.object_types.map((t) => `<label class="check"><input type="checkbox" data-obj="${t}" ${ctx.objects[t] ? "checked" : ""}> ${cap(t)}</label>`).join("")}</div>
          <small>Starts from your general settings. Untick what this camera should ignore.</small></fieldset>
        <p class="wiz-note">Mount height, tilt, frame skipping and notes are under Settings → Cameras.</p>`;
      el.oninput = (e) => {
        if (e.target.id === "wz-name") { ctx.name = e.target.value.trim(); ctx.nameTouched = true; }
        if (e.target.id === "wz-type") ctx.camera_type = e.target.value;
        if (e.target.id === "wz-group") ctx.group = e.target.value ? +e.target.value : null;
        if (e.target.dataset.obj) ctx.objects[e.target.dataset.obj] = e.target.checked;
      };
    },
    nextLabel: "Add",
    next() {
      if (!meta.object_types.some((t) => ctx.objects[t])) { ctx.error("Choose at least one thing to track."); return false; }
      return true;
    },
  };

  run({
    title: "Add a camera", ctx,
    steps: [kindStep, connectStep, placeStep, detailsStep].slice(kind ? 1 : 0),
    async finish() {
      const rules = {};   // only differences from the general settings become per-camera rules
      for (const t of meta.object_types) if (ctx.objects[t] !== meta.config.objects[t]) rules[t] = ctx.objects[t] ? "allow" : "block";
      const config = { camera_type: ctx.camera_type, objects: rules, make_model: ctx.make_model, profile: ctx.profile, group: ctx.group ?? null };
      // a lens angle that still matches the chosen profile is recorded as coming from it
      const fromProfile = ctx.kind === "stream" && ctx.profileHfov != null && ctx.hfov === ctx.profileHfov
        ? { hfov: ctx.hfov, range_m: ctx.range_m, fov_source: "profile" } : {};
      const where = ctx.lat == null ? {} : { lat: ctx.lat, lon: ctx.lon, heading: ctx.heading, range_m: ctx.range_m,
        ...(ctx.hfov !== meta.config.defaults.hfov ? { hfov: ctx.hfov } : {}) };
      let rows;
      if (ctx.kind === "file") {
        const opts = ctx.mode === "immediate" ? { speed: 0, mode: "immediate" } : { mode: "temporal" };
        if (ctx.files.length) {
          const fd = new FormData();
          for (const f of ctx.files) fd.append("files", f);
          rows = await api("/api/upload?start=false", { method: "POST", body: fd });
        } else rows = [await api("/api/sources", { method: "POST", json: { kind: "file", uri: ctx.path, start: false } })];
        for (const r of rows) {
          Object.assign(r, await api(`/api/sources/${r.id}`, { method: "PATCH", json: { ...where, config, opts, ...(ctx.name && rows.length === 1 ? { name: ctx.name } : {}) } }));
          await api(`/api/sources/${r.id}/start`, { method: "POST" });
        }
      } else {
        rows = [await api("/api/sources", { method: "POST", json: { kind: ctx.kind, uri: ctx.uri, name: ctx.name || null, ...where, ...fromProfile, config } })];
      }
      await refreshSources();
      return rows;
    },
    done(body, _ctx, rows, close) {
      const placed = ctx.lat != null, n = rows.length;
      body.innerHTML = `<div class="wiz-done"><div class="tick">✓</div>
        <h4>${n > 1 ? `${n} recordings added` : `${esc(rows[0].name)} added`}</h4>
        <p>${ctx.kind === "file" ? (ctx.mode === "immediate" ? "Every frame is being analysed now. Open the review to watch the timeline build." : "It is playing back in real time on the live wall, on the timeline at the moment it was recorded.") : "It is starting now and will appear on the live wall."}
          ${placed ? "" : " It is not on the map yet; you can place it any time from the Map view."}</p>
        <div class="wiz-actions">${ctx.kind === "file" && ctx.mode === "immediate" ? `<button class="primary" data-go="review">Open the review</button><button data-go="live">Show on the wall</button>` : `<button class="primary" data-go="live">Watch it live</button>`}
          <button data-go="another">Add another camera</button>
          ${placed ? `<button data-go="map">See it on the map</button>` : `<button data-go="place">Place it on the map</button>`}
          <button class="ghost" data-go="close">Done</button></div></div>`;
      body.onclick = (e) => {
        const go = e.target.dataset.go;
        if (!go) return;
        close();
        if (go === "another") addCamera({ onDone });
        else onDone?.(go, rows);
      };
    },
  });
}

// ---------------------------------------------------------------- general setup
const MOTION_CHOICES = [
  ["medium", "Balanced", "Pause the models while the picture is still; wake on normal movement. Recommended."],
  ["high", "Catch small or distant movement", "Wakes more easily. Use for far-away roads or small vehicles."],
  ["low", "Only large movement", "Ignores swaying trees and small changes. Lowest power use."],
  ["off", "Analyse every frame", "Never pause. Highest power use; only for testing or very busy scenes."],
];

export async function generalSetup({ onDone } = {}) {
  const meta = await loadMeta();
  const cfg = structuredClone(meta.config);
  const ctx = {};
  run({
    title: "General setup", ctx,
    steps: [
      { id: "objects", label: "Track", title: "What should the app look for?", help: "This applies to every camera. Individual cameras can differ later.",
        render(el) {
          el.innerHTML = `<div class="choices two">${meta.object_types.map((t) =>
            `<label class="choice check2 ${cfg.objects[t] ? "on" : ""}"><input type="checkbox" data-obj="${t}" ${cfg.objects[t] ? "checked" : ""}><strong>${cap(t)}</strong></label>`).join("")}</div>`;
          el.onchange = (e) => { const t = e.target.dataset.obj; if (t) { cfg.objects[t] = e.target.checked; e.target.closest(".choice").classList.toggle("on", e.target.checked); } };
        },
        next() { if (!meta.object_types.some((t) => cfg.objects[t])) { ctx.error("Choose at least one."); return false; } return true; } },
      { id: "motion", label: "Analyse", title: "When should the models run?", help: "A camera watching an empty driveway does not need analysing 30 times a second.",
        render(el) {
          const cur = cfg.motion.enabled ? cfg.motion.sensitivity : "off";
          el.innerHTML = `<div class="choices">${MOTION_CHOICES.map(([k, t, d]) =>
            `<button class="choice ${cur === k ? "on" : ""}" data-m="${k}"><strong>${t}</strong><span>${d}</span></button>`).join("")}</div>`;
          el.onclick = (e) => {
            const b = e.target.closest("[data-m]");
            if (!b) return;
            cfg.motion.enabled = b.dataset.m !== "off";
            if (cfg.motion.enabled) cfg.motion.sensitivity = b.dataset.m;
            el.querySelectorAll("[data-m]").forEach((x) => x.classList.toggle("on", x === b));
          };
        } },
      { id: "network", label: "Connections", title: "If a camera drops off the network?",
        render(el) {
          el.innerHTML = `<div class="choices">
            <button class="choice ${cfg.network.reconnect ? "on" : ""}" data-r="1"><strong>Keep trying to reconnect</strong><span>Retries after 2, 4, 8 … up to every 30 seconds until it is back. Recommended.</span></button>
            <button class="choice ${cfg.network.reconnect ? "" : "on"}" data-r="0"><strong>Stop and show an error</strong><span>The camera stays stopped until you start it again.</span></button></div>`;
          el.onclick = (e) => {
            const b = e.target.closest("[data-r]");
            if (!b) return;
            cfg.network.reconnect = b.dataset.r === "1";
            el.querySelectorAll("[data-r]").forEach((x) => x.classList.toggle("on", x === b));
          };
        }, nextLabel: "Save" },
    ],
    async finish() {
      await api("/api/config", { method: "PUT", json: cfg });
      try { localStorage.setItem("npr.setup", "1"); } catch {}
    },
    done(body, _c, _r, close) {
      body.innerHTML = `<div class="wiz-done"><div class="tick">✓</div><h4>General settings saved</h4>
        <p>They apply to every camera now. Fine-tuning (hold time, re-check interval, default lens angle) is under Settings → General.</p>
        <div class="wiz-actions"><button class="primary" data-go="close">Done</button><button data-go="settings">Open Settings</button></div></div>`;
      body.onclick = (e) => { const go = e.target.dataset.go; if (go) { close(); onDone?.(go); } };
    },
  });
}
