// Network view: a heartbeat card per connection (state strip + small-multiple line charts that
// share one time axis) and the step-by-step connection test.
import { $, $$, esc, fmtDur, state } from "./app.js";

// Series colours: validated pair for the dark surface (dataviz palette slots 1 and 2).
const SERIES = { recv: "#3987e5", proc: "#d95926" };
const INK = { text: "#dbe3ec", dim: "#8492a3", grid: "#222a34", axis: "#2e3945" };
// Connection state is a status, not a series: reserved status colours, always shown with a label.
const STATE_STYLE = {
  running: ["#0ca30c", "Receiving"], static: ["#3d6b4a", "Receiving · scene static"],
  starting: ["#fab219", "Starting"], reconnecting: ["#fab219", "Reconnecting"],
  error: ["#d03b3b", "Failed"], ended: ["#3a4654", "Ended"], stopped: ["#3a4654", "Stopped"], idle: ["#3a4654", "Not started"],
};
const pad = (n) => String(n).padStart(2, "0");
const clock = (t) => { const d = new Date(t * 1000); return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`; };
const num = (v, digits = 0) => (v == null ? "—" : v.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits }));

function niceMax(v) {
  if (!(v > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(v)), m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
}

// ---------------------------------------------------------------- connection test
const STEP_ICON = { ok: "✓", warn: "!", fail: "✕", skip: "–", run: "…" };
const STEP_TITLES = [["address", "Address"], ["dns", "Name lookup"], ["tcp", "Reach the device"],
  ["handshake", "Protocol handshake"], ["open", "Open the stream"], ["frames", "Receive video"]];

export async function runTest(box, body) {
  const steps = new Map(STEP_TITLES.map(([id, title]) => [id, { id, title, status: "pending" }]));
  const render = (verdict) => {
    box.innerHTML = `<ol class="steps">${[...steps.values()].map((s) => `
      <li class="step ${s.status}"><span class="ico" aria-hidden="true">${STEP_ICON[s.status] || ""}</span>
        <div><strong>${esc(s.title)}</strong><span class="st">${{ ok: "passed", warn: "warning", fail: "failed", skip: "skipped", run: "running…", pending: "waiting" }[s.status]}</span>
          ${s.ms != null ? `<span class="ms">${num(s.ms)} ms</span>` : ""}
          ${s.detail && s.status !== "skip" ? `<div class="detail">${esc(s.detail)}</div>` : ""}
          ${s.hint ? `<div class="hint2">${esc(s.hint)}</div>` : ""}</div></li>`).join("")}</ol>
      ${verdict ? `<p class="verdict ${verdict[0]}">${esc(verdict[1])}</p>` : ""}`;
  };
  const markNext = () => { const n = [...steps.values()].find((s) => s.status === "pending"); if (n) n.status = "run"; };
  markNext(); render();
  try {
    const r = await fetch("/api/nettest", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 1);
        if (!line) continue;
        const s = JSON.parse(line);
        steps.set(s.id, s);
        if (s.status !== "fail") markNext();
        render();
      }
    }
    const all = [...steps.values()];
    all.forEach((s) => { if (s.status === "run" || s.status === "pending") s.status = "skip"; });
    const bad = all.find((s) => s.status === "fail"), warn = all.some((s) => s.status === "warn");
    render(bad ? ["fail", `Failed at “${bad.title}”.`] : warn ? ["warn", "Connected, with warnings."] : ["ok", "Connection is healthy."]);
    return !bad;
  } catch (e) {
    box.innerHTML = `<p class="error">${esc(e.message)}</p>`;
    return false;
  }
}

// ---------------------------------------------------------------- charts
class LineChart {
  // series: [{ key, name, color }], one y axis, shared time axis supplied by the card
  constructor(host, { title, unit, series, digits = 0 }) {
    this.series = series; this.unit = unit; this.digits = digits;
    this.el = document.createElement("div");
    this.el.className = "chart";
    this.el.innerHTML = `<div class="chart-head"><span class="chart-title">${esc(title)}</span>
      ${series.length > 1 ? `<span class="legend">${series.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("")}</span>` : ""}</div><canvas></canvas>`;
    this.canvas = this.el.querySelector("canvas");
    host.append(this.el);
  }

  draw(rows, t0, t1, hoverT) {
    const dpr = window.devicePixelRatio || 1, cv = this.canvas;
    const W = cv.clientWidth, H = cv.clientHeight;
    if (!W || !H) return;
    if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const c = cv.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, W, H);
    const L = 38, R = 6, T = 6, B = 16, pw = W - L - R, ph = H - T - B;
    let max = 0;
    for (const r of rows) for (const s of this.series) if (r[s.key] != null && r[s.key] > max) max = r[s.key];
    max = niceMax(max);
    const x = (t) => L + ((t - t0) / (t1 - t0)) * pw, y = (v) => T + ph - (v / max) * ph;
    this.geom = { L, pw, t0, t1 };

    c.font = "10px ui-monospace, Menlo, monospace";
    c.textBaseline = "middle";
    for (const f of [0, 0.5, 1]) {   // recessive grid: three hairlines, labelled on the left
      const gy = Math.round(T + ph - f * ph) + 0.5;
      c.strokeStyle = f === 0 ? INK.axis : INK.grid; c.lineWidth = 1;
      c.beginPath(); c.moveTo(L, gy); c.lineTo(L + pw, gy); c.stroke();
      c.fillStyle = INK.dim; c.textAlign = "right";
      const tick = max * f;
      c.fillText(tick >= 10000 ? `${num(tick / 1000, tick % 1000 ? 1 : 0)}k` : num(tick, tick < 10 && f ? 1 : 0), L - 6, gy);
    }
    c.textBaseline = "alphabetic"; c.fillStyle = INK.dim;
    c.textAlign = "left"; c.fillText(clock(t0), L, H - 3);
    c.textAlign = "right"; c.fillText(clock(t1), L + pw, H - 3);

    c.save();
    c.beginPath(); c.rect(L, T - 2, pw, ph + 4); c.clip();
    c.lineWidth = 2; c.lineJoin = "round"; c.lineCap = "round";
    for (const s of this.series) {
      c.strokeStyle = s.color;
      c.beginPath();
      let pen = false, prevT = null;
      for (const r of rows) {
        const v = r[s.key];
        if (v == null || (prevT != null && r.t - prevT > 3)) pen = false;   // a hole in the data stays a hole
        if (v == null) continue;
        pen ? c.lineTo(x(r.t), y(Math.min(v, max))) : c.moveTo(x(r.t), y(Math.min(v, max)));
        pen = true; prevT = r.t;
      }
      c.stroke();
    }
    c.restore();

    if (hoverT != null && hoverT >= t0 && hoverT <= t1) {
      const hx = Math.round(x(hoverT)) + 0.5;
      c.strokeStyle = INK.dim; c.lineWidth = 1;
      c.beginPath(); c.moveTo(hx, T); c.lineTo(hx, T + ph); c.stroke();
      const row = nearest(rows, hoverT);
      if (row) for (const s of this.series) {
        if (row[s.key] == null) continue;
        c.beginPath(); c.arc(x(row.t), y(Math.min(row[s.key], max)), 4, 0, 7);
        c.fillStyle = s.color; c.fill();
        c.lineWidth = 2; c.strokeStyle = "#10151b"; c.stroke();   // surface ring keeps the marker legible on the line
      }
    }
  }
}

function nearest(rows, t) {
  let best = null, bd = 2.5;
  for (const r of rows) { const d = Math.abs(r.t - t); if (d < bd) { bd = d; best = r; } }
  return best;
}

function drawStrip(cv, rows, t0, t1, hoverT) {
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  if (!W) return;
  if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const c = cv.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, W, H);
  const L = 38, pw = W - L - 6, x = (t) => L + ((t - t0) / (t1 - t0)) * pw;
  c.fillStyle = "#161c24"; c.fillRect(L, 0, pw, H);
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i], a = Math.max(x(r.t - 1), L), b = Math.min(x(Math.min(rows[i + 1]?.t ?? r.t, r.t + 1.5) ), L + pw);
    if (b <= L) continue;
    c.fillStyle = (STATE_STYLE[r.state] || STATE_STYLE.idle)[0];
    c.fillRect(a, 0, Math.max(b - a, 1) + 0.5, H);
  }
  if (hoverT != null) { c.fillStyle = INK.text; c.fillRect(Math.round(x(hoverT)), 0, 1, H); }
}

// ---------------------------------------------------------------- cards
class Card {
  constructor(src) {
    this.id = src.id;
    this.rows = [];
    this.hoverT = null;
    this.el = document.createElement("article");
    this.el.className = "card net-card";
    this.el.dataset.id = src.id;
    this.el.innerHTML = `
      <header><h3></h3><span class="badge"></span><span class="spacer"></span>
        <button data-act="table" data-icon="table" data-tip="The last minute of samples as numbers.">Table</button>
        <button data-act="test" class="test" data-icon="health" data-tip="Six checks, and where it fails if it does.">Test connection</button></header>
      <div class="tiles"></div>
      <div class="plots">
        <div class="chart"><div class="chart-head"><span class="chart-title">Connection state</span><span class="legend states"></span></div><canvas class="strip"></canvas></div>
      </div>
      <div class="tip" hidden></div>
      <div class="table-wrap" hidden></div>
      <div class="test-result"></div>`;
    const plots = $(".plots", this.el);
    this.strip = $(".strip", this.el);
    this.charts = [
      new LineChart(plots, { title: "Frames per second", unit: "fps", digits: 1, series: [
        { key: "recv_fps", name: "Received from camera", color: SERIES.recv }, { key: "proc_fps", name: "Analysed", color: SERIES.proc }] }),
      new LineChart(plots, { title: "Longest wait between frames (ms)", unit: "ms", series: [{ key: "gap_ms", name: "Frame gap", color: SERIES.recv }] }),
      new LineChart(plots, { title: "Bitrate (kbit/s)", unit: "kbit/s", series: [{ key: "kbps", name: "Bitrate", color: SERIES.recv }] }),
    ];
    this.tip = $(".tip", this.el);
    plots.addEventListener("mousemove", (e) => this.hover(e));
    plots.addEventListener("mouseleave", () => { this.hoverT = null; this.tip.hidden = true; this.draw(); });
    this.el.addEventListener("click", (e) => {
      const act = e.target.dataset.act;
      if (act === "test") this.test(e.target);
      if (act === "table") { const w = $(".table-wrap", this.el); w.hidden = !w.hidden; e.target.classList.toggle("on", !w.hidden); this.table(); }
    });
    new ResizeObserver(() => this.draw()).observe(plots);
  }

  async test(btn) {
    btn.disabled = true;
    await runTest($(".test-result", this.el), { source_id: this.id });
    btn.disabled = false;
  }

  hover(e) {
    const g = this.charts[0].geom;
    if (!g) return;
    const r = this.charts[0].canvas.getBoundingClientRect();
    const f = (e.clientX - r.left - g.L) / g.pw;
    if (f < 0 || f > 1) { this.hoverT = null; this.tip.hidden = true; return this.draw(); }
    this.hoverT = g.t0 + f * (g.t1 - g.t0);
    const row = nearest(this.rows, this.hoverT);
    this.tip.hidden = !row;
    if (row) {
      const st = STATE_STYLE[row.state] || STATE_STYLE.idle;
      this.tip.innerHTML = `<strong>${clock(row.t)}</strong><span><i style="background:${st[0]}"></i>${st[1]}</span>
        <span><i style="background:${SERIES.recv}"></i>Received <b>${num(row.recv_fps, 1)}</b> fps</span>
        <span><i style="background:${SERIES.proc}"></i>Analysed <b>${num(row.proc_fps, 1)}</b> fps</span>
        <span>Frame gap <b>${num(row.gap_ms)}</b> ms</span><span>Bitrate <b>${num(row.kbps)}</b> kbit/s</span>
        <span>Model time <b>${num(row.infer_ms, 1)}</b> ms</span>`;
      const host = this.el.getBoundingClientRect();
      const left = Math.min(Math.max(e.clientX - host.left + 14, 8), host.width - 190);
      this.tip.style.left = `${left}px`;
      this.tip.style.top = `${e.clientY - host.top + 14}px`;
    }
    this.draw();
  }

  update(src, hb, fields, states, now, windowS) {
    const iT = fields.indexOf("t"), iS = fields.indexOf("state");
    this.rows = (hb?.samples || []).map((s) => {
      const o = {};
      fields.forEach((f, i) => (o[f] = s[i]));
      o.t = s[iT]; o.state = states[s[iS]];
      return o;
    });
    this.t1 = now; this.t0 = now - windowS;
    const st = src.status || {}, cur = this.rows.at(-1);
    const stateKey = cur?.state || (st.state === "running" && st.idle ? "static" : st.state) || "idle";
    const style = STATE_STYLE[stateKey] || STATE_STYLE.idle;
    $("h3", this.el).textContent = src.name;
    $(".badge", this.el).innerHTML = `<i style="background:${style[0]}"></i>${style[1]}`;
    $(".test", this.el).hidden = src.kind !== "stream";
    const last60 = this.rows.filter((r) => r.t >= now - 60 && r.gap_ms != null);
    const worst = last60.length ? Math.max(...last60.map((r) => r.gap_ms)) : null;
    const recv = this.rows.filter((r) => r.t >= now - 10 && (r.state === "running" || r.state === "static"));
    const avg = (k) => (recv.length ? recv.reduce((a, r) => a + (r[k] || 0), 0) / recv.length : null);
    const procShare = avg("recv_fps") ? Math.round((avg("proc_fps") / avg("recv_fps")) * 100) : null;
    const tiles = [
      ["Receiving", num(avg("recv_fps"), 1), "fps"],
      ["Analysed", procShare == null ? "—" : `${Math.min(procShare, 100)}`, "% of frames", "Frames that went through the models. Low while the scene is static."],
      ["Bitrate", cur?.kbps != null ? num(avg("kbps")) : "—", "kbit/s"],
      ["Worst gap, 1 min", num(worst), "ms"],
      ["Up for", hb?.uptime_s != null && (stateKey === "running" || stateKey === "static") ? fmtDur(hb.uptime_s) : "—", ""],
      ["Reconnects", num(hb?.reconnects ?? 0), ""],
      ["Available, 1 min", hb?.availability_60s != null ? num(hb.availability_60s * 100) : "—", "%"],
    ];
    $(".tiles", this.el).innerHTML = tiles.map(([k, v, u, tip]) => `<div class="tile2" ${tip ? `title="${esc(tip)}"` : ""}><span>${k}</span><b>${v}</b><em>${v === "—" ? "" : u}</em></div>`).join("")
      + (st.state === "reconnecting" || st.state === "error" ? `<p class="net-error">${esc(st.error || "")}</p>` : "");
    const seen = [...new Set(this.rows.map((r) => r.state))];
    $(".legend.states", this.el).innerHTML = seen.map((k) => { const s = STATE_STYLE[k] || STATE_STYLE.idle; return `<span><i style="background:${s[0]}"></i>${s[1]}</span>`; }).join("");
    this.charts[2].el.hidden = !this.rows.some((r) => r.kbps != null);
    $(".plots", this.el).classList.toggle("nodata", !this.rows.length);
    this.draw();
    if (!$(".table-wrap", this.el).hidden) this.table();
  }

  draw() {
    if (this.t0 == null) return;
    drawStrip(this.strip, this.rows, this.t0, this.t1, this.hoverT);
    for (const ch of this.charts) if (!ch.el.hidden) ch.draw(this.rows, this.t0, this.t1, this.hoverT);
  }

  table() {
    const rows = this.rows.slice(-60).reverse();
    $(".table-wrap", this.el).innerHTML = `<table><tr><th>Time</th><th>State</th><th class="n">Received fps</th><th class="n">Analysed fps</th><th class="n">Gap ms</th><th class="n">kbit/s</th><th class="n">Model ms</th><th class="n">Reconnects</th></tr>${
      rows.map((r) => `<tr><td>${clock(r.t)}</td><td>${(STATE_STYLE[r.state] || STATE_STYLE.idle)[1]}</td><td class="n">${num(r.recv_fps, 1)}</td><td class="n">${num(r.proc_fps, 1)}</td><td class="n">${num(r.gap_ms)}</td><td class="n">${num(r.kbps)}</td><td class="n">${num(r.infer_ms, 1)}</td><td class="n">${r.reconnects}</td></tr>`).join("")}</table>`;
  }
}

export class Monitor {
  constructor() {
    this.cards = new Map();
    this.window = 300;
    this.active = false;
    $("#net-window").addEventListener("click", (e) => {
      if (!e.target.dataset.window) return;
      this.window = +e.target.dataset.window;
      $$("#net-window button").forEach((b) => b.classList.toggle("on", b === e.target));
      this.poll();
    });
    $("#net-adhoc").addEventListener("submit", (e) => {
      e.preventDefault();
      const url = $("#net-url").value.trim();
      if (url) runTest($("#net-adhoc-result"), { uri: url });
    });
    setInterval(() => this.active && this.poll(), 2000);
  }

  setActive(on) { this.active = on; if (on) this.poll(); }

  focus(id, test = false) {
    const card = this.cards.get(id);
    if (!card) return;
    card.el.scrollIntoView({ block: "start", behavior: "smooth" });
    if (test) card.test($(".test", card.el));
  }

  async poll() {
    let data;
    try { data = await (await fetch(`/api/heartbeat?window=${this.window}`)).json(); } catch { return; }
    const box = $("#net-cards");
    for (const [id, card] of this.cards) if (!state.sources.has(id)) { card.el.remove(); this.cards.delete(id); }
    for (const src of state.sources.values()) {
      let card = this.cards.get(src.id);
      if (!card) { card = new Card(src); this.cards.set(src.id, card); box.append(card.el); }
      card.update(src, data.sources[src.id], data.fields, data.states, data.now, this.window);
    }
    $("#net-empty").hidden = state.sources.size > 0;
  }
}
