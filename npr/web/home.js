// Home: the jobs this app does, each with what it still needs and one button for the next step.
// A scenario is { title, what, check(facts) → { ready, status, action, label, also? } }.
import { $, esc } from "./app.js";

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

export const SCENARIOS = [
  {
    id: "live", title: "Watch cameras live",
    what: "A wall of live tiles with vehicles outlined, tracked and their plates read as they pass.",
    check: (f) => f.cameras === 0
      ? { status: "Needs a camera", action: "add", label: "Add a camera" }
      : { ready: true, status: `${f.camerasRunning} of ${plural(f.cameras, "camera")} running`, action: "live", label: "Open the live wall" },
  },
  {
    id: "gate", title: "Log plates at a gate or driveway",
    what: "One camera pointed at an entrance, with a running list of every plate read and when.",
    check: (f) => f.cameras === 0
      ? { status: "Needs a camera aimed at the entrance", action: "add-stream", label: "Add the gate camera" }
      : { ready: true, status: f.plates ? `${plural(f.plates, "plate")} read so far` : "Ready: no plates read yet", action: "gate", label: "Open the plate log" },
  },
  {
    id: "review", title: "Review recorded footage",
    what: "Drop in video files or photos from an incident and get the vehicles and plates in them, faster than real time.",
    check: (f) => f.files === 0
      ? { ready: true, status: "Ready whenever you have files", action: "add-file", label: "Add recordings" }
      : { ready: true, status: `${plural(f.files, "recording")} added`, action: "results", label: "See what was found", also: ["add-file", "Add more recordings"] },
  },
  {
    id: "follow", title: "Follow a vehicle across cameras",
    what: "Search a plate or click a vehicle and see every camera it passed, in order, drawn as a route on the map.",
    check: (f) => f.total < 2
      ? { status: `Needs at least 2 cameras or recordings (you have ${f.total})`, action: "add", label: f.total ? "Add another camera" : "Add a camera" }
      : f.placed < 2
        ? { status: `Needs 2 cameras on the map (${f.placed} placed)`, action: "place", label: "Place cameras on the map" }
        : { ready: true, status: f.linked ? `${plural(f.linked, "vehicle")} seen on more than one camera` : "Ready: waiting for a vehicle to pass two cameras", action: "follow", label: "Find a vehicle" },
  },
  {
    id: "patterns", title: "Understand movement patterns",
    what: "Which way traffic flows between cameras, how long it takes, typical speeds, common routes and repeat visitors.",
    check: (f) => f.total < 2
      ? { status: "Needs at least 2 cameras", action: "add", label: f.total ? "Add another camera" : "Add a camera" }
      : f.placed < 2
        ? { status: "Needs 2 cameras on the map for distances and speeds", action: "place", label: "Place cameras on the map" }
        : { ready: true, status: f.flows ? `${plural(f.flows, "flow")} measured between cameras` : "Ready: builds up as vehicles pass", action: "patterns", label: "See patterns" },
  },
  {
    id: "health", title: "Check that cameras are connected",
    what: "A heartbeat for every connection, and a step-by-step test that says exactly why a camera will not connect.",
    check: (f) => f.streams === 0
      ? { ready: true, status: "You can test an address before adding it", action: "health", label: "Test a camera address" }
      : f.down
        ? { ready: true, warn: true, status: `${plural(f.down, "camera")} not connected`, action: "health", label: "See what is wrong" }
        : { ready: true, status: `${plural(f.streams, "network camera")}, all connected`, action: "health", label: "Open connection health" },
  },
];

export class Home {
  constructor({ facts, act }) {
    this.facts = facts; this.act = act;
    $("#view-home").addEventListener("click", (e) => { const a = e.target.closest("[data-action]")?.dataset.action; if (a) this.act(a); });
  }

  render() {
    const el = $("#view-home");
    if (el.hidden) return;
    const f = this.facts();
    const first = f.total === 0;
    const steps = [
      [f.modelReady, "Models installed", f.modelReady ? "Vehicle and plate models are ready." : "Run “uv run python -m npr setup” once in the project folder.", null, null],
      [f.total > 0, "Add a camera or recording", f.total ? `${plural(f.total, "source")} added.` : "Guided: connect, place, name. About a minute.", "add", f.total ? "Add another" : "Add a camera"],
      [f.setupDone, "Choose general settings", f.setupDone ? "Done. Change them any time." : "Three questions: what to track, when to analyse, what to do if a camera drops.", "setup", f.setupDone ? "Review" : "Start"],
      [f.total > 0 && f.placed === f.total, "Place cameras on the map", f.total === 0 ? "After you add a camera." : f.placed === f.total ? "Every camera is placed." : `${f.total - f.placed} of ${f.total} not placed yet. Needed to follow vehicles between cameras.`, f.total && f.placed < f.total ? "place" : null, "Place them"],
    ];
    const todo = steps.filter((s) => !s[0]).length;
    const html = `
      <div class="home">
        <header class="home-head">
          <h1>${first ? "Welcome. Let’s get your first camera in." : "What do you want to do?"}</h1>
          <p>${first ? "Add a camera or some recorded footage, and the app starts reading plates and tracking vehicles straight away."
            : `${plural(f.total, "source")} · ${f.running} running · ${f.placed} on the map · ${plural(f.plates, "plate")} read`}</p>
          ${first ? `<div class="home-cta"><button class="primary big" data-action="add">Add a camera</button><button class="big" data-action="add-file">Add recorded footage</button></div>` : ""}
        </header>

        <section class="home-setup ${todo ? "" : "complete"}">
          <h2>${todo ? `Getting set up · ${steps.length - todo} of ${steps.length} done` : "Set up complete"}</h2>
          <ol>${steps.map(([ok, title, text, action, label]) => `<li class="${ok ? "ok" : ""}"><i>${ok ? "✓" : ""}</i>
            <div><strong>${title}</strong><span>${esc(text)}</span></div>
            ${action ? `<button data-action="${action}" class="${ok ? "ghost" : ""}">${label}</button>` : ""}</li>`).join("")}</ol>
        </section>

        <h2 class="home-h">${first ? "What you can do with it" : "Scenarios"}</h2>
        <div class="scenarios">${SCENARIOS.map((s) => {
          const c = s.check(f);
          return `<article class="scenario ${c.ready ? "ready" : "blocked"}">
            <h3>${s.title}</h3><p>${s.what}</p>
            <div class="sc-status ${c.warn ? "warn" : ""}"><i></i>${esc(c.status)}</div>
            <div class="sc-actions"><button class="primary" data-action="${c.action}">${c.label}</button>
              ${c.also ? `<button class="ghost" data-action="${c.also[0]}">${c.also[1]}</button>` : ""}</div>
          </article>`;
        }).join("")}</div>
      </div>`;
    if (this.last !== html) { this.last = html; el.innerHTML = html; }
  }
}
