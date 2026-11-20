// Icon buttons and tooltips.
//
// Any <button data-icon="name"> is turned into an icon that widens to show its name on hover or
// keyboard focus. Any element with data-tip gets a short tooltip (name, shortcut, one line of detail).
// Buttons added later (lists, cards) are picked up automatically, so templates only set attributes:
//   <button data-icon="edit" data-tip="Change this camera's settings">Edit</button>

const P = {   // 24×24, stroked with currentColor
  cameras: '<rect x="3" y="7" width="12" height="10" rx="2"/><path d="M15 10.5 21 7v10l-6-3.5"/>',
  live: '<rect x="3.5" y="3.5" width="7" height="7" rx="1.2"/><rect x="13.5" y="3.5" width="7" height="7" rx="1.2"/><rect x="3.5" y="13.5" width="7" height="7" rx="1.2"/><rect x="13.5" y="13.5" width="7" height="7" rx="1.2"/>',
  map: '<path d="M9 4 3 6.5v13.5l6-2.5 6 2.5 6-2.5V4l-6 2.5z"/><path d="M9 4v13.5M15 6.5V20"/>',
  health: '<path d="M2.5 12h4.5l2.2-6.5 4.2 13 2.2-6.5h6"/>',
  settings: '<path d="M4 6h9M18 6h2M4 12h3M12 12h8M4 18h11M19 18h1"/><circle cx="15.5" cy="6" r="2.2"/><circle cx="9.5" cy="12" r="2.2"/><circle cx="17" cy="18" r="2.2"/>',
  guide: '<circle cx="12" cy="12" r="9"/><path d="m15.5 8.5-2 5-5 2 2-5z"/>',
  masks: '<path d="M5.5 15.5 7 9l5-3 5.5 2.5L19 14l-5 5-6-.5z" fill="currentColor" fill-opacity=".28"/>',
  boxes: '<path d="M4 9V6h4M16 6h4v3M20 15v3h-4M8 18H4v-3"/>',
  trails: '<path d="M4 18c5 0 3-11 8-11s3 7 8 7"/><circle cx="20" cy="14" r="1.6" fill="currentColor"/>',
  plates: '<rect x="2.5" y="7.5" width="19" height="9" rx="1.8"/><path d="M6.5 12h2M11 12h2M15.5 12h2"/>',
  labels: '<path d="M3.5 5.5h8l9 6.5-9 6.5h-8z"/><circle cx="8" cy="12" r="1.4" fill="currentColor"/>',
  activity: '<path d="M8 6h12M8 12h12M8 18h12"/><circle cx="4" cy="6" r="1.2" fill="currentColor"/><circle cx="4" cy="12" r="1.2" fill="currentColor"/><circle cx="4" cy="18" r="1.2" fill="currentColor"/>',
  add: '<path d="M12 5v14M5 12h14"/>',
  fit: '<path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5"/>',
  flows: '<path d="M4 8h14l-3.5-3.5M20 16H6l3.5 3.5"/>',
  edit: '<path d="m4 20 4.5-1L20 7.5 16.5 4 5 15.5z"/><path d="m14 6.5 3.5 3.5"/>',
  start: '<path d="M7.5 5v14l11-7z" fill="currentColor" fill-opacity=".25"/>',
  stop: '<rect x="6.5" y="6.5" width="11" height="11" rx="1.5" fill="currentColor" fill-opacity=".25"/>',
  remove: '<path d="M4.5 7h15M10 7V4.5h4V7M6.5 7l1 13h9l1-13M10 11v5.5M14 11v5.5"/>',
  table: '<rect x="3.5" y="5" width="17" height="14" rx="1.5"/><path d="M3.5 10h17M3.5 14.5h17M10 5v14"/>',
  pin: '<path d="M12 21s-6.5-6-6.5-11a6.5 6.5 0 0 1 13 0c0 5-6.5 11-6.5 11z"/><circle cx="12" cy="10" r="2.3"/>',
  save: '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
  wand: '<path d="m4 20 11-11M13.5 7.5l3 3"/><path d="M18 3v3M16.5 4.5h3M20 10v2M19 11h2M9 3.5v2M8 4.5h2"/>',
  group: '<path d="M3.5 7.5a2 2 0 0 1 2-2h4l2 2.5h7a2 2 0 0 1 2 2v7.5a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/><path d="M12 11.5v5M9.5 14h5"/>',
  folder: '<path d="M3.5 7.5a2 2 0 0 1 2-2h4l2 2.5h7a2 2 0 0 1 2 2v7.5a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>',
  film: '<rect x="3.5" y="4.5" width="17" height="15" rx="2"/><path d="M8 4.5v15M16 4.5v15M3.5 9.5H8M3.5 14.5H8M16 9.5h4.5M16 14.5h4.5"/>',
  screen: '<rect x="3" y="4.5" width="18" height="12" rx="1.8"/><path d="M9 20h6M12 16.5V20"/>',
  all: '<path d="m12 3.5 8.5 4.5L12 12.5 3.5 8z"/><path d="m3.5 12 8.5 4.5 8.5-4.5M3.5 16l8.5 4.5 8.5-4.5"/>',
  car: '<path d="M4.5 16.5v-4l2-5.5h11l2 5.5v4"/><path d="M3.5 12.5h17v4.5h-17zM6 17v2M18 17v2"/><path d="M7 14.7h1.5M15.5 14.7H17"/>',
  filter: '<path d="M4 5.5h16l-6.2 7.5v5.5l-3.6-1.8V13z"/>',
  back: '<path d="M10 6 4 12l6 6M4.5 12H20"/>',
  loop: '<path d="M17 3.5 20.5 7 17 10.5"/><path d="M20.5 7H8a4.5 4.5 0 0 0-4.5 4.5v1M7 20.5 3.5 17 7 13.5"/><path d="M3.5 17H16a4.5 4.5 0 0 0 4.5-4.5v-1"/>',
  review: '<rect x="3" y="4" width="18" height="11" rx="1.8"/><path d="m10 7.3 4.5 2.2-4.5 2.2z" fill="currentColor"/><path d="M3 19h18M7 17.5v3M13 17.5v3M18 17.5v3"/>',
  rename: '<path d="M4 7V5h10v2M9 5v14M7 19h4"/><path d="M17 9v10M15 9h4M15 19h4"/>',
};

export const icon = (name) =>
  `<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${P[name] || ""}</svg>`;

function decorate(btn) {
  if (btn.dataset.iconDone === btn.dataset.icon && btn.querySelector(".ico")) return;
  const keep = [...btn.children].filter((c) => !c.matches(".ico, .lbl"));       // e.g. a count badge
  const label = btn.dataset.label || [...btn.childNodes].filter((n) => n.nodeType === 3).map((n) => n.textContent).join("").trim()
    || btn.querySelector(".lbl")?.textContent || "";
  btn.dataset.label = label;
  btn.dataset.iconDone = btn.dataset.icon;
  btn.setAttribute("aria-label", label);
  if (btn.title && !btn.dataset.tip) btn.dataset.tip = btn.title;   // one tooltip, not two
  btn.removeAttribute("title");
  btn.innerHTML = `${icon(btn.dataset.icon)}<span class="lbl">${label.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]))}</span>`;
  btn.append(...keep);
}

function decorateAll(root) {
  if (root.nodeType !== 1) return;
  if (root.matches?.("button[data-icon]")) decorate(root);
  root.querySelectorAll?.("button[data-icon]").forEach(decorate);
}

// ---------------------------------------------------------------- tooltip
let tipEl, timer, current;
function showTip(el) {
  const name = el.dataset.label || el.getAttribute("aria-label") || "", key = el.dataset.key, text = el.dataset.tip;
  if (!name && !text) return;
  tipEl.innerHTML = `<div class="tt-head"><b></b>${key ? "<kbd></kbd>" : ""}</div>${text ? "<p></p>" : ""}`;
  tipEl.querySelector("b").textContent = name || text;
  if (key) tipEl.querySelector("kbd").textContent = key;
  if (text && name) tipEl.querySelector("p").textContent = text; else tipEl.querySelector("p")?.remove();
  tipEl.hidden = false;
  const r = el.getBoundingClientRect(), t = tipEl.getBoundingClientRect();
  const below = r.bottom + 8 + t.height < window.innerHeight;
  tipEl.style.left = `${Math.round(Math.min(Math.max(r.left + r.width / 2 - t.width / 2, 8), window.innerWidth - t.width - 8))}px`;
  tipEl.style.top = `${Math.round(below ? r.bottom + 8 : r.top - t.height - 8)}px`;
}
function hideTip() { clearTimeout(timer); current = null; if (tipEl) tipEl.hidden = true; }

export function initIcons() {
  decorateAll(document.body);
  new MutationObserver((muts) => {
    for (const m of muts) {
      if (m.type === "attributes") { if (m.target.matches?.("button[data-icon]")) decorate(m.target); }
      else m.addedNodes.forEach(decorateAll);
    }
  }).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["data-icon"] });

  tipEl = document.createElement("div");
  tipEl.className = "tooltip";
  tipEl.setAttribute("role", "tooltip");
  tipEl.hidden = true;
  document.body.append(tipEl);
  const enter = (e, delay) => {
    const el = e.target.closest?.("[data-tip], button[data-icon]");
    if (!el || el === current) return;
    hideTip();
    current = el;
    timer = setTimeout(() => current === el && document.contains(el) && showTip(el), delay);
  };
  document.addEventListener("mouseover", (e) => enter(e, 450));
  document.addEventListener("mouseout", (e) => { if (current && !current.contains(e.relatedTarget)) hideTip(); });
  document.addEventListener("focusin", (e) => { if (e.target.matches?.(":focus-visible")) enter(e, 150); });
  document.addEventListener("focusout", hideTip);
  document.addEventListener("mousedown", hideTip, true);
  document.addEventListener("keydown", hideTip, true);
  window.addEventListener("scroll", hideTip, true);
}
