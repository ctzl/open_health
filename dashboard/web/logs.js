// open_oura activity logs — renders GET /api/logs (the dashboard's in-memory
// activity ring: syncs, torch model runs, live-HR sessions, feature toggles).
// Newest first; details expand on click; auto-refreshes while the tab is open.
"use strict";

const $ = (id) => document.getElementById(id);

function h(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v);
    }
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return n;
}
const clear = (node) => { node.replaceChildren(); return node; };

const KIND_LABEL = { sync: "sync", model: "model", "live-hr": "live-hr", feature: "feature" };
const AUTO_MS = 4000;

let kind = "";        // "" = all
let auto = true;
let timer = null;
let lastSeen = 0;     // newest ts rendered — used only for the "new entries" hint

function fmtTime(ts) {
  const d = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

async function getLogs() {
  const r = await fetch(`/api/logs?kind=${encodeURIComponent(kind)}&limit=300`, {
    headers: { "X-Oura-Dash": "1" },
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()).entries || [];
}

function entryRow(e) {
  const detail = String(e.detail || "");
  const first = detail.split("\n")[0];
  const rest = detail.slice(first.length).trim();
  const row = h("div", { class: `lg-row lg-${e.status}` },
    h("span", { class: "lg-ts" }, fmtTime(e.ts)),
    h("span", { class: `lg-kind lg-kind-${e.kind}` }, KIND_LABEL[e.kind] || e.kind),
    h("span", { class: `lg-status lg-status-${e.status}` }, e.status),
    h("span", { class: "lg-summary" }, first),
  );
  if (rest) {
    const pre = h("pre", { class: "lg-detail", hidden: true }, rest);
    row.append(h("button", {
      class: "lg-expand",
      onclick: (ev) => { pre.hidden = !pre.hidden; ev.currentTarget.textContent = pre.hidden ? "details +" : "details −"; },
    }, "details +"), pre);
  }
  return row;
}

async function render() {
  const status = $("lg-status");
  try {
    const entries = await getLogs();
    const list = clear($("lg-list"));
    if (!entries.length) {
      list.append(h("div", { class: "lg-empty" },
        "Nothing logged yet. Trigger a sync (or open the dashboard and let it load) and entries appear here."));
    } else {
      for (const e of entries) list.append(entryRow(e));
    }
    const fresh = entries.filter((e) => e.ts > lastSeen).length;
    if (fresh && lastSeen) status.textContent = `${entries.length} entries · ${fresh} new`;
    else status.textContent = `${entries.length} entries`;
    if (entries.length) lastSeen = Math.max(lastSeen, ...entries.map((e) => e.ts));
  } catch (err) {
    status.textContent = `log fetch failed: ${err.message}`;
  }
}

function setAuto(on) {
  auto = on;
  if (timer) { clearInterval(timer); timer = null; }
  if (auto) timer = setInterval(render, AUTO_MS);
}

document.querySelectorAll(".lg-filter").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".lg-filter").forEach((b) => b.classList.toggle("is-on", b === btn));
    kind = btn.dataset.kind;
    render();
  });
});
$("lg-refresh").addEventListener("click", render);
$("lg-auto").addEventListener("change", (ev) => setAuto(ev.target.checked));
$("lg-clear").addEventListener("click", async () => {
  if (!confirm("Clear the activity log?")) return;
  await fetch("/api/logs/clear", { method: "POST", headers: { "X-Oura-Dash": "1" } });
  lastSeen = 0;
  render();
});

render();
setAuto(true);
