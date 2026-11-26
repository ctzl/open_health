// open_oura ring diagnostics — renders GET /api/ring-diagnostics: one card per
// registered ring. Identity + key provenance, 14-day wearing-coverage heatmap
// (hours with heart-rate/motion samples), recording gaps classified by cause
// (power-off / charging / off-finger), per-stream last-seen, and the
// activity-log entries that mention the ring.
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

const DAY = 86400;

function fmtDateTime(ts) {
  const d = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}
function fmtAgo(ts) {
  if (!ts) return "never";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < DAY) return `${(s / 3600).toFixed(1)} h ago`;
  return `${(s / DAY).toFixed(1)} d ago`;
}
function fmtDur(s) {
  if (s < 3600) return `${Math.round(s / 60)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

async function getDiag() {
  const r = await fetch("/api/ring-diagnostics", { headers: { "X-Oura-Dash": "1" } });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

function kv(k, v, cls) {
  return h("div", { class: "rd-kv" + (cls ? " " + cls : "") },
    h("div", { class: "rd-k" }, k),
    h("div", { class: "rd-v" }, v));
}

// 14-day wearing-coverage heatmap: one cell per local hour, green = the ring
// produced heart-rate/motion samples that hour (server already mapped ring
// clock to wall clock via time_sync anchors; late sync delivery cannot fake
// coverage because buffered replays keep their original sample times).
// Hours that START past the ring's last sync are hatched: the ring may be
// recording fine, we simply have no data yet — that is unknown, not an
// off-hour. The hour containing the last sync itself stays NOT hatched:
// its data up to the sync was delivered, so silence there is known.
function coverageHeatmap(ring) {
  const live = new Set((ring.coverage || []).map((c) => c.h));
  const nowH = Math.floor(Date.now() / 1000 / 3600);
  const observedUntil = ring.observed_until_s || 0;
  const wrap = h("div", { class: "rd-heat" });
  const midnight = new Date();
  midnight.setHours(0, 0, 0, 0);
  for (let i = 13; i >= 0; i--) {
    const day = new Date(midnight);
    day.setDate(day.getDate() - i);
    const row = h("div", { class: "rd-hrow" },
      h("span", { class: "rd-hlab" }, `${day.getMonth() + 1}/${day.getDate()}`));
    let on = 0;
    for (let hr = 0; hr < 24; hr++) {
      const t = new Date(day);
      t.setHours(hr, 0, 0, 0);
      const he = Math.floor(t.getTime() / 1000 / 3600);
      const future = he > nowH;
      const unknown = !future && he * 3600 >= observedUntil;
      const isOn = live.has(he);
      if (isOn) on++;
      row.append(h("i", {
        class: "rd-hx " + (future ? "rd-hx-off" : unknown && !isOn ? "rd-hx-unknown" : isOn ? "rd-hx-on" : "rd-hx-gap"),
        title: `${fmtDateTime(he * 3600)} — ${isOn ? "recording" : future ? "" : unknown ? "after last sync — unknown" : "no samples"}`,
      }));
    }
    row.append(h("span", { class: "rd-hcount" }, `${on}h`));
    wrap.append(row);
  }
  return wrap;
}

const NS = "http://www.w3.org/2000/svg";
function sv(tag, attrs, ...kids) {
  const n = document.createElementNS(NS, tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) if (v != null) n.setAttribute(k, String(v));
  for (const kid of kids.flat()) if (kid != null) n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return n;
}

// Battery charge curve: the ring's fuel gauge emits battery_level_changed
// readings timed on the ring clock (server maps them through time_sync).
// Green bands = charge sessions (pct rising, merged across <30 min plateaus);
// red/gray bands reuse the gap classification; dashed line = raw pack mV.
function batteryChart(ring) {
  const pts = (ring.battery || []).filter((p) => Number.isFinite(p.pct));
  if (pts.length < 2) {
    return h("p", { class: "rd-ok" }, "No battery readings in this window.");
  }
  const now = Math.floor(Date.now() / 1000);
  const mid = new Date();
  mid.setHours(0, 0, 0, 0);
  const x0 = mid.getTime() / 1000 - 13 * DAY, x1 = now;
  const W = 1000, H = 250, ML = 34, MR = 50, MT = 12, MB = 22;
  const px = (t) => ML + ((Math.min(x1, Math.max(x0, t)) - x0) / (x1 - x0)) * (W - ML - MR);
  const py = (p) => MT + ((100 - p) / 100) * (H - MT - MB);
  const mvs = pts.filter((p) => p.mv > 0).map((p) => p.mv);
  let mvLo = Math.min(...mvs), mvHi = Math.max(...mvs);
  if (!(mvHi > mvLo)) { mvLo -= 50; mvHi += 50; }
  const pm = (mv) => MT + ((mvHi - mv) / (mvHi - mvLo)) * (H - MT - MB);

  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, class: "rd-batt-svg", role: "img" });
  for (const p of [0, 25, 50, 75, 100]) {
    svg.append(sv("line", { x1: ML, x2: W - MR, y1: py(p), y2: py(p), stroke: "var(--line-soft)", "stroke-width": 1 }));
    svg.append(sv("text", { x: ML - 5, y: py(p) + 3, class: "rd-batt-lab", "text-anchor": "end" }, `${p}%`));
  }
  for (let i = 13; i >= 0; i--) {
    const d = new Date(mid);
    d.setDate(d.getDate() - i);
    const t = d.getTime() / 1000;
    if (t < x0) continue;
    const X = px(t);
    svg.append(sv("line", { x1: X, x2: X, y1: MT, y2: H - MB, stroke: "var(--line-soft)", "stroke-width": 1, opacity: 0.5 }));
    svg.append(sv("text", { x: X + 2, y: H - 8, class: "rd-batt-lab" }, `${d.getMonth() + 1}/${d.getDate()}`));
  }
  for (const g of ring.gaps || []) {
    if (g.kind === "unknown") continue; // past the last sync: show nothing
    if (g.to_s <= x0 || g.from_s >= x1) continue;
    const what = `gap (${g.kind})`;
    svg.append(sv("rect", {
      x: px(g.from_s), y: MT,
      width: Math.max(1, px(g.to_s) - px(g.from_s)), height: H - MT - MB,
      class: "rd-batt-gap rd-batt-" + g.kind,
    }, sv("title", {}, `${what} ${fmtDateTime(g.from_s)} → ${fmtDateTime(g.to_s)}`)));
  }
  const sess = [];
  for (let i = 1; i < pts.length; i++) {
    if (pts[i].pct > pts[i - 1].pct && pts[i].t - pts[i - 1].t < 6 * 3600) {
      const cur = sess[sess.length - 1];
      if (cur && pts[i - 1].t - cur.e < 1800) { cur.e = pts[i].t; cur.p2 = pts[i].pct; }
      else sess.push({ s: pts[i - 1].t, e: pts[i].t, p1: pts[i - 1].pct, p2: pts[i].pct });
    }
  }
  for (const c of sess) {
    if (c.e <= x0 || c.s >= x1) continue;
    const durH = Math.max(0.1, (c.e - c.s) / 3600);
    svg.append(sv("rect", {
      x: px(c.s), y: MT, width: Math.max(1.5, px(c.e) - px(c.s)), height: H - MT - MB,
      class: "rd-batt-charge",
    }, sv("title", {}, `charging ${c.p1}% → ${c.p2}% in ${durH.toFixed(1)} h (${((c.p2 - c.p1) / durH).toFixed(0)}%/h)`)));
  }
  if (mvs.length > 1) {
    const dv = pts.filter((p) => p.mv > 0)
      .map((p, i) => `${i ? "L" : "M"}${px(p.t).toFixed(1)} ${pm(p.mv).toFixed(1)}`).join(" ");
    svg.append(sv("path", { d: dv, fill: "none", stroke: "var(--faint)", "stroke-width": 1, "stroke-dasharray": "3 3", opacity: 0.8 }));
    svg.append(sv("text", { x: W - MR + 4, y: pm(mvHi) + 3, class: "rd-batt-lab" }, `${mvHi}mV`));
    svg.append(sv("text", { x: W - MR + 4, y: pm(mvLo) + 3, class: "rd-batt-lab" }, `${mvLo}mV`));
  }
  const dp = pts.map((p, i) => `${i ? "L" : "M"}${px(p.t).toFixed(1)} ${py(p.pct).toFixed(1)}`).join(" ");
  svg.append(sv("path", { d: dp, fill: "none", stroke: "var(--accent)", "stroke-width": 1.6 }));

  // Hover crosshair: snapping to the nearest reading is the whole point — the
  // curve is event-driven, so the interesting datum is the change nearest the
  // hovered instant, not whatever the interpolated line pixel says.
  const cross = sv("line", {
    y1: MT, y2: H - MB, stroke: "var(--muted)", "stroke-width": 1,
    "stroke-dasharray": "2 2", visibility: "hidden",
  });
  const dot = sv("circle", {
    r: 4, fill: "var(--accent)", stroke: "var(--surface-2)", "stroke-width": 1.5,
    visibility: "hidden",
  });
  svg.append(cross, dot);
  const tip = h("div", { class: "rd-batt-tip", hidden: true });
  const hide = () => {
    tip.hidden = true;
    cross.setAttribute("visibility", "hidden");
    dot.setAttribute("visibility", "hidden");
  };
  const nearest = (t) => {
    let lo = 0, hi = pts.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (pts[m].t < t) lo = m; else hi = m; }
    return t - pts[lo].t <= pts[hi].t - t ? lo : hi;
  };
  svg.addEventListener("pointermove", (ev) => {
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const pnt = svg.createSVGPoint();
    pnt.x = ev.clientX;
    pnt.y = ev.clientY;
    const loc = pnt.matrixTransform(ctm.inverse());
    if (loc.x < ML || loc.x > W - MR) { hide(); return; }
    const t = x0 + ((loc.x - ML) / (W - ML - MR)) * (x1 - x0);
    const i = nearest(t);
    const q = pts[i];
    const prev = i > 0 ? pts[i - 1] : null;
    const inSess = sess.find((c) => q.t >= c.s && q.t <= c.e);
    const inGap = (ring.gaps || []).find((g) => q.t >= g.from_s && q.t <= g.to_s);
    cross.setAttribute("x1", loc.x);
    cross.setAttribute("x2", loc.x);
    cross.setAttribute("visibility", "");
    dot.setAttribute("cx", px(q.t));
    dot.setAttribute("cy", py(q.pct));
    dot.setAttribute("visibility", "");
    const age = now - q.t;
    const lines = [
      h("b", {}, `${q.pct}%${q.mv > 0 ? ` · ${q.mv} mV` : ""}`),
      h("span", {}, fmtDateTime(q.t) + (age > 2 * 3600 ? ` · nearest reading, ${fmtDur(age)} ago` : "")),
    ];
    if (prev) lines.push(h("span", {}, `Δ vs previous: ${q.pct - prev.pct >= 0 ? "+" : ""}${q.pct - prev.pct}% over ${fmtDur(q.t - prev.t)}`));
    if (inSess) lines.push(h("span", {}, `within charge session ${inSess.p1}% → ${inSess.p2}%`));
    if (inGap) lines.push(h("span", {}, `within ${inGap.kind} gap`));
    tip.replaceChildren(...lines);
    const rect = svg.getBoundingClientRect(), wrapRect = wrap.getBoundingClientRect();
    const sx = rect.width / W, sy = rect.height / H;
    let left = rect.left - wrapRect.left + loc.x * sx + 12;
    if (left > wrapRect.width - 235) left = rect.left - wrapRect.left + loc.x * sx - 247;
    tip.style.left = `${Math.max(0, left)}px`;
    tip.style.top = `${Math.max(0, rect.top - wrapRect.top + py(q.pct) * sy - 14)}px`;
    tip.hidden = false;
  });
  svg.addEventListener("pointerleave", hide);

  const last = pts[pts.length - 1];
  const minp = Math.min(...pts.map((p) => p.pct));
  const wrap = h("div", { class: "rd-batt" },
    h("div", { class: "rd-batt-sum" },
      `latest ${last.pct}%${last.mv > 0 ? ` (${last.mv}mV)` : ""} at ${fmtDateTime(last.t)} · ` +
      `${sess.length} charge session(s) · min ${minp}% in window · ${pts.length} readings`),
    svg,
    tip,
    h("div", { class: "rd-batt-legend" },
      "solid = battery % (left scale) · dashed = pack mV (right scale) · green band = charging · red/gray band = recording gap · hover the chart to read the nearest data point"));
  return wrap;
}

function card(ring) {
  const card = h("section", { class: "panel rd-card" },
    h("h2", {}, ring.name || ring.serial),
    h("p", { class: "rd-id" },
      `serial ${ring.serial}`,
      ring.mac ? ` · mac ${ring.mac}` : "",
      ring.firmware ? ` · fw ${ring.firmware}` : "",
      ring.api_version ? ` · api ${ring.api_version}` : ""));

  const kvs = h("div", { class: "rd-kvs" });
  const fresh = ring.last_sync_s && Date.now() / 1000 - ring.last_sync_s < 3 * DAY;
  kvs.append(kv("Last sync", `${ring.last_sync_s ? fmtDateTime(ring.last_sync_s) : "never"} (${fmtAgo(ring.last_sync_s)})`, fresh ? "" : "rd-warn"));
  kvs.append(kv("Last seen", fmtAgo(ring.last_seen_s)));
  if (ring.added_s) kvs.append(kv("Added", fmtDateTime(ring.added_s)));
  if (ring.key_source) kvs.append(kv("Key source", ring.key_source));
  if (ring.note) kvs.append(kv("Note", ring.note));
  card.append(kvs);

  const gaps = ring.gaps || [];
  card.append(h("h3", {},
    h("span", {}, "Recording gaps"),
    h("span", { class: "rd-sub" },
      "hours with no heart-rate/motion samples, anchored by the ring clock (time_sync). ",
      "power-off = the ring rebooted (a dead battery is off-finger time, not wear time); ",
      "charging = battery rose across the gap; off-finger = powered but recording nothing; ",
      "past the last sync nothing is knowable — that stretch is marked unknown, not off-finger.")));
  const ev = gaps.filter((g) => g.kind !== "unknown");
  const unk = gaps.filter((g) => g.kind === "unknown");
  if (!gaps.length) {
    card.append(h("p", { class: "rd-ok" }, "No empty hours over the last 14 days — the ring recorded continuously."));
  } else {
    const total = ev.reduce((a, g) => a + g.dur_s, 0);
    const worst = ev.length ? Math.max(...ev.map((g) => g.dur_s)) : 0;
    const KIND = {
      "power-off": "power-off", "charging": "charging", "idle": "off-finger",
      "unknown": "unknown (not synced)",
    };
    card.append(h("p", { class: "rd-stall-sum" },
      (ev.length
        ? `${ev.length} gap(s) totalling ${fmtDur(total)} · longest ${fmtDur(worst)}`
        : "No evidence-backed gaps in synced data") +
      (unk.length ? ` · ${fmtDur(unk.reduce((a, g) => a + g.dur_s, 0))} unknown since ${fmtDateTime(unk[0].from_s)} (no sync since)` : "")));
    const list = h("div", { class: "rd-gaps" });
    const ordered = [...gaps].reverse();
    const batt = (b) => (b && b.pct >= 0 ? `${b.pct}% ${b.mv}mV` : "no reading");
    const row = (g) => h("div", { class: "rd-gap" + (g.kind === "power-off" ? " rd-gap-poweroff" : "") },
      h("span", { class: "rd-stall-d", style: g.kind === "unknown" ? "color:var(--faint)" : "" }, fmtDur(g.dur_s)),
      h("div", {},
        h("span", { class: "rd-gap-kind rd-kind-" + g.kind }, KIND[g.kind] || g.kind),
        h("span", { class: "rd-gap-t" }, `${fmtDateTime(g.from_s)} → ${fmtDateTime(g.to_s)}`),
        h("div", { class: "rd-gap-batt" }, g.kind === "unknown"
          ? "no data delivered past the last sync — the ring may or may not have been recording"
          : `battery: ${batt(g.batt_before)} → ${batt(g.batt_after)}${g.boot ? " · ring rebooted in this gap" : ""}`)));
    for (const g of ordered.slice(0, 8)) list.append(row(g));
    if (ordered.length > 8) {
      const rest = h("div", { class: "rd-rest", hidden: true });
      for (const g of ordered.slice(8)) rest.append(row(g));
      const more = h("button", { class: "rd-more" }, `show ${ordered.length - 8} older`);
      more.addEventListener("click", () => {
        rest.hidden = !rest.hidden;
        more.textContent = rest.hidden ? `show ${ordered.length - 8} older` : "hide older";
      });
      list.append(rest, more);
    }
    card.append(list);
  }

  card.append(h("h3", {}, h("span", {}, "Battery charge"),
    h("span", { class: "rd-sub" }, "fuel-gauge readings timed on the ring clock · last 14 days · charge sessions and gap bands overlaid")));
  card.append(batteryChart(ring));

  card.append(h("h3", {}, h("span", {}, "Wearing coverage"), h("span", { class: "rd-sub" }, "one cell per hour · green = heart-rate/motion samples present · hatched = past last sync (unknown) · last 14 days")));
  card.append(coverageHeatmap(ring));

  const streams = ring.streams || [];
  if (streams.length) {
    card.append(h("h3", {}, h("span", {}, "Streams"), h("span", { class: "rd-sub" }, "oldest last-seen first — a stream that stopped is a clue")));
    const tbl = h("div", { class: "rd-streams" });
    for (const s of streams) {
      const staleH = (Date.now() / 1000 - s.last_s) / 3600;
      tbl.append(h("div", { class: "rd-stream" + (staleH > 24 ? " rd-stale" : "") },
        h("span", { class: "rd-s-name" }, s.name || `tag ${s.name}`),
        h("span", { class: "rd-s-last" }, fmtAgo(s.last_s)),
        h("span", { class: "rd-s-count" }, s.count.toLocaleString())));
    }
    card.append(tbl);
  }

  const log = ring.log || [];
  if (log.length) {
    card.append(h("h3", {}, h("span", {}, "Recent log entries for this ring")));
    const list = h("div", { class: "lg-list" });
    for (const e of log) {
      const detail = String(e.detail || "").split("\n")[0];
      list.append(h("div", { class: "lg-row" },
        h("span", { class: "lg-ts" }, fmtDateTime(e.ts)),
        h("span", { class: `lg-status lg-status-${e.status}` }, e.status),
        h("span", { class: "lg-summary" }, detail)));
    }
    card.append(list);
  }
  return card;
}

async function render() {
  const status = $("rd-status");
  status.textContent = "running…";
  try {
    const data = await getDiag();
    const rings = data.rings || [];
    clear($("rd-cards"));
    if (!rings.length) {
      status.textContent = data.error || "no rings registered";
      return;
    }
    status.textContent = `${rings.length} ring(s) · ${fmtDateTime(Math.floor(Date.now() / 1000))}`;
    for (const ring of rings) $("rd-cards").append(card(ring));
  } catch (e) {
    status.textContent = `error: ${e.message}`;
  }
}

$("rd-refresh").addEventListener("click", render);
render();
