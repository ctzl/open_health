// Shared profile switcher for the blood and DNA pages (docs/profiles.md).
// A "profile" is a ring serial: the dashboard's ring switcher already persists
// the choice in the `oura_ring` cookie, so these pages only read it, mirror it
// into their own header select, and stamp API calls with ?serial=.
"use strict";

const PROFILE_COOKIE = "oura_ring";
const readProfileCookie = () =>
  document.cookie.split(/;\s*/).find((c) => c.startsWith(PROFILE_COOKIE + "="))?.slice(PROFILE_COOKIE.length + 1) || "";

let PROFILE_SERIAL = readProfileCookie();
let PROFILE_RINGS = [];

// Query-string suffix for /api/* calls ("" = server default: everything).
const profileQS = () =>
  PROFILE_SERIAL ? `&serial=${encodeURIComponent(PROFILE_SERIAL)}` : "";

// Full URL for a path-only /api call: profileUrl("/api/blood/report").
const profileUrl = (path) =>
  PROFILE_SERIAL ? `${path}?serial=${encodeURIComponent(PROFILE_SERIAL)}` : path;

const profileName = (serial) => {
  const r = PROFILE_RINGS.find((x) => x.serial === serial);
  return (r && r.name) || (serial ? "Ring " + String(serial).slice(-4) : "All profiles");
};

function selectProfile(serial) {
  PROFILE_SERIAL = serial || "";
  document.cookie = `${PROFILE_COOKIE}=${encodeURIComponent(serial || "")}; path=/; max-age=${serial ? 31536000 : 0}; SameSite=Lax`;
}

// Populate #profile-sel when present; hidden with fewer than two rings.
async function initProfileSwitcher() {
  const sel = document.getElementById("profile-sel");
  try {
    const j = await (await fetch("/api/rings")).json();
    if (Array.isArray(j)) PROFILE_RINGS = j;
  } catch (_) { /* no registry; stay hidden */ }
  if (!sel) return;
  if (PROFILE_RINGS.length < 2) { sel.hidden = true; return; }
  sel.hidden = false;
  sel.replaceChildren();
  for (const r of PROFILE_RINGS) {
    const o = document.createElement("option");
    o.value = r.serial;
    o.textContent = r.name;
    sel.append(o);
  }
  if (!PROFILE_SERIAL || !PROFILE_RINGS.some((r) => r.serial === PROFILE_SERIAL)) {
    const first = [...PROFILE_RINGS].sort((a, b) => a.name.localeCompare(b.name))[0];
    selectProfile(first.serial);
  }
  sel.value = PROFILE_SERIAL;
  sel.addEventListener("change", () => {
    selectProfile(sel.value);
    location.reload(); // re-render this page under the new profile
  });
}
