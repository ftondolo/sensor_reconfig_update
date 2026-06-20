// Operator console (demo_v1) — the same BEV map, plus navigation controls.

const $ = (id) => document.getElementById(id);

$("detect-img").src = "/stream/detect";

// --------------------------------------------------------- map (clickable)
const bev = new BevMap($("map-canvas"));
let autoOn = false;
let followOn = false;

fetch("/api/map").then((r) => r.json()).then((m) => {
  bev.setMap(m);
  const s = m.rover_start || {};
  $("start-cell").textContent =
    `x=${(s.x / 1000).toFixed(1)} m, z=${(s.z / 1000).toFixed(1)} m`;
  initSpeed(m);
  initJogSpeed(m);
});

// --------------------------------------------------------- speed slider
// The selectable range + initial value come from the backend (NAV_SPEED_MIN/MAX
// and the live MAX_LINEAR). 'input' updates the label live; 'change' (release)
// POSTs the new cap, which the control loop applies on the next tick.
function initSpeed(m) {
  const el = $("speed-input");
  if (m.speed_min != null) el.min = m.speed_min;
  if (m.speed_max != null) el.max = m.speed_max;
  if (m.speed != null) el.value = m.speed;
  el.disabled = false;
  $("speed-val").textContent = parseFloat(el.value).toFixed(2);
  $("speed-range").textContent = `${parseFloat(el.min).toFixed(2)}–${parseFloat(el.max).toFixed(2)}`;
  el.oninput = () => { $("speed-val").textContent = parseFloat(el.value).toFixed(2); };
  el.onchange = async () => {
    const j = await post("/api/nav/speed", { mps: parseFloat(el.value) });
    if (j && j.ok && j.speed != null) {
      el.value = j.speed;                         // reflect the clamped value
      $("speed-val").textContent = parseFloat(j.speed).toFixed(2);
    }
  };
}

// --------------------------------------------------------- jog (Drive) speed
function initJogSpeed(m) {
  const el = $("jog-speed-input");
  if (m.jog_speed_min != null) el.min = m.jog_speed_min;
  if (m.jog_speed_max != null) el.max = m.jog_speed_max;
  if (m.jog_speed != null) el.value = m.jog_speed;
  el.disabled = false;
  $("jog-speed-val").textContent = parseFloat(el.value).toFixed(2);
  $("jog-speed-range").textContent = `${parseFloat(el.min).toFixed(2)}–${parseFloat(el.max).toFixed(2)}`;
  el.oninput = () => { $("jog-speed-val").textContent = parseFloat(el.value).toFixed(2); };
  el.onchange = async () => {
    const j = await post("/api/nav/jog_speed", { mps: parseFloat(el.value) });
    if (j && j.ok && j.jog_speed != null) {
      el.value = j.jog_speed;
      $("jog-speed-val").textContent = parseFloat(j.jog_speed).toFixed(2);
    }
  };
}

bev.onClick = (x, z) => {
  post("/api/nav/goto", { x: Math.round(x), z: Math.round(z) });
};

// --------------------------------------------------------- API helpers
async function post(url, body) {
  try {
    const r = await fetch(url, {
      method: "POST",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    const j = await r.json().catch(() => ({}));
    if (j.message || j.error) $("nav-msg-strip").textContent = j.message || j.error;
    return j;
  } catch (e) {
    $("nav-msg-strip").textContent = "request failed";
    return null;
  }
}

$("btn-go").onclick = () => post("/api/nav/go");
$("btn-cancel").onclick = () => post("/api/nav/cancel");
$("btn-home").onclick = async () => {
  // Return to start disables follow/auto server-side; reflect that in the UI.
  const j = await post("/api/nav/home");
  if (j && j.nav) { setFollow(j.nav.follow); setAuto(j.nav.auto); }
};
$("btn-reset").onclick = () => post("/api/nav/reset_pose", {});
$("btn-reload").onclick = async () => {
  const j = await post("/api/nav/reload_map");
  if (j && j.ok && j.map) {
    bev.setMap(j.map);
    bev.clearTrail();
    const s = j.map.rover_start || {};
    $("start-cell").textContent =
      `x=${(s.x / 1000).toFixed(1)} m, z=${(s.z / 1000).toFixed(1)} m`;
  }
};
$("btn-auto").onclick = async () => {
  const j = await post("/api/nav/auto", { enabled: !autoOn });
  if (j && j.ok) setAuto(j.auto);
};
$("btn-follow").onclick = async () => {
  const j = await post("/api/nav/follow", { enabled: !followOn });
  if (j && j.ok) { setFollow(j.follow); if (j.nav) setAuto(j.nav.auto); }
};

function setAuto(v) {
  autoOn = !!v;
  const b = $("btn-auto");
  b.textContent = "Auto: " + (autoOn ? "on" : "off");
  b.className = "btn" + (autoOn ? " toggle-on" : "");
}

function setFollow(v) {
  followOn = !!v;
  const b = $("btn-follow");
  b.textContent = "🎯 Follow: " + (followOn ? "on" : "off");
  b.className = "btn" + (followOn ? " toggle-on" : "");
}

// Hold-to-move drive pad: send /api/nav/jog every 200 ms while pressed (the
// backend dead-man is 0.5 s, so motion stops shortly after refreshes cease),
// and an explicit jog_stop on release for an immediate stop.
let jogTimer = null;
function jogStart(x, z) {
  if (jogTimer) return;
  const send = () => post("/api/nav/jog", { x, z });
  send();
  jogTimer = setInterval(send, 200);
}
function jogStop() {
  if (!jogTimer) return;
  clearInterval(jogTimer);
  jogTimer = null;
  post("/api/nav/jog_stop");
}
document.querySelectorAll(".nudge .jog").forEach((b) => {
  b.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    if (!b.disabled) jogStart(parseInt(b.dataset.x), parseInt(b.dataset.z));
  });
  for (const ev of ["pointerup", "pointerleave", "pointercancel"]) {
    b.addEventListener(ev, jogStop);
  }
  b.addEventListener("contextmenu", (e) => e.preventDefault());
});
window.addEventListener("blur", jogStop);

// --------------------------------------------------------- telemetry
function fmtMM(p) {
  if (!p || p.x == null) return "—";
  return `(${(p.x / 1000).toFixed(2)}, ${(p.z / 1000).toFixed(2)}) m`;
}

function updateNav(nav) {
  if (!nav) return;
  bev.setNav(nav);
  const st = (nav.status || "idle").toUpperCase();
  $("rover-xy").textContent = fmtMM(nav.rover);
  $("target-xy").textContent = fmtMM(nav.target);
  $("goal-xy").textContent = fmtMM(nav.goal) + (nav.goal && nav.goal.adjusted ? " (adj)" : "");
  $("nav-leg").textContent = nav.path ? `${nav.leg}/${nav.path.length - 1}` : "—";
  $("pose-src").textContent = nav.mock ? "MOCK" : "T265";
  if (nav.message) $("nav-msg-strip").textContent = nav.message;

  const badge = $("nav-badge");
  badge.textContent = st;
  badge.className = "badge " +
    (nav.status === "moving" ? "mode" :
     nav.status === "arrived" ? "on" :
     ["no_path", "blocked", "error"].includes(nav.status) ? "" : "wait");
  $("nav-chip").textContent = "nav: " + st.toLowerCase();
  setAuto(nav.auto);
  setFollow(nav.follow);

  const busy = nav.status === "moving" || nav.status === "planning";
  $("btn-go").disabled = busy;
  document.querySelectorAll(".nudge .btn").forEach((b) => { b.disabled = busy; });
}

function setBadge(id, status) {
  const el = $(id);
  if (!el) return;
  let cls = "badge", txt = "OFF";
  if (status === "streaming") { cls = "badge on"; txt = "LIVE"; }
  else if (status === "mock") { cls = "badge mock"; txt = "MOCK"; }
  else if (status === "opening" || status === "init") { cls = "badge wait"; txt = "…"; }
  el.textContent = txt;
  el.className = cls;
}

let ws = null;
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/telemetry`);
  ws.onopen = () => { $("ws-dot").className = "dot dot-on"; };
  ws.onclose = () => { $("ws-dot").className = "dot dot-off"; setTimeout(connectWS, 1500); };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.nav) updateNav(msg.nav);
    if (msg.status && msg.status.sensors && msg.status.sensors.detector) {
      setBadge("st-detector", msg.status.sensors.detector.status);
    }
  };
}

connectWS();
