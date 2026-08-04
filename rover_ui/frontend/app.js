// Operator console — the same BEV map, plus navigation controls.

const $ = (id) => document.getElementById(id);

$("detect-img").src = "/stream/detect";

// --------------------------------------------------------- map (clickable)
const bev = new BevMap($("map-canvas"));
let autoOn = false;
let followOn = false;
let ignoreObstaclesOn = false;

// Reflect the configured start cell in the hint text and pre-fill the
// "Set start position" inputs (mm in the payload -> metres in the UI).
function applyStartCell(s) {
  if (!s || s.x == null) return;
  $("start-cell").textContent =
    `x=${(s.x / 1000).toFixed(1)} m, z=${(s.z / 1000).toFixed(1)} m`;
  $("start-x").value = (s.x / 1000).toFixed(1);
  $("start-z").value = (s.z / 1000).toFixed(1);
}

fetch("/api/map").then((r) => r.json()).then((m) => {
  bev.setMap(m);
  applyStartCell(m.rover_start || {});
  initSpeed(m);
  initJogSpeed(m);
  initStandoff(m);
  initConfirmN(m);
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

// ------------------------------------------------- approach (standoff)
// How far short of the target the rover stops. A SOFT preference: obstacle
// avoidance still wins, so the achieved distance can be larger or smaller when
// the preferred spot is blocked. Applies to the next goal computation, which in
// AUTO/FOLLOW is the very next cycle.
function initStandoff(m) {
  const el = $("standoff-input");
  if (m.standoff_min_mm != null) el.min = m.standoff_min_mm;
  if (m.standoff_max_mm != null) el.max = m.standoff_max_mm;
  if (m.standoff_mm != null) el.value = m.standoff_mm;
  el.disabled = false;
  $("standoff-val").textContent = el.value;
  $("standoff-range").textContent = `${el.min}–${el.max}`;
  el.onchange = async () => {
    const j = await post("/api/nav/standoff", { mm: parseFloat(el.value) });
    if (j && j.ok && j.standoff_mm != null) {
      el.value = j.standoff_mm;                   // reflect the clamped value
      $("standoff-val").textContent = j.standoff_mm;
    }
  };
}

// ------------------------------------------------ ghost-guard confirm count
// How many consistent detector frames must accumulate before a new/moved
// target is trusted (Navigator.project_detection). Any whole number 1-10.
function initConfirmN(m) {
  const el = $("confirm-n-input");
  if (m.confirm_n != null) el.value = m.confirm_n;
  el.disabled = false;
  $("confirm-n-val").textContent = el.value;
  el.onchange = async () => {
    const j = await post("/api/nav/confirm_n", { n: parseInt(el.value, 10) });
    if (j && j.ok && j.confirm_n != null) {
      el.value = j.confirm_n;                     // reflect the clamped value
      $("confirm-n-val").textContent = j.confirm_n;
    }
  };
}

bev.onClick = (x, z) => {
  post("/api/nav/goto", { x: Math.round(x), z: Math.round(z) });
};

// Drag the rover icon itself to re-anchor the live pose to that map point --
// same effect as "Re-anchor pose to start cell", just at an arbitrary spot
// instead of the configured home cell. bev already refuses the drop
// client-side if it lands inside an obstacle's keep-out.
bev.onRoverDrag = (x, z) => {
  post("/api/nav/reset_pose", { x, z });
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
    applyStartCell(j.map.rover_start || {});
  }
};
$("btn-set-start").onclick = async () => {
  const xm = parseFloat($("start-x").value);
  const zm = parseFloat($("start-z").value);
  if (!Number.isFinite(xm) || !Number.isFinite(zm)) {
    $("nav-msg-strip").textContent = "enter numeric x and z (metres)";
    return;
  }
  const j = await post("/api/nav/set_start",
                       { x: Math.round(xm * 1000), z: Math.round(zm * 1000) });
  if (j && j.ok && j.map) {
    bev.setMap(j.map);
    applyStartCell(j.map.rover_start || {});
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
$("btn-ignore-obstacles").onclick = async () => {
  const j = await post("/api/nav/ignore_obstacles", { enabled: !ignoreObstaclesOn });
  if (j && j.ok) setIgnoreObstacles(j.ignore_obstacles);
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

function setIgnoreObstacles(v) {
  ignoreObstaclesOn = !!v;
  const b = $("btn-ignore-obstacles");
  b.textContent = "Ignore obstacles: " + (ignoreObstaclesOn ? "on" : "off");
  b.className = "btn" + (ignoreObstaclesOn ? " toggle-on" : "");
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

function fmtAccum(a) {
  if (!a || !a.phase || a.phase === "none") return "—";
  if (a.phase === "accumulating") return `accumulating ${a.n}/${a.need}`;
  return "confirmed";
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
  // Minimum body-to-obstacle distance the CURRENT route achieves. The planner
  // maximises this, so a low value means the geometry was tight, not that the
  // planner settled — worth watching against the e-stop margin.
  $("nav-clearance").textContent =
    (nav.plan_clearance_mm == null ? "—" : nav.plan_clearance_mm + " mm");
  // Drift health. pose_jumps = T265 steps REJECTED as glitches; reloc = steps
  // ACCEPTED while stationary as genuine relocalisations (non-zero is healthy:
  // the device is correcting its own drift and the fix is being kept).
  // zupt = drift measured with the rover provably still.
  const d = nav.drift_since_fix || {};
  const bits = [];
  if (d.margin_mm != null) bits.push("+" + d.margin_mm + "mm");
  if (nav.zupt_drift_mm_s != null) bits.push(nav.zupt_drift_mm_s + "mm/s");
  bits.push("jmp " + (nav.pose_jumps || 0) + "/reloc " + (nav.reloc_accepted || 0));
  $("nav-drift").textContent = bits.join(" · ");
  // ArUco fixes: the only thing here that BOUNDS drift rather than slowing it.
  // Shows tags used, residual corrected, and the heading error the solve saw
  // (reported but not acted on until TAGS_CORRECT_YAW is enabled).
  const tf = nav.tag_fix, tc = nav.tag_counts || {}, tl = nav.tag_live;
  // Show the LIVE solve whenever fixes are being refused: seeing "4 tags,
  // rms 11px" while parked in front of a panel says the offsets in map.json
  // disagree with the physical layout, which a bare reject count never would.
  if (tl) {
    // Show the CURRENT solve and only the verdict belonging to THAT solve.
    // tc.why is the reason the LAST rejection fired, which can be many frames
    // old — splicing it onto the live numbers produced lines like
    // "seeing 1 tag rms 0.3px — rms 14.6px > 9.0 limit (4 tags)", which read as
    // one observation but were two.
    // WHY a fix was refused is the actionable thing, so it wins over the audit
    // verdict. Showing only the verdict hid the real cause: a run of 132
    // rejections displayed "tag 6 has the largest residual" while the actual
    // reason was the yaw gate, which the operator had no way to see.
    const rejecting = (tc.rejected || 0) > 0 && (tc.applied || 0) === 0;
    const why = (rejecting && tc.why) ? (" — REJECTING: " + tc.why)
              : ((tl.audit && tl.audit.verdict) ? (" — " + tl.audit.verdict)
              : (tc.why ? (" — last reject: " + tc.why) : ""));
    const yw = nav.tag_yaw
      ? (" · heading spread " + nav.tag_yaw.min + "\u00b0.." + nav.tag_yaw.max +
         "\u00b0 (med " + nav.tag_yaw.med + "\u00b0)")
      : "";
    $("nav-tags").textContent =
      "seeing " + tl.n_tags + " tag" + (tl.n_tags === 1 ? "" : "s") +
      " [" + (tl.ids || []).join(",") + "] rms " + tl.rms_px + "px" + why +
      " · " + (tc.applied || 0) + " ok/" + (tc.rejected || 0) + " rej" + yw;
  } else if (!tf) {
    // Say WHY there is nothing, rather than a bare dash: "no camera
    // intrinsics" and "no known tags in view" need completely different fixes.
    const idle = nav.tag_idle ? (" — " + nav.tag_idle) : "";
    $("nav-tags").textContent = (tc.applied || tc.rejected)
      ? ("none yet" + idle + " · " + (tc.applied || 0) + " ok/" + (tc.rejected || 0) + " rej")
      : (nav.tag_idle ? nav.tag_idle : "—");
  } else {
    // age_s comes from the BACKEND (same clock that stamped the fix), so a
    // Jetson/browser clock skew can no longer show up as a bogus age.
    const age = (tf.age_s == null) ? null : tf.age_s;
    $("nav-tags").textContent =
      tf.n_tags + " tags seen · corrected " + tf.resid_mm + "mm · heading off " +
      (tf.yaw_err_deg >= 0 ? "+" : "") + tf.yaw_err_deg + "\u00b0 · " +
      (age == null ? "?" : (age < 90 ? age.toFixed(0) + "s"
                                     : (age / 60).toFixed(0) + "m")) + " ago · " +
      (tc.applied || 0) + " ok/" + (tc.rejected || 0) + " rej";
  }
  $("nav-accum").textContent = fmtAccum(nav.accum);
  const sEl = $("standoff-input");
  if (nav.standoff_mm != null && sEl && document.activeElement !== sEl) {
    sEl.value = nav.standoff_mm;                  // another client may have changed it
    $("standoff-val").textContent = nav.standoff_mm;
  }
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
  setIgnoreObstacles(nav.ignore_obstacles);

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
