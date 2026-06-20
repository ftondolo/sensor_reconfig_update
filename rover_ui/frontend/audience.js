// Audience display (demo_v1) — read-only view of the map-navigation demo.
//
//   * Big BEV map: pre-marked obstacles, live rover footprint, projected
//     detection target, standoff goal, planned rectilinear path.
//   * RGB+detections, thermal, depth as MJPEG <img> streams; mmWave canvas.
//   * Sensor-contribution bars + nav status from /ws/telemetry.

const $ = (id) => document.getElementById(id);

let savSmooth = null;   // EMA-smoothed compute-saving % for the prominent banner
let radarBox = null;    // {range, az} radar-frame track, set only when radar is
                        // the PRIMARY detection source — drawn on the mmWave plot

// --------------------------------------------------------- video streams
$("detect-img").src = "/stream/detect";
$("thermal-img").src = "/stream/detect_thermal";
$("depth-img").src = "/stream/detect_depth";

// --------------------------------------------------------- BEV map
const bev = new BevMap($("map-canvas"));
fetch("/api/map").then((r) => r.json()).then((m) => bev.setMap(m));

function fmtMM(p) {
  if (!p || p.x == null) return "—";
  return `(${(p.x / 1000).toFixed(2)}, ${(p.z / 1000).toFixed(2)}) m`;
}

function updateNav(nav) {
  if (!nav) return;
  bev.setNav(nav);

  const st = (nav.status || "idle").toUpperCase();
  const big = $("nav-status-big");
  big.textContent = st.replace("_", " ");
  big.className = "nav-big" +
    (nav.status === "moving" || nav.status === "arrived" ? " moving" : "") +
    (["no_path", "blocked", "error"].includes(nav.status) ? " bad" : "");
  $("nav-msg").textContent = nav.message ||
    ({ idle: "waiting for a detection…", moving: "driving the planned path",
       arrived: "standoff reached (2 m from target)",
       planning: "planning a rectilinear path…" }[nav.status] || "");

  $("rover-xy").textContent = fmtMM(nav.rover);
  $("target-xy").textContent = fmtMM(nav.target);
  $("goal-xy").textContent = fmtMM(nav.goal) + (nav.goal && nav.goal.adjusted ? " (adj)" : "");
  $("nav-leg").textContent = nav.path ? `${nav.leg}/${nav.path.length - 1}` : "—";
  $("pose-src").textContent = nav.mock ? "MOCK" : "T265";
  $("nav-auto").textContent = nav.auto ? "on" : "off";
  const fol = $("nav-follow");
  fol.textContent = nav.follow ? "ON" : "off";
  fol.classList.toggle("hot", !!nav.follow);
  $("nav-moving").textContent = nav.moving ? "yes" : "no";
  $("nav-ros").textContent = nav.mock ? "mock" : (nav.ros ? "ROS OK" : "no ROS");

  const badge = $("nav-badge");
  badge.textContent = st;
  badge.className = "badge " +
    (nav.status === "moving" ? "mode" :
     nav.status === "arrived" ? "on" :
     ["no_path", "blocked", "error"].includes(nav.status) ? "" : "wait");

  $("nav-chip").textContent = "nav: " + st.toLowerCase() +
    (nav.target && nav.target.range_m != null ? ` · target ${nav.target.range_m}m` : "");
}

// --------------------------------------------------------- mmWave plot
const mmCanvas = $("mmwave-canvas");
const mmCtx = mmCanvas.getContext("2d");

function drawMmwave(mm) {
  const points = mm && mm.points;
  const suppressed = mm && mm.suppressed;
  const c = mmCtx, W = mmCanvas.width, H = mmCanvas.height;
  c.fillStyle = "#000"; c.fillRect(0, 0, W, H);

  const maxRange = 6.0;
  const ox = W / 2, oy = H - 12;
  const scale = (H - 24) / maxRange;

  c.strokeStyle = "#1c2430"; c.lineWidth = 1;
  for (let r = 1; r <= maxRange; r++) {
    c.beginPath(); c.arc(ox, oy, r * scale, Math.PI, 2 * Math.PI); c.stroke();
    c.fillStyle = "#3a4150"; c.font = "10px sans-serif";
    c.fillText(r + "m", ox + 3, oy - r * scale + 12);
  }
  if (points) {
    for (const p of points) {
      const sx = ox + p.x * scale;
      const sy = oy - p.y * scale;
      const v = p.v || 0;
      c.fillStyle = v < -0.05 ? "#2f81f7" : v > 0.05 ? "#f85149" : "#3fb950";
      c.beginPath(); c.arc(sx, sy, 5, 0, Math.PI * 2); c.fill();
    }
  }

  // Radar target box — shown only when the radar is the primary detection
  // source. Same frame as the points: x=range*sin(az), y=range*cos(az).
  if (radarBox && radarBox.range != null && radarBox.range <= maxRange) {
    const az = (radarBox.az || 0) * Math.PI / 180;
    const tx = ox + radarBox.range * Math.sin(az) * scale;
    const ty = oy - radarBox.range * Math.cos(az) * scale;
    const half = 0.4 * scale;   // ~0.8 m box around the tracked target
    c.strokeStyle = "#f0883e";  // radar colour (matches the contribution bars)
    c.lineWidth = 2;
    c.strokeRect(tx - half, ty - half, 2 * half, 2 * half);
    c.fillStyle = "#f0883e";
    c.font = "11px sans-serif";
    c.fillText(`RADAR ${radarBox.range.toFixed(1)}m`, tx - half, ty - half - 4);
  }

  c.fillStyle = "#8b949e";
  c.beginPath(); c.arc(ox, oy, 5, 0, Math.PI * 2); c.fill();

  if (suppressed === "moving") {
    c.fillStyle = "#f0883e"; c.font = "12px sans-serif";
    c.fillText("radar paused — rover moving", 10, 18);
  }
}

// --------------------------------------------------------- status badges
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

// --------------------------------------------------------- detection panel
function pct(v) { return Math.round((v || 0) * 100); }

function updateContrib(det) {
  const c = det.contrib || [0, 0, 0];
  $("bar-rgb").style.width = pct(c[0]) + "%";
  $("bar-thermal").style.width = pct(c[1]) + "%";
  $("bar-radar").style.width = pct(c[2]) + "%";
  $("pct-rgb").textContent = pct(c[0]) + "%";
  $("pct-thermal").textContent = pct(c[1]) + "%";
  $("pct-radar").textContent = pct(c[2]) + "%";

  $("det-count").textContent = det.num != null ? det.num : 0;
  const t = det.target || {};
  // Box the radar target on the mmWave plot only when the radar is the primary
  // detection basis (vision saw nothing); otherwise the radar is just supplying
  // range to a vision box and we don't draw it.
  radarBox = (t.source === "radar" && det.radar_track) ? det.radar_track : null;
  $("det-range").textContent = t.range != null
    ? t.range.toFixed(2) + " m" + (t.range_src ? " (" + t.range_src + ")" : "")
    : "—";
  $("det-az").textContent = t.az != null ? t.az.toFixed(1) + "°" : "—";
  $("det-source").textContent = t.source || "—";

  // Adaptive-compute banner — EMA-smoothed so the big number eases, not jumps.
  const a = det.adaptive || {};
  if (a.savings != null) {
    savSmooth = (savSmooth == null) ? a.savings : 0.12 * a.savings + 0.88 * savSmooth;
    $("save-big").textContent = Math.round(savSmooth);
    $("save-fill").style.width = Math.max(0, Math.min(100, savSmooth)) + "%";
  }
  setTier("tiers-rgb", a.rgb_res);
  setTier("tiers-therm", a.therm_res);

  renderDetList(det.detections || []);
}

function setTier(containerId, label) {
  // Light up the box matching the DQN's committed tier. The tier itself is
  // already hysteresis-debounced backend-side, so the highlight steps calmly.
  const boxes = document.getElementById(containerId).children;
  for (const b of boxes) b.classList.toggle("active", !!label && b.dataset.t === label);
}

function renderDetList(dets) {
  const box = $("det-list");
  if (!dets.length) {
    box.innerHTML = '<div class="det-empty">No people detected.</div>';
    return;
  }
  box.innerHTML = "";
  dets.forEach((d, i) => {
    const c = d.contrib || [0, 0, 0];
    const item = document.createElement("div");
    item.className = "det-item";
    item.innerHTML =
      '<div class="det-top"><b>Person ' + (i + 1) + '</b><span>conf ' +
      Math.round((d.score || 0) * 100) + '%</span></div>' +
      '<div class="mini-bars">' +
        '<span class="rgb" style="width:' + pct(c[0]) + '%"></span>' +
        '<span class="thermal" style="width:' + pct(c[1]) + '%"></span>' +
        '<span class="radar" style="width:' + pct(c[2]) + '%"></span>' +
      '</div>' +
      '<div class="mini-legend">' +
        '<span><i style="color:var(--rgb)">RGB</i> ' + pct(c[0]) + '%</span>' +
        '<span><i style="color:var(--radar)">Therm</i> ' + pct(c[1]) + '%</span>' +
        '<span><i style="color:var(--thermal)">Radar</i> ' + pct(c[2]) + '%</span>' +
      '</div>';
    box.appendChild(item);
  });
}

function applyStatus(status) {
  if (!status) return;
  const s = status.sensors || {};
  for (const name of Object.keys(s)) setBadge("st-" + name, s[name].status);
}

// --------------------------------------------------------- websocket
let ws = null;
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/telemetry`);
  ws.onopen = () => { $("ws-dot").className = "dot dot-on"; };
  ws.onclose = () => { $("ws-dot").className = "dot dot-off"; setTimeout(connectWS, 1500); };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.mmwave) drawMmwave(msg.mmwave);
    if (msg.detection) updateContrib(msg.detection);
    if (msg.nav) updateNav(msg.nav);
    if (msg.status) applyStatus(msg.status);
  };
}

// --------------------------------------------------------- init
drawMmwave({ points: [] });
connectWS();
