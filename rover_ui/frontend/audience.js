// Audience display — read-only view of the map-navigation demo.
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

function fmtAccum(a) {
  if (!a || !a.phase || a.phase === "none") return "—";
  if (a.phase === "averaging") {
    const sp = a.spread_mm != null ? ` · ±${a.spread_mm} mm` : "";
    return `${a.n}/${a.need} frames${sp}`;
  }
  if (a.phase === "accumulating") return `accumulating ${a.n}/${a.need}`;
  return "confirmed";
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
  $("nav-accum").textContent = fmtAccum(nav.accum);
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

  c.strokeStyle = "#ffffff"; c.lineWidth = 1;
  for (let r = 1; r <= maxRange; r++) {
    c.beginPath(); c.arc(ox, oy, r * scale, Math.PI, 2 * Math.PI); c.stroke();
    c.fillStyle = "#ffffff"; c.font = "10px sans-serif";
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

  c.fillStyle = "#ffffff";
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

// Mirrors detector.py's _box_color() selection rule exactly (radar-only flag
// first, else the argmax of contrib [rgb, thermal, radar]) so the cam-card
// highlight always agrees with the box color drawn on the RGB stream --
// without touching that drawing logic at all.
function drivingSensor(d) {
  if (!d) return null;
  if (d.radar) return "radar";
  const c = d.contrib || [0, 0, 0];
  if (!c.some((v) => v)) return "rgb";
  let idx = 0;
  for (let i = 1; i < 3; i++) if (c[i] > c[idx]) idx = i;
  return ["rgb", "thermal", "radar"][idx];
}

function highlightCamCard(sensor) {
  const cards = { rgb: "card-rgb", thermal: "card-thermal", radar: "card-radar" };
  for (const key in cards) {
    const el = $(cards[key]);
    if (el) el.classList.toggle("driving-" + key, sensor === key);
  }
}

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

  const dets = det.detections || [];
  highlightCamCard(dets.length ? drivingSensor(dets[0]) : null);

  renderDetList(dets);
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

// --------------------------------------------------------- timing overlay
// Debug overlay for the RGB and thermal views, switched on from the OPERATOR
// page (server flag, arrives as msg.debug_overlay). While on, each view reads
// its /stream/*_ts twin with fetch(): the same MJPEG, plus per-frame headers
// with rover timestamps (frame received, boxes computed, box source frame,
// render, send). Frames are shown by swapping the <img> to a blob URL, so the
// overlay text always belongs to the frame on screen. While off, the views use
// the plain streams exactly as before (no extra cost).
//
// Clock offset: the rover and this browser are usually different machines.
// offset = rover_ms - browser_ms, estimated NTP-style from /api/time (lowest
// round trip of 5 samples, refreshed every 15 s). Rover times are converted
// to browser time before comparing with the live UI clock.
const TS_VIEWS = {
  rgb:     { img: "detect-img",  plain: "/stream/detect",         ts: "/stream/detect_ts",
             ovl: "ovl-rgb",     plot: "plot-rgb",     label: "RGB" },
  thermal: { img: "thermal-img", plain: "/stream/detect_thermal", ts: "/stream/detect_thermal_ts",
             ovl: "ovl-thermal", plot: "plot-thermal", label: "THERMAL" },
};
let overlayOn = false;
// Delay history plot (frame->screen), drawn along the bottom edge of each
// image. The plot's width always stands for PLOT_WINDOW_MS: it grows rightward
// from the moment the overlay is switched on, then scrolls left once 60 s of
// history exist. One sample per displayed frame, kept only in this browser.
const PLOT_WINDOW_MS = 60000;
const PLOT_H = 36;            // CSS px
const PLOT_GAP_MS = 1000;     // break the line across stalls longer than this
let plotT0 = null;            // when the overlay (and the history) started
const clockSync = { offset: null, rtt: null, timer: null };
let ovlTimer = null;

async function syncClock() {
  let best = null;
  for (let i = 0; i < 5; i++) {
    try {
      const t0 = Date.now();
      const r = await fetch("/api/time", { cache: "no-store" });
      const j = await r.json();
      const t1 = Date.now();
      const s = { rtt: t1 - t0, off: j.t * 1000 - (t0 + t1) / 2 };
      if (!best || s.rtt < best.rtt) best = s;
    } catch (e) { /* retry next sample */ }
  }
  if (best) { clockSync.offset = best.off; clockSync.rtt = best.rtt; }
}

function toLocalMs(roverMs) {
  if (roverMs == null) return null;
  return clockSync.offset == null ? roverMs : roverMs - clockSync.offset;
}

function fmtClock(ms) {
  if (ms == null) return "—";
  const d = new Date(ms);
  const p = (n, w) => String(n).padStart(w, "0");
  return `${p(d.getHours(), 2)}:${p(d.getMinutes(), 2)}:${p(d.getSeconds(), 2)}.${p(d.getMilliseconds(), 3)}`;
}

function fmtMs(v) { return v == null || !isFinite(v) ? "—" : `${Math.round(v)} ms`; }

const CRLF2 = [13, 10, 13, 10];
function findSeq(buf, seq, from) {
  outer: for (let i = from; i <= buf.length - seq.length; i++) {
    for (let j = 0; j < seq.length; j++) if (buf[i + j] !== seq[j]) continue outer;
    return i;
  }
  return -1;
}

function hdrMs(head, name) {
  const m = new RegExp(name + ":\\s*([0-9.]+)", "i").exec(head);
  return m ? parseFloat(m[1]) * 1000 : null;
}

function startTsView(key) {
  const v = TS_VIEWS[key];
  const ctrl = new AbortController();
  v.ctrl = ctrl;
  (async () => {
    try {
      const resp = await fetch(v.ts, { signal: ctrl.signal, cache: "no-store" });
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = new Uint8Array(0);
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        const nb = new Uint8Array(buf.length + value.length);
        nb.set(buf); nb.set(value, buf.length); buf = nb;
        for (;;) {
          const hEnd = findSeq(buf, CRLF2, 0);
          if (hEnd < 0) break;
          const head = dec.decode(buf.subarray(0, hEnd));
          const m = /Content-Length:\s*(\d+)/i.exec(head);
          const start = hEnd + 4;
          if (!m) { buf = buf.subarray(start); continue; }
          const len = parseInt(m[1], 10);
          if (buf.length < start + len) break;
          showTsFrame(v, head, buf.slice(start, start + len));
          buf = buf.subarray(start + len);
        }
      }
    } catch (e) { /* aborted or dropped */ }
    if (overlayOn && v.ctrl === ctrl) setTimeout(() => { if (overlayOn && v.ctrl === ctrl) startTsView(key); }, 1000);
  })();
}

function showTsFrame(v, head, jpg) {
  const url = URL.createObjectURL(new Blob([jpg], { type: "image/jpeg" }));
  const img = $(v.img);
  const meta = {
    frame: hdrMs(head, "X-Frame-T"), box: hdrMs(head, "X-Box-T"),
    boxSrc: hdrMs(head, "X-Box-Src-T"), render: hdrMs(head, "X-Render-T"),
    send: hdrMs(head, "X-Send-T"), shown: null,
  };
  img.onload = () => {
    meta.shown = Date.now();
    v.last = meta;
    const frame = toLocalMs(meta.frame);
    if (frame != null && v.hist) {
      v.hist.push({ t: meta.shown, v: Math.max(0, meta.shown - frame) });
      while (v.hist.length && v.hist[0].t < meta.shown - PLOT_WINDOW_MS) v.hist.shift();
    }
  };
  const prev = v.url;
  v.url = url;
  img.src = url;
  if (prev) setTimeout(() => URL.revokeObjectURL(prev), 1000);
}

function niceCeil(ms) {
  for (const s of [50, 100, 200, 250, 500, 1000, 2000, 2500, 5000]) if (ms <= s) return s;
  return Math.ceil(ms / 5000) * 5000;
}

function drawPlot(v, now) {
  const img = $(v.img), cv = $(v.plot);
  const nw = img.naturalWidth, nh = img.naturalHeight;
  const ew = img.clientWidth, eh = img.clientHeight;
  if (!nw || !nh || !ew || !eh) return;
  // The <img> letterboxes the picture (object-fit: contain): find the picture
  // itself inside the element, and pin the plot to ITS bottom edge.
  const sc = Math.min(ew / nw, eh / nh);
  const dw = nw * sc, dh = nh * sc;
  const w = Math.max(40, Math.round(dw - 12)), h = PLOT_H;
  const left = Math.round(img.offsetLeft + (ew - dw) / 2 + 6);
  const top = Math.round(img.offsetTop + (eh - dh) / 2 + dh - h - 4);
  const dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    cv.style.width = w + "px"; cv.style.height = h + "px";
  }
  cv.style.left = left + "px"; cv.style.top = top + "px";
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const x0 = Math.max(plotT0 || now, now - PLOT_WINDOW_MS);   // left edge time
  const pts = (v.hist || []).filter((p) => p.t >= x0);
  if (!pts.length) return;
  const vmax = Math.max(...pts.map((p) => p.v));
  const ymax = niceCeil(Math.max(vmax, 1));
  const labelH = 11, x = (t) => (t - x0) * w / PLOT_WINDOW_MS;
  const y = (d) => h - 1.5 - Math.min(d, ymax) / ymax * (h - labelH - 3);
  ctx.beginPath();
  pts.forEach((p, i) => {
    if (i === 0 || p.t - pts[i - 1].t > PLOT_GAP_MS) ctx.moveTo(x(p.t), y(p.v));
    else ctx.lineTo(x(p.t), y(p.v));
  });
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(0, 0, 0, .55)"; ctx.lineWidth = 3; ctx.stroke();       // contrast edge
  ctx.strokeStyle = "rgba(230, 237, 243, .95)"; ctx.lineWidth = 1.5; ctx.stroke();
  const label = `frame→screen max ${Math.round(vmax)} ms`;
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  ctx.textAlign = "right"; ctx.textBaseline = "top";
  ctx.lineWidth = 3; ctx.strokeStyle = "rgba(0, 0, 0, .7)"; ctx.strokeText(label, w - 1, 0);
  ctx.fillStyle = "#e6edf3"; ctx.fillText(label, w - 1, 0);
}

function renderOverlays() {
  const now = Date.now();
  for (const key of Object.keys(TS_VIEWS)) {
    const v = TS_VIEWS[key];
    const el = $(v.ovl);
    const m = v.last;
    const lines = [];
    if (!m) {
      lines.push(`${v.label}  waiting for frames…`);
    } else {
      const frame = toLocalMs(m.frame), box = toLocalMs(m.box);
      // frame->box uses rover times only (same clock, no offset needed).
      const detect = (m.box != null && m.boxSrc != null) ? m.box - m.boxSrc : null;
      const pad = (s) => s.padEnd(13, " ");
      lines.push(`${v.label} timing`);
      lines.push(pad("frame rx") + fmtClock(frame));
      lines.push(pad("boxes") + fmtClock(box));
      lines.push(pad("UI live") + fmtClock(now));
      lines.push(pad("frame→box") + fmtMs(detect));
      lines.push(pad("box→screen") + fmtMs(box != null ? m.shown - box : null));
      lines.push(pad("frame→screen") + fmtMs(frame != null ? m.shown - frame : null));
      lines.push(pad("frame age") + fmtMs(frame != null ? now - frame : null));
    }
    lines.push(clockSync.offset == null ? "clock Δ syncing…"
      : `clock Δ ${clockSync.offset >= 0 ? "+" : ""}${Math.round(clockSync.offset)} ms (rtt ${Math.round(clockSync.rtt)})`);
    el.textContent = lines.join("\n");
    drawPlot(v, now);
  }
}

function setOverlay(on) {
  overlayOn = !!on;
  for (const key of Object.keys(TS_VIEWS)) {
    const v = TS_VIEWS[key];
    $(v.ovl).hidden = !overlayOn;
    $(v.plot).hidden = !overlayOn;
    v.hist = [];                            // history restarts with the overlay
    if (overlayOn) {
      v.last = null;
      startTsView(key);
    } else {
      if (v.ctrl) v.ctrl.abort();
      v.ctrl = null;
      $(v.img).onload = null;
      $(v.img).src = v.plain;               // back to the plain stream
      if (v.url) { const u = v.url; setTimeout(() => URL.revokeObjectURL(u), 1000); v.url = null; }
    }
  }
  clearInterval(clockSync.timer); clearInterval(ovlTimer);
  clockSync.timer = ovlTimer = null;
  plotT0 = overlayOn ? Date.now() : null;
  if (overlayOn) {
    syncClock();
    clockSync.timer = setInterval(syncClock, 15000);
    ovlTimer = setInterval(renderOverlays, 50);
    renderOverlays();
  }
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
    if (typeof msg.debug_overlay === "boolean" && msg.debug_overlay !== overlayOn) {
      setOverlay(msg.debug_overlay);
    }
  };
}

// --------------------------------------------------------- init
drawMmwave({ points: [] });
connectWS();
