// Shared BEV map renderer (audience + operator pages).
//
// Map frame: mm, x right, z DOWN (matches map.json / rectilinear_mm.py).
// The canvas draws the map 1:1 in that orientation, so the rover (which heads
// "up" the map) appears at the bottom driving toward the top.
//
//   const view = new BevMap(canvas);
//   view.setMap(mapPayload);        // from GET /api/map
//   view.setNav(navState);          // from ws "nav" payloads
//   view.onClick = (x_mm, z_mm) => {...};   // optional (operator page)
//   view.onRoverDrag = (x_mm, z_mm) => {...};   // optional (operator page):
//     fires on drop when the rover icon itself was dragged to a new spot.
//
// The renderer runs its own requestAnimationFrame loop (target pulse etc).

class BevMap {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.map = null;       // {size:{width,depth}, obstacles, car, clearance, rover_start}
    this.nav = null;       // {rover, target, goal, path, status, ...}
    this.trail = [];       // recent rover positions [[x,z,t]]
    this.onClick = null;
    this.onRoverDrag = null;
    this._dragging = false;    // dragging the rover icon (vs. a plain click)
    this._dragPos = null;      // live [x,z] mm while dragging
    this._justDragged = false; // suppresses the click that follows a drag's pointerup

    canvas.addEventListener("pointerdown", (ev) => {
      if (!this.onRoverDrag || !this.map || !this.nav || !this.nav.rover) return;
      const p = this._eventToMap(ev, true);
      const r = this.nav.rover;
      const car = this.map.car || { width: 700, length: 750 };
      const grabR = Math.max(car.width, car.length); // generous grab radius, mm
      if (Math.hypot(p[0] - r.x, p[1] - r.z) > grabR) return;
      this._dragging = true;
      this._dragPos = p;
      canvas.setPointerCapture(ev.pointerId);
      ev.preventDefault();
    });
    canvas.addEventListener("pointermove", (ev) => {
      if (!this._dragging) return;
      this._dragPos = this._eventToMap(ev, true);
    });
    canvas.addEventListener("pointerup", (ev) => {
      if (!this._dragging) return;
      this._dragging = false;
      this._justDragged = true;
      try { canvas.releasePointerCapture(ev.pointerId); } catch (e) { /* ignore */ }
      const p = this._dragPos;
      this._dragPos = null;
      if (p && !this._footprintBlocked(p[0], p[1])) {
        this.onRoverDrag(Math.round(p[0]), Math.round(p[1]));
      }
    });
    canvas.addEventListener("click", (ev) => {
      if (this._justDragged) { this._justDragged = false; return; }
      if (!this.onClick || !this.map) return;
      const p = this._eventToMap(ev);
      if (p) this.onClick(p[0], p[1]);
    });

    const loop = () => { this.draw(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  setMap(map) { this.map = map; }

  setNav(nav) {
    this.nav = nav;
    const r = nav && nav.rover;
    if (r) {
      const now = performance.now();
      const last = this.trail[this.trail.length - 1];
      if (!last || Math.hypot(last[0] - r.x, last[1] - r.z) > 25) {
        this.trail.push([r.x, r.z, now]);
      }
      // keep ~3 minutes of motion
      while (this.trail.length > 1500) this.trail.shift();
    }
  }

  clearTrail() { this.trail = []; }

  // ---- coordinate transforms ------------------------------------------
  _fit() {
    const { width: W, height: H } = this.canvas;
    const mw = this.map.size.width, md = this.map.size.depth;
    const pad = 26;
    const s = Math.min((W - 2 * pad) / mw, (H - 2 * pad) / md);
    const ox = (W - mw * s) / 2;
    const oy = (H - md * s) / 2;
    return { s, ox, oy };
  }

  _px(f, x, z) { return [f.ox + x * f.s, f.oy + z * f.s]; }

  // clamp=true keeps returning a point (clamped into the map) instead of null
  // when the pointer strays outside the canvas/map — needed so a drag preview
  // keeps following the cursor smoothly even past the edge.
  _eventToMap(ev, clamp) {
    const rect = this.canvas.getBoundingClientRect();
    const cx = (ev.clientX - rect.left) * (this.canvas.width / rect.width);
    const cy = (ev.clientY - rect.top) * (this.canvas.height / rect.height);
    const f = this._fit();
    let x = (cx - f.ox) / f.s, z = (cy - f.oy) / f.s;
    if (clamp) {
      x = Math.max(0, Math.min(this.map.size.width, x));
      z = Math.max(0, Math.min(this.map.size.depth, z));
      return [x, z];
    }
    if (x < 0 || z < 0 || x > this.map.size.width || z > this.map.size.depth) return null;
    return [x, z];
  }

  // Same overlap test the rover-footprint render uses to turn red: would a
  // car-sized rectangle centered at (x, z) overlap any obstacle's clearance
  // halo? Used to block dropping the dragged rover into a keep-out.
  _footprintBlocked(x, z) {
    const car = this.map.car || { width: 700, length: 750 };
    const cw2 = car.width / 2, cl2 = car.length / 2;
    const clr = this.map.clearance || 0;
    for (const o of (this.map.obstacles || [])) {
      if (x + cw2 > o.x - clr && x - cw2 < o.x + o.w + clr &&
          z + cl2 > o.z - clr && z - cl2 < o.z + o.h + clr) {
        return true;
      }
    }
    return false;
  }

  // ---- drawing ----------------------------------------------------------
  draw() {
    const c = this.ctx, W = this.canvas.width, H = this.canvas.height;
    c.clearRect(0, 0, W, H);
    if (!this.map) return;
    const f = this._fit();
    const mw = this.map.size.width, md = this.map.size.depth;
    const [x0, y0] = this._px(f, 0, 0);
    const [x1, y1] = this._px(f, mw, md);

    // floor
    c.fillStyle = "#0b1018";
    c.fillRect(x0, y0, x1 - x0, y1 - y0);

    // 1 m grid (solid white, not grey — readable on a projector)
    c.strokeStyle = "#ffffff"; c.lineWidth = 1;
    c.beginPath();
    for (let gx = 0; gx <= mw; gx += 1000) {
      const [px] = this._px(f, gx, 0);
      c.moveTo(px, y0); c.lineTo(px, y1);
    }
    for (let gz = 0; gz <= md; gz += 1000) {
      const [, pz] = this._px(f, 0, gz);
      c.moveTo(x0, pz); c.lineTo(x1, pz);
    }
    c.stroke();

    // axis labels (meters)
    c.fillStyle = "#ffffff"; c.font = "10px sans-serif";
    for (let gx = 0; gx <= mw; gx += 1000) {
      const [px] = this._px(f, gx, 0);
      c.fillText((gx / 1000) + "m", px + 2, y1 + 12);
    }
    for (let gz = 1000; gz <= md; gz += 1000) {
      const [, pz] = this._px(f, 0, gz);
      c.fillText((gz / 1000) + "m", x0 - 22, pz + 3);
    }

    // obstacles + clearance halo. The halo is the FOOTPRINT keep-out
    // (obstacle grown by clearance only): the rover rectangle may touch but
    // never overlap it. (Planning additionally keeps the rover CENTER a
    // further car/2 away — that margin is implied, not drawn.)
    const car = this.map.car || { width: 800, length: 1200 };
    const hx = this.map.clearance || 0;
    const hz = this.map.clearance || 0;
    for (const o of (this.map.obstacles || [])) {
      const [ax, az] = this._px(f, o.x - hx, o.z - hz);
      const [bx, bz] = this._px(f, o.x + o.w + hx, o.z + o.h + hz);
      c.fillStyle = "rgba(248,81,73,.06)";
      c.fillRect(ax, az, bx - ax, bz - az);
      c.strokeStyle = "rgba(248,81,73,.18)";
      c.setLineDash([4, 4]);
      c.strokeRect(ax, az, bx - ax, bz - az);
      c.setLineDash([]);

      const [px, pz] = this._px(f, o.x, o.z);
      const [qx, qz] = this._px(f, o.x + o.w, o.z + o.h);
      c.fillStyle = "rgba(248,81,73,.30)";
      c.fillRect(px, pz, qx - px, qz - pz);
      c.strokeStyle = "#f85149"; c.lineWidth = 1.5;
      c.strokeRect(px, pz, qx - px, qz - pz);
    }

    // configured start cell (map.json rover_start): a hollow home marker so the
    // operator can see where "Return to start" / re-anchor will target.
    const start = this.map.rover_start;
    if (start && start.x != null) {
      const [sx, sz] = this._px(f, start.x, start.z);
      c.strokeStyle = "rgba(63,185,80,.7)"; c.lineWidth = 2;
      c.beginPath();
      c.moveTo(sx - 7, sz + 6); c.lineTo(sx - 7, sz - 2); c.lineTo(sx, sz - 8);
      c.lineTo(sx + 7, sz - 2); c.lineTo(sx + 7, sz + 6); c.closePath();
      c.stroke();
      this._label(c, sx, sz + 18, "START", "rgba(63,185,80,.85)");
    }

    // border (solid white, not grey)
    c.strokeStyle = "#ffffff"; c.lineWidth = 2;
    c.strokeRect(x0, y0, x1 - x0, y1 - y0);

    const nav = this.nav || {};

    // rover trail
    if (this.trail.length > 1) {
      c.strokeStyle = "rgba(47,129,247,.45)"; c.lineWidth = 2;
      c.beginPath();
      this.trail.forEach(([tx, tz], i) => {
        const [px, pz] = this._px(f, tx, tz);
        if (i === 0) c.moveTo(px, pz); else c.lineTo(px, pz);
      });
      c.stroke();
    }

    // planned path
    if (nav.path && nav.path.length > 1) {
      c.strokeStyle = "#d29922"; c.lineWidth = 2.5;
      c.setLineDash([8, 6]);
      c.beginPath();
      nav.path.forEach((w, i) => {
        const [px, pz] = this._px(f, w[0], w[1]);
        if (i === 0) c.moveTo(px, pz); else c.lineTo(px, pz);
      });
      c.stroke();
      c.setLineDash([]);
      for (const w of nav.path) {
        const [px, pz] = this._px(f, w[0], w[1]);
        c.fillStyle = "#d29922";
        c.beginPath(); c.arc(px, pz, 3, 0, Math.PI * 2); c.fill();
      }
    }

    // standoff link target -> goal
    if (nav.target && nav.goal) {
      const [tx, tz] = this._px(f, nav.target.x, nav.target.z);
      const [gx, gz] = this._px(f, nav.goal.x, nav.goal.z);
      c.strokeStyle = "rgba(63,185,80,.5)"; c.lineWidth = 1.5;
      c.setLineDash([3, 5]);
      c.beginPath(); c.moveTo(tx, tz); c.lineTo(gx, gz); c.stroke();
      c.setLineDash([]);
    }

    // goal marker (X in a circle)
    if (nav.goal) {
      const [gx, gz] = this._px(f, nav.goal.x, nav.goal.z);
      c.strokeStyle = nav.goal.adjusted ? "#d29922" : "#3fb950";
      c.lineWidth = 2.5;
      c.beginPath(); c.arc(gx, gz, 10, 0, Math.PI * 2); c.stroke();
      c.beginPath();
      c.moveTo(gx - 5, gz - 5); c.lineTo(gx + 5, gz + 5);
      c.moveTo(gx + 5, gz - 5); c.lineTo(gx - 5, gz + 5);
      c.stroke();
      this._label(c, gx, gz - 16, nav.goal.adjusted ? "GOAL (adjusted)" : "GOAL",
                  nav.goal.adjusted ? "#d29922" : "#3fb950");
    }

    // detection target (pulsing person dot)
    if (nav.target) {
      const [tx, tz] = this._px(f, nav.target.x, nav.target.z);
      const stale = !!nav.target.stale;
      // stale = solid white rather than grey; the missing pulse ring below
      // (and the "last seen" label) still distinguish it from a live target.
      const col = stale ? "#ffffff" : "#f85149";
      const pulse = 6 + 4 * (0.5 + 0.5 * Math.sin(performance.now() / 280));
      if (!stale) {
        c.strokeStyle = "rgba(248,81,73,.45)"; c.lineWidth = 2;
        c.beginPath(); c.arc(tx, tz, pulse + 8, 0, Math.PI * 2); c.stroke();
      }
      c.fillStyle = col;
      c.beginPath(); c.arc(tx, tz, 7, 0, Math.PI * 2); c.fill();
      c.fillStyle = "#0b1018"; c.font = "bold 9px sans-serif";
      c.textAlign = "center"; c.fillText("T", tx, tz + 3); c.textAlign = "left";
      const rng = nav.target.range_m != null ? ` ${nav.target.range_m}m` : "";
      this._label(c, tx, tz - 14, (stale ? "TARGET (last seen)" : "TARGET") + rng, col);
    }

    // rover footprint + heading. Footprint turns RED when it genuinely
    // overlaps an obstacle's clearance zone (i.e. the constraint is violated
    // — pose drift, bad anchor, or an off-map excursion).
    if (nav.rover) {
      const r = nav.rover;
      const [rx, rz] = this._px(f, r.x, r.z);
      const w2 = (car.width / 2) * f.s, l2 = (car.length / 2) * f.s;
      const yaw = (r.yaw_deg || 0) * Math.PI / 180;
      const cw2 = car.width / 2, cl2 = car.length / 2;
      const clr = this.map.clearance || 0;
      let violating = false;
      for (const o of (this.map.obstacles || [])) {
        if (r.x + cw2 > o.x - clr && r.x - cw2 < o.x + o.w + clr &&
            r.z + cl2 > o.z - clr && r.z - cl2 < o.z + o.h + clr) {
          violating = true; break;
        }
      }
      c.save();
      c.translate(rx, rz);
      c.rotate(yaw);
      c.fillStyle = violating ? "rgba(248,81,73,.9)" : "rgba(47,129,247,.85)";
      c.strokeStyle = violating ? "#ffb3ad" : "#9ecbff"; c.lineWidth = 1.5;
      this._roundRect(c, -w2, -l2, 2 * w2, 2 * l2, Math.min(5, w2 / 2));
      c.fill(); c.stroke();
      // heading wedge (rover noses "up" the map)
      c.fillStyle = "#eef3f9";
      c.beginPath();
      c.moveTo(0, -l2 + 3);
      c.lineTo(-w2 * 0.45, -l2 * 0.35);
      c.lineTo(w2 * 0.45, -l2 * 0.35);
      c.closePath(); c.fill();
      c.restore();
      this._label(c, rx, rz + l2 + 14, "ROVER", "#9ecbff");
    }

    // drag-in-progress preview: a translucent rover outline following the
    // pointer (red if the drop point would land in an obstacle keep-out),
    // drawn at the live rover's current heading since a drag doesn't change
    // orientation, only position.
    if (this._dragging && this._dragPos) {
      const [dx, dz] = this._dragPos;
      const blocked = this._footprintBlocked(dx, dz);
      const [px, pz] = this._px(f, dx, dz);
      const w2 = (car.width / 2) * f.s, l2 = (car.length / 2) * f.s;
      const yaw = ((nav.rover && nav.rover.yaw_deg) || 0) * Math.PI / 180;
      c.save();
      c.globalAlpha = 0.6;
      c.translate(px, pz);
      c.rotate(yaw);
      c.fillStyle = blocked ? "rgba(248,81,73,.9)" : "rgba(47,129,247,.85)";
      c.strokeStyle = blocked ? "#ffb3ad" : "#9ecbff"; c.lineWidth = 1.5;
      c.setLineDash([4, 3]);
      this._roundRect(c, -w2, -l2, 2 * w2, 2 * l2, Math.min(5, w2 / 2));
      c.fill(); c.stroke();
      c.restore();
    }
  }

  _roundRect(c, x, y, w, h, r) {
    c.beginPath();
    c.moveTo(x + r, y);
    c.arcTo(x + w, y, x + w, y + h, r);
    c.arcTo(x + w, y + h, x, y + h, r);
    c.arcTo(x, y + h, x, y, r);
    c.arcTo(x, y, x + w, y, r);
    c.closePath();
  }

  _label(c, px, pz, text, color) {
    c.font = "bold 11px sans-serif";
    const w = c.measureText(text).width;
    const x = Math.max(2, Math.min(px - w / 2, this.canvas.width - w - 4));
    const y = Math.max(12, pz);
    c.fillStyle = "rgba(0,0,0,.65)";
    c.fillRect(x - 3, y - 10, w + 6, 13);
    c.fillStyle = color;
    c.fillText(text, x, y);
  }
}

window.BevMap = BevMap;
