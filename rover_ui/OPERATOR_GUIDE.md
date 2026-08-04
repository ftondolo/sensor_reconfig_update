# Rover Operator UI — Technical Guide

This document describes the operator web console for the BEV-map navigation
demo: every panel, control, label, and status value, plus the navigation
behavior each control drives. It assumes familiarity with the system at a
code level (Python/JS) but not prior exposure to this particular UI.

Two pages are served by the same FastAPI backend:

| Page | URL | Audience |
|---|---|---|
| **Operator console** | `http://<host>:8000/` | the person driving the rover — clickable map, all controls |
| **Audience display** | `http://<host>:8000/audience` | read-only, large-format — for a crowd/demo screen, no controls |

Both pages render the same live map via a shared client-side renderer
(`map_view.js`, class `BevMap`) and receive state over one WebSocket
(`/ws/telemetry`, ~15 Hz). This guide focuses on the **operator console**
(`index.html` / `app.js`), with a shorter section at the end covering the
audience display and the underlying HTTP API, since a technical operator will
likely need both.

---

## 1. System model (context for the rest of this document)

Before describing the UI, the mental model it exposes:

- A **map** (`map.json`) is a pre-surveyed 6 m × 6 m (default) rectangular
  arena in millimetres, with a fixed set of rectangular **obstacles** and a
  **start cell** (`rover_start`).
- **Coordinate frame**: `x` increases to the right, `z` increases **downward**
  on the map (top edge is `z = 0`). The rover physically noses "up" the map —
  i.e., forward motion decreases `z`. This is not the usual "bottom-left
  origin" convention; the origin `(0, 0)` is the **top-left** corner.
- The rover's live position on this map is not measured directly — it is
  **anchored**: the T265 tracking camera's own world-frame pose is captured
  once (at startup, or whenever the operator re-anchors) and tied to a known
  map coordinate. All subsequent map positions are that anchor plus the
  T265's displacement since. This means map-frame accuracy depends entirely on
  the anchor being correct — if the physical rover was not actually standing
  where you told the software it was, every reported position is offset by
  that same error until the next re-anchor.
- A **detection** (a person seen by RGB, thermal, and/or mmWave radar) is
  fused into a single bearing + range reading and projected onto the map using
  the rover's current pose. This projected point is the **target**.
- The rover does not drive to the target directly — it drives to a
  **standoff goal**, nominally 2 m from the target on the rover's side, then
  stops. If that exact point would put the rover's footprint inside an
  obstacle's keep-out, the goal is nudged to the nearest clear point.
- Movement is planned as an **obstacle-avoiding rectilinear path** (a sequence
  of axis-aligned legs, not a straight line) and executed leg-by-leg, each leg
  re-derived from the live pose right before it runs.

Every control and readout on the operator console maps onto some part of this
pipeline. The sections below go panel by panel.

---

## 2. Page layout

The operator console (`/`) has three regions:

1. **Top bar** — page title, connection status, live nav-state chip, link to
   the audience view.
2. **Arena Map panel** (left/main) — the clickable/draggable BEV map, a
   status strip of live readouts beneath it, and a color legend.
3. **Side panel** (right) — four stacked cards: **Navigation**, **Pose
   anchor**, **Drive**, and **Detection** (a small live thumbnail).

### 2.1 Top bar

| Element | Meaning |
|---|---|
| Title text | Static label; the subtitle reminds the operator that clicking the map drives the rover and that the audience view mirrors it. |
| `nav: <status>` chip | Amber pill in the header showing the current navigation status word in lowercase (e.g. `nav: moving`). Mirrors the big status badge on the map panel; kept in the header so it's visible without looking at the map. |
| Connection dot | A small circle: **green and glowing** = the WebSocket to the backend is open and telemetry is flowing; **red** = disconnected. The frontend auto-reconnects every 1.5 s while red, so a brief red flash usually self-resolves. If it stays red, the backend process itself is down or unreachable, not just a slow network. |
| `Audience ↗` link | Opens the read-only audience display in a new tab. Useful for putting a second screen in front of onlookers while you operate from the console. |

---

## 3. Arena Map panel

### 3.1 The canvas itself

The map is drawn 1:1 by proportion inside a 900×900 px canvas (`BevMap.draw()`
in `map_view.js`), fitted to the arena's actual aspect ratio with a small
padding margin. Everything on it:

- **Floor** — dark background filling the arena's extent.
- **1 m grid** — faint grid lines every 1000 mm, with axis labels in metres
  along the bottom and left edges (white text).
- **Obstacles** (solid red rectangles, red border) — the raw obstacle regions
  from `map.json`.
- **Clearance halo** (faint red dashed rectangle around each obstacle) — the
  obstacle grown by the configured clearance margin. This is the zone the
  rover's *footprint* must never overlap; the planner additionally keeps the
  rover's *center* a further half-car-length away, which is not drawn (it's
  implied, not a hard visual boundary).
- **Start marker** — a small hollow green house/pin icon at the configured
  `rover_start` cell, labeled `START`. This is where "Return to start" drives
  to and where "Re-anchor pose to start cell" assumes the rover is physically
  sitting.
- **Rover trail** — a faint blue line tracing the last ~3 minutes of rover
  positions, useful for seeing the actual path just driven (as opposed to the
  planned one).
- **Planned path** (dashed amber/gold line with dots at each waypoint) — the
  active rectilinear plan, shown only while a navigation is planned/in
  progress.
- **Standoff link** (thin dashed green line) — connects the detected target to
  its standoff goal, when both are known.
- **Goal marker** (a circled X) — the standoff goal. **Green** if it's exactly
  2 m below the target (unmodified); **amber**, labeled `GOAL (adjusted)`, if
  the goal had to be nudged off that ideal point to clear an obstacle.
- **Target marker** (a pulsing dot labeled `T`) — the projected detection.
  **Red and pulsing** while fresh; **grey**, labeled `TARGET (last seen)`,
  once the detection has gone stale (no fresh confirmation within
  `NAV_TARGET_HOLD_S`, default 1.5 s) — the marker holds its last known
  position rather than disappearing, so the operator can see where the person
  was last seen even if the sensors have momentarily lost them. A small range
  readout (metres) is appended to the label when available.
- **Rover footprint** — a rounded rectangle at the rover's true footprint
  size, rotated to its live heading, with a small triangular wedge marking the
  front (the wedge points toward decreasing `z`, i.e. "up" the map). **Blue**
  under normal conditions; turns **solid red** if the rover's own footprint is
  currently overlapping an obstacle's clearance halo — this is a live
  constraint-violation indicator, not just decoration, and corresponds
  directly to the condition the E-STOP / start-blocked logic checks (see
  §4.5).
- **Drag preview** — while dragging the rover icon (see §3.3), a translucent
  outline follows the pointer; it turns red if dropping there would place the
  footprint inside an obstacle keep-out (in which case the drop is refused).

### 3.2 Click-to-navigate

Clicking anywhere on the map (when not dragging the rover icon) sends
`POST /api/nav/goto {x, z}` with the clicked map coordinates (millimetres).
This starts a fresh navigation to a **manual** target at that exact point —
it bypasses detection entirely; the "target" for this purpose is simply the
clicked point, and the standoff goal is computed 2 m below it exactly as for
a detected person. This is the fastest way to test obstacle-avoidance
routing or to drive the rover somewhere with no person actually present.

### 3.3 Drag-and-drop rover repositioning

The rover icon itself can be grabbed (pointer down within roughly one car-length
of the rover's current map position) and dragged to a new point. On release:

- If the drop point is inside an obstacle's clearance halo, the drop is
  **refused client-side** — nothing is sent to the backend, and the preview
  was already shown in red while hovering there.
- Otherwise, it calls `POST /api/nav/reset_pose {x, z}` with the dropped
  coordinates.

**This does not physically move the rover.** It has exactly the same effect
as clicking "Re-anchor pose to start cell" (§4.2), just at an arbitrary point
instead of the fixed start cell — it tells the software "the rover is
physically standing at this map point right now," re-anchoring the T265 pose
to that location. Use it when the rover's reported position has drifted from
its true physical position (e.g. after wheel slip or a bump) and you want to
correct the map-frame anchor to match reality, without walking back to the
official start cell. If you drag the icon to a spot where the rover *isn't*
actually standing, subsequent position reporting will be wrong until the next
correct re-anchor. This control is unaffected by whether a navigation is
currently in progress — dragging is not disabled while the rover is moving,
by design (a mid-move drag simply re-anchors around the rover's currently
reported position, which may itself be moving).

### 3.4 Map-strip readouts

The row of labels directly beneath the canvas, all populated live from the
`nav` object pushed over the WebSocket:

| Label | Meaning |
|---|---|
| **rover** | Live rover center position, `(x, z)` in metres, from the anchored pose. `—` until a pose exists. |
| **target** | Current projected detection target position in metres, or the manual click target if that's what's active. `—` if there is no live/held target. |
| **goal** | The standoff goal position in metres; ` (adj)` is appended if the goal was nudged off the ideal 2 m point to clear an obstacle. |
| **leg** | `current_leg/total_legs` of the active rectilinear path (e.g. `2/4`). `—` when there is no active path. |
| **pose** | `T265` under normal operation, or `MOCK` when running without real hardware (simulated pose service — see §8). |
| **accumulation** | Ghost-detection guard status (see §4.1 and §6). `—` (nothing being tracked yet), `accumulating n/need` (a new/relocated target is still building corroboration), or `confirmed` (the current track is trusted). |
| **msg** | The most recent human-readable navigation message from the backend — e.g. arrival confirmation, a planning failure reason, or an E-STOP explanation. This is the first place to look when something doesn't behave as expected. |

### 3.5 Legend

A static color key beneath the map: rover, target, standoff goal, obstacle,
planned path — matching the colors described in §3.1.

### 3.6 Status badge

The small badge in the Arena Map panel's header (`nav-badge`) always shows
the current navigation status word in uppercase, color-coded:

| Status | Badge color | Meaning |
|---|---|---|
| `IDLE` | grey | Nothing planned or moving. |
| `PLANNING` | grey | A path is being computed (very brief). |
| `MOVING` | blue | A leg is actively executing. |
| `ARRIVED` | green | The rover reached its goal within tolerance. |
| `NO_PATH` | red | The planner could not find any obstacle-free rectilinear route. |
| `BLOCKED` | red | An E-STOP fired mid-move, or a plan was refused because the rover's own position is in an obstacle zone (see §4.5). |
| `ERROR` | red | A leg failed to execute (e.g. the rover stalled — no measurable progress within the stall timeout — or lost its pose mid-move). |
| `CANCELLED` | grey | The operator (or the system) cancelled an in-progress navigation. |

This same status also drives whether **Navigate to detection** and the four
**Drive** jog buttons are disabled: they are greyed out while `status` is
`moving` or `planning`, since dispatching a new plan or manually jogging
during an active leg would conflict with it.

---

## 4. Navigation card

### 4.1 ▶ Navigate to detection

Calls `POST /api/nav/go`. Takes the **current** fused detection target (RGB
and thermal are preferred; radar range/bearing is only used standalone when
vision sees nothing), computes its standoff goal, plans a rectilinear path,
and starts driving it. This single dispatch **plans once** — it does not
continuously re-target if the person keeps walking after the plan is
computed (the rover will drive to wherever the person was standing at the
moment this button was pressed). If no detection is currently available, or
its position can't be confirmed yet (see Confirm-N below), the button call
fails with a message in the **msg** readout.

### 4.2 ■ Cancel / stop

Calls `POST /api/nav/cancel`. Immediately stops any in-progress or queued
navigation and sets status to `CANCELLED`. Safe to press at any time,
including mid-leg.

### 4.3 Auto: on/off

Toggles `POST /api/nav/auto {enabled}`. When **on**, the backend watches the
live detection continuously and automatically dispatches a navigation once a
target has held still (within `NAV_AUTO_STABLE_MM`, default 400 mm) for a
minimum dwell (`NAV_AUTO_STABLE_S`, default 1 s), subject to a cooldown
between dispatches (`NAV_AUTO_COOLDOWN_S`, default 2 s). AUTO **never**
interrupts a move already in progress — it only ever dispatches when the
rover is idle. It is designed around a **stationary** person (wait for them
to stop, then go once), not for chasing someone who is walking. Enabling
AUTO automatically disables FOLLOW (the two are mutually exclusive).

### 4.4 🎯 Follow: on/off

Toggles `POST /api/nav/follow {enabled}`. This is the mode built for a
**moving** person. Unlike AUTO, FOLLOW continuously re-targets and can
re-plan mid-move. Enabling it automatically disables AUTO and clears any
previously accumulated ghost-detection state so it locks onto whoever is
currently in front of the sensors (not a stale carry-over target).

FOLLOW runs a repeating cycle:

1. **Sense** (rover must be stationary): read the current fused target.
   Vision (RGB/thermal) targets are trusted immediately. A **radar**-sourced
   target additionally requires the rover to have been confirmed stationary
   for `NAV_FOLLOW_STILL_CONFIRM_S` (default 1 s) before it's trusted for a
   *fresh* lock — this is a genuine sensor constraint: the rover's own
   translation fakes a Doppler signature on static clutter, so a brand-new
   radar lock is only ever acquired while stationary. While this window is
   still counting down, the status line shows "follow: confirming stationary
   mmWave lock…".
2. **Commit + move**: dispatch a navigation to the confirmed target's
   standoff goal.
3. **Mid-move re-plan**: while driving, the live target is re-sampled. If it
   has moved more than `NAV_FOLLOW_REPLAN_MM` (default 300 mm) from the point
   the current move was committed to, the current leg is cancelled and a
   fresh plan is dispatched to the new position — this applies to **both**
   vision and radar sources. An already-**acquired** radar lock is
   maintained by the radar tracker through the rover's own motion (a purely
   spatial nearest-cluster association, independent of the Doppler signal
   that ego-motion would corrupt), so it stays valid for this re-plan check
   even while the rover is driving. `NAV_FOLLOW_CYCLE_S` (default 20 s) is a
   stuck-backstop: if a single commit runs this long without arriving, it's
   treated as done and the cycle restarts fresh.
4. Once the move ends (arrival, block, or the backstop), a fresh stationary
   dwell window opens and the cycle returns to Sense.

### 4.5 Ignore obstacles: on/off

Toggles `POST /api/nav/ignore_obstacles {enabled}`. This is an **operator
override for one specific situation**: the rover's own current position (or
starting position) being computed as inside an obstacle/clearance zone. It
does **not** disable obstacle avoidance generally — routing around obstacles
and the standoff-goal search still fully respect every obstacle when this is
on.

Concretely, enabling it suppresses two independent checks:

- The **planner's start-blocked refusal**: normally, if the rover's current
  position is itself flagged as being inside a keep-out (most often because
  of pose drift, a bad anchor, or the rover having been physically placed
  somewhere unexpected), the planner refuses to compute a path at all. With
  this on, it will plan a route starting from wherever the rover is anyway.
- The **mid-move footprint E-STOP**: normally, `_drive()` polls the rover's
  footprint at ~20 Hz while a leg executes and immediately stops the rover
  with a `BLOCKED` status and message
  `"E-STOP: rover footprint entered an obstacle zone at (x, z) mm"` if it
  detects an overlap. With this on, that safety stop is suppressed.

Use this deliberately and briefly — it exists for situations like "the pose
anchor is known to be slightly off and the rover is not actually near the
obstacle it thinks it's in," not as a general safety bypass. Takes effect
immediately, including on a move already underway.

### 4.6 Speed

A slider (m/s), bounded by `NAV_SPEED_MIN`/`NAV_SPEED_MAX` (defaults
0.05–0.80 m/s) and defaulting to the rover's currently configured translation
cap. Dragging it live-updates the number shown; releasing it (`onchange`)
posts `POST /api/nav/speed {mps}`, and the backend clamps and returns the
applied value (which the slider then snaps to, in case your requested value
was out of range). This caps translation speed for **automated** moves
(button click, AUTO, FOLLOW, manual map click) — it takes effect on the very
next control tick, even for a leg already in progress. It also acts as a hard
ceiling on the separate Drive jog speed (§5.2): the effective jog speed is
`min(jog speed, this cap)`.

### 4.7 Confirm-N

A number input (integer, 1–10), added specifically to make the
**ghost-detection guard** operator-tunable. Posts
`POST /api/nav/confirm_n {n}` on change; the backend clamps to [1, 10] and
applies it immediately (the very next detection uses the new value — no
restart needed).

What it controls: before a **new or relocated** detection target is trusted
enough to move the tracked position (as opposed to an already-locked,
continuously-agreeing target, which updates immediately), the backend
requires this many **mutually consistent** detector frames in a row — same
approximate position (within `NAV_TARGET_JUMP_MM`, default 700 mm) and, for
radar, similar reflected intensity (within `NAV_TARGET_SNR_TOL`, default
40% relative tolerance). This exists to stop a single bad frame (an
occasional false detection, sensor noise, a reflection) from teleporting the
tracked target somewhere the person never actually was. The **accumulation**
map-strip readout (§3.4) shows this process live.

- **Lower values** (e.g. 1) react fastest to a genuinely new or relocated
  target, at the cost of being more susceptible to a single spurious
  detection.
- **Higher values** (up to 10) are steadier/more resistant to noise, at the
  cost of slower response to a real relocation — the person has to be seen
  consistently in the new spot for longer before the rover treats it as real.

The default (3) is a compromise; this is meant to be tuned live for whichever
sensor conditions and lighting are in play at a given demo.

---

## 5. Pose anchor card

This card is where the map-frame position tracking gets set up and
corrected. Read §1's explanation of anchoring before using it.

### 5.1 ⌖ Re-anchor pose to start cell

Calls `POST /api/nav/reset_pose {}` (no coordinates). Anchors the *current*
T265 pose to the map's configured `rover_start` cell. The hint text beneath
it reminds the operator of the required physical step: **place the rover
physically on its start cell, facing the top of the map, before pressing
this** — the software has no way to verify the rover is actually there; it
simply takes whatever the T265 currently reports and calls it "the start
cell." Pressing this with the rover somewhere else silently miscalibrates
every subsequent position report.

The current configured start cell is shown inline (`x=… m, z=… m`) so the
operator knows exactly where to place the rover first.

### 5.2 ⟳ Reload map.json / config.json

Calls `POST /api/nav/reload_map`. Hot-reloads the map and plan-config files
from disk without restarting the process — useful after hand-editing
`map.json` to add/move an obstacle, or `config.json` for footprint/clearance
changes. Refused (409) while a navigation is in progress; cancel first. This
does **not** re-anchor the pose — the live position tracking continues from
wherever it already was; if the start cell moved as part of the edit, use
Re-anchor separately if you want the rover's current physical position
associated with the new start cell.

### 5.3 Set start position

A pair of numeric inputs (`x`, `z`, in **metres**) plus a **Set start
position** button. Calls `POST /api/nav/set_start {x, z}` (converted to mm),
which **persists** a new `rover_start` into `map.json` on disk. This changes
only the *configured* start cell — i.e. where "Return to start" drives to and
where the plain Re-anchor button will anchor to next time — it does **not**
move the rover and does **not** itself re-anchor the live pose. Refused if
the requested center would land outside the map or inside an obstacle
keep-out, and refused mid-navigation. The inputs are pre-filled from the
currently loaded map on page load and again after any reload/set-start
round-trip.

---

## 6. Drive card (manual jog)

A directional pad (▲ forward, ◀ left, ▶ right, ▼ back) for direct
hold-to-move control, independent of the map/planner entirely.

- **Hold** a direction button: the UI sends `POST /api/nav/jog {x, z}` (where
  `x`/`z` ∈ {-1, 0, 1} indicate direction) immediately, then every 200 ms
  while held.
- **Release** (pointerup/leave/cancel): the UI stops the repeat timer and
  sends `POST /api/nav/jog_stop` for an immediate stop.
- **Dead-man safety**: the backend independently stops the rover if it
  doesn't receive a fresh jog call within `NAV_JOG_DEADMAN_S` (default
  0.5 s) — so a dropped network connection or a closed browser tab stops the
  rover shortly after, even if `jog_stop` never arrives.
- Driving uses **T265 heading-hold**, so the rover corrects drift and holds
  its commanded direction while jogging.
- **Disabled while a navigation is running** (`moving`/`planning` status) —
  jog buttons grey out automatically to prevent conflicting commands.
- The **Speed** slider beneath the pad sets the jog speed
  (`NAV_JOG_SPEED_MIN`/`MAX`, default 0.05–0.40 m/s), same live-update /
  on-release-post pattern as the Navigation speed slider. It is additionally
  hard-capped by the Navigation card's Speed setting (§4.6) — whichever is
  lower actually applies.

---

## 7. Detection card

A small thumbnail (`/stream/detect`, MJPEG) showing the live RGB feed with
detection boxes drawn on it, plus a `st-detector` badge showing the
detector's own live status (`LIVE` when streaming normally, `MOCK` when
running without real camera hardware, `…` while opening/initializing, or
blank/`OFF` otherwise). This is a smaller version of the same feed shown
full-size on the audience display; it exists on the operator console mainly
so the operator can visually confirm what the detector currently sees
without switching tabs.

---

## 8. "MOCK" vs "T265" / "ROS"

Several readouts (the **pose** map-strip field, the Detection badge, and the
audience display's **Control** field) can show `MOCK` instead of a real
value. This means the corresponding subsystem (T265 pose service, a
sensor, or the ROS `cmd_vel` connection) isn't available — most commonly
because the software is running on a machine without the actual hardware
attached — and a software fallback is standing in: a fake pose service that
integrates commanded velocity into a simulated position (so the whole
click-to-navigate / plan / drive flow can be exercised and watched on a
laptop), or a synthetic detection feed. This is expected and useful for
development/testing; on the real robot in normal operation, these should
read `T265` / `ROS OK` / `LIVE`.

---

## 9. Audience display (`/audience`)

A larger, read-only version of the same map plus a fuller sensor dashboard,
meant for a screen facing onlookers rather than the operator. Differences
from the operator console:

- **No click-to-navigate, no drag-and-drop, no controls at all** — purely a
  live mirror of navigation + sensor state.
- Adds full-size **RGB+Detection**, **Thermal (FLIR)**, and **Depth
  (D435i)** MJPEG streams, plus a live **mmWave radar** polar plot (range
  rings out to 6 m, one dot per radar point, color-coded by radial velocity
  sign/magnitude: blue for one direction of motion, red for the other, green
  for near-zero — consult `audience.js`'s `drawMmwave()` for the exact sign
  convention if this distinction matters for your use case). A `RADAR ...m` box
  is drawn on that plot when radar is the *primary* detection source (not
  merely supplying range to a vision box). A "radar paused — rover moving"
  label appears on the plot whenever the display-side radar feed is
  suppressed because the rover is currently translating (the same ego-motion
  policy discussed in §4.4, applied here purely for the point-cloud
  visualization).
- A large **NAVIGATION** status readout (bigger version of the same status
  word, plus the same live message), and a compact status grid: **Standoff**
  (fixed 2.0 m label), **Auto**/**Follow** on-off, **Moving** yes/no, and
  **Control** (`ROS OK` / `no ROS` / `mock`).
- A **Sensor Contribution** panel: three bars (RGB/Thermal/Radar) showing
  each modality's fused weight in the *current* detection (from the
  gated-fusion model), plus **Detections** (count), **Range**, **Bearing**,
  and **Source** (which modality the active target actually came from).
  Whichever cam-card (RGB, Thermal, or Radar) is currently the *driving*
  sensor for the top detection gets a matching colored border, mirroring the
  same selection rule the detector uses to color the bounding box on the RGB
  stream.
- An **Adaptive Compute** panel: an overall GPU-savings percentage/bar (from
  dynamically downscaling inference resolution when the scene allows it),
  and per-modality resolution "tier" indicators for RGB and Thermal (`full`,
  `3/4`, `1/2`, `1/4` — the lit box is the currently committed tier).
- A **Per-detection** list: one card per detected person with a confidence
  percentage and a small per-modality contribution bar.

---

## 10. HTTP API reference (for scripting/automation)

All endpoints are relative to the backend host (default port 8000). POST
bodies are JSON; all POST nav endpoints return the full live `nav` state
(equivalent to `GET /api/status`'s `nav` field) so a caller doesn't need a
follow-up poll.

| Method & path | Body | Effect |
|---|---|---|
| `GET /` | — | Operator console HTML |
| `GET /audience` | — | Audience display HTML |
| `GET /api/status` | — | Snapshot: all sensor statuses + full nav state |
| `GET /api/map` | — | Static map payload: size, obstacles, car footprint, clearance, rover_start, speed/jog-speed ranges, confirm_n |
| `GET /stream/{flir,d435,depth,detect,detect_thermal,detect_depth}` | — | MJPEG streams |
| `WS /ws/telemetry` | — | Pushes `{mmwave, detection, nav, status}` at ~15 Hz |
| `POST /api/nav/go` | — | Navigate to the current detection target |
| `POST /api/nav/goto` | `{x, z}` (mm) | Navigate to a manual map target |
| `POST /api/nav/cancel` | — | Cancel/stop the active navigation |
| `POST /api/nav/home` | — | Return to the configured start cell (disables Auto/Follow first) |
| `POST /api/nav/auto` | `{enabled}` | Toggle AUTO |
| `POST /api/nav/follow` | `{enabled}` | Toggle FOLLOW |
| `POST /api/nav/ignore_obstacles` | `{enabled}` | Toggle the rover's-own-position obstacle override |
| `POST /api/nav/speed` | `{mps}` | Set the navigation speed cap (clamped) |
| `POST /api/nav/jog_speed` | `{mps}` | Set the Drive-pad hold-to-move speed (clamped) |
| `POST /api/nav/confirm_n` | `{n}` | Set the ghost-guard confirm count, 1-10 (clamped) |
| `POST /api/nav/reload_map` | — | Hot-reload map.json/config.json (refused mid-nav) |
| `POST /api/nav/set_start` | `{x, z}` (mm) | Persist a new configured start cell |
| `POST /api/nav/reset_pose` | `{x?, z?}` (mm) | Anchor the current pose to a map cell (default: configured start) |
| `POST /api/nav/jog` | `{x, z}` (each -1/0/1) | One hold-to-move tick; call every ~200 ms while held |
| `POST /api/nav/jog_stop` | — | Immediate jog stop |
| `POST /api/nav/nudge` | `{axis, mm}` | Small (≤1000 mm) single-axis open-loop move, no obstacle planning |

### Key `nav` state fields (from `GET /api/status` or the WS `nav` payload)

| Field | Meaning |
|---|---|
| `status` | One of `idle`, `planning`, `moving`, `arrived`, `no_path`, `blocked`, `cancelled`, `error` |
| `message` | Latest human-readable status detail |
| `rover` | `{x, z, yaw_deg}` mm/degrees, or `null` if unanchored |
| `target` | `{x, z, age_s, range_m, az_deg, source, stale?}` or `null` |
| `goal` | `{x, z, adjusted}` or `null` |
| `path` | List of `[x, z]` mm waypoints of the active plan, or `null` |
| `leg` | Index of the waypoint currently being driven to |
| `accum` | `{phase: "none"\|"accumulating"\|"confirmed", n, need}` — ghost-guard state |
| `auto`, `follow`, `ignore_obstacles` | Current toggle states |
| `confirm_n` | Current ghost-guard threshold (1-10) |
| `speed`, `jog_speed` | Current applied speed caps, m/s |
| `moving` | Whether the rover service is currently busy executing a leg |
| `mock` | Whether the pose service is running in simulated (no-hardware) mode |
| `ros` | Whether the `cmd_vel` ROS connection is live |
| `pose_jumps` | Count of detected pose discontinuities (diagnostic) |

---

## 11. Quick troubleshooting reference

| Symptom | Likely cause / where to look |
|---|---|
| Status stuck at `BLOCKED`, message mentions "E-STOP" | Rover's footprint drifted into an obstacle keep-out mid-move. Check the pose anchor is correct; consider Ignore obstacles (§4.5) only if you're confident the rover isn't actually near a real obstacle. |
| Status `NO_PATH` | The planner found no obstacle-free rectilinear route to the goal — check the goal isn't genuinely walled in, or that the obstacle layout in `map.json` matches reality. |
| Status `ERROR` after a move | A leg failed — commonly a stall (no measurable progress within the stall timeout) or lost pose tracking; see the `message` field for the specific reason. |
| Target marker frozen on "last seen" | The live detection hasn't been re-confirmed recently (accumulation may be stuck at "accumulating" rather than reaching "confirmed") — check the **accumulation** readout and consider lowering **Confirm-N** if the sensors are consistently accurate but slow to agree. |
| Rover footprint drawn in red on the map | It is currently overlapping an obstacle's clearance halo — either the pose anchor is wrong or the rover is genuinely too close to an obstacle. |
| Everything reads `MOCK` | Running without the real T265/camera hardware attached — expected in a dev/test environment (§8). |
| Connection dot stuck red | WebSocket can't reach the backend — check the process is running and the host/port are correct. |

---

*This guide reflects the UI as implemented in `rover_ui/frontend/` (`index.html`,
`app.js`, `map_view.js`, `audience.html`, `audience.js`) and the backend in
`rover_ui/backend/` (`app.py`, `nav/navigator.py`, `config.py`) at the time of
writing. Default numeric values shown are the code's built-in defaults; any of
them may be overridden per-deployment via environment variables of the same
name (see `config.py`).*
