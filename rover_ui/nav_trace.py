"""Lightweight nav-state tracer. Polls /api/status at 5 Hz and appends a line
to /tmp/nav_trace.log whenever the nav status OR message changes — so the exact
reason a move stops (E-STOP / leg timeout / cancelled / no_path / arrived) is
captured with the pose, target azimuth/range and source at that instant.
"""
import json, time, urllib.request

URL = "http://127.0.0.1:8000/api/status"
OUT = "/tmp/nav_trace.log"
last = None

with open(OUT, "a") as fh:
    fh.write("=== trace started ===\n"); fh.flush()
    while True:
        try:
            s = json.load(urllib.request.urlopen(URL, timeout=2))
            n = s.get("nav") or {}
            key = (n.get("status"), n.get("message"))
            if key != last:
                last = key
                r = n.get("rover") or {}
                t = n.get("target") or {}
                line = ("%.1f  status=%-9s moving=%s  rover=(%s,%s,%s°)  "
                        "leg=%s  target az=%s rng=%s src=%s stale=%s  | %s\n" % (
                    time.time() % 100000,
                    n.get("status"), n.get("moving"),
                    r.get("x"), r.get("z"), r.get("yaw_deg"),
                    n.get("leg"),
                    t.get("az_deg"), t.get("range_m"), t.get("source"), t.get("stale"),
                    n.get("message") or ""))
                fh.write(line); fh.flush()
        except Exception:
            pass
        time.sleep(0.2)
