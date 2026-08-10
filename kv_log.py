#!/usr/bin/env python3
"""Sample vLLM's Prometheus endpoint during a run.

Plan §4 E1 asks for KV-cache utilisation at every load point, and identifies
saturation as "growing queue" -- neither is visible in the load generator's
output, only in the server's own metrics. This polls /metrics alongside the
power logger and is summarised over the same measurement window.

  kv_log.py poll --port 8000 --out run.kv.csv --interval 2
  kv_log.py summarize --log run.kv.csv --start ISO --end ISO

Scraping is one small HTTP GET every two seconds against a text endpoint. It is
constant across every point in a series, so it cannot bias a comparison between
them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import sys
import time
import urllib.request
from datetime import datetime

# Gauges worth having per point. Counters are recorded as levels; the summary
# reports the delta across the window, which is what "preemptions during this
# run" means.
GAUGES = [
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_prefix_cache_hit_rate",
]
COUNTERS = [
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
]
FIELDS = ["timestamp"] + GAUGES + COUNTERS

LINE_RE = re.compile(r"^(?P<name>vllm:[a-z_0-9]+)(?:\{[^}]*\})?\s+(?P<value>[-0-9.eE+]+)$")


def scrape(url: str, timeout: float = 5.0) -> dict[str, float]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", "replace")
    out: dict[str, float] = {}
    for line in body.splitlines():
        if not line or line[0] == "#":
            continue
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        name = m.group("name")
        if name not in GAUGES and name not in COUNTERS:
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        # One engine per server here; if a name repeats, sum it.
        out[name] = out.get(name, 0.0) + value
    return out


def poll(host: str, port: int, out_path: str, interval: float,
         duration: float | None) -> int:
    url = f"http://{host}:{port}/metrics"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    stop = {"flag": False}

    def _handle(signum, frame):  # noqa: ARG001
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    started = time.time()
    n, errors = 0, 0
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        while not stop["flag"]:
            try:
                vals = scrape(url)
                w.writerow([datetime.now().isoformat()]
                           + [vals.get(k, "") for k in GAUGES + COUNTERS])
                n += 1
                if n % 10 == 0:
                    fh.flush()
            except Exception:  # noqa: BLE001
                errors += 1
                if errors > 100:
                    break
            if duration is not None and (time.time() - started) >= duration:
                break
            time.sleep(interval)
    print(f"[kv_log] wrote {n} samples ({errors} scrape errors) to {out_path}")
    return 0


def _f(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def summarize(path: str, start: str | None, end: str | None) -> dict:
    lo = datetime.fromisoformat(start) if start else None
    hi = datetime.fromisoformat(end) if end else None
    rows = []
    with open(path, newline="") as fh:
        for raw in csv.DictReader(fh):
            try:
                ts = datetime.fromisoformat(raw["timestamp"])
            except (ValueError, KeyError):
                continue
            if lo and ts < lo:
                continue
            if hi and ts > hi:
                continue
            rows.append(raw)

    if not rows:
        return {"available": False, "reason": "no samples in the measurement window"}

    out: dict = {"available": True, "n_samples": len(rows)}
    for name in GAUGES:
        vals = [v for v in (_f(r.get(name, "")) for r in rows) if v is not None]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        key = name.split(":", 1)[1]
        out[key] = {
            "min": round(min(vals), 4),
            "mean": round(sum(vals) / len(vals), 4),
            "p95": round(vals_sorted[min(len(vals_sorted) - 1,
                                         int(0.95 * (len(vals_sorted) - 1)))], 4),
            "max": round(max(vals), 4),
        }
    for name in COUNTERS:
        vals = [v for v in (_f(r.get(name, "")) for r in rows) if v is not None]
        if not vals:
            continue
        key = name.split(":", 1)[1]
        # Counters only rise; the delta across the window is what happened
        # during this run rather than since the server started.
        out[key + "_delta"] = round(max(vals) - min(vals), 3)

    # Saturation signal: a queue that grows and stays grown, rather than one
    # that spikes and drains.
    waiting = [v for v in (_f(r.get("vllm:num_requests_waiting", "")) for r in rows)
               if v is not None]
    if len(waiting) >= 4:
        q = len(waiting) // 4
        first, last = waiting[:q], waiting[-q:]
        out["queue_growth"] = {
            "first_quarter_mean": round(sum(first) / len(first), 2),
            "last_quarter_mean": round(sum(last) / len(last), 2),
            "growing": (sum(last) / len(last)) > (sum(first) / len(first)) + 1.0,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("poll")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--out", required=True)
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--duration", type=float, default=None)

    s = sub.add_parser("summarize")
    s.add_argument("--log", required=True)
    s.add_argument("--start")
    s.add_argument("--end")
    s.add_argument("--out")

    a = ap.parse_args()
    if a.cmd == "poll":
        return poll(a.host, a.port, a.out, a.interval, a.duration)
    summary = summarize(a.log, a.start, a.end)
    text = json.dumps(summary, indent=2)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
        print(f"[kv_log] wrote {a.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
