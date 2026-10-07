#!/usr/bin/env python3
"""Sample a serving stack's Prometheus endpoint during a run.

Plan §4 E1 asks for KV-cache utilisation at every load point, and identifies
saturation as "growing queue" -- neither is visible in the load generator's
output, only in the server's own metrics. This polls /metrics alongside the
power logger and is summarised over the same measurement window.

  kv_log.py poll --port 8000 --out run.kv.csv --interval 2
  kv_log.py summarize --log run.kv.csv --start ISO --end ISO

All three stacks in this project are scraped through the same columns:
llama.cpp's `llamacpp:*` and the PyTorch arm's `pytorch:*` names are mapped
onto vLLM's where they mean the same thing (see ALIASES), so the stack table
can put them side by side without a second code path per stack. Names with no
vLLM counterpart (the PyTorch arm's allocator gauges and CUDA-time counters)
keep their own prefix and are summarised the same way. Where a stack genuinely has no counterpart the column is simply
absent -- llama.cpp does not preempt, so it reports no preemption counter, and
that shows up as "not measured" rather than as zero.

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
    # pytorch arm only: the allocator's own view of the process, which the
    # nvidia-smi figure in the power log cannot give for a pre-allocating
    # server. Summarised as max over the window == the peak.
    "pytorch:memory_allocated_bytes",
    "pytorch:max_memory_allocated_bytes",
    "pytorch:memory_reserved_bytes",
    "pytorch:max_memory_reserved_bytes",
    "pytorch:batch_size_last",
]
COUNTERS = [
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    # pytorch arm only: CUDA-event time inside generate(), and how many
    # generate() calls (static batches) the window contained.
    "pytorch:cuda_time_total_ms",
    "pytorch:prefill_time_total_ms",
    "pytorch:generate_calls_total",
]
FIELDS = ["timestamp"] + GAUGES + COUNTERS

# llama.cpp's server exposes the same quantities under its own names. Mapping
# them here rather than in the tables keeps one schema on disk for both stacks.
#
# Two columns stay empty for llama.cpp, and both absences are findings rather
# than gaps to paper over with a zero:
#
#   preemptions        llama.cpp has no such counter because it does not
#                      preempt. A request that finds no free slot is deferred
#                      before it starts, never evicted after it starts.
#   KV utilisation     this build (b1-6b4344e) exports no kv_cache_usage_ratio,
#                      and the quantity would not mean the same thing anyway:
#                      llama.cpp partitions its KV into fixed per-slot budgets
#                      rather than sharing one pool the way vLLM does. The
#                      alias is kept for forward compatibility and currently
#                      never matches.
ALIASES = {
    "llamacpp:kv_cache_usage_ratio": "vllm:kv_cache_usage_perc",
    "llamacpp:requests_processing": "vllm:num_requests_running",
    "llamacpp:requests_deferred": "vllm:num_requests_waiting",
    "llamacpp:prompt_tokens_total": "vllm:prompt_tokens_total",
    "llamacpp:tokens_predicted_total": "vllm:generation_tokens_total",
    # The pytorch arm (pytorch_server.py). "running" is rows in the batch
    # currently inside generate(); "waiting" is requests queued for the next
    # static batch -- the queue that static batching creates and that the
    # stack table needs to show. KV utilisation has no counterpart: a
    # DynamicCache has no fixed pool to be a fraction of, so the column stays
    # empty, as it does for llama.cpp. Preemptions: none, it never evicts.
    "pytorch:num_requests_running": "vllm:num_requests_running",
    "pytorch:num_requests_waiting": "vllm:num_requests_waiting",
    "pytorch:prompt_tokens_total": "vllm:prompt_tokens_total",
    "pytorch:generation_tokens_total": "vllm:generation_tokens_total",
}

LINE_RE = re.compile(
    r"^(?P<name>(?:vllm|llamacpp|pytorch):[a-z_0-9]+)(?:\{[^}]*\})?\s+(?P<value>[-0-9.eE+]+)$")


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
        name = ALIASES.get(m.group("name"), m.group("name"))
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
    if lo and hi:
        # Lets a per-window rate be formed from a counter delta downstream.
        out["window_s"] = round((hi - lo).total_seconds(), 3)
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
