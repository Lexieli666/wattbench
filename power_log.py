#!/usr/bin/env python3
"""GPU power logger and joule integrator for WattBench.

Three modes:

  poll       Sample nvidia-smi telemetry to a CSV until stopped (or for --duration).
  idle       Measure the idle-power baseline and write a small JSON summary.
  integrate  Turn a CSV from `poll` into energy (J), mean power, clock/thermal
             stats and throttle flags, optionally restricted to a time window.

Energy is integrated trapezoidally over sample timestamps rather than taken as
mean-power x duration, so an uneven sample cadence (which WSL2 does produce)
does not bias the result. `integrate` reports both so the two can be compared;
they should agree to well under a percent on a healthy log.

All power numbers are whole-board draw as reported by NVML. There is no
software way to separate SM from memory/VRM draw on this card, so J/token is
reported both raw and idle-subtracted and never claimed to be compute-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta

# Fields requested from nvidia-smi, in order. Keep in sync with FIELD_NAMES.
QUERY_FIELDS = [
    "timestamp",
    "power.draw",
    "power.limit",
    "clocks.sm",
    "clocks.mem",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "pstate",
    "clocks_throttle_reasons.sw_power_cap",
    "clocks_throttle_reasons.hw_thermal_slowdown",
    "clocks_throttle_reasons.sw_thermal_slowdown",
    "clocks_throttle_reasons.hw_slowdown",
]

FIELD_NAMES = [
    "timestamp",
    "power_w",
    "power_limit_w",
    "clock_sm_mhz",
    "clock_mem_mhz",
    "temp_c",
    "util_gpu_pct",
    "util_mem_pct",
    "mem_used_mib",
    "pstate",
    "throttle_sw_power_cap",
    "throttle_hw_thermal",
    "throttle_sw_thermal",
    "throttle_hw_slowdown",
]

NVIDIA_SMI = os.environ.get("WATTBENCH_NVIDIA_SMI", "nvidia-smi")

# nvidia-smi timestamp format, e.g. "2026/08/09 22:18:32.963"
TS_FORMAT = "%Y/%m/%d %H:%M:%S.%f"


# --------------------------------------------------------------------------
# poll
# --------------------------------------------------------------------------


def poll(out_path: str, interval_ms: int, duration_s: float | None) -> int:
    """Stream nvidia-smi telemetry into a CSV until SIGTERM/SIGINT or timeout."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    cmd = [
        NVIDIA_SMI,
        f"--query-gpu={','.join(QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
        f"--loop-ms={interval_ms}",
    ]

    stop = {"flag": False}

    def _handle(signum, frame):  # noqa: ARG001
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    started = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
    )

    n = 0
    failed = False
    try:
        with open(out_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(FIELD_NAMES)
            fh.flush()
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != len(FIELD_NAMES):
                    # Malformed / warning line from the driver: keep going, but
                    # do not silently pretend it was a sample.
                    print(f"[power_log] skipping malformed line: {line}", file=sys.stderr)
                    continue
                writer.writerow(parts)
                n += 1
                if n % 20 == 0:
                    fh.flush()
                if stop["flag"]:
                    break
                if duration_s is not None and (time.time() - started) >= duration_s:
                    break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        err = ""
        if proc.stderr is not None:
            err = proc.stderr.read()
        if n == 0:
            print(
                f"[power_log] ERROR: no samples captured. nvidia-smi stderr:\n{err}",
                file=sys.stderr,
            )
            failed = True

    if failed:
        return 1
    print(f"[power_log] wrote {n} samples to {out_path}")
    return 0


# --------------------------------------------------------------------------
# parsing / integration
# --------------------------------------------------------------------------


def _f(value: str) -> float | None:
    """Parse a numeric telemetry field; nvidia-smi emits [N/A] for unsupported."""
    value = value.strip()
    if not value or value.startswith("[") or value.lower() in {"n/a", "na"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _active(value: str) -> bool:
    return value.strip().lower() == "active"


def read_log(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            ts_raw = (raw.get("timestamp") or "").strip()
            try:
                ts = datetime.strptime(ts_raw, TS_FORMAT)
            except ValueError:
                continue
            power = _f(raw.get("power_w", ""))
            if power is None:
                continue
            rows.append(
                {
                    "ts": ts,
                    "power_w": power,
                    "power_limit_w": _f(raw.get("power_limit_w", "")),
                    "clock_sm_mhz": _f(raw.get("clock_sm_mhz", "")),
                    "clock_mem_mhz": _f(raw.get("clock_mem_mhz", "")),
                    "temp_c": _f(raw.get("temp_c", "")),
                    "util_gpu_pct": _f(raw.get("util_gpu_pct", "")),
                    "mem_used_mib": _f(raw.get("mem_used_mib", "")),
                    "throttle_sw_power_cap": _active(raw.get("throttle_sw_power_cap", "")),
                    "throttle_hw_thermal": _active(raw.get("throttle_hw_thermal", "")),
                    "throttle_sw_thermal": _active(raw.get("throttle_sw_thermal", "")),
                    "throttle_hw_slowdown": _active(raw.get("throttle_hw_slowdown", "")),
                }
            )
    rows.sort(key=lambda r: r["ts"])
    return rows


def _pct(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def integrate(
    rows: list[dict],
    idle_baseline_w: float | None = None,
    max_gap_s: float = 5.0,
) -> dict:
    """Trapezoidally integrate power over time.

    Gaps longer than max_gap_s are excluded from both energy and duration and
    counted, so a stalled poller inflates neither. A run whose log has gaps is
    reported as such rather than quietly patched.
    """
    if len(rows) < 2:
        return {
            "error": "fewer than 2 samples",
            "n_samples": len(rows),
        }

    energy_j = 0.0
    left_j = 0.0     # left-Riemann sum over the same intervals
    right_j = 0.0    # right-Riemann sum over the same intervals
    integrated_s = 0.0
    gaps = 0
    gap_s_total = 0.0
    dts: list[float] = []

    for a, b in zip(rows, rows[1:]):
        dt = (b["ts"] - a["ts"]).total_seconds()
        if dt <= 0:
            continue
        if dt > max_gap_s:
            gaps += 1
            gap_s_total += dt
            continue
        energy_j += 0.5 * (a["power_w"] + b["power_w"]) * dt
        left_j += a["power_w"] * dt
        right_j += b["power_w"] * dt
        integrated_s += dt
        dts.append(dt)

    powers = [r["power_w"] for r in rows]
    clocks = [r["clock_sm_mhz"] for r in rows if r["clock_sm_mhz"] is not None]
    temps = [r["temp_c"] for r in rows if r["temp_c"] is not None]
    utils = [r["util_gpu_pct"] for r in rows if r["util_gpu_pct"] is not None]
    mems = [r["mem_used_mib"] for r in rows if r["mem_used_mib"] is not None]
    limits = sorted({r["power_limit_w"] for r in rows if r["power_limit_w"] is not None})

    powers_sorted = sorted(powers)
    clocks_sorted = sorted(clocks)
    dts_sorted = sorted(dts)

    wall_s = (rows[-1]["ts"] - rows[0]["ts"]).total_seconds()
    mean_power_integrated = energy_j / integrated_s if integrated_s > 0 else None
    mean_power_arith = sum(powers) / len(powers)

    n_pcap = sum(1 for r in rows if r["throttle_sw_power_cap"])
    n_hwt = sum(1 for r in rows if r["throttle_hw_thermal"])
    n_swt = sum(1 for r in rows if r["throttle_sw_thermal"])
    n_hws = sum(1 for r in rows if r["throttle_hw_slowdown"])

    # Clock sag: fraction of *busy* samples (util >= 50%) whose SM clock sits
    # below 90% of the busy-sample p95 clock. This catches a card that boosts
    # fine at first and settles lower as it heats, which a max/min pair hides.
    busy = [r for r in rows if (r["util_gpu_pct"] or 0) >= 50 and r["clock_sm_mhz"] is not None]
    busy_clocks_sorted = sorted(r["clock_sm_mhz"] for r in busy)
    clock_sag_frac = None
    busy_clock_p95 = _pct(busy_clocks_sorted, 0.95)
    if busy and busy_clock_p95:
        thresh = 0.90 * busy_clock_p95
        clock_sag_frac = sum(1 for c in busy_clocks_sorted if c < thresh) / len(busy_clocks_sorted)

    out = {
        "n_samples": len(rows),
        "t_start": rows[0]["ts"].isoformat(),
        "t_end": rows[-1]["ts"].isoformat(),
        "wall_s": round(wall_s, 3),
        "integrated_s": round(integrated_s, 3),
        "sample_gaps": gaps,
        "sample_gap_s_total": round(gap_s_total, 3),
        "sample_interval_s": {
            "p50": round(_pct(dts_sorted, 0.50), 4) if dts_sorted else None,
            "p95": round(_pct(dts_sorted, 0.95), 4) if dts_sorted else None,
            "max": round(max(dts_sorted), 4) if dts_sorted else None,
        },
        "energy_j": round(energy_j, 2),
        "mean_power_w": round(mean_power_integrated, 3) if mean_power_integrated else None,
        "mean_power_w_arithmetic": round(mean_power_arith, 3),
        "power_w": {
            "min": round(min(powers), 2),
            "p50": round(_pct(powers_sorted, 0.50), 2),
            "p95": round(_pct(powers_sorted, 0.95), 2),
            "max": round(max(powers), 2),
        },
        "power_limit_w_observed": limits,
        "clock_sm_mhz": {
            "min": round(min(clocks), 1) if clocks else None,
            "p50": round(_pct(clocks_sorted, 0.50), 1) if clocks else None,
            "max": round(max(clocks), 1) if clocks else None,
        },
        "temp_c": {
            "min": round(min(temps), 1) if temps else None,
            "p50": round(_pct(temps, 0.50), 1) if temps else None,
            "max": round(max(temps), 1) if temps else None,
        },
        "util_gpu_pct_mean": round(sum(utils) / len(utils), 1) if utils else None,
        "mem_used_mib_max": round(max(mems), 0) if mems else None,
        "throttle": {
            "sw_power_cap_samples": n_pcap,
            "hw_thermal_samples": n_hwt,
            "sw_thermal_samples": n_swt,
            "hw_slowdown_samples": n_hws,
            "sw_power_cap_frac": round(n_pcap / len(rows), 4),
            "thermal_any_frac": round((n_hwt + n_swt) / len(rows), 4),
            "busy_clock_p95_mhz": round(busy_clock_p95, 1) if busy_clock_p95 else None,
            "busy_clock_sag_frac": round(clock_sag_frac, 4) if clock_sag_frac is not None else None,
            "n_busy_samples": len(busy),
        },
    }

    # Two independent cross-checks on the integral.
    #
    # discretisation_bound is the one that gates a run. Left- and right-Riemann
    # sums over the same intervals bracket the true integral for a monotone
    # segment, and the trapezoid is their midpoint, so half their spread over
    # the total is a bound on the error introduced by sampling at ~2Hz rather
    # than continuously. It answers the question that matters: is this sample
    # rate fast enough to integrate this signal?
    #
    # mean_power_crosscheck is the plan's "integrated energy ~= mean power x
    # duration". It is reported but does NOT gate, because the unweighted mean
    # of samples is only equal to the time-weighted mean when the sample
    # cadence is uniform -- and it is not under WSL2, where nvidia-smi returns
    # faster at idle than under load. On a short window containing a cold-start
    # ramp the two legitimately differ by a few percent with nothing wrong.
    if mean_power_integrated is not None and energy_j:
        naive_j = mean_power_arith * integrated_s
        out["crosscheck"] = {
            "naive_energy_j": round(naive_j, 2),
            "abs_rel_diff": round(abs(naive_j - energy_j) / energy_j, 5),
            "left_riemann_j": round(left_j, 2),
            "right_riemann_j": round(right_j, 2),
            "discretisation_bound": round(abs(right_j - left_j) / (2 * energy_j), 5),
        }

    # A thermally throttled or clock-sagging run is flagged, not discarded.
    flags = []
    if n_hwt or n_swt:
        flags.append("thermal_throttle")
    if clock_sag_frac is not None and clock_sag_frac > 0.10:
        flags.append("clock_sag")
    if gaps:
        flags.append("sample_gaps")
    if len(limits) > 1:
        flags.append("power_limit_changed_mid_run")
    out["flags"] = flags

    if idle_baseline_w is not None:
        idle_j = idle_baseline_w * integrated_s
        out["idle_baseline_w"] = idle_baseline_w
        out["idle_energy_j"] = round(idle_j, 2)
        out["incremental_energy_j"] = round(energy_j - idle_j, 2)

    return out


def window(rows: list[dict], start: str | None, end: str | None) -> list[dict]:
    """Restrict rows to [start, end] given ISO-8601 local timestamps."""
    lo = datetime.fromisoformat(start) if start else None
    hi = datetime.fromisoformat(end) if end else None
    out = [r for r in rows if (lo is None or r["ts"] >= lo) and (hi is None or r["ts"] <= hi)]
    # Guard against a window that lands between samples: fall back to the
    # nearest bracketing samples rather than returning nothing.
    if not out and rows and lo is not None:
        nearest = min(rows, key=lambda r: abs((r["ts"] - lo).total_seconds()))
        out = [nearest]
    return out


# --------------------------------------------------------------------------
# idle baseline
# --------------------------------------------------------------------------


def idle(out_path: str, duration_s: float, interval_ms: int, settle_s: float,
         series: str | None = None) -> int:
    """Measure idle draw. The first settle_s seconds are discarded.

    Idle draw is not a constant of the machine: it moves with what the Windows
    desktop happens to be doing, driver state, and ambient temperature. A
    baseline is therefore measured per series and stamped with the series name
    and time, and every run records which baseline it subtracted.
    """
    tmp = out_path + ".samples.csv"
    print(f"[power_log] measuring idle baseline for {duration_s:.0f}s "
          f"(discarding first {settle_s:.0f}s)...")
    rc = poll(tmp, interval_ms, duration_s)
    if rc != 0:
        return rc
    rows = read_log(tmp)
    if not rows:
        print("[power_log] ERROR: idle log empty", file=sys.stderr)
        return 1
    cutoff = rows[0]["ts"] + timedelta(seconds=settle_s)
    kept = [r for r in rows if r["ts"] >= cutoff] or rows
    summary = integrate(kept)
    summary["mode"] = "idle_baseline"
    summary["series"] = series
    summary["measured_at"] = datetime.now().astimezone().isoformat()
    summary["settle_s_discarded"] = settle_s
    summary["samples_csv"] = os.path.basename(tmp)
    summary["idle_power_w"] = summary.get("mean_power_w")

    # An idle baseline taken while something is still using the GPU is worse
    # than none: it would be subtracted from every J/token in the series.
    util = summary.get("util_gpu_pct_mean")
    if util is not None and util > 25:
        summary["warning"] = (
            f"mean GPU utilisation was {util}% during the idle measurement; "
            f"something else is using the card"
        )
        print(f"[power_log] WARNING: {summary['warning']}", file=sys.stderr)

    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[power_log] idle baseline = {summary.get('idle_power_w')} W "
          f"over {summary.get('integrated_s')}s -> {out_path}")
    return 0


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("poll", help="stream telemetry to CSV")
    p.add_argument("--out", required=True)
    p.add_argument("--interval-ms", type=int, default=500)
    p.add_argument("--duration", type=float, default=None,
                   help="stop after N seconds (default: run until signalled)")

    i = sub.add_parser("idle", help="measure idle baseline")
    i.add_argument("--out", required=True)
    i.add_argument("--duration", type=float, default=120.0)
    i.add_argument("--interval-ms", type=int, default=500)
    i.add_argument("--settle", type=float, default=15.0)
    i.add_argument("--series", default=None,
                   help="experiment series this baseline belongs to, e.g. E0 or E1-chat")

    g = sub.add_parser("integrate", help="CSV -> energy summary JSON")
    g.add_argument("--log", required=True)
    g.add_argument("--idle-baseline-w", type=float, default=None)
    g.add_argument("--start", default=None, help="ISO-8601 local start of window")
    g.add_argument("--end", default=None, help="ISO-8601 local end of window")
    g.add_argument("--out", default=None, help="write JSON here (default: stdout)")
    g.add_argument("--max-gap-s", type=float, default=5.0)

    a = ap.parse_args()

    if a.cmd == "poll":
        return poll(a.out, a.interval_ms, a.duration)
    if a.cmd == "idle":
        return idle(a.out, a.duration, a.interval_ms, a.settle, a.series)
    if a.cmd == "integrate":
        rows = read_log(a.log)
        rows = window(rows, a.start, a.end)
        summary = integrate(rows, a.idle_baseline_w, a.max_gap_s)
        summary["source_log"] = os.path.basename(a.log)
        if a.start or a.end:
            summary["window"] = {"start": a.start, "end": a.end}
        text = json.dumps(summary, indent=2)
        if a.out:
            with open(a.out, "w") as fh:
                fh.write(text + "\n")
            print(f"[power_log] wrote {a.out}")
        else:
            print(text)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
