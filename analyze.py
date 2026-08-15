#!/usr/bin/env python3
"""Turn results/raw/*.json into the tables and plots the README reports.

Nothing here invents a number. Every measured value is read from an assembled
raw result; every external value is read from pricing.yaml and is labelled
"reported, not measured" wherever it appears. A quantity that was not measured
prints as "not run" rather than an estimate.

  ./analyze.py summary                  # every run, one line each
  ./analyze.py variance --experiment E0 # run-to-run coefficient of variation
  ./analyze.py sweep --experiment E1    # load sweep table
  ./analyze.py frontier --experiment E3 # model-size frontier table
  ./analyze.py stacks --experiment E4   # vLLM vs llama.cpp
  ./analyze.py economics                # $/1M tokens and break-even volumes
  ./analyze.py plots                    # the three headline charts
  ./analyze.py all                      # everything, into results/
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from typing import Any

import harness
import power_log

REPO = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(REPO, "results", "raw")
PLOTS = os.path.join(REPO, "results", "plots")
TABLES = os.path.join(REPO, "results", "tables")

# Categorical slots, fixed order, from the validated reference palette.
# Assigned by entity and never cycled or re-ordered when a series is filtered out.
C = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"

NOT_RUN = "not run"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_results(experiment: str | None = None, include_failed: bool = False,
                 include_superseded: bool = False) -> list[dict]:
    """Load raw results, keeping only the newest run of each point by default.

    Re-running a point (after a harness fix, or a config correction) leaves the
    earlier run on disk and in git -- raw data is append-only -- but analysis
    should use the current one. Superseded runs are counted and announced, never
    silently dropped.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(RAW, "*.json"))):
        if path.endswith(".power.json"):
            continue
        try:
            with open(path) as fh:
                r = json.load(fh)
        except json.JSONDecodeError:
            print(f"[analyze] WARN: unreadable JSON, skipping: {path}", file=sys.stderr)
            continue
        if r.get("schema_version") != 1:
            continue
        r["_path"] = os.path.basename(path)
        if not include_failed and r.get("status") != "ok":
            continue
        if experiment and r.get("experiment") != experiment:
            continue
        out.append(r)

    if include_superseded:
        return out

    # Filenames are <point_id>__<YYYYmmddTHHMMSS>.json, so lexical order on the
    # filename is chronological order.
    newest: dict[str, dict] = {}
    superseded = 0
    for r in out:
        pid = r.get("point_id") or r["_path"]
        prev = newest.get(pid)
        if prev is None or r["_path"] > prev["_path"]:
            if prev is not None:
                superseded += 1
            newest[pid] = r
        else:
            superseded += 1
    if superseded:
        print(f"[analyze] {superseded} superseded run(s) excluded; "
              f"use --include-superseded to see them", file=sys.stderr)
    return sorted(newest.values(), key=lambda r: r["_path"])


def load_pricing() -> dict:
    import yaml
    with open(os.path.join(REPO, "pricing.yaml")) as fh:
        return yaml.safe_load(fh)


def g(d: dict, *path: str, default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def fmt(v: Any, spec: str = ".3g") -> str:
    if v is None:
        return NOT_RUN
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return NOT_RUN
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(headers), sep] + [line(r) for r in rows])


# --------------------------------------------------------------------------
# per-run derived values
# --------------------------------------------------------------------------


def recomputed_goodput(r: dict) -> dict:
    """Goodput recomputed from the preserved per-request arrays.

    Points measured before 2026-08-10 01:50 carry a goodput that counted failed
    requests as SLO-meeting (a failure reports ttft 0.0, which passes every
    threshold). The per-request arrays are committed, so the correct value is
    recoverable without re-running anything -- and recomputing here rather than
    rewriting results/raw/ keeps the raw files an honest record of what the
    harness computed at the time, with git and METHODOLOGY.md carrying the
    correction.
    """
    stored = r.get("goodput") or {}
    m = r.get("metrics") or {}
    if not m.get("ttfts"):
        return stored
    slo = (r.get("config") or {}).get("slo") or {}
    try:
        fresh = harness.compute_goodput(
            m, slo.get("ttft_p95_s", 1.0), slo.get("e2e_p95_s", 10.0))
    except Exception:  # noqa: BLE001
        return stored
    if not fresh.get("available"):
        return stored
    if stored.get("goodput_req_per_s") != fresh.get("goodput_req_per_s"):
        fresh["superseded_stored_value"] = stored.get("goodput_req_per_s")
        fresh["recomputed"] = True
    return fresh


_BASELINE_CACHE: dict[str, dict] = {}
_INCR_RECOMPUTED: dict[str, float] = {}  # point_id -> relative change vs stored


def load_baseline(name: str) -> dict | None:
    """One idle baseline JSON from results/idle/, by filename."""
    if name not in _BASELINE_CACHE:
        path = os.path.join(REPO, "results", "idle", name)
        try:
            with open(path) as fh:
                _BASELINE_CACHE[name] = json.load(fh)
        except (OSError, json.JSONDecodeError):
            _BASELINE_CACHE[name] = {}
    return _BASELINE_CACHE[name] or None


def recomputed_incremental(r: dict) -> dict:
    """Incremental energy recomputed under the current idle-baseline policy.

    Runs measured before 2026-08-14 subtracted the *mean* of their idle window.
    Idle draw here is bimodal, so the mean is set by how many 1.7-4.0 s bursts
    from an external consumer happened to land in the 130 s window: across
    twelve baselines the mean ranged 18.8-27.9 W while the median never left
    18.7-22.2 W. The policy is now median subtraction (decided 2026-08-14).

    Applying it here rather than rewriting results/raw/ follows the same rule as
    recomputed_goodput: the raw file stays an honest record of what the harness
    computed at the time, and git plus METHODOLOGY.md carry the correction. The
    baseline files themselves are unchanged raw data -- only which statistic is
    read out of them changed.

    Raw J/token is untouched by any of this; only the idle-subtracted figure
    moves, and for 34 of 47 runs -- every run on an excursion-free baseline --
    it moves by less than 0.05%.
    """
    e = r.get("energy") or {}
    stored_j = e.get("incremental_energy_j")
    ej, win = e.get("energy_j"), e.get("window_s")
    src = (e.get("idle_baseline_source") or {}).get("file")
    if not (ej and win and src):
        return {"incremental_energy_j": stored_j,
                "j_per_output_token_incremental": e.get("j_per_output_token_incremental"),
                "idle_baseline_w": e.get("idle_baseline_w")}
    baseline = load_baseline(src)
    idle_w = power_log.baseline_idle_w(baseline) if baseline else None
    if idle_w is None:
        return {"incremental_energy_j": stored_j,
                "j_per_output_token_incremental": e.get("j_per_output_token_incremental"),
                "idle_baseline_w": e.get("idle_baseline_w")}
    incr = round(ej - idle_w * win, 2)
    out_tok = (r.get("metrics") or {}).get("total_output_tokens")
    fresh = {
        "incremental_energy_j": incr,
        "j_per_output_token_incremental": round(incr / out_tok, 5) if out_tok else None,
        "idle_baseline_w": idle_w,
        "idle_baseline_stat": "median",
    }
    if stored_j:
        rel = (incr - stored_j) / stored_j
        if abs(rel) > 1e-6:
            fresh["superseded_stored_value"] = stored_j
            _INCR_RECOMPUTED[r.get("point_id") or "?"] = rel
    return fresh


def precision_of(model: str | None) -> str:
    """Weight precision, read from the checkpoint name.

    The configs deliberately leave `quantization` unset so that vLLM reads
    quantization_config from the checkpoint and picks its own kernel -- forcing
    one would measure our choice rather than what a user gets. That means the
    config cannot answer this question and `server.quantization or "bf16"`
    silently labels every int4 rung as BF16. The checkpoint name can answer it;
    the kernel actually selected is recorded in each run's server log.
    """
    m = (model or "").lower()
    if "awq" in m:
        return "awq"
    if "gptq" in m:
        return "gptq"
    return "bf16"


def row_view(r: dict) -> dict:
    """Flatten one raw result into the fields every table and plot reads."""
    m = r.get("metrics") or {}
    e = r.get("energy") or {}
    gp = recomputed_goodput(r)
    incr = recomputed_incremental(r)
    sm = r.get("server_metrics") or {}
    cfg = r.get("config") or {}
    load = cfg.get("load") or {}
    server = cfg.get("server") or {}
    return {
        "point_id": r.get("point_id"),
        "experiment": r.get("experiment"),
        "status": r.get("status"),
        "file": r.get("_path"),
        # Absent on every E0-E3 run: the field postdates them, and they are all
        # vLLM. Defaulted rather than backfilled, so no raw file is rewritten.
        "stack": server.get("stack") or "vllm",
        "model": server.get("model"),
        "quantization": server.get("quantization") or precision_of(server.get("model")),
        "max_model_len": server.get("max_model_len"),
        "input_len": load.get("input_len"),
        "output_len": load.get("output_len"),
        "request_rate": load.get("request_rate"),
        "max_concurrency": load.get("max_concurrency"),
        "duration_s": m.get("duration"),
        "completed": m.get("completed"),
        "req_throughput": m.get("request_throughput"),
        "out_tok_throughput": m.get("output_throughput"),
        "total_tok_throughput": m.get("total_token_throughput"),
        "total_output_tokens": m.get("total_output_tokens"),
        "total_input_tokens": m.get("total_input_tokens"),
        "ttft_p50_ms": m.get("median_ttft_ms"),
        "ttft_p95_ms": m.get("p95_ttft_ms") or m.get("p99_ttft_ms"),
        "itl_p50_ms": m.get("median_itl_ms"),
        "itl_p95_ms": m.get("p95_itl_ms") or m.get("p99_itl_ms"),
        "e2e_p50_ms": m.get("median_e2el_ms"),
        "e2e_p95_ms": m.get("p95_e2el_ms") or m.get("p99_e2el_ms"),
        "goodput_rps": gp.get("goodput_req_per_s") if gp.get("available") else None,
        "slo_attainment": gp.get("slo_attainment_frac") if gp.get("available") else None,
        "energy_j": e.get("energy_j"),
        # Idle-subtracted energy is recomputed under the current baseline
        # policy (median, decided 2026-08-14); raw energy is untouched.
        "incremental_energy_j": incr.get("incremental_energy_j"),
        "idle_baseline_w": incr.get("idle_baseline_w"),
        "mean_power_w": e.get("mean_power_w"),
        "j_per_out_tok": e.get("j_per_output_token"),
        "j_per_out_tok_incr": incr.get("j_per_output_token_incremental"),
        "temp_max_c": g(e, "temp_c", "max"),
        "clock_p50_mhz": g(e, "clock_sm_mhz", "p50"),
        "flags": e.get("flags") or [],
        "sanity": g(r, "sanity", "overall"),
        # Present only on runs whose server never started; carries the root
        # cause and memory-budget lines lifted from that server's own log.
        "failure": r.get("failure") or {},
        # Server-side view. Saturation is a growing queue, which only the
        # server can report; the load generator cannot see it.
        "kv_util_mean": g(sm, "kv_cache_usage_perc", "mean"),
        "kv_util_max": g(sm, "kv_cache_usage_perc", "max"),
        "batch_mean": g(sm, "num_requests_running", "mean"),
        "batch_max": g(sm, "num_requests_running", "max"),
        "queue_mean": g(sm, "num_requests_waiting", "mean"),
        "queue_max": g(sm, "num_requests_waiting", "max"),
        "queue_growing": g(sm, "queue_growth", "growing"),
        "preemptions": sm.get("num_preemptions_total_delta"),
    }


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------


def cmd_summary(args: argparse.Namespace) -> str:
    rows = [row_view(r) for r in load_results(args.experiment, include_failed=True,
                                          include_superseded=getattr(args, "include_superseded", False))]
    if not rows:
        return "No raw results yet. Nothing measured; nothing to report."
    body = [
        [
            r["point_id"] or "?",
            r["experiment"] or "?",
            r["status"] or "?",
            fmt(r["request_rate"]),
            fmt(r["req_throughput"], ".3g"),
            fmt(r["out_tok_throughput"], ".4g"),
            fmt(r["ttft_p95_ms"], ".4g"),
            fmt(r["j_per_out_tok"], ".3g"),
            r["sanity"] or "?",
            ",".join(r["flags"]) or "-",
        ]
        for r in rows
    ]
    return md_table(
        ["point", "exp", "status", "rate", "req/s", "out tok/s",
         "TTFT p95 ms", "J/tok", "sanity", "flags"],
        body,
    )


# --------------------------------------------------------------------------
# E0 variance
# --------------------------------------------------------------------------


def cv(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    mean = statistics.fmean(vals)
    if mean == 0:
        return None
    return statistics.stdev(vals) / mean


def cmd_variance(args: argparse.Namespace) -> str:
    rows = [row_view(r) for r in load_results(args.experiment or "E0",
        include_superseded=getattr(args, "include_superseded", False))]
    if len(rows) < 2:
        return (f"E0 variance: {NOT_RUN} — need at least 2 repeats of the reference "
                f"configuration, found {len(rows)}.")

    metrics = [
        ("Output throughput (tok/s)", "out_tok_throughput", ".4g"),
        ("Request throughput (req/s)", "req_throughput", ".4g"),
        ("TTFT p50 (ms)", "ttft_p50_ms", ".4g"),
        ("TTFT p95 (ms)", "ttft_p95_ms", ".4g"),
        ("ITL p50 (ms)", "itl_p50_ms", ".4g"),
        ("E2E p95 (ms)", "e2e_p95_ms", ".4g"),
        ("Mean GPU power (W)", "mean_power_w", ".4g"),
        ("Energy per run (J)", "energy_j", ".5g"),
        ("J per output token", "j_per_out_tok", ".4g"),
    ]
    body = []
    for label, key, spec in metrics:
        vals = [r[key] for r in rows if r[key] is not None]
        if len(vals) < 2:
            body.append([label, NOT_RUN, NOT_RUN, NOT_RUN, NOT_RUN])
            continue
        c = cv(vals)
        body.append([
            label,
            fmt(statistics.fmean(vals), spec),
            fmt(min(vals), spec),
            fmt(max(vals), spec),
            f"{c * 100:.2f}%" if c is not None else NOT_RUN,
        ])

    header = (f"## E0 stability baseline — {len(rows)} repeats of the reference "
              f"configuration\n\n"
              f"Runs: {', '.join(r['file'] for r in rows)}\n\n")
    table = md_table(["Metric", "mean", "min", "max", "CV"], body)

    return header + table + variance_verdict(rows)


def variance_verdict(rows: list[dict]) -> str:
    """Turn the per-metric CVs into resolution limits, with the caveats that apply.

    Throughput is deliberately NOT used as the headline. Below saturation, with
    a fixed seed, every repeat replays an identical Poisson arrival schedule and
    the server keeps up, so the run's duration -- and therefore its throughput --
    is set by the generator rather than by the card. Its CV then measures the
    workload generator's determinism, not serving stability, and quoting it as a
    variance bar would overstate the precision of every later comparison.
    """
    out = ["\n\n### What this bounds\n"]

    tput_cv = cv([r["out_tok_throughput"] for r in rows])
    energy_cv = cv([r["j_per_out_tok"] for r in rows])
    power_cv = cv([r["mean_power_w"] for r in rows])
    tail_cv = cv([r["ttft_p95_ms"] for r in rows])
    e2e_cv = cv([r["e2e_p95_ms"] for r in rows])

    # Is throughput pinned by the arrival schedule rather than by the server?
    pinned = []
    for r in rows:
        rate, achieved = r["request_rate"], r["req_throughput"]
        if rate in (None, "inf") or achieved is None:
            continue
        pinned.append(achieved >= 0.95 * float(rate))
    schedule_pinned = bool(pinned) and all(pinned)

    if tput_cv is not None:
        if schedule_pinned:
            out.append(
                f"- **Throughput CV is {tput_cv * 100:.2f}%, and that number is not a "
                f"variance bar.** Every repeat replays the same seeded Poisson "
                f"schedule, and the server kept up with it (achieved ≥ 95% of "
                f"offered), so the run ends when the last request was *scheduled*, "
                f"not when the card finished working. Throughput is therefore "
                f"pinned by the load generator here. It becomes a real measure of "
                f"the card only at saturation, where the queue grows and duration "
                f"is set by service rate.\n"
            )
        else:
            out.append(
                f"- **Throughput CV {tput_cv * 100:.2f}%** — meaningful here, since "
                f"at least one repeat did not keep up with offered load, so "
                f"duration reflects service rate.\n"
            )

    limits = []
    if energy_cv is not None:
        limits.append(("Energy per token", energy_cv))
    if power_cv is not None:
        limits.append(("Mean power", power_cv))
    if tail_cv is not None:
        limits.append(("TTFT p95", tail_cv))
    if e2e_cv is not None:
        limits.append(("E2E p95", e2e_cv))

    for name, c in limits:
        out.append(f"- **{name}: CV {c * 100:.2f}%** — differences smaller than "
                   f"~{c * 200:.1f}% between later points are inside noise.\n")

    if not limits:
        return "".join(out) + "\nNo metric had enough repeats to bound.\n"

    worst_name, worst_cv = max(limits, key=lambda x: x[1])
    binding = worst_cv * 200
    if worst_cv < 0.02:
        grade = "**Verdict: tight.**"
    elif worst_cv < 0.05:
        grade = "**Verdict: acceptable.**"
    else:
        grade = ("**Verdict: poor — diagnose before trusting later numbers.** "
                 "Check thermal throttling, background GPU consumers, and power "
                 "readout stability.")

    out.append(
        f"\n{grade} The binding constraint is {worst_name} at "
        f"{worst_cv * 100:.2f}% CV, so **this project does not claim any "
        f"difference smaller than about {binding:.1f}%** on that metric. "
        f"Energy comparisons are resolvable to ~{(energy_cv or 0) * 200:.1f}%.\n"
    )

    thermal = [r for r in rows if r["flags"]]
    if thermal:
        out.append("\nFlagged runs: "
                   + "; ".join(f"{r['file']} ({','.join(r['flags'])})" for r in thermal)
                   + "\n")
    else:
        out.append("\nNo run tripped a thermal, clock-sag, sample-gap or "
                   "power-limit flag.\n")
    return "".join(out)


# --------------------------------------------------------------------------
# E1 sweep
# --------------------------------------------------------------------------


def shape_of(r: dict) -> str:
    return f"{r['input_len']}in/{r['output_len']}out"


def cmd_sweep(args: argparse.Namespace) -> str:
    rows = [row_view(r) for r in load_results(args.experiment or "E1",
        include_superseded=getattr(args, "include_superseded", False))]
    if not rows:
        return f"E1 load sweep: {NOT_RUN}."

    out = []
    shapes = sorted({shape_of(r) for r in rows})
    for shape in shapes:
        srows = sorted(
            [r for r in rows if shape_of(r) == shape],
            key=lambda r: (float("inf") if r["request_rate"] in (None, "inf")
                           else float(r["request_rate"])),
        )
        body = [
            [
                fmt(r["request_rate"]),
                fmt(r["req_throughput"], ".3f"),
                fmt(r["goodput_rps"], ".3f"),
                fmt(r["out_tok_throughput"], ".4g"),
                fmt(r["ttft_p50_ms"], ".4g"),
                fmt(r["ttft_p95_ms"], ".4g"),
                fmt(r["e2e_p95_ms"], ".4g"),
                fmt(r["mean_power_w"], ".4g"),
                fmt(r["j_per_out_tok"], ".3g"),
                fmt(r["j_per_out_tok_incr"], ".3g"),
                f"{r['kv_util_mean'] * 100:.1f}%" if r["kv_util_mean"] is not None else NOT_RUN,
                fmt(r["batch_mean"], ".3g"),
                fmt(r["queue_max"], ".4g"),
                "yes" if r["queue_growing"] else ("no" if r["queue_growing"] is not None else "-"),
                fmt(r["preemptions"], ".4g"),
            ]
            for r in srows
        ]
        out.append(
            f"### E1 — {shape}\n\n"
            + md_table(
                ["offered req/s", "achieved req/s", "goodput req/s", "out tok/s",
                 "TTFT p50 ms", "TTFT p95 ms", "E2E p95 ms",
                 "mean W", "J/tok raw", "J/tok net",
                 "KV util", "batch mean", "queue max", "queue growing", "preempt"],
                body,
            )
        )
        sustainable = max_sustainable_rate(srows)
        if sustainable is None:
            out.append(f"\nMax sustainable rate under the SLO: {NOT_RUN}\n")
        else:
            out.append(f"\nMax sustainable rate under the SLO: **{sustainable:g} req/s** "
                       f"(highest offered rate where achieved throughput tracks offered "
                       f"load and the SLO still holds for a majority of requests).\n")
    return "\n\n".join(out)


def max_sustainable_rate(srows: list[dict]) -> float | None:
    """Highest offered rate the server actually sustains under the SLO.

    Three conditions, all measured rather than inferred:
      - achieved throughput within 5% of offered load (it is keeping up),
      - the server's own request queue is not growing across the window
        (plan §4's definition of saturation), and
      - at least half of requests met both SLO clauses.

    The queue condition is the load-bearing one. Throughput can track offered
    load while latency quietly degrades, and a queue that grows across the
    window is the unambiguous signal that the point is past saturation.
    """
    best = None
    for r in srows:
        rate = r["request_rate"]
        if rate in (None, "inf"):
            continue
        rate = float(rate)
        achieved = r["req_throughput"]
        att = r["slo_attainment"]
        if achieved is None:
            continue
        keeping_up = achieved >= 0.95 * rate
        queue_ok = not r.get("queue_growing")     # None (uncollected) counts as ok
        slo_ok = att is None or att >= 0.5
        if keeping_up and queue_ok and slo_ok:
            best = rate if best is None else max(best, rate)
    return best


# --------------------------------------------------------------------------
# E2 quantization ablation
# --------------------------------------------------------------------------


def arm_of(r: dict) -> str:
    """Which precision arm a run belongs to, from its checkpoint name."""
    return precision_of(r["model"])


STACK_LABEL = {
    "vllm": "vLLM",
    "llamacpp": "llama.cpp",
}
# What each stack was actually serving. E4's whole caveat lives in this dict:
# the two arms are not the same weights.
STACK_FORMAT = {
    "vllm": "AWQ int4",
    "llamacpp": "GGUF Q4_K_M",
}
E4_CAVEAT = (
    "**GGUF Q4_K_M and AWQ int4 are different quantization formats.** This "
    "compares two serving stacks each at its own native int4, not one set of "
    "weights on two servers, so a throughput or energy gap here includes "
    "whatever the formats themselves cost. Everything else is held equal: same "
    "model family and size (Qwen2.5-7B-Instruct), same traffic shape "
    "(512 in / 128 out), same tokenizer, same load generator, same idle "
    "baseline, same session."
)


def stack_of(row: dict) -> str:
    """Serving stack of a row_view. Absent means vLLM: E0-E3 predate the field."""
    return str(row.get("stack") or "vllm")


def load_gsm8k() -> dict[str, dict]:
    """Guard results, newest per arm."""
    out: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(RAW, "gsm8k_*.json"))):
        with open(path) as fh:
            try:
                r = json.load(fh)
            except json.JSONDecodeError:
                continue
        label = (r.get("config") or {}).get("label")
        if label:
            out[label] = r          # sorted() means the newest wins
    return out


def cmd_ablation(args: argparse.Namespace) -> str:
    rows = [row_view(r) for r in load_results(args.experiment or "E2",
            include_superseded=getattr(args, "include_superseded", False))]
    guards = load_gsm8k()
    if not rows and not guards:
        return f"E2 quantization ablation: {NOT_RUN}."

    out = ["## E2 — quantization ablation (Qwen2.5-7B, chat shape)", ""]

    rates = sorted({r["request_rate"] for r in rows if r["request_rate"] is not None},
                   key=float)
    arms = ["bf16", "awq", "gptq"]
    body = []
    for rate in rates:
        for arm in arms:
            match = [r for r in rows if r["request_rate"] == rate and arm_of(r) == arm]
            if not match:
                continue
            r = match[0]
            base = next((x for x in rows
                         if x["request_rate"] == rate and arm_of(x) == "bf16"), None)
            def rel(field: str) -> str:
                if not base or base[field] in (None, 0) or r[field] is None:
                    return "-"
                if arm == "bf16":
                    return "baseline"
                return f"{(r[field] / base[field] - 1) * 100:+.1f}%"
            body.append([
                fmt(rate), arm,
                fmt(r["out_tok_throughput"], ".4g"), rel("out_tok_throughput"),
                fmt(r["ttft_p95_ms"], ".4g"),
                fmt(r["j_per_out_tok"], ".3g"), rel("j_per_out_tok"),
                fmt(r["mean_power_w"], ".4g"),
                fmt(r["slo_attainment"], ".3f"),
            ])
    if body:
        out.append(md_table(
            ["req/s", "arm", "out tok/s", "vs BF16", "TTFT p95 ms",
             "J/tok", "vs BF16", "mean W", "SLO attain"], body))
        out.append("")
        out.append("Relative columns compare against the BF16 arm at the same offered "
                   "rate. **E0's variance bar is regime-specific, and each row must be "
                   "read against its own regime.** Unsaturated rows (2 and 8 req/s): "
                   "energy CV 0.16%, so differences below ~0.4% are noise. Saturated "
                   "rows (16 req/s), from the three E0-sat repeats: throughput CV "
                   "1.20%, energy CV 1.08%, so differences below ~1.2% are noise. "
                   "Borrowing the unsaturated bar for a saturated row would manufacture "
                   "significance that is not there.")
        out.append("")

    # --- quality guard ---
    out.append("### Quality guard — GSM8K exact match")
    out.append("")
    if not guards:
        out.append(f"{NOT_RUN}. Speed without a quality number is not a tradeoff, "
                   f"it is half of one.")
    else:
        gbody = []
        for arm in arms:
            g = guards.get(arm)
            if not g:
                gbody.append([arm, NOT_RUN, NOT_RUN, NOT_RUN, NOT_RUN])
                continue
            m = g["metrics"]
            n, k = m["n_items"], m["n_correct"]
            # Wilson 95% interval: with n=50 the normal approximation is poor.
            p = k / n
            z = 1.96
            denom = 1 + z * z / n
            centre = (p + z * z / (2 * n)) / denom
            half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
            gbody.append([
                arm, f"{k}/{n}", f"{p:.1%}",
                f"{max(0.0, centre - half):.1%} – {min(1.0, centre + half):.1%}",
                str(m.get("n_errors", 0)),
            ])
        out.append(md_table(
            ["arm", "correct", "exact match", "95% Wilson interval", "request errors"],
            gbody))
        out.append("")
        out.append("**50 items. The intervals overlap unless the gap is large**, so "
                   "this guard can only rule out a big quality collapse from "
                   "quantization — it cannot certify parity. It says nothing about "
                   "any capability other than grade-school arithmetic word problems.")

    # --- what int4 buys structurally ---
    out.append("")
    out.append("### Longest servable context")
    out.append("")
    lim = []
    for path in sorted(glob.glob(os.path.join(RAW, "limits_context_*.json"))):
        with open(path) as fh:
            try:
                r = json.load(fh)
            except json.JSONDecodeError:
                continue
        label = (r.get("config") or {}).get("label", "?")
        best = (r.get("metrics") or {}).get("longest_servable_context")
        attempts = (r.get("metrics") or {}).get("attempts") or []
        failed = next((a for a in attempts if not a.get("served")), None)
        lim.append([
            label,
            fmt(best) if best else "none served",
            str(next((a.get("kv_cache_tokens") for a in attempts
                      if a.get("max_model_len") == best), "-") or "-"),
            (failed or {}).get("error") or "-",
        ])
    if lim:
        out.append(md_table(
            ["arm", "longest servable context", "KV cache tokens there",
             "why the next rung failed"], lim))
    else:
        out.append(f"{NOT_RUN}.")
    return "\n".join(out)


# --------------------------------------------------------------------------
# E3 frontier
# --------------------------------------------------------------------------


def cmd_stacks(args: argparse.Namespace) -> str:
    """E4: vLLM vs llama.cpp at fixed concurrency."""
    results = load_results(args.experiment or "E4", include_failed=True,
                           include_superseded=getattr(args, "include_superseded", False))
    if not results:
        return f"E4 stack comparison: {NOT_RUN}."
    rows = [row_view(r) for r in results]

    out = ["## E4 — serving-stack comparison (Qwen2.5-7B int4, chat shape)", "",
           E4_CAVEAT, ""]

    concs = sorted({r["max_concurrency"] for r in rows
                    if r["max_concurrency"] is not None}, key=int)
    stacks = ["vllm", "llamacpp"]
    body = []
    for c in concs:
        for st in stacks:
            match = [r for r in rows
                     if r["max_concurrency"] == c and stack_of(r) == st]
            if not match:
                continue
            r = match[0]
            base = next((x for x in rows if x["max_concurrency"] == c
                         and stack_of(x) == "vllm"), None)

            def rel(field: str, r=r, base=base, st=st) -> str:
                if st == "vllm":
                    return "baseline"
                if not base or not base.get(field) or r.get(field) is None:
                    return "-"
                return f"{(r[field] / base[field] - 1) * 100:+.1f}%"

            if r["status"] != "ok":
                body.append([str(c), STACK_LABEL[st], STACK_FORMAT[st], r["status"]]
                            + ["--"] * 8)
                continue
            body.append([
                str(c), STACK_LABEL[st], STACK_FORMAT[st], "served",
                fmt(r["out_tok_throughput"], ".4g"), rel("out_tok_throughput"),
                fmt(r["ttft_p50_ms"], ".4g"),
                fmt(r["itl_p50_ms"], ".3g"),
                fmt(r["e2e_p95_ms"], ".4g"),
                fmt(r["mean_power_w"], ".4g"),
                fmt(r["j_per_out_tok"], ".3g"), rel("j_per_out_tok"),
            ])
    if body:
        out.append(md_table(
            ["concurrency", "stack", "format", "status", "out tok/s", "vs vLLM",
             "TTFT p50 ms", "ITL p50 ms", "E2E p95 ms", "mean W",
             "J/tok", "vs vLLM"], body))
        out.append("")
        out.append(
            "Relative columns compare llama.cpp against vLLM at the same "
            "concurrency. Concurrency is held by the load generator "
            "(`--max-concurrency`), so it is the number of requests in flight, "
            "not an offered rate: there is no queue to grow and no SLO column, "
            "because every request is admitted as soon as a slot frees.")
        out.append("")

    # --- what each stack reports about itself ---
    sbody = []
    for c in concs:
        for st in stacks:
            match = [r for r in rows if r["max_concurrency"] == c
                     and stack_of(r) == st and r["status"] == "ok"]
            if not match:
                continue
            r = match[0]
            completed, issued = r["completed"], None
            cfg_load = None
            for res in results:
                if res.get("point_id") == r["point_id"]:
                    cfg_load = (res.get("config") or {}).get("load") or {}
            if cfg_load:
                issued = cfg_load.get("num_prompts")
            sbody.append([
                str(c), STACK_LABEL[st],
                fmt(r["batch_mean"], ".3g"),
                f"{r['kv_util_mean'] * 100:.1f}%" if r["kv_util_mean"] is not None else NOT_RUN,
                fmt(r["preemptions"], ".4g") if r["preemptions"] is not None else NOT_RUN,
                f"{completed}/{issued}" if issued else fmt(completed, ".4g"),
            ])
    if sbody:
        out.append("### What each stack reports about itself")
        out.append("")
        out.append(md_table(
            ["concurrency", "stack", "requests running (mean)", "KV utilisation",
             "preemptions", "completed"], sbody))
        out.append("")
        out.append(
            f"`{NOT_RUN}` in the last two columns is a difference between the "
            "stacks rather than a gap in the measurement. llama.cpp reports no "
            "preemption counter because it does not preempt: a request that "
            "finds no free slot is deferred before it starts rather than "
            "evicted after it starts. It also exports no KV-utilisation ratio, "
            "and the quantity would not mean the same thing if it did — "
            "llama.cpp partitions KV into fixed per-slot budgets where vLLM "
            "shares one pool.")
        out.append("")
        dropped = [r for r in rows if r["status"] == "ok" and r["completed"]
                   and r["max_concurrency"]
                   and r["completed"] < (next(
                       ((res.get("config") or {}).get("load") or {}).get("num_prompts")
                       for res in results if res.get("point_id") == r["point_id"]) or 0)]
        if dropped:
            out.append(
                "A `completed` count below the issued count is a dropped request, "
                "kept visible rather than quietly excluded; each one also trips "
                "this project's `all_requests_completed` sanity check.")
        else:
            out.append(
                "**Every point completed every request it issued.** That is worth "
                "stating because it was not true on the first attempt: with HTTP "
                "connection reuse left on, llama.cpp dropped 10.9% of requests at "
                "concurrency 8 and 11.2% at concurrency 32 — see "
                "`e4_transport.md`, and the superseded runs in `results/raw/`.")
    return "\n".join(out)


def cmd_frontier(args: argparse.Namespace) -> str:
    results = load_results(args.experiment or "E3", include_failed=True,
        include_superseded=getattr(args, "include_superseded", False))
    if not results:
        return f"E3 model-size frontier: {NOT_RUN}."
    rows = [row_view(r) for r in results]

    def size_key(r: dict) -> float:
        name = (r["model"] or "").lower()
        for tok in name.replace("/", "-").split("-"):
            if tok.endswith("b"):
                try:
                    return float(tok[:-1])
                except ValueError:
                    continue
        return 1e9

    rows.sort(key=size_key)

    def cell(r: dict, field: str, spec: str) -> str:
        # A rung whose server never started has no metrics, but it is not
        # "not run" -- this project reserves that phrase for work never
        # attempted, and these were attempted and failed. Say so.
        if r["status"] != "ok":
            return "--"
        return fmt(r[field], spec)

    def ms_cell(r: dict, field: str) -> str:
        # A saturated rung queues for minutes. ".4g" renders that as 3.664e+05,
        # which hides the single most important number in the table.
        if r["status"] != "ok":
            return "--"
        v = r[field]
        if v is None:
            return NOT_RUN
        return f"{v:,.0f}" if v >= 1000 else f"{v:.4g}"

    body = [
        [
            (r["model"] or "?").split("/")[-1],
            r["quantization"],
            fmt(r["max_model_len"], "d"),
            r["status"] if r["status"] != "ok" else "served",
            cell(r, "out_tok_throughput", ".4g"),
            ms_cell(r, "ttft_p50_ms"),
            ms_cell(r, "ttft_p95_ms"),
            cell(r, "mean_power_w", ".4g"),
            cell(r, "j_per_out_tok", ".3g"),
        ]
        for r in rows
    ]
    note = ("\n\nRows marked with a non-`served` status are configurations that "
            "failed to serve. They are kept: what the card *cannot* do is part of "
            "the frontier. `--` means the run was attempted and produced no "
            "measurement, which is not the same as not run.\n")

    # Why a rung failed is the whole content of that row, and it is recorded in
    # the run's own server log rather than inferred here.
    failed = [r for r in rows if r["status"] != "ok" and r["failure"].get("root_cause")]
    if failed:
        note += "\nWhy each non-`served` rung failed, from its server log:\n\n"
        for r in failed:
            name = (r["model"] or "?").split("/")[-1]
            note += f"- **{name}** (`{r['file']}`): {r['failure']['root_cause']}\n"
            for ev in r["failure"].get("evidence") or []:
                note += f"  - {ev}\n"
    return ("## E3 — model-size frontier\n\n"
            + md_table(
                ["model", "precision", "ctx", "status", "out tok/s", "TTFT p50 ms",
                 "TTFT p95 ms", "mean W", "J/tok"],
                body,
            ) + note)


# --------------------------------------------------------------------------
# economics
# --------------------------------------------------------------------------

J_PER_KWH = 3.6e6
HOURS_PER_YEAR = 365.25 * 24


def energy_cost_per_1m_out(j_per_out_tok: float, usd_per_kwh: float) -> float:
    """$ of electricity per 1M output tokens, from measured J/token."""
    return (j_per_out_tok * 1e6 / J_PER_KWH) * usd_per_kwh


def owned_amortization_usd_per_hour(card_price_usd: float, years: float) -> float:
    return card_price_usd / (years * HOURS_PER_YEAR)


def usd_per_1m_out(usd_per_hour: float, out_tok_per_s: float) -> float | None:
    """Fixed hourly cost spread over the tokens actually produced in that hour."""
    if not out_tok_per_s or out_tok_per_s <= 0:
        return None
    tokens_per_hour = out_tok_per_s * 3600.0
    return usd_per_hour / (tokens_per_hour / 1e6)


def api_effective_out_price(price_in: float, price_out: float,
                            in_tokens: float, out_tokens: float) -> float:
    """API $/1M output tokens at the measured input:output ratio.

    Self-hosting pays for prefill as well as decode, so comparing against an
    API's output price alone would flatter the API. Folding the input charge in
    at the shape actually benchmarked is the like-for-like comparison.
    """
    if not out_tokens:
        return price_out
    return price_out + price_in * (in_tokens / out_tokens)


def breakeven_tokens_per_day(amort_usd_per_day: float,
                             self_host_marginal_per_1m: float,
                             api_per_1m: float) -> float | None:
    """Daily output-token volume at which owning beats paying per token.

    amort_per_day + marginal*(V/1e6) = api*(V/1e6)
      => V = amort_per_day * 1e6 / (api - marginal)
    Returns None when the API is cheaper than the marginal cost of running the
    card, in which case owning never wins at any volume.
    """
    spread = api_per_1m - self_host_marginal_per_1m
    if spread <= 0:
        return None
    return amort_usd_per_day * 1e6 / spread


def pick_economics_point(rows: list[dict]) -> dict | None:
    """The load point the headline numbers are quoted at.

    The best-case operating point: highest output throughput among runs that
    still met the SLO for a majority of requests. Quoting the peak of a
    saturated server would overstate what an owner can actually run at.
    """
    usable = [
        r for r in rows
        if r["out_tok_throughput"] and r["j_per_out_tok"]
        and (r["slo_attainment"] is None or r["slo_attainment"] >= 0.5)
    ]
    if not usable:
        return None
    return max(usable, key=lambda r: r["out_tok_throughput"])


def cmd_economics(args: argparse.Namespace) -> str:
    pricing = load_pricing()
    rows = [row_view(r) for r in load_results(args.experiment or "E1",
        include_superseded=getattr(args, "include_superseded", False))]
    if not rows:
        return (f"Economics: {NOT_RUN} — no E1 results to price. The cost model "
                f"needs a measured throughput and a measured J/token.")

    point = pick_economics_point(rows)
    if point is None:
        return f"Economics: {NOT_RUN} — no SLO-meeting load point with both throughput and energy."

    rate = pricing["electricity"]["primary"]["usd_per_kwh"]
    band = pricing["electricity"]["sensitivity_band_usd_per_kwh"]
    card = pricing["hardware"]["gpu"]["street_price_usd"]
    card_band = pricing["hardware"]["gpu"]["street_price_band_usd"]
    years = pricing["hardware"]["amortization_years"]

    tput = point["out_tok_throughput"]
    jtok = point["j_per_out_tok"]
    jtok_net = point["j_per_out_tok_incr"]

    e_cost = energy_cost_per_1m_out(jtok, rate)
    e_cost_lo = energy_cost_per_1m_out(jtok, band[0])
    e_cost_hi = energy_cost_per_1m_out(jtok, band[1])
    amort_hr = owned_amortization_usd_per_hour(card, years)
    amort_day = amort_hr * 24
    amort_per_1m = usd_per_1m_out(amort_hr, tput) or float("nan")
    owned_total = amort_per_1m + e_cost

    lines = [
        "## E5 — serving economics",
        "",
        f"Quoted at the best SLO-meeting load point measured: **{point['point_id']}** "
        f"({shape_of(point)}, {fmt(point['request_rate'])} req/s offered), which "
        f"sustained **{fmt(tput, '.4g')} output tok/s**.",
        "",
        "### Measured inputs (this machine)",
        "",
        md_table(
            ["Quantity", "Value", "Source"],
            [
                ["Output throughput", f"{fmt(tput, '.4g')} tok/s", point["file"]],
                ["Energy per output token (raw)", f"{fmt(jtok, '.3g')} J", point["file"]],
                ["Energy per output token (idle-subtracted)",
                 f"{fmt(jtok_net, '.3g')} J", point["file"]],
                ["Mean GPU power", f"{fmt(point['mean_power_w'], '.4g')} W", point["file"]],
            ],
        ),
        "",
        "### Sourced inputs (reported, not measured)",
        "",
        md_table(
            ["Quantity", "Value", "Source", "Dated"],
            [
                ["Electricity", f"${rate}/kWh",
                 pricing["electricity"]["primary"]["name"],
                 pricing["electricity"]["primary"]["snapshot_date"]],
                ["RTX 4090 street price", f"${card:,.0f}",
                 "retail/used listing trackers",
                 pricing["hardware"]["gpu"]["snapshot_date"]],
                ["Amortisation", f"{years} years, card only", "assumption", "-"],
            ],
        ),
        "",
        "### Cost per 1M output tokens",
        "",
    ]

    cost_rows = [
        ["Electricity only (marginal cost of running the card)",
         f"${e_cost:.3f}", f"${e_cost_lo:.3f} – ${e_cost_hi:.3f}"],
        [f"Card amortisation at {fmt(tput, '.4g')} tok/s sustained",
         f"${amort_per_1m:.3f}",
         f"${usd_per_1m_out(owned_amortization_usd_per_hour(card_band[0], years), tput):.3f} – "
         f"${usd_per_1m_out(owned_amortization_usd_per_hour(card_band[1], years), tput):.3f}"],
        ["**Owned hardware, total**", f"**${owned_total:.3f}**", "-"],
    ]
    for rent in pricing["rental"]["options"]:
        rc = usd_per_1m_out(rent["usd_per_hour"], tput)
        cost_rows.append([
            f"Hypothetical rental — {rent['provider']} {rent['sku']} "
            f"(${rent['usd_per_hour']}/hr, electricity included)",
            f"${rc:.3f}" if rc else NOT_RUN, "-",
        ])
    lines.append(md_table(["Line", "$/1M output tokens", "sensitivity band"], cost_rows))
    lines.append("")
    lines.append("The amortisation line assumes the card is **busy at this throughput "
                 "every hour of its three-year life**. That is the most generous "
                 "possible assumption for owning; at 10% duty cycle the amortised "
                 f"figure is 10x higher (${amort_per_1m * 10:.2f}/1M).")
    lines.append("")

    # --- break-even -------------------------------------------------------
    lines += [
        "### Break-even volume against API list prices",
        "",
        "Formula, with $V$ the daily output-token volume:",
        "",
        "```",
        "owned daily cost  =  amortisation_per_day + energy_per_1M * V/1e6",
        "api daily cost    =  api_price_per_1M     * V/1e6",
        "break-even V*     =  amortisation_per_day * 1e6 / (api_price_per_1M - energy_per_1M)",
        "```",
        "",
        f"Amortisation is ${amort_day:.3f}/day; the marginal (electricity) cost of "
        f"self-hosting is ${e_cost:.3f}/1M output tokens. API prices are blended to "
        f"an effective output price at the measured "
        f"{fmt(point['input_len'])}in/{fmt(point['output_len'])}out shape, because "
        f"self-hosting pays for prefill too.",
        "",
    ]

    in_tok = point["total_input_tokens"] or (point["input_len"] or 0) * (point["completed"] or 0)
    out_tok = point["total_output_tokens"] or (point["output_len"] or 0) * (point["completed"] or 0)
    daily_capacity = tput * 86400

    be_rows = []
    for group in ("open_weight_hosted", "frontier"):
        for api in pricing["api_prices"].get(group, []):
            eff = api_effective_out_price(
                api["usd_per_1m_input"], api["usd_per_1m_output"], in_tok, out_tok
            )
            v = breakeven_tokens_per_day(amort_day, e_cost, eff)
            if v is None:
                verdict = "never — API is below the card's electricity cost"
                vstr = "-"
            else:
                vstr = (f"{v / 1e6:,.2f}M tok/day" if v >= 1e6
                        else f"{v / 1e3:,.0f}k tok/day")
                pct = 100 * v / daily_capacity
                if v > daily_capacity:
                    verdict = (f"unreachable — {pct:.0f}% of this card's "
                               f"{daily_capacity / 1e6:,.2f}M tok/day ceiling")
                elif pct < 1:
                    verdict = "<1% of the card's daily ceiling"
                else:
                    verdict = f"{pct:.0f}% of the card's daily ceiling"
            be_rows.append([
                f"{api['provider']} {api['model']}",
                "yes" if api.get("weight_class_comparable") else "**no**",
                f"${eff:.3f}",
                vstr,
                verdict,
            ])
    lines.append(md_table(
        ["API (list price, reported not measured)", "comparable weight class",
         "effective $/1M out", "break-even", "feasibility"],
        be_rows,
    ))
    lines += [
        "",
        "The *comparable weight class* column is the one that decides whether a row "
        "means anything. A frontier model is not a substitute for a locally served "
        "7B; those rows show the price ceiling of the market, not an available "
        "trade. The only capability evidence in this project is the E2 GSM8K guard, "
        "and it licenses no claim beyond that task.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "font.family": "DejaVu Sans",
        "figure.dpi": 150,
    })
    return plt


def _finish(ax, plt, path: str, footnote: str | None = None, bottom: float = 0.0):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if footnote:
        ax.figure.text(0.01, 0.005, footnote, ha="left", va="bottom",
                       fontsize=7.5, color=INK_MUTED, linespacing=1.45)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    reserve = bottom + (0.035 if footnote else 0.0)
    ax.figure.tight_layout(rect=(0, reserve, 1, 1) if reserve else None)
    ax.figure.savefig(path)
    plt.close(ax.figure)
    print(f"[analyze] wrote {os.path.relpath(path, REPO)}")


def plot_energy_vs_load(rows: list[dict], path: str) -> bool:
    """J per output token against offered load, one line per traffic shape."""
    plt = _style()
    shapes = sorted({shape_of(r) for r in rows})
    series = []
    for shape in shapes:
        pts = sorted(
            [r for r in rows if shape_of(r) == shape
             and r["j_per_out_tok"] and r["request_rate"] not in (None, "inf")],
            key=lambda r: float(r["request_rate"]),
        )
        if pts:
            series.append((shape, pts))
    if not series:
        return False

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    slots = [C["blue"], C["orange"], C["aqua"]]
    for i, (shape, pts) in enumerate(series):
        x = [float(p["request_rate"]) for p in pts]
        y = [p["j_per_out_tok"] for p in pts]
        ax.plot(x, y, marker="o", color=slots[i % len(slots)], label=shape,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        # Direct-label the endpoints only; never a number on every point.
        ax.annotate(f"{y[-1]:.2g} J", (x[-1], y[-1]), textcoords="offset points",
                    xytext=(8, 0), va="center", fontsize=9, color=INK_2)

    ax.set_xscale("log")
    # Room on the right so the direct labels are not clipped by the axes.
    xmax = max(max(float(p["request_rate"]) for p in pts) for _, pts in series)
    ax.set_xlim(right=xmax * 1.9)
    ax.set_xlabel("Offered load (requests/s, Poisson arrivals)")
    ax.set_ylabel("Energy per output token (J)")
    ax.set_title("Batching collapses energy per token\nRTX 4090, Qwen2.5-7B BF16, vLLM")
    ax.set_ylim(bottom=0)
    if len(series) >= 2:
        ax.legend(loc="upper right")
    _finish(ax, plt, path,
            "Measured on this machine. Whole-board GPU draw integrated over the "
            "measurement window; raw, not idle-subtracted.")
    return True


def plot_breakeven(rows: list[dict], pricing: dict, path: str,
                   shape: str | None = None) -> bool:
    """$/1M output tokens vs offered load, with API list prices overlaid.

    ONE traffic shape per chart. Cost depends on throughput, which depends on
    the request shape, so plotting chat and RAG points on a shared x-axis would
    put two different measurements at the same offered rate and read as one
    line. The shape is named in the subtitle so the chart cannot be quoted
    without it.
    """
    plt = _style()
    candidates = [r for r in rows if r["out_tok_throughput"] and r["j_per_out_tok"]
                  and r["request_rate"] not in (None, "inf")]
    if shape is None:
        # Default to the shape with the most measured points; ties go to the
        # smaller input length, which is the chat shape.
        by_shape: dict[str, list[dict]] = {}
        for r in candidates:
            by_shape.setdefault(shape_of(r), []).append(r)
        if not by_shape:
            return False
        shape = sorted(by_shape, key=lambda k: (-len(by_shape[k]), k))[0]
    pts = sorted([r for r in candidates if shape_of(r) == shape],
                 key=lambda r: float(r["request_rate"]))
    if not pts:
        return False

    rate = pricing["electricity"]["primary"]["usd_per_kwh"]
    card = pricing["hardware"]["gpu"]["street_price_usd"]
    years = pricing["hardware"]["amortization_years"]
    amort_hr = owned_amortization_usd_per_hour(card, years)

    x = [float(p["request_rate"]) for p in pts]
    owned = []
    for p in pts:
        e = energy_cost_per_1m_out(p["j_per_out_tok"], rate)
        a = usd_per_1m_out(amort_hr, p["out_tok_throughput"]) or float("nan")
        owned.append(e + a)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.plot(x, owned, marker="o", color=C["blue"],
            label="Owned 4090 (amortised + electricity)",
            markeredgecolor=SURFACE, markeredgewidth=1.5)

    rentals = pricing["rental"]["options"]
    if rentals:
        rent = rentals[0]
        rvals = [usd_per_1m_out(rent["usd_per_hour"], p["out_tok_throughput"]) for p in pts]
        if all(v is not None for v in rvals):
            ax.plot(x, rvals, marker="s", color=C["orange"],
                    label=f"Rented 4090 (${rent['usd_per_hour']}/hr, reported)",
                    markeredgecolor=SURFACE, markeredgewidth=1.5)

    # API list prices as horizontal reference lines. Comparable-weight-class
    # rows only -- a frontier price on this axis would imply a trade that the
    # capability evidence does not support. These are direct-labelled at the
    # right edge rather than pushed into the legend: they are reference levels,
    # and a legend covering them would defeat the point of drawing them.
    ref = pts[-1]
    in_tok = ref["total_input_tokens"] or 1
    out_tok = ref["total_output_tokens"] or 1
    api_slots = [C["aqua"], C["violet"], C["magenta"]]
    api_lines = []
    for api in pricing["api_prices"]["open_weight_hosted"]:
        if not api.get("weight_class_comparable") or len(api_lines) >= len(api_slots):
            continue
        eff = api_effective_out_price(api["usd_per_1m_input"], api["usd_per_1m_output"],
                                      in_tok, out_tok)
        name = api["model"].split("/")[-1]
        for suffix in ("-Instruct-Turbo", "-Instruct-Lite", "-Instruct"):
            name = name.replace(suffix, "")
        api_lines.append((eff, f"{api['provider']} {name}", api_slots[len(api_lines)]))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(x) * 0.75, max(x) * 1.25)
    for eff, label, colour in api_lines:
        ax.axhline(eff, color=colour, linestyle=(0, (5, 3)), linewidth=1.6)
        # Sit the label just above its own line, anchored at the left edge, so
        # it can never run off the right of the axes however long the name is.
        ax.annotate(f"{label} — ${eff:.2f}", (min(x) * 0.79, eff),
                    xytext=(0, 3), textcoords="offset points",
                    va="bottom", ha="left", fontsize=8.5, color=colour)

    ax.set_xlabel("Offered load (requests/s, Poisson arrivals)")
    ax.set_ylabel("Cost per 1M output tokens (USD)")
    ax.set_title("Self-hosting cost falls with load; API list prices do not\n"
                 f"RTX 4090, Qwen2.5-7B BF16, vLLM — {shape} traffic")
    # Legend carries only the measured lines; the dashed levels are labelled
    # in place, and the footnote says what they are.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2,
              frameon=False, fontsize=9)
    _finish(ax, plt, path,
            "Solid lines measured on this machine, priced with dated public tariffs.\n"
            "Dashed levels are vendor list prices — reported, not measured — blended to\n"
            "an effective output price at the benchmarked input:output ratio, because\n"
            "self-hosting pays for prefill too.",
            bottom=0.11)
    return True


def plot_goodput(rows: list[dict], path: str, shape: str | None = None) -> bool:
    """Offered vs achieved vs SLO-meeting throughput — where saturation begins.

    One shape per chart, for the same reason as the cost chart: two shapes share
    an x-axis but not a service rate.
    """
    plt = _style()
    candidates = [r for r in rows
                  if r["req_throughput"] and r["request_rate"] not in (None, "inf")]
    if shape is None:
        by_shape: dict[str, list[dict]] = {}
        for r in candidates:
            by_shape.setdefault(shape_of(r), []).append(r)
        if not by_shape:
            return False
        shape = sorted(by_shape, key=lambda k: (-len(by_shape[k]), k))[0]
    pts = sorted([r for r in candidates if shape_of(r) == shape],
                 key=lambda r: float(r["request_rate"]))
    if not pts:
        return False
    x = [float(p["request_rate"]) for p in pts]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x, x, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.4,
            label="Offered load (ideal)")
    ax.plot(x, [p["req_throughput"] for p in pts], marker="o", color=C["blue"],
            label="Achieved throughput", markeredgecolor=SURFACE, markeredgewidth=1.5)
    gp = [p["goodput_rps"] for p in pts]
    if any(v is not None for v in gp):
        ax.plot(x, gp, marker="s", color=C["orange"],
                label="Goodput (TTFT ≤ 1s and E2E ≤ 10s)",
                markeredgecolor=SURFACE, markeredgewidth=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    # A log axis cannot render zero, so points where goodput collapsed entirely
    # would silently vanish off the bottom. Say so instead.
    dead = [(float(p["request_rate"])) for p in pts if p["goodput_rps"] == 0]
    if dead:
        ax.annotate(f"goodput = 0 at ≥ {min(dead):g} req/s\n(no request met the SLO)",
                    (min(dead), ax.get_ylim()[0]), xytext=(6, 26),
                    textcoords="offset points", fontsize=8.5, color=C["orange"],
                    va="bottom", ha="left")
    ax.set_xlabel("Offered load (requests/s)")
    ax.set_ylabel("Requests/s")
    ax.set_title("Goodput saturates before throughput does\n"
                 f"RTX 4090, Qwen2.5-7B BF16, vLLM — {shape} traffic")
    ax.legend(loc="upper left")
    _finish(ax, plt, path, "Measured on this machine.")
    return True


def plot_stacks(rows: list[dict], path: str) -> bool:
    """E4: throughput and energy per token against concurrency, one line per stack.

    Two panels rather than two axes on one: throughput and J/token have no
    common scale, and a twin axis invites the reader to compare their slopes as
    if they did.
    """
    plt = _style()
    series = []
    for st in ("vllm", "llamacpp"):
        pts = sorted([r for r in rows
                      if stack_of(r) == st and r["status"] == "ok"
                      and r["out_tok_throughput"] and r["max_concurrency"]],
                     key=lambda r: int(r["max_concurrency"]))
        if pts:
            series.append((st, pts))
    if len(series) < 1:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    colour = {"vllm": C["blue"], "llamacpp": C["orange"]}
    # Endpoint labels are nudged apart per series: at high concurrency the two
    # energy curves converge to within a few hundredths of a joule and the
    # labels would otherwise print on top of each other.
    nudge = {"vllm": -9, "llamacpp": 9}
    for st, pts in series:
        x = [int(p["max_concurrency"]) for p in pts]
        label = f"{STACK_LABEL[st]} ({STACK_FORMAT[st]})"
        for ax, field, fmt_end in ((axes[0], "out_tok_throughput", lambda v: f"{v:,.0f}"),
                                   (axes[1], "j_per_out_tok", lambda v: f"{v:.3g} J")):
            y = [p[field] for p in pts]
            ax.plot(x, y, marker="o", color=colour[st], label=label,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(fmt_end(y[-1]), (x[-1], y[-1]),
                        textcoords="offset points", xytext=(9, nudge[st]),
                        va="center", fontsize=9, color=colour[st])

    ticks = sorted({int(p["max_concurrency"]) for _, pts in series for p in pts})
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlabel("Concurrent requests in flight")
        ax.set_xlim(right=max(ticks) * 2.4)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Output tokens/s")
    axes[0].set_title("Throughput")
    axes[1].set_ylabel("Energy per output token (J)")
    axes[1].set_title("Energy per token")
    axes[0].legend(loc="upper left")
    fig.suptitle("Two stacks at their native int4 — RTX 4090, Qwen2.5-7B, 512 in / 128 out",
                 x=0.01, ha="left", fontsize=13, color=INK, weight="600")

    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    fig.text(0.01, 0.005,
             "GGUF Q4_K_M and AWQ int4 are different quantization formats: this "
             "compares stacks at their native int4, not identical weights.",
             ha="left", va="bottom", fontsize=7.5, color=INK_MUTED)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)
    print(f"[analyze] wrote {os.path.relpath(path, REPO)}")
    return True


def cmd_plots(args: argparse.Namespace) -> str:
    pricing = load_pricing()
    rows = [row_view(r) for r in load_results("E1")]
    if not rows:
        return f"Plots: {NOT_RUN} — no E1 results."
    made = []
    if plot_energy_vs_load(rows, os.path.join(PLOTS, "e1_energy_per_token_vs_load.png")):
        made.append("e1_energy_per_token_vs_load.png")
    shapes = sorted({shape_of(r) for r in rows if r["out_tok_throughput"]})
    for shape in shapes:
        slug = shape.replace("/", "_").replace(" ", "")
        name = ("e5_cost_per_1m_vs_load.png" if len(shapes) == 1
                else f"e5_cost_per_1m_vs_load__{slug}.png")
        if plot_breakeven(rows, pricing, os.path.join(PLOTS, name), shape=shape):
            made.append(name)
    for shape in shapes:
        slug = shape.replace("/", "_").replace(" ", "")
        name = ("e1_goodput_vs_load.png" if len(shapes) == 1
                else f"e1_goodput_vs_load__{slug}.png")
        if plot_goodput(rows, os.path.join(PLOTS, name), shape=shape):
            made.append(name)
    e4 = [row_view(r) for r in load_results("E4")]
    if e4 and plot_stacks(e4, os.path.join(PLOTS, "e4_stacks_vs_concurrency.png")):
        made.append("e4_stacks_vs_concurrency.png")
    return "Wrote: " + (", ".join(made) if made else NOT_RUN)


# --------------------------------------------------------------------------
# all
# --------------------------------------------------------------------------


def cmd_all(args: argparse.Namespace) -> str:
    os.makedirs(TABLES, exist_ok=True)
    ns = argparse.Namespace(experiment=None,
                            include_superseded=getattr(args, "include_superseded", False))
    sections = {
        "summary.md": cmd_summary(ns),
        "e0_variance.md": cmd_variance(ns),
        "e1_sweep.md": cmd_sweep(ns),
        "e2_ablation.md": cmd_ablation(ns),
        "e3_frontier.md": cmd_frontier(ns),
        "e4_stacks.md": cmd_stacks(ns),
        "e5_economics.md": cmd_economics(ns),
    }
    for name, text in sections.items():
        with open(os.path.join(TABLES, name), "w") as fh:
            fh.write(text + "\n")
        print(f"[analyze] wrote results/tables/{name}")
    print(cmd_plots(ns))
    return "done"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("summary", "variance", "sweep", "ablation", "frontier",
                 "stacks", "economics", "plots", "all"):
        p = sub.add_parser(name)
        p.add_argument("--experiment", default=None)
        p.add_argument("--include-superseded", action="store_true",
                       help="include earlier runs of a point that was re-run")
    args = ap.parse_args()
    fn = {
        "summary": cmd_summary, "variance": cmd_variance, "sweep": cmd_sweep,
        "ablation": cmd_ablation, "frontier": cmd_frontier,
        "stacks": cmd_stacks,
        "economics": cmd_economics, "plots": cmd_plots,
        "all": cmd_all,
    }[args.cmd]
    out = fn(args)
    if out:
        print(out)
    if _INCR_RECOMPUTED:
        big = {k: v for k, v in _INCR_RECOMPUTED.items() if abs(v) >= 0.001}
        worst = max(_INCR_RECOMPUTED.items(), key=lambda kv: abs(kv[1]))
        print(f"[analyze] idle-subtracted energy recomputed under the median "
              f"baseline policy for {len(_INCR_RECOMPUTED)} run(s), of which "
              f"{len(big)} moved by more than 0.1% (worst {worst[0]} "
              f"{worst[1]:+.2%}); raw J/token unchanged. See METHODOLOGY "
              f"'Idle baseline'.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
