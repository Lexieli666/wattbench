#!/usr/bin/env python3
"""Compare two measured points against E0's resolution limits.

Used to decide whether a change in conditions actually moved a measurement, or
only moved it by less than the run-to-run noise E0 established. Prints CLEAN or
PERTURBED on the last line so a shell script can branch on it.

  ./compare_points.py --baseline e1_chat_7b_bf16_r8 \\
                      --candidate e1_chat_7b_bf16_r8_dltest
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(REPO, "results", "raw")

# (label, path into the result, direction-agnostic)
METRICS = [
    ("output throughput (tok/s)", ("metrics", "output_throughput")),
    ("TTFT p50 (ms)", ("metrics", "median_ttft_ms")),
    ("TTFT p95 (ms)", ("metrics", "p95_ttft_ms")),
    ("ITL p50 (ms)", ("metrics", "median_itl_ms")),
    ("E2E p95 (ms)", ("metrics", "p95_e2el_ms")),
    ("mean GPU power (W)", ("energy", "mean_power_w")),
    ("J per output token", ("energy", "j_per_output_token")),
]


def newest(point_id: str) -> dict | None:
    best = None
    for path in sorted(glob.glob(os.path.join(RAW, f"{point_id}__*.json"))):
        if path.endswith((".power.json", ".kv.json")):
            continue
        with open(path) as fh:
            try:
                r = json.load(fh)
            except json.JSONDecodeError:
                continue
        if r.get("point_id") == point_id and "metrics" in r:
            best = r
    return best


def dig(r: dict, path: tuple[str, ...]):
    cur = r
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def e0_limits() -> dict[str, float]:
    """Per-metric CV from the E0 repeats, as fractions."""
    # Newest run per point_id. Sorting paths alone would interleave superseded
    # and current runs of different repeats (a-old, a-new, b-old, b-new, ...)
    # and a tail slice would mix configurations -- which is exactly the kind of
    # blend this project refuses elsewhere.
    newest_by_point: dict[str, tuple[str, dict]] = {}
    for path in sorted(glob.glob(os.path.join(RAW, "e0_ref_*__*.json"))):
        if path.endswith((".power.json", ".kv.json")):
            continue
        with open(path) as fh:
            try:
                r = json.load(fh)
            except json.JSONDecodeError:
                continue
        if r.get("experiment") != "E0" or "metrics" not in r:
            continue
        pid = r.get("point_id") or path
        prev = newest_by_point.get(pid)
        if prev is None or path > prev[0]:
            newest_by_point[pid] = (path, r)
    runs = [r for _, r in newest_by_point.values()]
    out: dict[str, float] = {}
    if len(runs) < 2:
        return out
    for label, path in METRICS:
        vals = [dig(r, path) for r in runs]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if len(vals) < 2:
            continue
        mean = statistics.fmean(vals)
        if mean:
            out[label] = statistics.stdev(vals) / mean
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out")
    # A metric is "moved" if it exceeds this many E0 standard deviations.
    ap.add_argument("--sigma", type=float, default=3.0)
    # Below this relative difference nothing is flagged however tight E0 was:
    # a 0.2% shift is not operationally meaningful even at 20 sigma.
    ap.add_argument("--floor", type=float, default=0.01)
    a = ap.parse_args()

    base, cand = newest(a.baseline), newest(a.candidate)
    if base is None or cand is None:
        print(f"missing point: baseline={base is not None} candidate={cand is not None}",
              file=sys.stderr)
        print("PERTURBED (could not compare)")
        return 2

    limits = e0_limits()
    rows, moved = [], []
    for label, path in METRICS:
        b, c = dig(base, path), dig(cand, path)
        if not isinstance(b, (int, float)) or not isinstance(c, (int, float)) or not b:
            continue
        rel = (c - b) / b
        cv = limits.get(label)
        sigmas = abs(rel) / cv if cv else None
        flag = (sigmas is not None and sigmas > a.sigma and abs(rel) > a.floor)
        if flag:
            moved.append(label)
        rows.append((label, b, c, rel, cv, sigmas, flag))

    verdict = "CLEAN" if not moved else "PERTURBED"
    lines = [
        f"# Download-interference control",
        "",
        f"`{a.candidate}` measured while a weight download saturated the link, "
        f"against `{a.baseline}` measured on a quiet machine. Both are the same "
        f"configuration and the same offered load.",
        "",
        f"A metric counts as moved if it exceeds **{a.sigma:g}x the E0 run-to-run "
        f"standard deviation** for that metric *and* differs by more than "
        f"{a.floor:.0%} — E0 is tight enough that a fraction of a percent can be "
        f"many sigma without mattering operationally.",
        "",
        "| Metric | quiet | with download | change | E0 CV | sigma | moved |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, b, c, rel, cv, sig, flag in rows:
        lines.append(
            f"| {label} | {b:.4g} | {c:.4g} | {rel * 100:+.2f}% | "
            f"{'—' if cv is None else f'{cv * 100:.2f}%'} | "
            f"{'—' if sig is None else f'{sig:.1f}'} | {'**yes**' if flag else 'no'} |"
        )
    lines += [
        "",
        f"**Verdict: {verdict}.** "
        + ("No metric moved beyond the noise floor, so weight downloads may run "
           "alongside measured benchmarks and the schedule can overlap them."
           if verdict == "CLEAN" else
           f"Moved: {', '.join(moved)}. Downloads are paused during measured "
           f"runs, and the affected points are not compared across that boundary."),
    ]
    text = "\n".join(lines)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    print(text)
    print(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
