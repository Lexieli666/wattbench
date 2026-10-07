#!/usr/bin/env python3
"""A record's status is decided by what the run delivered, not by the caller.

  ~/wattbench-venv/bin/python tests/test_harness_status.py

Synthetic bench JSON through the real `harness.py assemble` CLI, then the
records through analyze.py's loader and status column. No GPU, no server.
The case this exists for: on 2026-10-07 a compiled server died during warmup,
all 800 requests failed at connect, and the record said `ok` -- so the sweep
driver counted it a success and the stack table rendered it as served at
0 tok/s.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import analyze  # noqa: E402
import harness  # noqa: E402

CONFIG = os.path.join(REPO, "configs", "e4", "pytorch_c8.yaml")
REQUESTED = harness.effective_num_prompts(harness.load_config(CONFIG))

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("  ok   " if ok else "  FAIL ") + name + (f": {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def bench(completed: int, p50: float = 20.0, p99: float = 30.0) -> dict:
    """The fields sanity_checks reads, shaped like `vllm bench serve` output.
    No per-request arrays, so goodput is SKIP rather than computed."""
    out_tok = completed * 128
    return {
        "duration": 600.0 if completed else 0.65,
        "completed": completed,
        "num_prompts": REQUESTED,
        "total_input_tokens": completed * 512,
        "total_output_tokens": out_tok,
        "request_throughput": completed / 600.0,
        "output_throughput": out_tok / 600.0,
        **{f"median_{b}_ms": p50 for b in ("ttft", "tpot", "itl", "e2el")},
        **{f"p99_{b}_ms": p99 for b in ("ttft", "tpot", "itl", "e2el")},
    }


def assemble(tmp: str, name: str, b: dict, status: str = "ok") -> dict:
    bench_json = os.path.join(tmp, f"{name}.bench.json")
    prov_json = os.path.join(tmp, "provenance.json")
    out = os.path.join(tmp, "raw", f"e4_pytorch_bf16_c8__2026010{len(name)}T000000_{name}.json")
    with open(bench_json, "w") as fh:
        json.dump(b, fh)
    with open(prov_json, "w") as fh:
        json.dump({"synthetic": True}, fh)
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "harness.py"), "assemble",
         "--config", CONFIG, "--bench-json", bench_json, "--provenance", prov_json,
         "--status", status, "--out", out],
        capture_output=True, text=True)
    check(f"{name}: assemble exits 0 (a failure record is still written)",
          proc.returncode == 0, proc.stderr.strip()[-200:] if proc.returncode else "")
    with open(out) as fh:
        return json.load(fh)


def main() -> int:
    print("--- derive_status ---")
    passed = {"overall": "PASS"}
    for args, want in (
        (("ok", REQUESTED, REQUESTED, passed), "ok"),
        (("ok", REQUESTED - 87, REQUESTED, passed), "partial"),
        (("ok", 0, REQUESTED, passed), "failed"),
        (("ok", None, REQUESTED, passed), "failed"),
        (("bench_failed", 0, REQUESTED, passed), "failed"),
        (("bench_failed", REQUESTED, REQUESTED, passed), "bench_failed"),
        (("ok", REQUESTED, REQUESTED, {"overall": "FAIL"}), "sanity_failed"),
    ):
        got = harness.derive_status(*args)
        check(f"derive_status{args[:3]} -> {want}", got == want, got)

    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "raw"))
        print("--- assemble, synthetic bench JSON ---")
        ok = assemble(tmp, "ok", bench(REQUESTED))
        check("all completed, sanity clean -> ok", ok["status"] == "ok",
              f"{ok['status']}, sanity {ok['sanity']['overall']}")

        part = assemble(tmp, "partial", bench(REQUESTED - 87))
        check("some completed -> partial", part["status"] == "partial", part["status"])

        # The 2026-10-07 shape: the load generator exits 0 with a result file
        # in which every request failed.
        dead = assemble(tmp, "failed", bench(0))
        check("none completed -> failed", dead["status"] == "failed", dead["status"])
        check("failed record keeps its sanity block",
              dead["sanity"]["overall"] == "FAIL", dead["sanity"]["overall"])

        bad = assemble(tmp, "sanity", bench(REQUESTED, p50=40.0, p99=30.0))
        check("all completed but a sanity FAIL -> not ok",
              bad["status"] == "sanity_failed", bad["status"])

        print("--- analyze.py loader and status column ---")
        analyze.RAW = os.path.join(tmp, "raw")
        everything = analyze.load_results(include_failed=True, include_superseded=True)
        statuses = sorted(r["status"] for r in everything)
        check("failed is excluded even with include_failed",
              statuses == ["ok", "partial", "sanity_failed"], str(statuses))
        only_ok = analyze.load_results(include_superseded=True)
        check("default loader keeps only ok", [r["status"] for r in only_ok] == ["ok"],
              str([r["status"] for r in only_ok]))

        labels = {r["status"]: analyze.status_label(analyze.row_view(r)) for r in everything}
        check("ok renders as served", labels.get("ok") == "served", str(labels))
        check("partial renders as partial", labels.get("partial") == "partial", str(labels))
        check("sanity_failed renders as its status",
              labels.get("sanity_failed") == "sanity_failed", str(labels))

        # Records assembled before the harness derived status: `ok` on disk.
        legacy_zero = dict(analyze.row_view(dead), status="ok")
        legacy_part = dict(analyze.row_view(part), status="ok")
        check("legacy ok with 0 completed renders as failed",
              analyze.status_label(legacy_zero) == "failed", analyze.status_label(legacy_zero))
        check("legacy ok with dropped requests is never served",
              analyze.status_label(legacy_part) == "partial", analyze.status_label(legacy_part))
        zero_tps = dict(analyze.row_view(ok), out_tok_throughput=0.0)
        check("0 tok/s is never served", analyze.status_label(zero_tps) != "served",
              analyze.status_label(zero_tps))

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
