#!/usr/bin/env python3
"""Config parsing, provenance capture, and raw-result assembly for WattBench.

`run.sh` owns process lifecycle; this owns everything that is easier to get
right in Python than in bash. Sub-commands:

  export-env    config YAML -> shell `export` lines for run.sh to eval
  provenance    capture stack/driver/GPU/git state as JSON
  assemble      bench JSON + power JSON + provenance -> one raw result file
  validate      sanity-check an assembled raw result

The assembled file in results/raw/ is the only thing downstream analysis is
allowed to read, so everything needed to interpret or reproduce a number has
to be inside it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime

try:
    import yaml
except ImportError:  # pragma: no cover - surfaced immediately by run.sh
    print("[harness] PyYAML missing: pip install pyyaml", file=sys.stderr)
    raise

REPO = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

DEFAULTS = {
    "server": {
        "model": None,
        "revision": None,
        "quantization": None,
        "dtype": "bfloat16",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.90,
        "max_num_seqs": 256,
        "enable_prefix_caching": False,
        "port": 8000,
        "extra_args": [],
        "load_timeout_s": 900,
    },
    "load": {
        "backend": "vllm",
        "dataset": "random",
        "input_len": 512,
        "output_len": 128,
        "request_rate": 2.0,
        "burstiness": 1.0,  # 1.0 == Poisson arrivals
        "num_prompts": 240,
        "max_concurrency": None,
        "seed": 0,
        "ignore_eos": True,
        "extra_args": [],
    },
    "protocol": {
        "warmup_s": 30,
        "warmup_prompts": 32,
        "min_duration_s": 180,
        "min_requests": 200,
    },
    "slo": {
        "ttft_p95_s": 1.0,
        "e2e_p95_s": 10.0,
    },
}


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> dict:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    # A config may inherit from another via `extends:` (relative to this file).
    if raw.get("extends"):
        parent_path = os.path.join(os.path.dirname(os.path.abspath(path)), raw["extends"])
        parent = load_config(parent_path)
        parent.pop("point_id", None)
        parent.pop("description", None)
        raw = deep_merge(parent, {k: v for k, v in raw.items() if k != "extends"})

    cfg = deep_merge(DEFAULTS, raw)
    if not cfg.get("point_id"):
        cfg["point_id"] = os.path.splitext(os.path.basename(path))[0]
    if not cfg["server"].get("model"):
        raise SystemExit(f"[harness] {path}: server.model is required")
    cfg["_config_path"] = os.path.relpath(os.path.abspath(path), REPO)
    return cfg


def server_fingerprint(cfg: dict) -> str:
    """Identity of the served model+flags. Points sharing it can share a server."""
    s = cfg["server"]
    keys = [
        "model", "revision", "quantization", "dtype", "max_model_len",
        "gpu_memory_utilization", "max_num_seqs", "enable_prefix_caching",
        "extra_args", "port",
    ]
    payload = json.dumps({k: s.get(k) for k in keys}, sort_keys=True, default=str)
    import hashlib
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def server_args(cfg: dict) -> list[str]:
    """Flags for `vllm serve <model> ...`; the model itself is positional."""
    s = cfg["server"]
    args = [
        "--port", str(s["port"]),
        "--max-model-len", str(s["max_model_len"]),
        "--gpu-memory-utilization", str(s["gpu_memory_utilization"]),
        "--max-num-seqs", str(s["max_num_seqs"]),
    ]
    if s.get("revision"):
        args += ["--revision", str(s["revision"])]
    if s.get("quantization"):
        args += ["--quantization", str(s["quantization"])]
    else:
        args += ["--dtype", str(s["dtype"])]
    if s.get("enable_prefix_caching"):
        args += ["--enable-prefix-caching"]
    else:
        args += ["--no-enable-prefix-caching"]
    args += [str(x) for x in (s.get("extra_args") or [])]
    return args


def effective_num_prompts(cfg: dict) -> int:
    """Prompt count that satisfies both protocol minimums at this offered rate.

    `num_prompts: auto` sizes the run so it lasts at least min_duration_s at
    the configured arrival rate and still issues at least min_requests. A
    saturated server takes longer than N/rate, so this is a floor, not a
    prediction.
    """
    l, p = cfg["load"], cfg["protocol"]
    n = l.get("num_prompts")
    if n not in (None, "auto"):
        return int(n)
    rate = l.get("request_rate")
    min_req = int(p.get("min_requests", 200))
    if rate in (None, "inf", "infinity") or float(rate) == float("inf"):
        return max(min_req, int(l.get("max_concurrency") or 0) * 8 or min_req)
    by_time = int(float(rate) * float(p.get("min_duration_s", 180)))
    return max(min_req, by_time)


def warmup_prompts(cfg: dict) -> int:
    """Prompts for the warmup pass: warmup_s worth of load at the same rate."""
    l, p = cfg["load"], cfg["protocol"]
    floor = int(p.get("warmup_prompts", 32))
    rate = l.get("request_rate")
    if rate in (None, "inf", "infinity") or float(rate) == float("inf"):
        return floor
    return max(floor, int(float(rate) * float(p.get("warmup_s", 30))))


def export_env(cfg: dict) -> str:
    """Emit shell exports consumed by run.sh."""
    s, l, p = cfg["server"], cfg["load"], cfg["protocol"]
    lines = [
        f"WB_POINT_ID={shlex.quote(str(cfg['point_id']))}",
        f"WB_EXPERIMENT={shlex.quote(str(cfg.get('experiment', 'NA')))}",
        f"WB_MODEL={shlex.quote(str(s['model']))}",
        f"WB_PORT={shlex.quote(str(s['port']))}",
        f"WB_SERVER_FP={shlex.quote(server_fingerprint(cfg))}",
        f"WB_SERVER_ARGS={shlex.quote(' '.join(shlex.quote(a) for a in server_args(cfg)))}",
        f"WB_LOAD_TIMEOUT_S={shlex.quote(str(s['load_timeout_s']))}",
        f"WB_WARMUP_S={shlex.quote(str(p['warmup_s']))}",
        f"WB_WARMUP_PROMPTS={shlex.quote(str(warmup_prompts(cfg)))}",
        f"WB_INPUT_LEN={shlex.quote(str(l['input_len']))}",
        f"WB_OUTPUT_LEN={shlex.quote(str(l['output_len']))}",
        f"WB_REQUEST_RATE={shlex.quote(str(l['request_rate']))}",
        f"WB_BURSTINESS={shlex.quote(str(l['burstiness']))}",
        f"WB_NUM_PROMPTS={shlex.quote(str(effective_num_prompts(cfg)))}",
        f"WB_SEED={shlex.quote(str(l['seed']))}",
        f"WB_MAX_CONCURRENCY={shlex.quote('' if l.get('max_concurrency') in (None, '') else str(l['max_concurrency']))}",
        f"WB_IGNORE_EOS={shlex.quote('1' if l.get('ignore_eos') else '')}",
        f"WB_DATASET={shlex.quote(str(l['dataset']))}",
        f"WB_LOAD_EXTRA={shlex.quote(' '.join(shlex.quote(str(a)) for a in (l.get('extra_args') or [])))}",
        f"WB_SLO_TTFT_S={shlex.quote(str(cfg['slo']['ttft_p95_s']))}",
        f"WB_SLO_E2E_S={shlex.quote(str(cfg['slo']['e2e_p95_s']))}",
    ]
    return "\n".join(f"export {x}" for x in lines)


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def sh(cmd: list[str], timeout: int = 30) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or out.stderr or "").strip()
    except Exception as exc:  # noqa: BLE001
        return f"<error: {exc}>"


def pkg_version(mod: str) -> str | None:
    try:
        import importlib.metadata as md
        return md.version(mod)
    except Exception:  # noqa: BLE001
        return None


def resolve_hf_revision(model: str, revision: str | None) -> dict:
    """Resolve the exact commit of the weights actually on disk.

    A model id without a pinned commit is not reproducible, so the resolved
    sha is recorded in every run whether or not the config pinned one.
    """
    info: dict = {"requested_revision": revision, "resolved_commit": None, "source": None}
    if os.path.isdir(model):
        info["source"] = "local_path"
        return info
    hub = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    repo_dir = os.path.join(hub, "models--" + model.replace("/", "--"))
    refs = os.path.join(repo_dir, "refs", revision or "main")
    if os.path.isfile(refs):
        with open(refs) as fh:
            info["resolved_commit"] = fh.read().strip()
        info["source"] = "hf_cache_refs"
        return info
    snap = os.path.join(repo_dir, "snapshots")
    if os.path.isdir(snap):
        entries = sorted(os.listdir(snap))
        if len(entries) == 1:
            info["resolved_commit"] = entries[0]
            info["source"] = "hf_cache_snapshot"
        elif entries:
            info["resolved_commit"] = None
            info["source"] = f"hf_cache_ambiguous:{len(entries)}_snapshots"
    return info


def gpu_state() -> dict:
    fields = ("name,driver_version,memory.total,power.limit,power.max_limit,"
              "power.min_limit,persistence_mode,pcie.link.gen.current,"
              "pcie.link.width.current,clocks.max.sm,clocks.max.mem")
    raw = sh(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"])
    keys = fields.split(",")
    vals = [v.strip() for v in raw.split(",")]
    state = dict(zip(keys, vals)) if len(vals) == len(keys) else {"raw": raw}
    state["cuda_version_smi"] = None
    header = sh(["nvidia-smi"])
    m = re.search(r"CUDA Version:\s*([0-9.]+)", header)
    if m:
        state["cuda_version_smi"] = m.group(1)
    return state


def git_state() -> dict:
    def g(*a: str) -> str:
        return sh(["git", "-C", REPO, *a])
    status = g("status", "--porcelain")
    return {
        "commit": g("rev-parse", "HEAD"),
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status.strip()) and not status.startswith("<error"),
        "dirty_files": [l[3:] for l in status.splitlines()][:20] if status.strip() else [],
    }


def provenance() -> dict:
    return {
        "captured_at": datetime.now().astimezone().isoformat(),
        "host": {
            "hostname": sh(["hostname"]),
            "kernel": sh(["uname", "-r"]),
            "os": sh(["bash", "-lc", ". /etc/os-release && echo $PRETTY_NAME"]),
            "cpu": sh(["bash", "-lc", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-"]).strip(),
            "ram_gib_total": sh(["bash", "-lc", "free -g | awk '/^Mem:/{print $2}'"]),
        },
        "gpu": gpu_state(),
        "stack": {
            "python": sys.version.split()[0],
            "vllm": pkg_version("vllm"),
            "torch": pkg_version("torch"),
            "transformers": pkg_version("transformers"),
            "flashinfer": pkg_version("flashinfer-python") or pkg_version("flashinfer"),
            "xformers": pkg_version("xformers"),
            "torch_cuda": _torch_cuda(),
        },
        "harness_git": git_state(),
        "env": {
            k: os.environ.get(k)
            for k in ("HF_HOME", "VLLM_ATTENTION_BACKEND", "CUDA_VISIBLE_DEVICES",
                      "VLLM_USE_V1", "WATTBENCH_NVIDIA_SMI")
            if os.environ.get(k)
        },
    }


def _torch_cuda() -> str | None:
    try:
        import torch
        return torch.version.cuda
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# assemble
# --------------------------------------------------------------------------


def _load_json(path: str | None) -> dict | None:
    if not path or not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def compute_goodput(bench: dict, slo_ttft_s: float, slo_e2e_s: float) -> dict:
    """Per-request SLO attainment from vllm bench serve's per-request arrays.

    Goodput here means: completed requests that met BOTH the TTFT and the
    end-to-end latency SLO, expressed as requests/s over the benchmark
    duration. Requests the server failed outright are counted as misses, not
    dropped from the denominator -- a server that sheds load is not thereby
    faster.
    """
    ttfts = bench.get("ttfts") or []
    e2es = bench.get("e2els") or bench.get("e2es") or []
    dur = bench.get("duration")
    completed = bench.get("completed")
    requested = bench.get("num_prompts") or completed

    if not ttfts or not e2es or not dur:
        return {
            "available": False,
            "reason": "per-request ttft/e2e arrays not present in bench output",
        }

    n = min(len(ttfts), len(e2es))
    met_both = sum(
        1 for i in range(n) if ttfts[i] <= slo_ttft_s and e2es[i] <= slo_e2e_s
    )
    met_ttft = sum(1 for t in ttfts[:n] if t <= slo_ttft_s)
    met_e2e = sum(1 for e in e2es[:n] if e <= slo_e2e_s)

    denom = requested or n
    return {
        "available": True,
        "slo_ttft_s": slo_ttft_s,
        "slo_e2e_s": slo_e2e_s,
        "n_evaluated": n,
        "n_requested": requested,
        "n_completed": completed,
        "n_met_both": met_both,
        "n_met_ttft": met_ttft,
        "n_met_e2e": met_e2e,
        "goodput_req_per_s": round(met_both / dur, 4),
        "slo_attainment_frac": round(met_both / denom, 4) if denom else None,
        "note": "denominator is requests issued, so failed requests count as SLO misses",
    }


def reduce_metrics(bench: dict) -> tuple[dict, dict]:
    """Drop the two unbounded arrays before the result is committed.

    `itls` holds one float per generated token per request (millions of values
    at high load) and `generated_texts` holds the completions themselves. Both
    are dropped; vLLM's own ITL percentiles, computed from the full arrays, are
    kept, and what was dropped is recorded so the reduction is visible in the
    artifact rather than only in the docs.
    """
    dropped = {}
    out = dict(bench)
    for key in ("itls", "generated_texts"):
        if key in out:
            val = out.pop(key)
            try:
                if key == "itls":
                    dropped[key] = {"n_requests": len(val),
                                    "n_values": sum(len(x) for x in val)}
                else:
                    dropped[key] = {"n_requests": len(val)}
            except TypeError:
                dropped[key] = {"n_requests": None}
    return out, dropped


def assemble(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    bench = _load_json(args.bench_json)
    power = _load_json(args.power_json)
    prov = _load_json(args.provenance) or provenance()

    if bench is None:
        raise SystemExit(f"[harness] bench JSON missing: {args.bench_json}")

    # Freeze `auto` sizing to the value actually used, so the record is exact.
    cfg["load"]["num_prompts"] = effective_num_prompts(cfg)
    bench, dropped_arrays = reduce_metrics(bench)

    slo_ttft = cfg["slo"]["ttft_p95_s"]
    slo_e2e = cfg["slo"]["e2e_p95_s"]
    good = compute_goodput(bench, slo_ttft, slo_e2e)

    out_tokens = bench.get("total_output_tokens")
    in_tokens = bench.get("total_input_tokens")
    energy_j = (power or {}).get("energy_j")
    incr_j = (power or {}).get("incremental_energy_j")

    energy = {}
    if power:
        energy = {
            "energy_j": energy_j,
            "incremental_energy_j": incr_j,
            "idle_baseline_w": power.get("idle_baseline_w"),
            "mean_power_w": power.get("mean_power_w"),
            "window_s": power.get("integrated_s"),
            "flags": power.get("flags", []),
            "throttle": power.get("throttle"),
            "clock_sm_mhz": power.get("clock_sm_mhz"),
            "temp_c": power.get("temp_c"),
            "crosscheck": power.get("crosscheck"),
            "power_limit_w_observed": power.get("power_limit_w_observed"),
            "sample_gaps": power.get("sample_gaps"),
        }
        if out_tokens:
            if energy_j is not None:
                energy["j_per_output_token"] = round(energy_j / out_tokens, 5)
            if incr_j is not None:
                energy["j_per_output_token_incremental"] = round(incr_j / out_tokens, 5)
        if out_tokens and in_tokens is not None and energy_j is not None:
            total_tok = out_tokens + in_tokens
            if total_tok:
                energy["j_per_total_token"] = round(energy_j / total_tok, 5)

    result = {
        "schema_version": 1,
        "point_id": cfg["point_id"],
        "experiment": cfg.get("experiment"),
        "description": cfg.get("description"),
        "status": args.status,
        "run_started_at": args.started_at,
        "run_finished_at": args.finished_at,
        "measurement_window": {"start": args.window_start, "end": args.window_end},
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_path": cfg["_config_path"],
        "model_revision": resolve_hf_revision(
            cfg["server"]["model"], cfg["server"].get("revision")
        ),
        "provenance": prov,
        "metrics": bench,
        "metrics_arrays_dropped": dropped_arrays,
        "goodput": good,
        "energy": energy,
        "artifacts": {
            "bench_json": os.path.basename(args.bench_json) if args.bench_json else None,
            "power_csv": os.path.basename(args.power_csv) if args.power_csv else None,
            "power_json": os.path.basename(args.power_json) if args.power_json else None,
            "server_log": os.path.basename(args.server_log) if args.server_log else None,
        },
        "notes": args.note or [],
    }

    result["sanity"] = sanity_checks(result)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[harness] wrote {args.out}")

    failed = [c for c in result["sanity"]["checks"] if c["result"] == "FAIL"]
    if failed:
        print("[harness] SANITY FAILURES:", file=sys.stderr)
        for c in failed:
            print(f"  - {c['name']}: {c['detail']}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------
# sanity
# --------------------------------------------------------------------------


def sanity_checks(result: dict) -> dict:
    checks: list[dict] = []

    def add(name: str, ok: bool | None, detail: str) -> None:
        checks.append({
            "name": name,
            "result": "PASS" if ok else ("SKIP" if ok is None else "FAIL"),
            "detail": detail,
        })

    m = result.get("metrics") or {}
    g = result.get("goodput") or {}
    e = result.get("energy") or {}

    # goodput <= throughput
    tput = m.get("request_throughput")
    gp = g.get("goodput_req_per_s")
    if tput is not None and gp is not None:
        add("goodput_le_throughput", gp <= tput + 1e-9,
            f"goodput={gp} req/s vs throughput={tput} req/s")
    else:
        add("goodput_le_throughput", None, "throughput or goodput missing")

    # percentile ordering
    for base in ("ttft", "tpot", "itl", "e2el"):
        p50 = m.get(f"median_{base}_ms")
        p99 = m.get(f"p99_{base}_ms")
        mean = m.get(f"mean_{base}_ms")
        if p50 is not None and p99 is not None:
            add(f"percentiles_ordered_{base}", p50 <= p99 + 1e-9,
                f"p50={p50}ms p99={p99}ms mean={mean}ms")

    # energy cross-check: trapezoid vs mean x duration
    cc = e.get("crosscheck") or {}
    rel = cc.get("abs_rel_diff")
    if rel is not None:
        add("energy_integration_crosscheck", rel < 0.02,
            f"|trapezoid - mean*dt| / trapezoid = {rel:.4f} (tolerance 0.02)")
    else:
        add("energy_integration_crosscheck", None, "no power data")

    # power window should cover the benchmark duration
    dur = m.get("duration")
    win = e.get("window_s")
    if dur and win:
        cover = win / dur
        add("power_window_covers_benchmark", 0.9 <= cover <= 1.5,
            f"power window {win:.1f}s vs benchmark {dur:.1f}s (ratio {cover:.3f})")
    else:
        add("power_window_covers_benchmark", None, "duration or window missing")

    # completion
    completed = m.get("completed")
    requested = (result.get("config") or {}).get("load", {}).get("num_prompts")
    if completed is not None and requested:
        add("all_requests_completed", completed == requested,
            f"{completed}/{requested} completed")

    # protocol minimums
    proto = (result.get("config") or {}).get("protocol", {})
    if dur is not None:
        add("meets_min_duration", dur >= proto.get("min_duration_s", 0) * 0.95,
            f"duration {dur:.1f}s vs min {proto.get('min_duration_s')}s")
    if completed is not None:
        add("meets_min_requests", completed >= proto.get("min_requests", 0),
            f"{completed} completed vs min {proto.get('min_requests')}")

    # no unflagged thermal problems
    flags = e.get("flags")
    if flags is not None:
        add("no_thermal_or_sampling_flags", len(flags) == 0,
            f"flags={flags}" if flags else "none")

    n_fail = sum(1 for c in checks if c["result"] == "FAIL")
    return {
        "checks": checks,
        "n_pass": sum(1 for c in checks if c["result"] == "PASS"),
        "n_fail": n_fail,
        "n_skip": sum(1 for c in checks if c["result"] == "SKIP"),
        "overall": "PASS" if n_fail == 0 else "FAIL",
    }


def validate(args: argparse.Namespace) -> int:
    with open(args.raw) as fh:
        result = json.load(fh)
    s = sanity_checks(result)
    print(json.dumps(s, indent=2))
    return 0 if s["overall"] == "PASS" else 1


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export-env")
    e.add_argument("config")

    sub.add_parser("provenance")

    a = sub.add_parser("assemble")
    a.add_argument("--config", required=True)
    a.add_argument("--bench-json", required=True)
    a.add_argument("--power-json")
    a.add_argument("--power-csv")
    a.add_argument("--provenance")
    a.add_argument("--server-log")
    a.add_argument("--out", required=True)
    a.add_argument("--status", default="ok")
    a.add_argument("--started-at")
    a.add_argument("--finished-at")
    a.add_argument("--window-start")
    a.add_argument("--window-end")
    a.add_argument("--note", action="append")

    v = sub.add_parser("validate")
    v.add_argument("--raw", required=True)

    args = ap.parse_args()
    if args.cmd == "export-env":
        print(export_env(load_config(args.config)))
        return 0
    if args.cmd == "provenance":
        print(json.dumps(provenance(), indent=2))
        return 0
    if args.cmd == "assemble":
        return assemble(args)
    if args.cmd == "validate":
        return validate(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
