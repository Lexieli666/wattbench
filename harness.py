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
import glob
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
        # "vllm" or "llamacpp". Everything below the divider is read only by
        # the stack it names; a vLLM config that never mentions llama.cpp is
        # unaffected by its presence, which is why E0-E3 need no re-running.
        "stack": "vllm",
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
        # --- llamacpp only ---------------------------------------------------
        "gguf_repo": None,        # HF repo holding the GGUF
        "gguf_file": None,        # filename within it, e.g. *-q4_k_m.gguf
        "n_gpu_layers": 99,       # 99 == every layer on the GPU
        "parallel": 1,            # server-side slots; must cover concurrency
        "ctx_size": None,         # TOTAL KV context, split across slots
        "cont_batching": True,
        "flash_attn": "auto",
        # The load generator needs a tokenizer to build the random dataset and
        # to count tokens. A GGUF path is not an HF repo, so it is named here
        # and is the same tokenizer the vLLM arm uses -- identical prompts by
        # construction, which is what makes the two arms comparable at all.
        "tokenizer": None,
        "served_model_name": None,
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
        # null means "whatever the server defaults to" -- a per-checkpoint
        # property from generation_config.json, so it differs across the E3
        # ladder. The base configs pin it to 0 for that reason; the default
        # stays null so a config that does not pin it is visibly not pinned.
        "temperature": None,
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
        "stack", "model", "revision", "quantization", "dtype", "max_model_len",
        "gpu_memory_utilization", "max_num_seqs", "enable_prefix_caching",
        "extra_args", "port",
        "gguf_repo", "gguf_file", "n_gpu_layers", "parallel", "ctx_size",
        "cont_batching", "flash_attn",
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


def gguf_path(cfg: dict) -> str:
    """Local path of the GGUF weights, resolved from the HF cache.

    Resolved rather than downloaded: a fetch here would run inside a measured
    window, and this project has measured what that does (TTFT p95 +819%).
    Missing weights are a hard error telling you to fetch them first.
    """
    s = cfg["server"]
    repo, fname = s.get("gguf_repo"), s.get("gguf_file")
    if not (repo and fname):
        raise SystemExit("[harness] llamacpp stack needs server.gguf_repo and "
                         "server.gguf_file")
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    root = os.path.join(hf_home, "hub", "models--" + repo.replace("/", "--"))
    hits = sorted(glob.glob(os.path.join(root, "snapshots", "*", fname)))
    if not hits:
        raise SystemExit(
            f"[harness] {fname} not found under {root}. Fetch it before the run:\n"
            f"  HF_HUB_DISABLE_XET=1 hf download {repo} --include '{fname}'")
    return hits[-1]


def llamacpp_server_args(cfg: dict) -> list[str]:
    """Flags for `llama-server`.

    Deliberately minimal: each stack is run at its own defaults except where
    the *workload* forces a choice (how many concurrent slots, how much KV
    context, all layers on the GPU). Tuning one stack and not the other would
    measure the tuning.
    """
    s = cfg["server"]
    ctx = s.get("ctx_size")
    if ctx is None:
        # llama.cpp divides its context across slots, so a per-slot budget has
        # to be multiplied out. Default: the same 4096 per sequence vLLM gets
        # from max_model_len.
        ctx = int(s.get("max_model_len") or 4096) * int(s.get("parallel") or 1)
    args = [
        "-m", gguf_path(cfg),
        "--host", "127.0.0.1",
        "--port", str(s["port"]),
        "-ngl", str(s.get("n_gpu_layers", 99)),
        "-c", str(ctx),
        "--parallel", str(s.get("parallel") or 1),
        "--metrics",                      # /metrics, scraped by kv_log.py
        "--alias", str(s.get("served_model_name") or s["model"]),
    ]
    if s.get("flash_attn") is not None:
        args += ["--flash-attn", str(s["flash_attn"])]
    if s.get("cont_batching") is False:
        args += ["--no-cont-batching"]
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


WARMUP_PROMPT_CAP = 600


def warmup_prompts(cfg: dict) -> int:
    """Prompts for the warmup pass: warmup_s worth of load at the same rate.

    Capped, because at an offered rate the server cannot sustain, "30s of
    offered load" is far more than 30s of work -- the warmup would outlast the
    measurement. The cap still delivers a saturated card at steady clocks,
    which is what the warmup is for.
    """
    l, p = cfg["load"], cfg["protocol"]
    floor = int(p.get("warmup_prompts", 32))
    rate = l.get("request_rate")
    if rate in (None, "inf", "infinity") or float(rate) == float("inf"):
        return floor
    return min(WARMUP_PROMPT_CAP,
               max(floor, int(float(rate) * float(p.get("warmup_s", 30)))))


def export_env(cfg: dict) -> str:
    """Emit shell exports consumed by run.sh."""
    s, l, p = cfg["server"], cfg["load"], cfg["protocol"]
    stack = str(s.get("stack") or "vllm")
    args = llamacpp_server_args(cfg) if stack == "llamacpp" else server_args(cfg)
    # The name the endpoint answers to. vLLM serves under the HF repo id; the
    # llama.cpp arm is given the same string via --alias so the load generator
    # sends identical request bodies to both.
    served = str(s.get("served_model_name") or s["model"])
    lines = [
        f"WB_POINT_ID={shlex.quote(str(cfg['point_id']))}",
        f"WB_EXPERIMENT={shlex.quote(str(cfg.get('experiment', 'NA')))}",
        f"WB_STACK={shlex.quote(stack)}",
        f"WB_MODEL={shlex.quote(str(s['model']))}",
        f"WB_SERVED_MODEL_NAME={shlex.quote(served)}",
        f"WB_TOKENIZER={shlex.quote(str(s.get('tokenizer') or s['model']))}",
        # vLLM's own backend speaks to a vLLM server; anything else is driven
        # through the OpenAI-compatible path, which llama.cpp implements.
        f"WB_BENCH_BACKEND={shlex.quote('vllm' if stack == 'vllm' else 'openai')}",
        f"WB_PORT={shlex.quote(str(s['port']))}",
        f"WB_SERVER_FP={shlex.quote(server_fingerprint(cfg))}",
        f"WB_SERVER_ARGS={shlex.quote(' '.join(shlex.quote(a) for a in args))}",
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
        f"WB_TEMPERATURE={shlex.quote('' if l.get('temperature') is None else str(l['temperature']))}",
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
            **({"llamacpp": _llamacpp_build()} if os.environ.get("WB_LLAMACPP_BIN") else {}),
        },
        "harness_git": git_state(),
        "env": {
            k: os.environ.get(k)
            for k in ("HF_HOME", "VLLM_ATTENTION_BACKEND", "CUDA_VISIBLE_DEVICES",
                      "VLLM_USE_V1", "VLLM_USE_V2_MODEL_RUNNER",
                      "VLLM_USE_FLASHINFER_SAMPLER", "WATTBENCH_NVIDIA_SMI")
            if os.environ.get(k)
        },
    }


def _llamacpp_build() -> dict:
    """Version of the llama.cpp binary actually about to serve.

    llama.cpp has no release cadence to cite, so the source commit is the
    version: `llama-server --version` prints the build number and commit it was
    compiled from, which is what makes an E4 number reproducible.
    """
    binary = os.environ.get("WB_LLAMACPP_BIN", "")
    out = sh([binary, "--version"]) if binary else ""
    build = {"binary": binary, "version_output": out}
    m = re.search(r"commit\s+([0-9a-f]{7,40})", out)
    if m:
        build["commit"] = m.group(1)
    # The binary's own build number is 1 here because it was compiled from a
    # shallow clone with no tags, so the source tree is the authority on which
    # commit this is.
    src = os.environ.get("WB_LLAMACPP_SRC", "")
    if src and os.path.isdir(src):
        build["source_dir"] = src
        build["source_commit"] = sh(["git", "-C", src, "rev-parse", "HEAD"])
        build["source_commit_date"] = sh(
            ["git", "-C", src, "log", "-1", "--format=%cI"])
    return build


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


def derive_e2els(bench: dict) -> dict:
    """Reconstruct per-request end-to-end latency, in seconds.

    `vllm bench serve --save-detailed` emits `ttfts` and `itls` but no
    per-request end-to-end array, and end-to-end is half of the goodput SLO.
    For a streamed request the total is exactly the time to first token plus
    every inter-token gap after it, so the array is recoverable -- but only
    before `itls` is dropped from the committed artifact, hence doing it here.
    """
    if bench.get("e2els") or not bench.get("ttfts") or not bench.get("itls"):
        return bench
    ttfts, itls = bench["ttfts"], bench["itls"]
    if len(ttfts) != len(itls):
        return bench
    # Both arrays are in SECONDS. vLLM derives its *_ms summaries by scaling
    # them (mean_itl_ms == mean(itls) * 1000), so no conversion belongs here.
    # Verified against a real run: mean(ttft) 0.0274s + mean(sum itls) 0.704s
    # == mean_e2el_ms 731.5ms.
    bench = dict(bench)
    bench["e2els"] = [t + sum(gaps) for t, gaps in zip(ttfts, itls)]
    bench["e2els_derived"] = (
        "ttft + sum(itl), seconds; vLLM emits no per-request e2e array"
    )
    return bench


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


def compute_goodput(bench: dict, slo_ttft_s: float, slo_e2e_s: float) -> dict:
    """Per-request SLO attainment from vllm bench serve's per-request arrays.

    Goodput here means: requests that actually completed AND met BOTH the TTFT
    and the end-to-end latency SLO, expressed as requests/s over the benchmark
    duration.

    A failed request must be excluded from the numerator explicitly. vLLM
    records one as ttft = 0.0, output_len = 0 and a non-empty error string --
    and 0.0 passes every latency threshold, so a naive comparison counts a
    connection failure as the best possible response. Observed for real: one
    request in an E1 point failed with a transport traceback and pushed goodput
    above measured throughput, which is what surfaced the bug.

    Failures stay in the denominator: a server that drops requests is not
    thereby faster.
    """
    ttfts = bench.get("ttfts") or []
    e2es = bench.get("e2els") or bench.get("e2es") or []
    errors = bench.get("errors") or []
    out_lens = bench.get("output_lens") or []
    dur = bench.get("duration")
    completed = bench.get("completed")
    requested = bench.get("num_prompts") or completed

    if not ttfts or not e2es or not dur:
        return {
            "available": False,
            "reason": "per-request ttft/e2e arrays not present in bench output",
        }

    n = min(len(ttfts), len(e2es))

    def failed(i: int) -> bool:
        if i < len(errors) and errors[i]:
            return True
        if i < len(out_lens) and not out_lens[i]:
            return True
        return ttfts[i] <= 0.0

    n_failed = sum(1 for i in range(n) if failed(i))
    ok = [i for i in range(n) if not failed(i)]
    met_both = sum(1 for i in ok if ttfts[i] <= slo_ttft_s and e2es[i] <= slo_e2e_s)
    met_ttft = sum(1 for i in ok if ttfts[i] <= slo_ttft_s)
    met_e2e = sum(1 for i in ok if e2es[i] <= slo_e2e_s)

    denom = requested or n
    return {
        "available": True,
        "slo_ttft_s": slo_ttft_s,
        "slo_e2e_s": slo_e2e_s,
        "n_evaluated": n,
        "n_requested": requested,
        "n_completed": completed,
        "n_failed": n_failed,
        "n_met_both": met_both,
        "n_met_ttft": met_ttft,
        "n_met_e2e": met_e2e,
        "goodput_req_per_s": round(met_both / dur, 6),
        "slo_attainment_frac": round(met_both / denom, 4) if denom else None,
        "note": ("failed requests are excluded from the numerator and kept in the "
                 "denominator: a server that sheds load is not thereby faster"),
    }


# vLLM says why it refused to start in a handful of recognisable lines. The
# whole server log is kept next to the record; these are lifted out so the
# reason survives into a table without anyone reading the traceback.
_ERR_RE = re.compile(r"^(?:\w+\.)*\w*(?:Error|Exception):\s")
_FACT_RE = re.compile(
    r"Available KV cache memory|estimated maximum model length"
    r"|Free memory on device|Model loading took|Estimated CUDA graph memory"
)


def _strip_log_prefix(line: str) -> str:
    """Drop the "(EngineCore pid=N) ERROR 08-11 10:48:40 [core.py:1330] " noise."""
    line = re.sub(r"^\([A-Za-z]+ pid=\d+\)\s*", "", line.rstrip())
    line = re.sub(r"^(?:ERROR|INFO|WARNING)\s+[\d-]+\s+[\d:]+\s+\[[^\]]+\]\s*", "",
                  line)
    return line.strip()


def read_failure(server_log: str | None) -> dict:
    """Recover, from a server's own log, why it never served."""
    out: dict = {"root_cause": None, "evidence": [], "server_log_lines": 0}
    if not server_log or not os.path.exists(server_log):
        return out
    causes: list[str] = []
    facts: list[str] = []
    with open(server_log, errors="replace") as fh:
        for raw in fh:
            out["server_log_lines"] += 1
            line = _strip_log_prefix(raw)
            if not line:
                continue
            if _ERR_RE.match(line):
                if line not in causes:
                    causes.append(line)
            elif _FACT_RE.search(line) and line not in facts:
                facts.append(line)
    # The innermost exception is raised first and re-raised outward, so the
    # first non-wrapper line is the actual cause. "Engine core initialization
    # failed" is the wrapper and explains nothing on its own.
    inner = [c for c in causes if "Engine core initialization failed" not in c]
    out["root_cause"] = (inner or causes or [None])[0]
    out["evidence"] = facts[-12:]
    return out


def assemble_unserved(args: argparse.Namespace, cfg: dict, prov: dict) -> int:
    """Write a record for a configuration that never served a request.

    Same schema as a served run, with the measured sections empty rather than
    absent, so the row sorts and renders alongside the rungs that did serve.
    """
    failure = read_failure(args.server_log)
    result = {
        "schema_version": 1,
        "point_id": cfg["point_id"],
        "experiment": cfg.get("experiment"),
        "description": cfg.get("description"),
        "status": args.status,
        "run_started_at": args.started_at,
        "run_finished_at": args.finished_at,
        "measurement_window": {"start": None, "end": None},
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_path": cfg["_config_path"],
        "model_revision": resolve_hf_revision(
            cfg["server"]["model"], cfg["server"].get("revision")
        ),
        "provenance": prov,
        "metrics": {},
        "server_metrics": {"available": False, "reason": args.status},
        "goodput": {"available": False, "reason": args.status},
        "energy": {},
        "failure": failure,
        "artifacts": {
            "bench_json": None,
            "power_csv": None,
            "kv_csv": None,
            "power_json": None,
            "server_log": (os.path.basename(args.server_log)
                           if args.server_log else None),
        },
        "notes": args.note or [],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[harness] wrote {args.out} (status={args.status})")
    if failure["root_cause"]:
        print(f"[harness] root cause: {failure['root_cause']}")
    return 0


def assemble(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    bench = _load_json(args.bench_json)
    power = _load_json(args.power_json)
    kv = _load_json(args.kv_json)
    prov = _load_json(args.provenance) or provenance()

    if bench is None:
        if args.status == "ok":
            raise SystemExit(f"[harness] bench JSON missing: {args.bench_json}")
        # A configuration that never produced load -- typically one whose
        # server would not start -- is still a result. cmd_frontier already
        # renders non-`served` statuses and keeps them, because an absent row
        # in a frontier table reads as "does not exist" rather than "was not
        # measured". Without this, the only rung the card cannot serve is the
        # one rung the table fails to mention.
        return assemble_unserved(args, cfg, prov)

    # Freeze `auto` sizing to the value actually used, so the record is exact.
    cfg["load"]["num_prompts"] = effective_num_prompts(cfg)
    bench = derive_e2els(bench)
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
            # Which baseline measurement was subtracted. Idle draw drifts
            # between sessions, so an incremental J/token is only interpretable
            # alongside the baseline it was computed against.
            "idle_baseline_source": {
                "file": args.idle_baseline_file,
                "series": args.idle_baseline_series,
                "measured_at": args.idle_baseline_measured_at,
            },
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
        # Server-side view: KV-cache utilisation and queue depth over the same
        # measurement window. Saturation is a growing queue, and that is only
        # visible from the server, not from the load generator.
        "server_metrics": kv or {"available": False, "reason": "not collected"},
        "goodput": good,
        "energy": energy,
        "artifacts": {
            "bench_json": os.path.basename(args.bench_json) if args.bench_json else None,
            "power_csv": os.path.basename(args.power_csv) if args.power_csv else None,
            "kv_csv": os.path.basename(args.kv_csv) if args.kv_csv else None,
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
        # Tolerance absorbs the rounding applied to the stored goodput; at 100%
        # SLO attainment the two are the same quantity computed twice.
        add("goodput_le_throughput", gp <= tput * (1 + 1e-4) + 1e-9,
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

    # Energy integration. The gate is the discretisation bound (half the spread
    # between left- and right-Riemann sums), which measures whether the ~2Hz
    # sample rate is fast enough to integrate this power trace.
    cc = e.get("crosscheck") or {}
    bound = cc.get("discretisation_bound")
    rel = cc.get("abs_rel_diff")
    if bound is not None:
        add("energy_integration_bound", bound < 0.02,
            f"half-spread of Riemann sums / trapezoid = {bound:.4f} (tolerance 0.02)")
    else:
        add("energy_integration_bound", None, "no power data")
    if rel is not None:
        # Informational: unweighted sample mean is not the time-weighted mean
        # when the cadence is uneven, so this is reported, never a FAIL.
        checks.append({
            "name": "energy_vs_mean_power_duration",
            "result": "INFO",
            "detail": f"|mean*dt - trapezoid| / trapezoid = {rel:.4f} "
                      f"(not a gate; see power_log.py)",
        })

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

    # Protocol minimum is plan §2's "each point >= 3 minutes OR >= 200 requests".
    # A saturated high-rate point can satisfy the request count in well under
    # three minutes; that is a valid point, not a short run.
    proto = (result.get("config") or {}).get("protocol", {})
    min_dur = proto.get("min_duration_s", 0)
    min_req = proto.get("min_requests", 0)
    if dur is not None or completed is not None:
        by_time = dur is not None and dur >= min_dur * 0.95
        by_count = completed is not None and completed >= min_req
        satisfied_by = "+".join(
            n for n, ok in (("duration", by_time), ("count", by_count)) if ok
        ) or "neither"
        add("meets_protocol_minimum", by_time or by_count,
            f"duration {dur if dur is None else round(dur, 1)}s (min {min_dur}s), "
            f"{completed} requests (min {min_req}); satisfied by {satisfied_by}")

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
        "n_info": sum(1 for c in checks if c["result"] == "INFO"),
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
    # Not required: a configuration whose server never started produces no
    # bench output, and that outcome still has to be recorded (see
    # assemble_unserved). A missing bench JSON with --status ok is still fatal.
    a.add_argument("--bench-json")
    a.add_argument("--power-json")
    a.add_argument("--power-csv")
    a.add_argument("--kv-json")
    a.add_argument("--kv-csv")
    a.add_argument("--provenance")
    a.add_argument("--server-log")
    a.add_argument("--out", required=True)
    a.add_argument("--status", default="ok")
    a.add_argument("--idle-baseline-file")
    a.add_argument("--idle-baseline-series")
    a.add_argument("--idle-baseline-measured-at")
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
