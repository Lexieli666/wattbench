#!/usr/bin/env bash
# Find what a checkpoint can actually be served at on this card.
#
#   ./probe_limits.sh context Qwen/Qwen2.5-7B-Instruct bf16
#   ./probe_limits.sh context Qwen/Qwen2.5-7B-Instruct-AWQ awq
#
# This answers the half of E2 that is not a throughput number: what int4 buys
# that BF16 cannot on 24GB. Both probes are recorded as results, including the
# configurations that fail -- a config that will not load is the finding.
#
# context : largest --max-model-len that loads and answers one request.
#           Walks a fixed ladder upward and stops at the first failure, so the
#           answer is "the largest rung that worked", not an interpolation.
#
# Weights are already local, so each attempt costs a model load (~1 min), not a
# download.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${WATTBENCH_VENV:-$HOME/wattbench-venv}"
PY="$VENV/bin/python"
VLLM="$VENV/bin/vllm"
RAW="$REPO/results/raw"
TMP="${WATTBENCH_TMP:-/tmp/wattbench}"
PORT="${WB_PROBE_PORT:-8000}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

MODE="${1:-}"; MODEL="${2:-}"; LABEL="${3:-}"
if [[ -z "$MODE" || -z "$MODEL" || -z "$LABEL" ]]; then
  echo "usage: ./probe_limits.sh context <model> <label>" >&2
  exit 1
fi

mkdir -p "$RAW" "$TMP"
GPU_UTIL="${WB_PROBE_GPU_UTIL:-0.90}"
CTX_LADDER="${WB_CTX_LADDER:-4096 8192 16384 32768 65536 131072}"

log() { printf '[probe %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

stop_server() {
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 5
}

# Start the server at a given context length; return 0 if it becomes healthy
# AND answers one real request. "Loads" is not the same as "serves": a config
# can pass startup and still fail the first forward pass.
try_context() {
  local ctx="$1" logf="$TMP/probe_${LABEL}_${ctx}.log"
  : > "$logf"
  setsid "$VLLM" serve "$MODEL" --port "$PORT" \
      --max-model-len "$ctx" \
      --gpu-memory-utilization "$GPU_UTIL" \
      --max-num-seqs 256 \
      --dtype bfloat16 --no-enable-prefix-caching >>"$logf" 2>&1 &
  local pid=$! waited=0
  while true; do
    if curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      FAIL_REASON="$(grep -oE 'ValueError:.*|RuntimeError:.*|torch\.OutOfMemoryError:.*|No available memory for the cache blocks.*' "$logf" | tail -1)"
      [[ -z "$FAIL_REASON" ]] && FAIL_REASON="server exited during startup"
      return 1
    fi
    sleep 3; waited=$((waited + 3))
    if (( waited >= 600 )); then
      FAIL_REASON="did not become healthy within 600s"
      kill -KILL "-$pid" 2>/dev/null || true
      return 1
    fi
  done

  # Healthy is not enough -- make it actually generate.
  local probe_tokens=$(( ctx > 64 ? 16 : 4 ))
  if ! curl -sf -m 120 "http://127.0.0.1:${PORT}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"prompt\":\"hello\",\"max_tokens\":${probe_tokens}}" \
        >/dev/null 2>&1; then
    FAIL_REASON="healthy but failed to complete a request"
    kill -KILL "-$pid" 2>/dev/null || true
    return 1
  fi

  KV_TOKENS="$(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$logf" | tail -1 | grep -oE '[0-9,]+' || true)"
  CONCURRENCY="$(grep -oE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' "$logf" | tail -1 || true)"
  kill -KILL "-$pid" 2>/dev/null || true
  return 0
}

STAMP="$(date +%Y%m%dT%H%M%S)"
OUT="$RAW/limits_${MODE}_${LABEL}__${STAMP}.json"
ATTEMPTS="[]"

case "$MODE" in
context)
  log "probing longest servable context for $MODEL (gpu_mem_util=$GPU_UTIL)"
  BEST=""
  for ctx in $CTX_LADDER; do
    stop_server
    KV_TOKENS=""; CONCURRENCY=""; FAIL_REASON=""
    log "  trying --max-model-len $ctx"
    if try_context "$ctx"; then
      log "    served. KV cache: ${KV_TOKENS:-?} tokens. ${CONCURRENCY:-}"
      BEST="$ctx"
      ATTEMPTS="$("$PY" -c "
import json,sys
a=json.loads(sys.argv[1]); a.append({'max_model_len':int(sys.argv[2]),'served':True,
 'kv_cache_tokens':sys.argv[3] or None,'reported_concurrency':sys.argv[4] or None,'error':None})
print(json.dumps(a))" "$ATTEMPTS" "$ctx" "$KV_TOKENS" "$CONCURRENCY")"
    else
      log "    FAILED: ${FAIL_REASON}"
      ATTEMPTS="$("$PY" -c "
import json,sys
a=json.loads(sys.argv[1]); a.append({'max_model_len':int(sys.argv[2]),'served':False,
 'kv_cache_tokens':None,'reported_concurrency':None,'error':sys.argv[3]})
print(json.dumps(a))" "$ATTEMPTS" "$ctx" "$FAIL_REASON")"
      break
    fi
  done
  stop_server
  "$PY" - "$OUT" "$MODEL" "$LABEL" "$BEST" "$GPU_UTIL" "$ATTEMPTS" <<'PYEOF'
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])) or ".")
sys.path.insert(0, os.environ.get("WB_REPO", "."))
import harness
out, model, label, best, util, attempts = sys.argv[1:7]
json.dump({
    "schema_version": 1,
    "point_id": f"limits_context_{label}",
    "experiment": "E2-limits",
    "status": "ok" if best else "no_context_served",
    "description": (
        "Longest servable context: largest --max-model-len that both loads and "
        "completes one request, walking a fixed ladder upward and stopping at "
        "the first failure."
    ),
    "config": {"model": model, "label": label, "gpu_memory_utilization": float(util)},
    "model_revision": harness.resolve_hf_revision(model, None),
    "provenance": harness.provenance(),
    "metrics": {
        "longest_servable_context": int(best) if best else None,
        "attempts": json.loads(attempts),
    },
}, open(out, "w"), indent=2)
print(f"[probe] wrote {out}")
PYEOF
  log "longest servable context for $LABEL: ${BEST:-none}"
  ;;
*)
  echo "unknown mode: $MODE (expected: context)" >&2
  exit 1
  ;;
esac
