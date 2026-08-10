#!/usr/bin/env bash
# WattBench run driver: one config YAML in, one raw result JSON out.
#
#   ./run.sh configs/e1/chat_r2.yaml
#   ./run.sh --force configs/e0/ref_a.yaml
#   ./run.sh --stop-server configs/e1/chat_r32.yaml
#
# Resumable: a point whose raw result already exists is skipped unless --force,
# so an interrupted sweep can simply be relaunched. The vLLM server is kept
# alive between points that share a server fingerprint (model + serving flags),
# because reloading weights per point would cost more wall-clock than the
# measurements themselves.
#
# What happens per point:
#   1. reuse or start the server, wait for /health
#   2. start the power poller (covers warmup + measurement)
#   3. warmup at the measurement offered rate
#   4. record window start, run the measurement, record window end
#   5. stop the poller, integrate power over exactly the measurement window
#   6. assemble config + metrics + energy + provenance into results/raw/

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${WATTBENCH_VENV:-$HOME/wattbench-venv}"
PY="$VENV/bin/python"
VLLM="$VENV/bin/vllm"
RAW="$REPO/results/raw"
STATE_DIR="$REPO/results/.state"
TMP="${WATTBENCH_TMP:-/tmp/wattbench}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# vLLM 0.26's default V2 GPU model runner allocates UVA (unified virtual
# addressing) host buffers, which WSL2's CUDA passthrough does not provide:
# the engine dies at init with "RuntimeError: UVA is not available". The V1
# runner has no such requirement. This is set for every run in the project, so
# the choice is constant across all experiments and comparability holds.
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"

# FlashInfer JIT-compiles its top-k/top-p sampling kernel on first use and needs
# nvcc to do it. No CUDA toolkit is installed here (the wheels ship their own
# kernels), so the engine dies with "Could not find nvcc". Fall back to vLLM's
# PyTorch-native sampler, which is what vLLM's own error message recommends.
# Sampling is a negligible share of decode time, and the setting is constant
# across every run in this project.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"   # ~2x faster here; see fetch_models.sh
# Weights must live on ext4; NTFS passthrough makes model load pathologically slow.
case "$HF_HOME" in
  /mnt/*) echo "FATAL: HF_HOME=$HF_HOME is on a Windows mount. Use an ext4 path." >&2; exit 1 ;;
esac

FORCE=0
STOP_SERVER=0
DRY=0
NOTES=()

usage() {
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

CONFIG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --stop-server) STOP_SERVER=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --note) NOTES+=("$2"); shift 2 ;;
    -h|--help) usage 0 ;;
    -*) echo "unknown flag: $1" >&2; usage 1 ;;
    *) CONFIG="$1"; shift ;;
  esac
done
[[ -n "$CONFIG" ]] || usage 1
[[ -f "$CONFIG" ]] || { echo "FATAL: no such config: $CONFIG" >&2; exit 1; }
[[ -x "$PY" ]] || { echo "FATAL: no venv python at $PY (set WATTBENCH_VENV)" >&2; exit 1; }

mkdir -p "$RAW" "$STATE_DIR" "$TMP"

# --- config -> environment ------------------------------------------------
eval "$("$PY" "$REPO/harness.py" export-env "$CONFIG")"

log() { printf '[run %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
iso() { date +%Y-%m-%dT%H:%M:%S.%3N; }

# --- resumability ---------------------------------------------------------
existing="$(find "$RAW" -maxdepth 1 -name "${WB_POINT_ID}__*.json" -print -quit 2>/dev/null || true)"
if [[ -n "$existing" && $FORCE -eq 0 ]]; then
  log "SKIP ${WB_POINT_ID}: already have $(basename "$existing") (use --force to rerun)"
  exit 0
fi

STAMP="$(date +%Y%m%dT%H%M%S)"
BASE="$RAW/${WB_POINT_ID}__${STAMP}"
POWER_CSV="${BASE}.power.csv"
POWER_JSON="${BASE}.power.json"
SERVER_LOG="${BASE}.server.log"
RESULT_JSON="${BASE}.json"
PROV_JSON="$TMP/${WB_POINT_ID}.provenance.json"
BENCH_JSON="$TMP/${WB_POINT_ID}.bench.json"
SERVER_LOG_LIVE="$TMP/server.log"

# --- server lifecycle -----------------------------------------------------
PID_FILE="$STATE_DIR/server.pid"
FP_FILE="$STATE_DIR/server.fingerprint"

server_healthy() { curl -sf -m 5 "http://127.0.0.1:${WB_PORT}/health" >/dev/null 2>&1; }

stop_server() {
  if [[ -f "$PID_FILE" ]]; then
    local pid; pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      log "stopping server pid $pid"
      kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE" "$FP_FILE"
  fi
  # Anything still holding the port would silently serve the wrong model.
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3
}

start_server() {
  log "starting vLLM: $WB_MODEL"
  log "  flags: $WB_SERVER_ARGS"
  : > "$SERVER_LOG_LIVE"
  # setsid so the whole engine process group can be signalled as one.
  setsid "$VLLM" serve "$WB_MODEL" $WB_SERVER_ARGS >>"$SERVER_LOG_LIVE" 2>&1 &
  local pid=$!
  echo "$pid" > "$PID_FILE"
  echo "$WB_SERVER_FP" > "$FP_FILE"

  local waited=0
  until server_healthy; do
    if ! kill -0 "$pid" 2>/dev/null; then
      log "FATAL: server exited during startup; last 40 lines:"
      tail -40 "$SERVER_LOG_LIVE" >&2 || true
      cp "$SERVER_LOG_LIVE" "$SERVER_LOG" 2>/dev/null || true
      rm -f "$PID_FILE" "$FP_FILE"
      return 1
    fi
    sleep 3
    waited=$((waited + 3))
    if (( waited % 30 == 0 )); then log "  waiting for /health (${waited}s)"; fi
    if (( waited >= WB_LOAD_TIMEOUT_S )); then
      log "FATAL: server did not become healthy in ${WB_LOAD_TIMEOUT_S}s"
      tail -40 "$SERVER_LOG_LIVE" >&2 || true
      cp "$SERVER_LOG_LIVE" "$SERVER_LOG" 2>/dev/null || true
      stop_server
      return 1
    fi
  done
  log "server healthy after ${waited}s"
}

ensure_server() {
  local cur_fp=""
  [[ -f "$FP_FILE" ]] && cur_fp="$(cat "$FP_FILE")"
  if [[ "$cur_fp" == "$WB_SERVER_FP" ]] && server_healthy; then
    log "reusing running server (fingerprint $WB_SERVER_FP)"
    return 0
  fi
  stop_server
  start_server
}

# --- power poller ---------------------------------------------------------
POLLER_PID=""
start_poller() {
  "$PY" "$REPO/power_log.py" poll --out "$POWER_CSV" --interval-ms 500 &
  POLLER_PID=$!
  sleep 2   # let the first samples land before warmup begins
}
stop_poller() {
  if [[ -n "$POLLER_PID" ]] && kill -0 "$POLLER_PID" 2>/dev/null; then
    kill -TERM "$POLLER_PID" 2>/dev/null || true
    wait "$POLLER_PID" 2>/dev/null || true
  fi
  POLLER_PID=""
}

cleanup() {
  local rc=$?
  stop_poller
  if [[ $STOP_SERVER -eq 1 ]]; then stop_server; fi
  exit $rc
}
trap cleanup EXIT INT TERM

# --- load generator -------------------------------------------------------
bench_args() {
  local nprompts="$1" out_json="$2"
  local -a a=(
    bench serve
    --backend vllm
    --model "$WB_MODEL"
    --host 127.0.0.1 --port "$WB_PORT"
    --endpoint /v1/completions
    --dataset-name "$WB_DATASET"
    --num-prompts "$nprompts"
    --seed "$WB_SEED"
    --percentile-metrics ttft,tpot,itl,e2el
    --metric-percentiles 50,90,95,99
    # vLLM's own goodput, in ms; cross-checks the one harness.py computes
    # from the per-request arrays.
    --goodput "ttft:$(awk "BEGIN{printf \"%d\", $WB_SLO_TTFT_S*1000}")"
              "e2el:$(awk "BEGIN{printf \"%d\", $WB_SLO_E2E_S*1000}")"
  )
  if [[ "$WB_DATASET" == "random" ]]; then
    a+=(--random-input-len "$WB_INPUT_LEN" --random-output-len "$WB_OUTPUT_LEN"
        --random-range-ratio 0.0)
  fi
  a+=(--request-rate "$WB_REQUEST_RATE")
  if [[ "$WB_REQUEST_RATE" != "inf" ]]; then a+=(--burstiness "$WB_BURSTINESS"); fi
  [[ -n "$WB_MAX_CONCURRENCY" ]] && a+=(--max-concurrency "$WB_MAX_CONCURRENCY")
  [[ -n "$WB_IGNORE_EOS" ]] && a+=(--ignore-eos)
  if [[ -n "$out_json" ]]; then
    a+=(--save-result --save-detailed
        --result-dir "$(dirname "$out_json")"
        --result-filename "$(basename "$out_json")")
  fi
  # shellcheck disable=SC2086
  [[ -n "$WB_LOAD_EXTRA" ]] && eval "a+=($WB_LOAD_EXTRA)"
  printf '%s\n' "${a[@]}"
}

if [[ $DRY -eq 1 ]]; then
  echo "point_id     : $WB_POINT_ID"
  echo "model        : $WB_MODEL"
  echo "server flags : $WB_SERVER_ARGS"
  echo "warmup       : $WB_WARMUP_PROMPTS prompts @ ${WB_REQUEST_RATE} req/s"
  echo "measurement  : $WB_NUM_PROMPTS prompts @ ${WB_REQUEST_RATE} req/s, ${WB_INPUT_LEN}in/${WB_OUTPUT_LEN}out"
  echo "bench argv   :"; mapfile -t _a < <(bench_args "$WB_NUM_PROMPTS" "$BENCH_JSON"); printf '  %s\n' "${_a[@]}"
  echo "would write  : $RESULT_JSON"
  exit 0
fi

# --- execute --------------------------------------------------------------
STARTED_AT="$(iso)"
log "=== ${WB_POINT_ID} (${WB_EXPERIMENT}) ==="

ensure_server || { log "FATAL: server unavailable"; exit 1; }

"$PY" "$REPO/harness.py" provenance > "$PROV_JSON"

start_poller
log "poller pid $POLLER_PID -> $(basename "$POWER_CSV")"

log "warmup: $WB_WARMUP_PROMPTS prompts @ ${WB_REQUEST_RATE} req/s"
mapfile -t WARM_ARGS < <(bench_args "$WB_WARMUP_PROMPTS" "")
if ! "$VLLM" "${WARM_ARGS[@]}" >"$TMP/warmup.log" 2>&1; then
  log "WARN: warmup pass returned nonzero; last 20 lines:"
  tail -20 "$TMP/warmup.log" >&2 || true
fi
sleep 5   # let queues drain and power settle before the measured window

rm -f "$BENCH_JSON"
WINDOW_START="$(iso)"
log "measuring: $WB_NUM_PROMPTS prompts @ ${WB_REQUEST_RATE} req/s"
mapfile -t RUN_ARGS < <(bench_args "$WB_NUM_PROMPTS" "$BENCH_JSON")
STATUS="ok"
if ! "$VLLM" "${RUN_ARGS[@]}" 2>&1 | tee "$TMP/bench.log"; then
  STATUS="bench_failed"
  log "WARN: load generator returned nonzero"
fi
WINDOW_END="$(iso)"
FINISHED_AT="$WINDOW_END"

stop_poller
tail -n 2000 "$SERVER_LOG_LIVE" > "$SERVER_LOG" 2>/dev/null || true

# --- energy ---------------------------------------------------------------
IDLE_ARGS=()
IDLE_FILE="$REPO/results/idle_baseline.json"
if [[ -f "$IDLE_FILE" ]]; then
  IDLE_W="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['idle_power_w'])" "$IDLE_FILE")"
  IDLE_ARGS=(--idle-baseline-w "$IDLE_W")
  log "idle baseline: ${IDLE_W} W"
else
  log "WARN: no results/idle_baseline.json; incremental energy will be absent"
fi

"$PY" "$REPO/power_log.py" integrate \
  --log "$POWER_CSV" --start "$WINDOW_START" --end "$WINDOW_END" \
  "${IDLE_ARGS[@]}" --out "$POWER_JSON" >/dev/null

# --- assemble -------------------------------------------------------------
if [[ ! -f "$BENCH_JSON" ]]; then
  log "FATAL: load generator produced no result JSON at $BENCH_JSON"
  STATUS="bench_no_output"
  "$PY" - "$RESULT_JSON" "$WB_POINT_ID" "$STATUS" <<'PYEOF'
import json, sys
path, point, status = sys.argv[1], sys.argv[2], sys.argv[3]
json.dump({"schema_version": 1, "point_id": point, "status": status,
           "note": "load generator produced no output; kept as a failure record"},
          open(path, "w"), indent=2)
PYEOF
  exit 1
fi

NOTE_ARGS=()
for n in ${NOTES+"${NOTES[@]}"}; do NOTE_ARGS+=(--note "$n"); done

"$PY" "$REPO/harness.py" assemble \
  --config "$CONFIG" \
  --bench-json "$BENCH_JSON" \
  --power-json "$POWER_JSON" \
  --power-csv "$POWER_CSV" \
  --provenance "$PROV_JSON" \
  --server-log "$SERVER_LOG" \
  --status "$STATUS" \
  --started-at "$STARTED_AT" --finished-at "$FINISHED_AT" \
  --window-start "$WINDOW_START" --window-end "$WINDOW_END" \
  ${NOTE_ARGS+"${NOTE_ARGS[@]}"} \
  --out "$RESULT_JSON"

log "done: $(basename "$RESULT_JSON")"
