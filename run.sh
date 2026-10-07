#!/usr/bin/env bash
# WattBench run driver: one config YAML in, one raw result JSON out.
#
#   ./run.sh configs/e1/chat_r2.yaml
#   ./run.sh --force configs/e0/ref_a.yaml
#   ./run.sh --stop-server configs/e1/chat_r32.yaml
#
# Resumable: a point whose raw result already exists is skipped unless --force,
# so an interrupted sweep can simply be relaunched. The server is kept alive
# between points that share a server fingerprint (stack + model + serving
# flags), because reloading weights per point would cost more wall-clock than
# the measurements themselves.
#
# Three stacks share this driver, selected by `server.stack` in the config:
# vllm (default), llamacpp (E4), and pytorch -- HF transformers + model.generate
# behind pytorch_server.py. Everything from the power poller onward is the same
# code for all three; only start_server and the quiescence probe know which one
# is running.
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
# E4's second stack. Built from source (see SETUP-WSL2.md); the exact commit is
# recorded in every llama.cpp run's provenance.
LLAMACPP_BIN="${WATTBENCH_LLAMACPP_BIN:-$HOME/src/llama.cpp/build/bin/llama-server}"
LLAMACPP_SRC="${WATTBENCH_LLAMACPP_SRC:-$HOME/src/llama.cpp}"
# The third stack. Runs on the same venv python: it needs only torch and
# transformers, both of which vLLM already pulls in, and recording the same
# torch version for both arms is the point.
PYTORCH_SERVER="$REPO/pytorch_server.py"
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

if [[ "${WB_STACK:-vllm}" == "llamacpp" ]]; then
  [[ -x "$LLAMACPP_BIN" ]] || {
    echo "FATAL: no llama-server at $LLAMACPP_BIN (set WATTBENCH_LLAMACPP_BIN)" >&2
    exit 1; }
  export WB_LLAMACPP_BIN="$LLAMACPP_BIN" WB_LLAMACPP_SRC="$LLAMACPP_SRC"
elif [[ "${WB_STACK:-vllm}" == "pytorch" ]]; then
  "$PY" -c "import torch, transformers" 2>/dev/null || {
    echo "FATAL: the pytorch stack needs torch and transformers in $VENV" >&2
    exit 1; }
fi

log() { printf '[run %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
iso() { date +%Y-%m-%dT%H:%M:%S.%3N; }

# --- idle baseline: fresh, per series -------------------------------------
# Idle draw drifts between sessions, so a stale baseline would quietly bias
# every idle-subtracted J/token. Refuse to run against one rather than silently
# subtracting a number measured under different conditions.
BASELINE_MAX_AGE_H="${WATTBENCH_BASELINE_MAX_AGE_H:-12}"
BASELINE_PTR="$STATE_DIR/current_baseline"
WB_IDLE_FILE=""
if [[ -f "$BASELINE_PTR" ]]; then
  candidate="$(cat "$BASELINE_PTR")"
  if [[ -f "$candidate" ]]; then
    age_h=$(( ( $(date +%s) - $(stat -c %Y "$candidate") ) / 3600 ))
    if (( age_h < BASELINE_MAX_AGE_H )); then
      WB_IDLE_FILE="$candidate"
    else
      log "FATAL: idle baseline $(basename "$candidate") is ${age_h}h old (limit ${BASELINE_MAX_AGE_H}h)."
      log "       Measure a fresh one for this series:  ./baseline.sh <series>"
      exit 1
    fi
  fi
fi
if [[ -z "$WB_IDLE_FILE" ]]; then
  log "FATAL: no idle baseline for this session."
  log "       Measure one before the series:  ./baseline.sh <series>"
  log "       (override for a throwaway run with WATTBENCH_ALLOW_NO_BASELINE=1)"
  [[ "${WATTBENCH_ALLOW_NO_BASELINE:-}" == "1" ]] || exit 1
  log "WARN: proceeding without a baseline; incremental energy will be absent"
fi

# --- resumability ---------------------------------------------------------
existing="$(find "$RAW" -maxdepth 1 -name "${WB_POINT_ID}__*.json" -print -quit 2>/dev/null || true)"
if [[ -n "$existing" && $FORCE -eq 0 ]]; then
  log "SKIP ${WB_POINT_ID}: already have $(basename "$existing") (use --force to rerun)"
  exit 0
fi

STAMP="$(date +%Y%m%dT%H%M%S)"
BASE="$RAW/${WB_POINT_ID}__${STAMP}"
POWER_CSV="${BASE}.power.csv"
KV_CSV="${BASE}.kv.csv"
KV_JSON="${BASE}.kv.json"
POWER_JSON="${BASE}.power.json"
PROFILE_JSON="${BASE}.profile.json"
MEMORY_JSON="$TMP/${WB_POINT_ID}.memory.json"
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
  # Anything still holding the port would silently serve the wrong model --
  # and in E4 the two stacks share a port, so this covers both by name.
  pkill -f "vllm serve" 2>/dev/null || true
  # Matched on the binary's path, not its name: a bare "llama-server" pattern
  # also matches any shell whose command line merely mentions it, including
  # this one.
  pkill -f "bin/llama-server" 2>/dev/null || true
  pkill -f "pytorch_server.py --model" 2>/dev/null || true
  sleep 3
}

start_server() {
  if [[ "${WB_STACK:-vllm}" == "llamacpp" ]]; then
    log "starting llama.cpp: $WB_MODEL"
    log "  flags: $WB_SERVER_ARGS"
    : > "$SERVER_LOG_LIVE"
    setsid "$LLAMACPP_BIN" $WB_SERVER_ARGS >>"$SERVER_LOG_LIVE" 2>&1 &
  elif [[ "${WB_STACK:-vllm}" == "pytorch" ]]; then
    log "starting pytorch (HF transformers + model.generate): $WB_MODEL"
    log "  flags: $WB_SERVER_ARGS"
    : > "$SERVER_LOG_LIVE"
    # -u: unbuffered, so the server log is complete if the process is killed.
    setsid "$PY" -u "$PYTORCH_SERVER" $WB_SERVER_ARGS >>"$SERVER_LOG_LIVE" 2>&1 &
  else
    log "starting vLLM: $WB_MODEL"
    log "  flags: $WB_SERVER_ARGS"
    : > "$SERVER_LOG_LIVE"
    # setsid so the whole engine process group can be signalled as one.
    setsid "$VLLM" serve "$WB_MODEL" $WB_SERVER_ARGS >>"$SERVER_LOG_LIVE" 2>&1 &
  fi
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

# A server is only safe to reuse if it is actually idle. A previous run that
# was starved, aborted or killed can leave requests in flight, and vLLM will
# happily keep decoding them underneath the next measurement -- which shows up
# as a larger batch, roughly halved per-token speed and much lower power, with
# no error anywhere. Observed for real: a reused server measured 33.25ms ITL at
# 192W where a fresh one measured 16.90ms at 312W for the identical config.
server_quiescent() {
  local metrics running waiting
  metrics="$(curl -sf -m 5 "http://127.0.0.1:${WB_PORT}/metrics" 2>/dev/null)" || return 1
  if [[ "${WB_STACK:-vllm}" == "llamacpp" ]]; then
    running="$(awk '/^llamacpp:requests_processing[{ ]/ {print $2; exit}' <<<"$metrics")"
    waiting="$(awk '/^llamacpp:requests_deferred[{ ]/ {print $2; exit}' <<<"$metrics")"
  elif [[ "${WB_STACK:-vllm}" == "pytorch" ]]; then
    running="$(awk '/^pytorch:num_requests_running[{ ]/ {print $2; exit}' <<<"$metrics")"
    waiting="$(awk '/^pytorch:num_requests_waiting[{ ]/ {print $2; exit}' <<<"$metrics")"
  else
    running="$(awk '/^vllm:num_requests_running/ {print $2; exit}' <<<"$metrics")"
    waiting="$(awk '/^vllm:num_requests_waiting[{ ]/ {print $2; exit}' <<<"$metrics")"
  fi
  [[ -z "$running" || -z "$waiting" ]] && return 1
  awk -v r="$running" -v w="$waiting" 'BEGIN{exit !(r+0==0 && w+0==0)}'
}

ensure_server() {
  local cur_fp=""
  [[ -f "$FP_FILE" ]] && cur_fp="$(cat "$FP_FILE")"
  if [[ "$cur_fp" == "$WB_SERVER_FP" ]] && server_healthy; then
    local waited=0
    until server_quiescent; do
      if (( waited >= 60 )); then
        log "server has outstanding work after ${waited}s; restarting it rather"
        log "than measuring on top of another run's leftovers"
        stop_server
        start_server
        return $?
      fi
      sleep 5; waited=$((waited + 5))
    done
    [[ $waited -gt 0 ]] && log "server drained after ${waited}s"
    log "reusing running server (fingerprint $WB_SERVER_FP, quiescent)"
    return 0
  fi
  stop_server
  start_server
}

# --- power poller ---------------------------------------------------------
POLLER_PID=""
KV_PID=""
start_poller() {
  "$PY" "$REPO/power_log.py" poll --out "$POWER_CSV" --interval-ms 500 &
  POLLER_PID=$!
  # Server-side metrics: KV-cache utilisation and queue depth are only visible
  # here, and plan §4 asks for both. One small scrape every 2s, identical for
  # every point, so it cannot bias a comparison between them.
  "$PY" "$REPO/kv_log.py" poll --port "$WB_PORT" --out "$KV_CSV" --interval 2 &
  KV_PID=$!
  sleep 2   # let the first samples land before warmup begins
}
stop_poller() {
  for pid_var in POLLER_PID KV_PID; do
    pid="${!pid_var}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
    printf -v "$pid_var" '%s' ""
  done
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
    # vLLM's native backend for the vLLM arm; the OpenAI-compatible path for
    # anything else. Both send to /v1/completions, and both build their
    # prompts from the same tokenizer, so the two arms of E4 receive the same
    # token sequences.
    --backend "${WB_BENCH_BACKEND:-vllm}"
    --model "$WB_SERVED_MODEL_NAME"
    --tokenizer "$WB_TOKENIZER"
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
  # vLLM 0.26 no longer forces greedy decoding; unset means the server's own
  # per-checkpoint default, which would differ across the E3 ladder.
  [[ -n "$WB_TEMPERATURE" ]] && a+=(--temperature "$WB_TEMPERATURE")
  # Progress-bar rendering only; keeps committed sweep logs readable.
  a+=(--disable-tqdm)
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
  echo "stack        : ${WB_STACK:-vllm}"
  echo "model        : $WB_MODEL"
  echo "server flags : $WB_SERVER_ARGS"
  echo "warmup       : $WB_WARMUP_PROMPTS prompts @ ${WB_REQUEST_RATE} req/s"
  echo "measurement  : $WB_NUM_PROMPTS prompts @ ${WB_REQUEST_RATE} req/s, ${WB_INPUT_LEN}in/${WB_OUTPUT_LEN}out"
  echo "bench argv   :"; mapfile -t _a < <(bench_args "$WB_NUM_PROMPTS" "$BENCH_JSON"); printf '  %s\n' "${_a[@]}"
  if [[ "${WB_STACK:-vllm}" == "pytorch" ]]; then
    echo "profile pass : batch ${WB_PROFILE_BATCH_SIZE} x ${WB_PROFILE_INPUT_LEN}in/${WB_PROFILE_OUTPUT_LEN}out, after the window"
  fi
  echo "would write  : $RESULT_JSON"
  exit 0
fi

# --- execute --------------------------------------------------------------
STARTED_AT="$(iso)"
log "=== ${WB_POINT_ID} (${WB_EXPERIMENT}) ==="

if ! ensure_server; then
  log "FATAL: server unavailable"
  # Record it. A configuration this card cannot serve is a finding -- it is
  # half of what a frontier is -- and exiting bare here used to leave nothing
  # behind but a server log, so the rung vanished from the table entirely.
  "$PY" "$REPO/harness.py" assemble \
    --config "$CONFIG" \
    --server-log "$SERVER_LOG" \
    --status "server_failed_to_start" \
    --started-at "$STARTED_AT" --finished-at "$(iso)" \
    --out "$RESULT_JSON" || log "WARN: could not write failure record"
  exit 1
fi

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

# --- pytorch arm: allocator snapshot, then the profiler pass -----------------
# Both happen AFTER the window has closed and the pollers have stopped, so
# neither the profiler's overhead nor its synthetic batch touches a measured
# number. The allocator snapshot is read first, before the profile pass can
# move the high-water mark.
PROFILE_ARGS=()
if [[ "${WB_STACK:-vllm}" == "pytorch" ]]; then
  if curl -sf -m 10 "http://127.0.0.1:${WB_PORT}/wattbench/memory" -o "$MEMORY_JSON"; then
    PROFILE_ARGS+=(--memory-json "$MEMORY_JSON")
  else
    log "WARN: could not read /wattbench/memory"
  fi
  log "profile pass: batch ${WB_PROFILE_BATCH_SIZE} x ${WB_PROFILE_INPUT_LEN}in/${WB_PROFILE_OUTPUT_LEN}out under torch.profiler"
  if curl -sf -m 900 -X POST "http://127.0.0.1:${WB_PORT}/wattbench/profile" \
       -H 'Content-Type: application/json' \
       -d "{\"batch_size\": ${WB_PROFILE_BATCH_SIZE}, \"input_len\": ${WB_PROFILE_INPUT_LEN}, \"output_len\": ${WB_PROFILE_OUTPUT_LEN}}" \
       -o "$PROFILE_JSON"; then
    PROFILE_ARGS+=(--profile-json "$PROFILE_JSON")
  else
    log "WARN: profile pass failed; the record will say so"
    rm -f "$PROFILE_JSON"
  fi
fi

# --- energy ---------------------------------------------------------------
IDLE_ARGS=()
IDLE_NOTE_ARGS=()
if [[ -n "${WB_IDLE_FILE:-}" && -f "$WB_IDLE_FILE" ]]; then
  # The median of the idle window, for baselines measured from 2026-08-14; for
  # older ones idle_power_w is the mean and analyze.py recomputes at read time.
  IDLE_W="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['idle_power_w'])" "$WB_IDLE_FILE")"
  IDLE_SERIES="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('series') or '?')" "$WB_IDLE_FILE")"
  IDLE_AT="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('measured_at') or '?')" "$WB_IDLE_FILE")"
  IDLE_ARGS=(--idle-baseline-w "$IDLE_W")
  IDLE_NOTE_ARGS=(--idle-baseline-file "$(basename "$WB_IDLE_FILE")"
                  --idle-baseline-series "$IDLE_SERIES"
                  --idle-baseline-measured-at "$IDLE_AT")
  log "idle baseline: ${IDLE_W} W (series '${IDLE_SERIES}', measured ${IDLE_AT})"
fi

"$PY" "$REPO/power_log.py" integrate \
  --log "$POWER_CSV" --start "$WINDOW_START" --end "$WINDOW_END" \
  "${IDLE_ARGS[@]}" --out "$POWER_JSON" >/dev/null

if [[ -f "$KV_CSV" ]]; then
  "$PY" "$REPO/kv_log.py" summarize --log "$KV_CSV" \
    --start "$WINDOW_START" --end "$WINDOW_END" --out "$KV_JSON" >/dev/null || true
fi

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
  --kv-json "$KV_JSON" \
  --kv-csv "$KV_CSV" \
  --provenance "$PROV_JSON" \
  --server-log "$SERVER_LOG" \
  --status "$STATUS" \
  --started-at "$STARTED_AT" --finished-at "$FINISHED_AT" \
  --window-start "$WINDOW_START" --window-end "$WINDOW_END" \
  ${IDLE_NOTE_ARGS+"${IDLE_NOTE_ARGS[@]}"} \
  ${NOTE_ARGS+"${NOTE_ARGS[@]}"} \
  ${PROFILE_ARGS+"${PROFILE_ARGS[@]}"} \
  --out "$RESULT_JSON"

log "done: $(basename "$RESULT_JSON")"
