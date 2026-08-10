#!/usr/bin/env bash
# Run a list of configs through run.sh, in order, unattended.
#
#   ./sweep.sh configs/e1/chat/*.yaml
#   ./sweep.sh --force configs/e0/*.yaml
#
# Safe to interrupt and relaunch: run.sh skips any point whose raw result
# already exists, so a sweep resumes where it stopped. A point that fails does
# not abort the sweep -- it is recorded and the next point runs, because an OOM
# at high concurrency is a data point about the card, not a reason to lose the
# rest of the night.
#
# The server is left running between points so that points sharing a model and
# serving flags do not pay the load cost repeatedly; it is stopped at the end.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORCE_ARGS=()
CONFIGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE_ARGS+=(--force); shift ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) CONFIGS+=("$1"); shift ;;
  esac
done

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "usage: ./sweep.sh [--force] CONFIG.yaml [CONFIG.yaml ...]" >&2
  exit 1
fi

LOG_DIR="$REPO/results/sweep_logs"
mkdir -p "$LOG_DIR"
SWEEP_LOG="$LOG_DIR/sweep_$(date +%Y%m%dT%H%M%S).log"

n=0; ok=0; failed=0
FAILED_POINTS=()

say() { printf '%s\n' "$*" | tee -a "$SWEEP_LOG"; }

say "sweep started $(date -Is) with ${#CONFIGS[@]} points"
say "log: $SWEEP_LOG"
started=$(date +%s)

for cfg in "${CONFIGS[@]}"; do
  n=$((n + 1))
  say ""
  say "----- [$n/${#CONFIGS[@]}] $cfg -----"
  if bash "$REPO/run.sh" ${FORCE_ARGS+"${FORCE_ARGS[@]}"} "$cfg" 2>&1 | tee -a "$SWEEP_LOG"; then
    ok=$((ok + 1))
  else
    failed=$((failed + 1))
    FAILED_POINTS+=("$cfg")
    say "!! point failed: $cfg (continuing)"
  fi
done

# Free the GPU once the whole list is done.
if [[ -f "$REPO/results/.state/server.pid" ]]; then
  pid="$(cat "$REPO/results/.state/server.pid")"
  kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  sleep 5
  kill -KILL "-$pid" 2>/dev/null || true
  rm -f "$REPO/results/.state/server.pid" "$REPO/results/.state/server.fingerprint"
fi
pkill -f "vllm serve" 2>/dev/null || true

elapsed=$(( $(date +%s) - started ))
say ""
say "sweep finished $(date -Is): $ok ok, $failed failed, $(( elapsed / 60 )) min elapsed"
for f in ${FAILED_POINTS+"${FAILED_POINTS[@]}"}; do say "  failed: $f"; done
