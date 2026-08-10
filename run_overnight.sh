#!/usr/bin/env bash
# Post-E1 sequence, unattended.
#
#   ./run_overnight.sh
#
# E3 needs ~26GB of weights that are not downloaded yet, and downloading them
# takes hours. Running that alongside measured benchmarks would save most of
# those hours -- but only if it demonstrably does not perturb a measurement.
#
# So: start the download, measure one point that has already been measured on a
# quiet machine, and compare. If every metric lands inside E0's resolution
# limits, keep downloading through E2. If not, pause downloads until the GPU
# work is done. The decision is recorded either way.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${WATTBENCH_VENV:-$HOME/wattbench-venv}"
PY="$VENV/bin/python"
LOG="$REPO/results/sweep_logs/overnight_$(date +%Y%m%dT%H%M%S).log"
mkdir -p "$(dirname "$LOG")"

say() { printf '[overnight %s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

say "starting E3 weight download in the background"
nohup bash "$REPO/fetch_models.sh" e3 > /tmp/wattbench/fetch_e3.log 2>&1 &
DL_PID=$!
say "download pid $DL_PID"
sleep 30   # let it reach steady-state throughput before measuring against it

say "measuring interference control point (E1 chat r8, download active)"
bash "$REPO/baseline.sh" E1-dltest >>"$LOG" 2>&1
bash "$REPO/run.sh" --force "$REPO/configs/e1/chat/r8_dltest.yaml" >>"$LOG" 2>&1 \
  || say "!! control point failed"

DECISION="$("$PY" "$REPO/compare_points.py" \
    --baseline e1_chat_7b_bf16_r8 --candidate e1_chat_7b_bf16_r8_dltest \
    --out "$REPO/results/tables/download_interference.md" 2>&1 | tail -1)"
say "interference verdict: $DECISION"

if [[ "$DECISION" == CLEAN* ]]; then
  say "downloads may continue during E2"
else
  say "pausing downloads for the duration of E2"
  kill "$DL_PID" 2>/dev/null || true
  pkill -f "hf download" 2>/dev/null || true
  sleep 3
fi

say "starting E2"
bash "$REPO/run_e2.sh" E2 >>"$LOG" 2>&1
say "E2 finished"

if [[ "$DECISION" != CLEAN* ]]; then
  say "resuming E3 downloads now that the GPU work is done"
  nohup bash "$REPO/fetch_models.sh" e3 > /tmp/wattbench/fetch_e3.log 2>&1 &
  say "download pid $!"
fi

say "done. E3 runs once its weights finish downloading."
