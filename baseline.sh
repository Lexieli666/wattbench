#!/usr/bin/env bash
# Measure the idle-power baseline for an experiment series.
#
#   ./baseline.sh E0
#   ./baseline.sh E1-chat --duration 180
#
# Idle draw is not a property of the card alone -- it moves with what the
# Windows desktop is doing, driver state, and ambient temperature. Assuming one
# session's baseline still holds in the next would silently bias every
# idle-subtracted J/token in the series, so a fresh baseline is measured at the
# start of each session or series and every run records which one it used.
#
# Writes results/idle/idle__<series>__<stamp>.json (committed as raw data) and
# points results/.state/current_baseline at it. run.sh refuses to run against a
# baseline older than WATTBENCH_BASELINE_MAX_AGE_H hours (default 12).
#
# The number subtracted is the window's MEDIAN, not its mean: idle draw here is
# bimodal, so the mean is decided by how many brief bursts from an external host
# consumer land in the window (decided 2026-08-14, see METHODOLOGY). The mean is
# kept in the file as idle_power_w_mean, and the excursion fraction is recorded.
#
# Nothing else may be using the GPU while this runs. The measurement warns if
# mean utilisation exceeds 25%, but on this machine utilisation is not a usable
# contamination proxy -- judge quiescence by power.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${WATTBENCH_VENV:-$HOME/wattbench-venv}"
PY="$VENV/bin/python"
IDLE_DIR="$REPO/results/idle"
STATE_DIR="$REPO/results/.state"

SERIES=""
DURATION=150
SETTLE=20

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration) DURATION="$2"; shift 2 ;;
    --settle) SETTLE="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) SERIES="$1"; shift ;;
  esac
done

if [[ -z "$SERIES" ]]; then
  echo "usage: ./baseline.sh <series> [--duration S] [--settle S]" >&2
  echo "  e.g. ./baseline.sh E0" >&2
  exit 1
fi

mkdir -p "$IDLE_DIR" "$STATE_DIR"
STAMP="$(date +%Y%m%dT%H%M%S)"
OUT="$IDLE_DIR/idle__${SERIES}__${STAMP}.json"

echo "[baseline] series '$SERIES': measuring ${DURATION}s of idle draw."
echo "[baseline] close other GPU consumers now if you have not already."

"$PY" "$REPO/power_log.py" idle \
  --out "$OUT" --duration "$DURATION" --settle "$SETTLE" --series "$SERIES"

echo "$OUT" > "$STATE_DIR/current_baseline"
W="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['idle_power_w'])" "$OUT")"
echo "[baseline] series '$SERIES' baseline = ${W} W (median) -> $(basename "$OUT")"
echo "[baseline] run.sh will subtract this until a newer baseline is measured."
