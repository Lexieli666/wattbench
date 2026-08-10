#!/usr/bin/env bash
# E2: quantization ablation, BF16 vs int4, plus the quality guard.
#
#   ./run_e2.sh
#
# Each arm's load points run first, then the GSM8K guard runs against the same
# still-running server -- so speed and quality are measured on the identical
# process, not on two separate loads of the same checkpoint.
#
# The GPTQ arm is a single point whose only job is to settle the int4 format
# question plan §3 leaves open ("AWQ or GPTQ, whichever vLLM serves faster for
# this family; note which"). Its quality is still guarded, because a format that
# is faster and worse is not a win.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${WATTBENCH_VENV:-$HOME/wattbench-venv}"
PY="$VENV/bin/python"
SERIES="${1:-E2}"

echo "[e2] idle baseline for series '$SERIES' (GPU must be quiet now)"
bash "$REPO/baseline.sh" "$SERIES" || exit 1

run_arm() {
  local label="$1" model="$2"; shift 2
  echo
  echo "=============================================================="
  echo "[e2] arm '$label' -> $model"
  echo "=============================================================="
  local ok=1
  for cfg in "$@"; do
    echo "[e2] --- $cfg"
    bash "$REPO/run.sh" --force "$cfg" || { echo "[e2] !! point failed: $cfg"; ok=0; }
  done

  # The server from the last point is still up; guard it before moving on.
  if curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "[e2] GSM8K guard for '$label'"
    "$PY" "$REPO/gsm8k_guard.py" --label "$label" --model "$model" \
      || echo "[e2] !! guard failed for $label"
  else
    echo "[e2] !! no healthy server after '$label' points; guard skipped (recorded as not run)"
    ok=0
  fi
  return $((1 - ok))
}

run_arm bf16 Qwen/Qwen2.5-7B-Instruct \
  "$REPO/configs/e2/bf16/r2.yaml" "$REPO/configs/e2/bf16/r8.yaml" "$REPO/configs/e2/bf16/r16.yaml"

run_arm awq Qwen/Qwen2.5-7B-Instruct-AWQ \
  "$REPO/configs/e2/awq/r2.yaml" "$REPO/configs/e2/awq/r8.yaml" "$REPO/configs/e2/awq/r16.yaml"

run_arm gptq Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
  "$REPO/configs/e2/gptq/r8.yaml"

echo
echo "[e2] longest servable context per format (each attempt costs a model load)"
bash "$REPO/probe_limits.sh" context Qwen/Qwen2.5-7B-Instruct bf16 || true
bash "$REPO/probe_limits.sh" context Qwen/Qwen2.5-7B-Instruct-AWQ awq || true

pkill -f "vllm serve" 2>/dev/null || true
echo "[e2] done"
