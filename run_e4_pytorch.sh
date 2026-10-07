#!/usr/bin/env bash
# E4, third arm: HF transformers + plain PyTorch (model.generate) at
# concurrency 1, 8 and 32, plus a torch.compile control at 8, then the GSM8K
# guard at n=200 against the same server.
#
#   ./run_e4_pytorch.sh                 # all four points + guard
#   ./run_e4_pytorch.sh --skip-guard
#   ./run_e4_pytorch.sh --skip-compile
#
# Run it in the same session as a fresh idle baseline (./baseline.sh E4-pytorch)
# and, ideally, right after re-running one vLLM point as a drift check -- the
# E4 arms were measured 2026-08-15 and anything measured now carries
# cross-session drift (~2% on tail latency, see METHODOLOGY) on top of the
# within-session bar. The record carries the session either way.
#
# The caveat that belongs on every number this produces: this arm serves BF16,
# because plain transformers has no int4 path that is still "just PyTorch",
# while the other two E4 arms serve their native int4. Like E4 itself, it
# compares stacks at their native format. The like-for-like control for the
# format is the vLLM BF16 reference server (E1/E2's BF16 arm): same weights,
# different stack.
#
# Order: the three uncompiled points share one server fingerprint and so one
# model load; the compiled control is a different fingerprint and loads again.
# The guard runs against whichever server is up last, which is why the
# compiled point runs before the guard only when --skip-compile is NOT given
# -- the guard then measures the compiled server, and says so in its label.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${WATTBENCH_VENV:-$HOME/wattbench-venv}"
PY="$VENV/bin/python"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

say() { printf '[e4-pytorch %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

SKIP_GUARD=0
SKIP_COMPILE=0
for a in "$@"; do
  case "$a" in
    --skip-guard) SKIP_GUARD=1 ;;
    --skip-compile) SKIP_COMPILE=1 ;;
    *) say "unknown flag: $a"; exit 1 ;;
  esac
done

[[ -x "$PY" ]] || { say "FATAL: no venv python at $PY"; exit 1; }
"$PY" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)" \
  || { say "FATAL: torch/transformers missing from $VENV"; exit 1; }

# Weights must already be on disk: a download inside a measured window is the
# most destructive mistake available here (TTFT p95 +819%, measured).
say "checking weights are local before starting"
SNAP_GLOB="$HF_HOME/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/config.json"
# shellcheck disable=SC2086
if ! compgen -G "$SNAP_GLOB" >/dev/null; then
  say "FATAL: Qwen/Qwen2.5-7B-Instruct not in the HF cache. Fetch BEFORE running:"
  say "  HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen2.5-7B-Instruct"
  exit 1
fi

POINTS=(
  configs/e4/pytorch_c1.yaml
  configs/e4/pytorch_c8.yaml
  configs/e4/pytorch_c32.yaml
)
GUARD_LABEL="hf_bf16_n200"
if [[ $SKIP_COMPILE -eq 0 ]]; then
  # The guard runs against the last server left up, so put the uncompiled
  # server last: the quality number should describe the plain path, and the
  # compiled control is a speed control, not a second arm.
  POINTS=(configs/e4/pytorch_c8_compile.yaml "${POINTS[@]}")
fi

FORCE="${WATTBENCH_E4_FORCE:-}"
FAILED=()
for i in "${!POINTS[@]}"; do
  cfg="${POINTS[$i]}"
  extra=()
  [[ -n "$FORCE" ]] && extra=(--force)
  # Stop the compiled server before the plain one loads, so the two never
  # hold VRAM at once. The last plain server stays up for the guard.
  [[ "$cfg" == *compile* ]] && extra+=(--stop-server)
  if [[ $SKIP_GUARD -eq 1 && "$i" == $(( ${#POINTS[@]} - 1 )) ]]; then
    extra+=(--stop-server)
  fi
  say "=== $cfg"
  if ! "$REPO/run.sh" "${extra[@]}" "$cfg"; then
    say "point FAILED: $cfg (kept as a failure record)"
    FAILED+=("$cfg")
  fi
done

if [[ $SKIP_GUARD -eq 0 ]]; then
  if curl -sf -m 5 "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
    say "=== GSM8K guard, n=200, label $GUARD_LABEL"
    "$PY" "$REPO/gsm8k_guard.py" --label "$GUARD_LABEL" --n 200 --experiment E4-guard \
      --model Qwen/Qwen2.5-7B-Instruct \
      --note "served by pytorch_server.py (HF transformers + model.generate), BF16, SDPA, uncompiled" \
      || FAILED+=("gsm8k_guard")
  else
    say "no server up for the guard (every pytorch point failed?); skipping it"
  fi
  # run.sh's stop_server, inlined: the guard was the last user of this server.
  pkill -f "pytorch_server.py --model" 2>/dev/null || true
  rm -f "$REPO/results/.state/server.pid" "$REPO/results/.state/server.fingerprint"
fi

say "done. ${#FAILED[@]} point(s) failed."
for f in ${FAILED+"${FAILED[@]}"}; do say "  $f"; done
say "next: $PY ./analyze.py stacks --experiment E4 && $PY ./analyze.py memory"
