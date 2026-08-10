#!/usr/bin/env bash
# E3: model-size frontier at int4, fixed load.
#
#   ./run_e3.sh
#
# Every rung is the same precision, the same serving flags and the same offered
# load, so the only variable is model size. A rung whose weights are missing is
# skipped and reported as "not run" -- never silently omitted, because an
# absent row in a frontier table reads as "does not exist" rather than "was not
# measured".
#
# The 32B rung is expected to be the interesting one: at ~19GB of int4 weights
# there is little left for KV cache, so it may fail to serve at the ladder's
# shared 4096-token context. If it does, the reduced-context config runs
# instead and the gap between them is the KV squeeze the plan asks to document.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${WATTBENCH_VENV:-$HOME/wattbench-venv}"
PY="$VENV/bin/python"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

say() { printf '[e3 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

have_weights() {
  # The HF cache stores blobs under blobs/ by hash and exposes them as symlinks
  # named *.safetensors under snapshots/. `find -size` on the symlink measures
  # the link, not the target, so this needs -L -- without it every model looks
  # absent, including ones that are fully downloaded.
  local repo="$1"
  local dir="$HF_HOME/hub/models--${repo//\//--}"
  [[ -d "$dir" ]] || return 1
  find -L "$dir" -name "*.safetensors" -size +100M 2>/dev/null | grep -q . || return 1
  # A .incomplete blob means a download is still in flight or was interrupted.
  find "$dir" -name "*.incomplete" 2>/dev/null | grep -q . && return 1
  return 0
}

say "idle baseline for series E3 (GPU must be quiet)"
bash "$REPO/baseline.sh" E3 || exit 1

RUNGS=(
  "1p5b:Qwen/Qwen2.5-1.5B-Instruct-AWQ"
  "3b:Qwen/Qwen2.5-3B-Instruct-AWQ"
  "7b:Qwen/Qwen2.5-7B-Instruct-AWQ"
  "14b:Qwen/Qwen2.5-14B-Instruct-AWQ"
  "32b:Qwen/Qwen2.5-32B-Instruct-AWQ"
)

SKIPPED=()
for entry in "${RUNGS[@]}"; do
  slug="${entry%%:*}"; model="${entry##*:}"
  if ! have_weights "$model"; then
    say "SKIP rung $slug: weights for $model are absent or incomplete"
    SKIPPED+=("$slug ($model)")
    continue
  fi
  say "--- rung $slug"
  if ! bash "$REPO/run.sh" --force "$REPO/configs/e3/${slug}.yaml"; then
    say "!! rung $slug failed to serve at the shared context"
    if [[ "$slug" == "32b" ]]; then
      say "   retrying 32B at a reduced context -- the gap is the KV squeeze"
      bash "$REPO/run.sh" --force "$REPO/configs/e3/32b_short_ctx.yaml" \
        || say "   !! 32B failed at the reduced context too; recorded as unservable"
    fi
  fi
done

say "longest servable context for the largest rungs"
for entry in "14b:Qwen/Qwen2.5-14B-Instruct-AWQ" "32b:Qwen/Qwen2.5-32B-Instruct-AWQ"; do
  slug="${entry%%:*}"; model="${entry##*:}"
  have_weights "$model" && bash "$REPO/probe_limits.sh" context "$model" "$slug" || true
done

pkill -f "vllm serve" 2>/dev/null || true
say "done"
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
  say "rungs not run (weights absent): ${SKIPPED[*]}"
fi
