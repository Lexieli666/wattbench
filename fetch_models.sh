#!/usr/bin/env bash
# Pre-fetch the model ladder into the ext4 HF cache, in the order the
# experiments need it, so a sweep never stalls mid-run on a download.
#
#   ./fetch_models.sh            # everything, in priority order
#   ./fetch_models.sh e0 e1      # only what those experiments need
#
# Measured on this connection at ~2.9 MB/s, so the full ladder is a multi-hour
# unattended job. Downloads resume after an interrupt; rerunning is cheap.
#
# Transfer backend, measured here on the same 3.1 GB file, 60 s each:
#   Xet (hub default)          76 MB   1.27 MB/s
#   HF_HUB_DISABLE_XET=1      171 MB   2.85 MB/s   <- used
#   HF_XET_HIGH_PERFORMANCE=1   1 MB   0.02 MB/s
# hf_transfer is not a lever anymore: huggingface_hub deprecated
# HF_HUB_ENABLE_HF_TRANSFER and ignores it.

set -uo pipefail

VENV="${WATTBENCH_VENV:-$HOME/wattbench-venv}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1

case "$HF_HOME" in
  /mnt/*) echo "FATAL: HF_HOME=$HF_HOME is on a Windows mount." >&2; exit 1 ;;
esac

# group:repo — ordered by which experiment blocks on it first
MODELS=(
  "smoke:Qwen/Qwen2.5-1.5B-Instruct"
  "e0e1:Qwen/Qwen2.5-7B-Instruct"
  "e2:Qwen/Qwen2.5-7B-Instruct-AWQ"
  "e2:Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
  "e3:Qwen/Qwen2.5-14B-Instruct-AWQ"
  "e3:Qwen/Qwen2.5-3B-Instruct-AWQ"
  "e3:Qwen/Qwen2.5-1.5B-Instruct-AWQ"
  "e3:Qwen/Qwen2.5-32B-Instruct-AWQ"
)

want=("$@")
match() {
  [[ ${#want[@]} -eq 0 ]] && return 0
  local g="$1"
  for w in "${want[@]}"; do [[ "$g" == *"$w"* ]] && return 0; done
  return 1
}

for entry in "${MODELS[@]}"; do
  group="${entry%%:*}"; repo="${entry#*:}"
  match "$group" || continue
  echo "=== [$(date +%H:%M:%S)] $repo ($group)"
  start=$(date +%s)
  # No --exclude: the Qwen2.5 repos ship only safetensors plus tokenizer
  # files, and passing extra patterns makes the CLI read them as filenames and
  # silently download nothing.
  if "$VENV/bin/hf" download "$repo" \
        >/dev/null 2>"/tmp/wattbench/fetch_${repo//\//_}.err"; then
    echo "    ok in $(( $(date +%s) - start ))s"
  else
    echo "    FAILED (see /tmp/wattbench/fetch_${repo//\//_}.err)"
    tail -3 "/tmp/wattbench/fetch_${repo//\//_}.err" || true
  fi
done

echo "=== cache now: $(du -sh "$HF_HOME/hub" | cut -f1)"
