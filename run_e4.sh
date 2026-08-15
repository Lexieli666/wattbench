#!/usr/bin/env bash
# E4: serving-stack comparison, vLLM vs llama.cpp, at concurrency 1, 8 and 32.
#
#   ./run_e4.sh
#
# The caveat that belongs on every number this produces: **GGUF Q4_K_M and AWQ
# int4 are different quantization formats**. This compares two stacks each at
# its own native int4, not one set of weights on two servers. Same model
# family, same size, same traffic shape, same tokenizer, same load generator --
# different weight encodings, because that is what each stack actually serves.
#
# Order matters for wall-clock: concurrency is client-side for vLLM, so all
# three vLLM points share one server and one model load. llama.cpp sizes its
# slots and KV context at startup, so each of its points needs its own load.
#
# SGLang is the plan's optional third stack and is deliberately not run: it
# would be a third quantization format and a third set of serving defaults for
# a comparison that is already caveat-heavy.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMACPP_BIN="${WATTBENCH_LLAMACPP_BIN:-$HOME/src/llama.cpp/build/bin/llama-server}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

say() { printf '[e4 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

[[ -x "$LLAMACPP_BIN" ]] || {
  say "FATAL: no llama-server at $LLAMACPP_BIN"
  say "       build it first; see SETUP-WSL2.md 'Building llama.cpp'"
  exit 1
}

# Both arms need weights already on disk. A download inside a measured window
# is the single most destructive mistake available here (measured: TTFT p95
# +819% while throughput and energy still looked normal).
say "checking weights are local before starting"
GGUF_GLOB="$HF_HOME/hub/models--Qwen--Qwen2.5-7B-Instruct-GGUF/snapshots/*/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
# shellcheck disable=SC2086
if ! compgen -G "$GGUF_GLOB" >/dev/null; then
  say "FATAL: GGUF weights missing. Fetch them BEFORE running (never during):"
  say "  HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen2.5-7B-Instruct-GGUF --include '*q4_k_m*'"
  exit 1
fi

POINTS=(
  configs/e4/vllm_c1.yaml
  configs/e4/vllm_c8.yaml
  configs/e4/vllm_c32.yaml
  configs/e4/llamacpp_c1.yaml
  configs/e4/llamacpp_c8.yaml
  configs/e4/llamacpp_c32.yaml
)

FAILED=()
for i in "${!POINTS[@]}"; do
  cfg="${POINTS[$i]}"
  # Stop the server after the last vLLM point and after the last point overall,
  # so the two stacks never hold VRAM at the same time.
  extra=()
  [[ "$cfg" == *vllm_c32* || "$i" == $(( ${#POINTS[@]} - 1 )) ]] && extra=(--stop-server)
  say "=== $cfg"
  if ! "$REPO/run.sh" "${extra[@]}" "$cfg"; then
    say "point FAILED: $cfg (kept as a failure record)"
    FAILED+=("$cfg")
  fi
done

say "done. ${#FAILED[@]} point(s) failed."
for f in ${FAILED+"${FAILED[@]}"}; do say "  $f"; done
say "next: ~/wattbench-venv/bin/python ./analyze.py stacks --experiment E4"
