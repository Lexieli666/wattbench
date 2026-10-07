# Config schema

One YAML per experiment point. `run.sh <config>` turns it into exactly one raw
result in `results/raw/`. Configs may inherit with `extends: <relative path>`;
the child is deep-merged over the parent.

```yaml
experiment: E1                  # experiment group, recorded in the result
point_id: e1_chat_bf16_r4       # unique; names the raw file. Defaults to filename.
description: free text

server:                         # identity of the served endpoint
  stack: vllm                   # vllm | llamacpp (E4) | pytorch (E4 third arm). Absent means vllm.
  model: Qwen/Qwen2.5-7B-Instruct
  revision: null                # HF commit; resolved and recorded either way
  quantization: null            # null | awq | awq_marlin | gptq_marlin | fp8
  dtype: bfloat16               # ignored when quantization is set
  max_model_len: 4096
  gpu_memory_utilization: 0.90
  max_num_seqs: 256
  enable_prefix_caching: false  # off for E0-E3 so numbers are cache-free
  port: 8000
  extra_args: []                # verbatim extra flags for the server binary
  load_timeout_s: 900

  # llamacpp only; ignored by the vLLM path
  gguf_repo: Qwen/Qwen2.5-7B-Instruct-GGUF
  gguf_file: qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf  # first shard
  tokenizer: Qwen/Qwen2.5-7B-Instruct   # a GGUF path is not an HF repo
  served_model_name: qwen2.5-7b-instruct-q4_k_m   # --alias, and what the client asks for
  n_gpu_layers: 99              # 99 == every layer on the GPU
  parallel: 8                   # server slots; must cover the offered concurrency
  ctx_size: null                # TOTAL KV context; null => max_model_len * parallel
  cont_batching: true
  flash_attn: auto

  # pytorch only (HF transformers + model.generate, pytorch_server.py); ignored
  # by the other two. dtype, max_model_len, revision, tokenizer and
  # served_model_name above apply to it too.
  max_batch_size: 32            # rows per static batch; the analogue of max_num_seqs
  batch_wait_ms: 0              # hold a forming batch for late arrivals; 0 = none
  attn_implementation: sdpa     # sdpa | eager | flash_attention_2 | null (checkpoint default)
  compile: false                # static KV cache + generate()'s decode step compiled (mode=compile_mode)
  compile_mode: reduce-overhead
  profile_batch_size: null      # batch for the post-window torch.profiler pass; null => max_concurrency, else 1
  profile_input_len: null       # null => load.input_len
  profile_output_len: null      # null => load.output_len

load:
  dataset: random               # random | sharegpt | sonnet
  input_len: 512                # prompt tokens (random dataset)
  output_len: 128               # generated tokens
  request_rate: 4.0             # req/s; "inf" sends everything at t=0
  burstiness: 1.0               # 1.0 == Poisson arrivals
  num_prompts: auto             # auto = max(min_requests, rate * min_duration_s)
  max_concurrency: null         # null = unbounded
  seed: 0
  ignore_eos: true              # forces exactly output_len tokens per request
  temperature: 0.0              # null = server default (per-checkpoint!); pin it
  extra_args: []

protocol:                       # plan §2 stability protocol
  warmup_s: 30
  warmup_prompts: 32            # floor; actual = max(this, rate * warmup_s)
  min_duration_s: 180
  min_requests: 200

slo:                            # goodput definition
  ttft_p95_s: 1.0
  e2e_p95_s: 10.0
```

## Server reuse

Points sharing a *server fingerprint* (stack, model, revision, quantization,
dtype, max_model_len, gpu_memory_utilization, max_num_seqs, prefix caching,
extra args, port, plus the llama.cpp fields and the pytorch fields other than
the profile shape) reuse one running server process.
Change any of those and the server restarts — which is also why changing one
mid-series breaks comparability and requires rerunning the affected points.

This is why E4's arms cost different amounts of wall clock. Concurrency is a
*client-side* limit for vLLM and for the pytorch arm, so their three points
each share one fingerprint and one model load; llama.cpp sizes its slots and
KV context at startup, so each of its points needs its own. The pytorch
`compile: true` control is a different fingerprint and loads again.

## Compiled pytorch points are fixed-concurrency only

`compile: true` uses transformers' static KV cache and recompiles for every new
(batch size, cache length) pair. At a fixed `max_concurrency` with identical
request shapes the warmup absorbs that — size `protocol.warmup_prompts` up for
it, as `e4/pytorch_c8_compile.yaml` does. Under Poisson arrivals the batch
size varies request to request and the server would recompile inside the
measured window, so do not write a compiled point with a finite
`request_rate`.

## Before running any point

`run.sh` refuses to start without a fresh idle-power baseline for the current
session (see `results/idle/README.md`). Measure one with `./baseline.sh <series>`,
or let `./sweep.sh --series <name>` do it before it loads any weights.

## Conventions

- `ignore_eos: true` everywhere in E0–E3. Output length is then an input, not a
  model behaviour, so throughput and J/token compare across models and
  quantizations rather than across sampling luck.
- `random-range-ratio` is fixed at 0.0 by `run.sh`, so every request has
  exactly the configured shape.
- Two shapes are used throughout: **chat** 512 in / 128 out and **RAG** 2048 in
  / 256 out (plan §4 E1).
