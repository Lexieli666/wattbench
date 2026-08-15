# Config schema

One YAML per experiment point. `run.sh <config>` turns it into exactly one raw
result in `results/raw/`. Configs may inherit with `extends: <relative path>`;
the child is deep-merged over the parent.

```yaml
experiment: E1                  # experiment group, recorded in the result
point_id: e1_chat_bf16_r4       # unique; names the raw file. Defaults to filename.
description: free text

server:                         # identity of the served endpoint
  stack: vllm                   # vllm | llamacpp (E4). Absent means vllm.
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
extra args, port, plus the llama.cpp fields) reuse one running server process.
Change any of those and the server restarts — which is also why changing one
mid-series breaks comparability and requires rerunning the affected points.

This is why E4's two arms cost different amounts of wall clock. Concurrency is
a *client-side* limit for vLLM, so all three vLLM points share one fingerprint
and one model load; llama.cpp sizes its slots and KV context at startup, so
each of its points needs its own.

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
