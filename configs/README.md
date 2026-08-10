# Config schema

One YAML per experiment point. `run.sh <config>` turns it into exactly one raw
result in `results/raw/`. Configs may inherit with `extends: <relative path>`;
the child is deep-merged over the parent.

```yaml
experiment: E1                  # experiment group, recorded in the result
point_id: e1_chat_bf16_r4       # unique; names the raw file. Defaults to filename.
description: free text

server:                         # identity of the served endpoint
  model: Qwen/Qwen2.5-7B-Instruct
  revision: null                # HF commit; resolved and recorded either way
  quantization: null            # null | awq | awq_marlin | gptq_marlin | fp8
  dtype: bfloat16               # ignored when quantization is set
  max_model_len: 4096
  gpu_memory_utilization: 0.90
  max_num_seqs: 256
  enable_prefix_caching: false  # off for E0-E3 so numbers are cache-free
  port: 8000
  extra_args: []                # verbatim extra flags for `vllm serve`
  load_timeout_s: 900

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

Points sharing a *server fingerprint* (model, revision, quantization, dtype,
max_model_len, gpu_memory_utilization, max_num_seqs, prefix caching, extra
args, port) reuse one running vLLM process. Change any of those and the server
restarts — which is also why changing one mid-series breaks comparability and
requires rerunning the affected points.

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
