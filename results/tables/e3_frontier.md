## E3 — model-size frontier

| model                     | precision | ctx  | status                 | out tok/s | TTFT p50 ms | TTFT p95 ms | mean W | J/tok |
|---------------------------|-----------|------|------------------------|-----------|-------------|-------------|--------|-------|
| Qwen2.5-1.5B-Instruct-AWQ | awq       | 4096 | served                 | 511.1     | 29.66       | 42.93       | 182.9  | 0.371 |
| Qwen2.5-3B-Instruct-AWQ   | awq       | 4096 | served                 | 510.6     | 35.5        | 58.88       | 227    | 0.461 |
| Qwen2.5-7B-Instruct-AWQ   | awq       | 4096 | served                 | 509.8     | 65.38       | 111.3       | 319.7  | 0.65  |
| Qwen2.5-14B-Instruct-AWQ  | awq       | 4096 | served                 | 507.7     | 140         | 368.4       | 369.2  | 0.752 |
| Qwen2.5-32B-Instruct-AWQ  | awq       | 4096 | server_failed_to_start | --        | --          | --          | --     | --    |
| Qwen2.5-32B-Instruct-AWQ  | awq       | 1024 | served                 | 109.9     | 366,370     | 693,006     | 370.7  | 3.4   |

Rows marked with a non-`served` status are configurations that failed to serve. They are kept: what the card *cannot* do is part of the frontier. `--` means the run was attempted and produced no measurement, which is not the same as not run.

Why each non-`served` rung failed, from its server log:

- **Qwen2.5-32B-Instruct-AWQ** (`e3_32b_awq_chat_r4__20260811T110705.json`): ValueError: To serve at least one request with the model's max seq len (4096), (1.0 GiB KV cache is needed, which is larger than the available KV cache memory (0.58 GiB). Based on the available memory, the estimated maximum model length is 2368. Try increasing `gpu_memory_utilization` (which also controls CPU memory on the CPU backend) or decreasing `max_model_len` when initializing the engine. See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ for more details.
  - Model loading took 18.14 GiB memory and 11.465340 seconds
  - Estimated CUDA graph memory: 1.59 GiB total
  - Available KV cache memory: 0.58 GiB

