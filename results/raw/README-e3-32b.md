# The 32B rung: six attempts, and why they are all kept

Qwen2.5-32B-Instruct-AWQ is the one E3 rung whose result is a *distribution*
rather than a number, so the raw directory holds more files for it than for any
other rung. This is the key to them.

18.14 GiB of int4 weights load into a 21.6 GiB budget (`gpu_memory_utilization
0.90` of a 24 GB card). What is left has to cover CUDA graphs, activations and
KV cache. vLLM's **CUDA-graph memory estimate is not stable between identical
invocations** — measured at 0.62, 0.86, 1.59 and 7.34 GiB on the same flags —
and that variation, not the config, decides whether a given context loads.

| when | ctx | outcome | graph mem | KV left | artifact |
|---|---|---|---|---|---|
| 10:47 | 4096 | failed | 0.62 GiB | 0.54 GiB | `e3_32b_awq_chat_r4__20260811T104715.server.log` |
| 10:48 | 1024 | failed before loading weights | — | — | `e3_32b_awq_chat_r4_ctx1024__20260811T104844.server.log` |
| 10:59 | 4096 | **served**, aborted by operator | — | — | `e3_32b_awq_chat_r4__20260811T105944.{power,kv}.csv` |
| 11:07 | 4096 | failed | 1.59 GiB | 0.58 GiB | `e3_32b_awq_chat_r4__20260811T110705.json` |
| 11:08 | 1024 | **served**, full 800-prompt run | 0.86 GiB | 0.56 GiB | `e3_32b_awq_chat_r4_ctx1024__20260811T110842.json` |
| 11:34 | 512 | **served** (probe) | — | 4,432 tok | `limits_context_32b_fine__20260811T113344.json` |
| 11:35 | 1024 | failed (probe) | 7.34 GiB | — | same file, `attempts[]` |

Three of these need explicit marking, because none is a clean measurement:

- **10:48, ctx 1024.** Not a KV-capacity result. The config asked for
  `gpu_memory_utilization: 0.95` (22.79 GiB), which exceeds the 22.45 GiB free
  on a machine where the Windows desktop permanently holds ~1.5 GB, so vLLM
  refused at startup *before loading any weights*. That is a statement about
  the flag, not about the card. The config was corrected to 0.90 — which also
  restores the series' fixed-flags rule — and re-run at 11:08.
- **10:59, ctx 4096.** Genuine telemetry, no result JSON: the operator killed
  it mid-benchmark on a too-short timeout. It is kept because it is the only
  evidence that 32B served at 4096 at all, and the evidence is real — the KV
  log records 96% cache use, 8-9 running, 429 waiting, 67 preemptions and
  62,832 tokens generated at a median 372 W. It is **not** a completed
  measurement and must not be quoted as one.
- **11:35, ctx 1024 (probe).** Contradicts the successful 11:08 run at the same
  context. Both are real. This is the variance, not an error in either.

`limits_context_32b__20260811T105306.json` is the first context probe, whose
ladder starts at 4096 and so learned only "not 4096". The `_fine` probe re-ran
it on a 512/1024/2048/4096 ladder.

Reporting a single "longest servable context" for 32B would be false precision.
Only 512 served on every attempt.
