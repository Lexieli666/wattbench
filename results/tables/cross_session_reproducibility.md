# Cross-session reproducibility

`e1_chat_7b_bf16_r4` against `e0_ref_7b_bf16_chat_r4_a`: byte-identical configurations measured about an hour apart, in separate sessions, against separately measured idle baselines. No download was running for either.

A metric counts as moved if it exceeds **3x the E0 run-to-run standard deviation** for that metric *and* differs by more than 1% — E0 is tight enough that a fraction of a percent can be many sigma without mattering operationally.

| Metric | E0 session | E1 session | change | within-session CV | sigma | beyond bar |
|---|---|---|---|---|---|---|
| output throughput (tok/s) | 506.5 | 506.5 | -0.00% | 0.01% | 0.2 | no |
| TTFT p50 (ms) | 87.72 | 88.06 | +0.39% | 0.54% | 0.7 | no |
| TTFT p95 (ms) | 155.2 | 158.3 | +2.00% | 0.48% | 4.2 | **yes** |
| ITL p50 (ms) | 16.73 | 16.78 | +0.31% | 0.23% | 1.3 | no |
| E2E p95 (ms) | 3075 | 3103 | +0.91% | 0.41% | 2.2 | no |
| mean GPU power (W) | 344.4 | 343.7 | -0.19% | 0.25% | 0.8 | no |
| J per output token | 0.7015 | 0.7021 | +0.09% | 0.16% | 0.5 | no |

**Reading:** throughput, power, energy per token and ITL reproduce across sessions essentially exactly. Tail latency does not — TTFT p95 moved 2.0%, 4.2x the within-session CV. So cross-sweep tail-latency comparisons use a ~2% floor, and any control must run in the same session as its treatment.
