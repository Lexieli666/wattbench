## E0 stability baseline — 3 repeats of the reference configuration

Runs: e0_ref_7b_bf16_chat_r4_a__20260810T005712.json, e0_ref_7b_bf16_chat_r4_b__20260810T010159.json, e0_ref_7b_bf16_chat_r4_c__20260810T010616.json

| Metric                     | mean   | min    | max   | CV    |
|----------------------------|--------|--------|-------|-------|
| Output throughput (tok/s)  | 506.5  | 506.5  | 506.6 | 0.01% |
| Request throughput (req/s) | 3.957  | 3.957  | 3.957 | 0.01% |
| TTFT p50 (ms)              | 88.24  | 87.72  | 88.66 | 0.54% |
| TTFT p95 (ms)              | 156    | 155.2  | 156.7 | 0.48% |
| ITL p50 (ms)               | 16.76  | 16.73  | 16.8  | 0.23% |
| E2E p95 (ms)               | 3086   | 3075   | 3100  | 0.41% |
| Mean GPU power (W)         | 343.4  | 342.8  | 344.4 | 0.25% |
| Energy per run (J)         | 71864  | 71772  | 71989 | 0.16% |
| J per output token         | 0.7018 | 0.7009 | 0.703 | 0.16% |

### What this bounds
- **Throughput CV is 0.01%, and that number is not a variance bar.** Every repeat replays the same seeded Poisson schedule, and the server kept up with it (achieved ≥ 95% of offered), so the run ends when the last request was *scheduled*, not when the card finished working. Throughput is therefore pinned by the load generator here. It becomes a real measure of the card only at saturation, where the queue grows and duration is set by service rate.
- **Energy per token: CV 0.16%** — differences smaller than ~0.3% between later points are inside noise.
- **Mean power: CV 0.25%** — differences smaller than ~0.5% between later points are inside noise.
- **TTFT p95: CV 0.48%** — differences smaller than ~1.0% between later points are inside noise.
- **E2E p95: CV 0.41%** — differences smaller than ~0.8% between later points are inside noise.

**Verdict: tight.** The binding constraint is TTFT p95 at 0.48% CV, so **this project does not claim any difference smaller than about 1.0%** on that metric. Energy comparisons are resolvable to ~0.3%.

No run tripped a thermal, clock-sag, sample-gap or power-limit flag.

