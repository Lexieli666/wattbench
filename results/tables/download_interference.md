# Download-interference control

`e1_chat_7b_bf16_r8_dltest` measured while a weight download saturated the link, against `e1_chat_7b_bf16_r8_quiet` measured on a quiet machine. Both are the same configuration and the same offered load.

A metric counts as moved if it exceeds **3x the E0 run-to-run standard deviation** for that metric *and* differs by more than 1% — E0 is tight enough that a fraction of a percent can be many sigma without mattering operationally.

| Metric | quiet | with download | change | E0 CV | sigma | moved |
|---|---|---|---|---|---|---|
| output throughput (tok/s) | 1001 | 1010 | +0.91% | 0.01% | 88.7 | no |
| TTFT p50 (ms) | 166.3 | 395.1 | +137.55% | 0.54% | 253.4 | **yes** |
| TTFT p95 (ms) | 470.8 | 4327 | +819.11% | 0.48% | 1700.8 | **yes** |
| ITL p50 (ms) | 20.1 | 36.24 | +80.28% | 0.23% | 342.0 | **yes** |
| E2E p95 (ms) | 1.059e+04 | 2.682e+04 | +153.19% | 0.41% | 369.3 | **yes** |
| mean GPU power (W) | 359.3 | 336.2 | -6.44% | 0.25% | 26.2 | **yes** |
| J per output token | 0.3748 | 0.3491 | -6.87% | 0.16% | 43.8 | **yes** |

**Verdict: PERTURBED.** Moved: TTFT p50 (ms), TTFT p95 (ms), ITL p50 (ms), E2E p95 (ms), mean GPU power (W), J per output token. Downloads are paused during measured runs, and the affected points are not compared across that boundary.
