# Status and remaining work

Updated 2026-08-11. Everything below is recoverable from the repo alone.

## Done (measured, committed, analysed)

| Milestone | State |
|---|---|
| M0 environment | `results/environment.md` |
| M1 harness | `run.sh`, `power_log.py`, `kv_log.py`, `harness.py`, `analyze.py` |
| M2 E0 variance | 3 repeats + 3 saturated repeats |
| M3 energy validation | `validate_energy.py`, `results/tables/m3_energy_validation.md` |
| M4 E1 load sweep | 16 points, both shapes |
| M5 E2 ablation | 7 rate points, 3 GSM8K guards, 8-point concurrency ladder, 2 context probes |
| M7 write-up | README, METHODOLOGY, SETUP-WSL2, pricing.yaml |

Regenerate every table and plot from raw data:

```bash
./analyze.py all
```

## Remaining

1. **E3 model-size frontier (M6).** Chained: `run_e3.sh` fires once the 32B AWQ
   download finishes. It measures 1.5B/3B/7B/14B/32B int4 at 4 req/s, retries
   32B at reduced context if it will not serve at 4096, and probes longest
   servable context for 14B/32B. Rungs whose weights are missing are skipped and
   reported, never silently omitted. Afterwards, replace the "Model-size
   frontier — Not run" section of README.md with `results/tables/e3_frontier.md`.

2. **Re-measure two provisional points.** `configs/e2/bf16/r16.yaml` and
   `configs/e2/awq/r16.yaml`. E2's BF16 r16 is an 18% outlier against four other
   measurements of the same config — see `results/tables/e2_r16_anomaly.md`. Run
   with the GPU otherwise idle:

   ```bash
   ./baseline.sh E2-recheck
   ./run.sh --force configs/e2/bf16/r16.yaml
   ./run.sh --force configs/e2/awq/r16.yaml
   ```

   Until then the r16 row of the ablation table is provisional and the
   defensible saturated throughput figure is the concurrency ladder's c256
   point (+6.8%), not the withdrawn +21.7%.

## Rules that bit hard here — do not relax them

- **Never run a download during a measured run.** Measured: TTFT p95 +819%,
  while throughput and energy look almost normal. Corrupted data that looks
  clean is the failure mode.
- **A fresh idle baseline per session.** Observed drift 19.0–25.7 W across
  sessions; one baseline was contaminated (10.8% GPU util while "idle").
- **Never reuse a non-idle vLLM server.** A server carrying another run's
  requests measured 33 ms ITL at 192 W where a fresh one measured 17 ms at
  312 W. `run.sh` now gates on quiescence.
- **E0's variance bar is regime-specific.** Unsaturated: 0.16% energy, 0.48%
  TTFT p95. Saturated: 1.09% energy, 5.83% TTFT p95. Cross-session tail latency
  drifts ~2%. Do not borrow one regime's bar for another.
- **Compare a new point against an existing measurement of the same config.**
  That reflex caught three defects nothing else did.
