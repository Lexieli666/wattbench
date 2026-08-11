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
| M6 E3 frontier | 1.5B/3B/7B/14B at 4096 + 32B at 1024, 3 recorded 32B serve failures, 14B and 32B context probes, 7B control |
| M7 write-up | README, METHODOLOGY, SETUP-WSL2, pricing.yaml |

Regenerate every table and plot from raw data:

```bash
~/wattbench-venv/bin/python ./analyze.py all
```

## Remaining

1. **Decide the idle-baseline statistic.** Open question, not a defect to fix
   blindly — see METHODOLOGY §"the idle *mean* is not a robust statistic". Idle
   draw here is bimodal: every one of the ten committed baselines has a median
   of 18.7–22.2 W, but the mean ranges 18.8–27.9 W depending on how many ~62 W
   excursions from an intermittent host consumer land in the 130 s window.
   Re-measuring does not fix it — the E3 baseline was re-measured and came back
   *worse* (27.878 vs 24.558 W) with a *lower* median. Options: subtract the
   median rather than the mean, or require a window with zero samples above
   40 W. Either changes every `*_incremental` figure across E0–E5 and the M3
   energy validation, so it needs a deliberate decision and a re-analysis pass,
   not a quiet edit.

   This affects **only** incremental J/token. Raw J/token, the E3 frontier and
   the E5 cost model are baseline-independent. The E3 runs themselves are clean:
   the 7B rung re-measured an hour later under a different baseline came back
   within **0.13% on power and 0.02% on J/token**
   (`e3_7b_awq_chat_r4_recheck`), against E0's 0.16% unsaturated bar.

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
  That reflex caught three defects nothing else did — now four: it is what
  established the E3 sweep was clean despite a visibly suspect idle baseline.
- **A failure that writes no record is a silent omission.** `run.sh` used to
  exit bare when a server would not start, so the one rung the card *cannot*
  serve was the one rung missing from the frontier table — reading as "does not
  exist" rather than "was measured and failed". `harness.py assemble` now
  accepts a run with no bench output and writes a `server_failed_to_start`
  record carrying the root cause lifted from the server's own log.
- **`util_gpu_pct` is not a contamination proxy on this machine.** It reads
  19–41% at a genuinely quiescent ~20 W with clocks at their 210 MHz floor,
  because desktop compositing registers as utilisation while costing almost no
  power. The 25% guard in `power_log.py` cannot catch what it was meant to
  catch. Judge idle by power, not utilisation.
- **Never let a config differ from its series in two variables at once.**
  `configs/e3/32b_short_ctx.yaml` combined a reduced context with
  `gpu_memory_utilization: 0.95`, which is unsatisfiable here (~1.5 GB is
  permanently held by the desktop), so it died before loading weights and was
  recorded as "32B unservable at reduced context" — a statement about the flag,
  not the card. At 0.90 it serves.
- **`analyze.py all` needs the venv python.** System `python3` lacks
  matplotlib; the tables are written before it reaches the plots, so a partial
  run looks like a success. Use `~/wattbench-venv/bin/python ./analyze.py all`.
