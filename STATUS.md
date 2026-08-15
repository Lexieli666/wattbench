# Status and remaining work

Updated 2026-08-14. Everything below is recoverable from the repo alone.

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

## Closed since 2026-08-11

1. **The idle-baseline statistic is decided: subtract the median.** Owner's
   decision, 2026-08-14, applied in one re-analysis pass. Twelve baselines over
   six days all have a median in 18.7–22.2 W while their means range
   18.8–27.9 W; the excursions that move the mean are 1.7–4.0 s bursts recurring
   every ~30–50 s, and the longest excursion-free stretch inside a contaminated
   window is 44–99 s against a 130 s measurement, so "re-measure until clean" is
   a lottery, not a protocol. Effect across 47 runs: median +0.03%, max +2.56%;
   in the published tables only `e1_sweep.md`'s J/tok-net column moves, in the
   third significant figure of two rows. Rationale, evidence and limits are in
   METHODOLOGY §"The idle *mean* is not a robust statistic here".

   The M3 gate moved with it: `(p95 − min)/mean < 0.35` failed four in-use
   baselines once the corpus grew to twelve, because on a bimodal window the p95
   *is* the excursion. It is now `IQR/median < 0.35`, which reads 2.1–15.8% over
   the same twelve. M3 is back to 0 failures.

   Raw J/token, the E3 frontier and the E5 cost model were and remain
   baseline-independent.

2. **The two provisional E2 r16 points are re-measured** (2026-08-13). The 18%
   BF16 outlier does not reproduce: 1605.9 tok/s against the withdrawn 1262.4,
   inside the five-run consensus of 1535–1614. The field that settles it is
   `sw_power_cap_frac` — 93–95% in every healthy saturated BF16 run, 55.6% in
   the outlier, 94.2% in the re-measurement. On clean same-session data the two
   arms are at **+0.1% throughput, i.e. parity**, so int4's saturated advantage
   on this card is 0–7% depending on how load is applied, not the withdrawn
   +21.7%. Full account in `results/tables/e2_r16_anomaly.md`.

## Remaining

1. **M8 / E4 stack comparison.** vLLM vs. llama.cpp server at concurrency 1, 8
   and 32, per plan §4 E4. In progress as of 2026-08-15. SGLang stays optional
   and default-skipped. The caveat belongs in every E4 table and plot: GGUF
   Q4_K_M and AWQ are different quantization formats, so this compares **stacks
   at their native int4**, not identical weights.

2. **Close v0.1**: regenerate from raw, trace every README number to a raw file
   or a dated source, tag.

## Rules that bit hard here — do not relax them

- **Never run a download during a measured run.** Measured: TTFT p95 +819%,
  while throughput and energy look almost normal. Corrupted data that looks
  clean is the failure mode.
- **A fresh idle baseline per session, and subtract its median.** Twelve
  baselines: medians 18.7–22.2 W, means 18.8–27.9 W. The gap is an external host
  consumer, not the card. Two E4 windows sixteen minutes apart came back 7.9%
  and 0.0% contaminated, which is why the statistic has to be robust rather than
  the window clean.
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
