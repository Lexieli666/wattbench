# Idle power baselines

One file per measurement, named `idle__<series>__<timestamp>.json`, with the
raw samples alongside it. Committed as raw data: these are measurements, and
every incremental J/token in the project is computed against one of them.

## Why there is more than one

Idle GPU draw is not a constant of the card. It moves with what the Windows
desktop is doing, with driver state, and with ambient temperature. Two
baselines measured on this machine ~2 hours apart:

| Series | Measured | Idle draw |
|---|---|---|
| `m1-smoke` | 2026-08-09 22:55 | 25.72 W |
| `E0` | 2026-08-10 00:30 | 20.57 W |

That is a **5.1 W drift, about 20%**, with no configuration change between them.
Carrying the first into the second series would have over-subtracted 5 W from
every point — silently, and invisibly in the output.

So a fresh baseline is measured at the start of each session or experiment
series, before any weights load, with the GPU otherwise idle. `run.sh` refuses
to run when there is no baseline or the current one is over 12 hours old,
rather than quietly subtracting a stale number.

## Reading one

- `idle_power_w` — the value subtracted, time-weighted over the kept window.
- `series` / `measured_at` — what it belongs to. Every raw result echoes these
  back in `energy.idle_baseline_source`, so an incremental figure can always be
  traced to the baseline behind it.
- `settle_s_discarded` — leading seconds dropped before averaging.
- `util_gpu_pct_mean` — sanity: this should be low single digits. The
  measurement warns above 25%, because a baseline taken while something else is
  using the card is worse than no baseline at all.

## Measuring one

```bash
./baseline.sh E1-chat            # 150s, 20s settle
./baseline.sh E3 --duration 240  # longer if the machine is noisy
```

`sweep.sh --series NAME` does this automatically before it loads any weights.
