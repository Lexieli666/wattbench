# Methodology

How WattBench measures what it measures, and what it refuses to claim.

The short version: every number in this repository is either **measured on one
RTX 4090 in this machine** and traceable to a timestamped file in
`results/raw/`, or **sourced from a dated public price card** in `pricing.yaml`
and labelled *reported, not measured*. The two are never averaged, blended, or
plotted as if they were the same kind of thing.

---

## 1. What is measured, and on what

One consumer GPU, in a Windows desktop, serving through WSL2. The full verified
inventory — card, driver, CUDA, kernel, Python, vLLM, disk, RAM — is in
[`results/environment.md`](results/environment.md), read off the machine rather
than assumed. Every raw result additionally embeds its own provenance block, so
a result file is interpretable on its own without trusting the docs.

Recorded in **every** run:

| Recorded | Why it matters |
|---|---|
| Model id **and resolved HF commit** | A model id without a commit is not reproducible |
| vLLM, PyTorch, transformers, FlashInfer versions | Serving performance is a stack property, not a model property |
| Driver version, GPU name, VRAM | |
| Power limit observed on every sample | A mid-run change invalidates the point (see §3) |
| SM clock and temperature distribution | Detects throttling that would otherwise look like a slow model |
| Harness git commit, and whether the tree was dirty | A number produced by uncommitted code is flagged as such |
| The exact config, with `auto` values frozen to what was used | |
| Which idle baseline was subtracted | See §4 |

---

## 2. Stability protocol

Consumer cards boost opportunistically; a benchmark that ignores this measures
the weather. Every measured point follows the same protocol:

1. **Fixed serving flags across a series.** Model, quantization, `max_model_len`,
   `gpu_memory_utilization`, `max_num_seqs`, prefix caching and port form a
   *server fingerprint*. Points sharing a fingerprint reuse one vLLM process;
   changing any of them starts a new server — and, by the same logic, makes the
   point non-comparable with points measured under the old flags. Such a change
   mid-series requires rerunning the affected points, never blending.
2. **Warmup at the measurement's own offered rate**, for 30 s of offered load,
   immediately before the measured window. Warming at a different rate would
   leave the card in the wrong thermal and clock state. The warmup is capped at
   600 prompts, because at an offered rate the server cannot sustain, "30 s of
   offered load" is far more than 30 s of work.
3. **A 5 s settle** between warmup and measurement so queues drain.
4. **Each point runs to ≥ 3 minutes or ≥ 200 requests**, whichever comes first —
   a saturated high-rate point satisfies the request count in well under three
   minutes and is a valid point, not a short run.
5. **Poisson arrivals** (`burstiness = 1.0`), fixed seed, fixed request shape
   (`--random-range-ratio 0.0`, so every request has exactly the configured
   shape rather than a distribution around it).
6. **`ignore_eos: true` throughout E0–E3.** Output length becomes an input
   rather than a model behaviour, so throughput and J/token compare across
   models and quantizations instead of across sampling luck.
7. **Prefix caching explicitly disabled** for E0–E3, so results are cache-free
   by construction rather than by whatever the default happens to be.
8. **Repeat the reference configuration three times** and report its
   run-to-run coefficient of variation *before* trusting any other number.
   That is E0, and it sets the resolution limit for the entire project: any
   later difference smaller than the CV is noise and is not claimed as a result.

**Other GPU consumers are closed before measured runs.** The desktop's residual
draw is part of the measured idle baseline and is subtracted; a browser
decoding video mid-run is not, and would corrupt the point.

### Two environment-forced deviations

Both are constant across every run in the project, so no comparison within
WattBench is affected — but both mean these numbers should not be compared
against published vLLM figures from a native-Linux box.

- **`VLLM_USE_V2_MODEL_RUNNER=0`.** vLLM 0.26's default GPU model runner
  allocates UVA host buffers, which WSL2's CUDA passthrough does not provide;
  the engine dies at init. The V1 runner has no such dependency. Every
  throughput number here is a V1-runner number.
- **`VLLM_USE_FLASHINFER_SAMPLER=0`.** FlashInfer JIT-compiles its top-k/top-p
  sampling kernel and needs `nvcc`, which is not installed. vLLM's PyTorch-native
  sampler is used instead — vLLM's own error message recommends exactly this.

### The power limit is disclosed, not controlled

`nvidia-smi -pl` requires elevation from both WSL2 and the Windows side, and
neither is available to the harness. The card therefore runs at its stock
450 W limit, which is also its maximum and its power-on default.

What replaces control is **verification**: `power.limit` is read on every
telemetry sample, the set of observed values is stored in each result, and a
`power_limit_changed_mid_run` flag fires if it ever varies. A point whose limit
moved is invalidated rather than blended.

---

## 3. Energy measurement

### Instrument

`nvidia-smi --query-gpu=power.draw,...` polled at a requested 500 ms.
**The delivered cadence is ~620 ms** under WSL2, because every NVML query pays
passthrough overhead — and it is not even uniform: queries return faster at idle
than under load. Every run records the observed p50/p95/max sample interval so
the cadence is auditable rather than assumed.

This is whole-board draw as NVML reports it. There is no software way to
separate SM from memory and VRM draw on this card, and **host power (CPU, RAM,
PSU losses) is not measured at all**. J/token here means *GPU-rail joules per
token*; a wall-socket figure would be higher.

### Integration

Joules are integrated **trapezoidally against the timestamps the driver
reports**, never as mean power × duration. With a cadence that varies with load,
an unweighted sample mean is not the time-weighted mean, and using it would bias
energy toward whatever the card was doing when NVML happened to answer fastest.

Sample gaps longer than 5 s are excluded from both energy and duration, counted,
and flagged — a stalled poller then inflates neither.

### Two cross-checks, only one of which gates

- **Discretisation bound (gates the run).** Left- and right-Riemann sums are
  computed over the same intervals; the trapezoid is their midpoint, so half
  their spread is a bound on the error from sampling at ~2 Hz rather than
  continuously. It answers the question that actually matters — *is this sample
  rate fast enough to integrate this trace?* Observed values are 0.6–0.9 % on
  real runs; the gate is 2 %.
- **Mean-power × duration (reported, does not gate).** The plan's original
  cross-check. It is kept as an `INFO` line because on a short window containing
  a cold-start ramp it legitimately disagrees by a few percent for the cadence
  reason above — treating that as a failure would have flagged healthy runs and
  taught us to ignore the check.

### The measurement window

Power is polled across warmup *and* measurement, but integrated over **exactly
the measured benchmark window**, bracketed by timestamps written immediately
before and after the load generator runs. A sanity check confirms the power
window covers the benchmark duration.

### Idle baseline — measured per series, never assumed

Idle draw is not a constant of the card. It moves with what the Windows desktop
is doing, with driver state, and with ambient temperature. Assuming one
session's baseline still holds in the next would silently bias every
idle-subtracted J/token, and the bias would be invisible in the output.

So: **a fresh idle baseline is measured at the start of each session or
experiment series**, before any weights are loaded, with the GPU otherwise idle.
It is stored in `results/idle/` as committed raw data, stamped with its series
and time, and **every run records which baseline it subtracted**.

`run.sh` refuses to run when there is no baseline, or when the current one is
older than 12 hours. Refusing beats silently subtracting a stale number. The
idle measurement itself warns if mean GPU utilisation exceeded 25 % while it
ran, since a baseline taken while something else is using the card is worse
than no baseline at all.

Both figures are reported everywhere:

- **raw J/token** — total GPU-rail energy ÷ output tokens. This is what the card
  costs you to run, including the idle floor you pay whether or not you serve.
- **incremental J/token** — with `idle_power_W × window_seconds` subtracted.
  This is the marginal energy of the serving work itself.

Neither is "the" answer; the raw figure drives the cost model, because an owner
pays for the idle floor too.

### Throttle detection

Each run reports the fraction of samples in `sw_power_cap`, `hw_thermal_slowdown`,
`sw_thermal_slowdown` and `hw_slowdown`, plus **clock sag**: the fraction of
*busy* samples (≥ 50 % utilisation) whose SM clock sits below 90 % of the busy
p95 clock. A max/min clock pair hides a card that boosts fine at first and
settles lower as it heats; the sag fraction does not. Runs are **flagged, not
discarded** — a thermally limited run is a fact about this hardware.

---

## 4. Goodput, and why it is computed here rather than taken

Throughput counts requests the server finished. **Goodput counts requests it
finished well enough to be worth anything**: TTFT ≤ 1 s *and* end-to-end ≤ 10 s.

Two details matter:

- **The denominator is requests issued, not requests completed.** A server that
  sheds load is not thereby faster.
- **The end-to-end array is reconstructed, not read.** `vllm bench serve
  --save-detailed` emits per-request `ttfts` and `itls` but no per-request
  end-to-end array, so goodput could not be computed at all from its output
  as-is. Each request's end-to-end latency is `ttft + Σ itl` (both arrays are in
  seconds). Verified against a real run: mean TTFT 0.0274 s + mean Σ ITL
  0.704 s = the 731.5 ms `mean_e2el_ms` vLLM reports. The resulting goodput
  matches vLLM's own independently computed `request_goodput` to seven
  significant figures — the two are computed from different code paths, which is
  what makes the agreement worth stating.

**Max sustainable rate** is the highest offered rate at which achieved
throughput is still within 5 % of offered load (beyond that the queue is
growing) *and* at least half of requests met both SLO clauses.

---

## 5. What gets committed, and what gets dropped

`results/raw/` is **append-only**. Files are timestamped; a completed run is
never overwritten or deleted. Failed and anomalous runs are kept and marked —
an OOM at concurrency 64 is a data point about the card, not garbage. The three
M1 smoke runs are all present, including the first one with its sanity FAIL.

Each point commits four files: the assembled result JSON, the raw power CSV, the
integrated power JSON, and the tail of the server log.

**Two arrays are dropped before commit**, and the drop is recorded inside the
artifact itself rather than only in this document:

- `itls` — one float per generated token per request (millions of values at
  high load). vLLM's own ITL percentiles, computed from the full arrays, are
  kept, and the arrays are used to derive `e2els` before being dropped.
- `generated_texts` — the completions themselves, which with `ignore_eos` are
  not meaningful text anyway.

Per-request `ttfts`, `e2els`, `input_lens`, `output_lens` and `errors` survive
in full, so goodput and the latency distribution can be recomputed from the
committed data.

---

## 6. Sanity checks

Every assembled result is checked before it is trusted. A `FAIL` is printed
loudly and stored in the file:

| Check | What it catches |
|---|---|
| goodput ≤ throughput | Arithmetic or unit errors in the SLO computation |
| p50 ≤ p99 for TTFT, TPOT, ITL, E2E | Percentile mix-ups |
| Energy discretisation bound < 2 % | Sample rate too slow for the trace |
| Power window covers the benchmark | Misaligned measurement window |
| All issued requests completed | Silent request failures |
| ≥ 3 min or ≥ 200 requests | Runs too short to be stable |
| No thermal / sampling flags | Throttling, clock sag, sample gaps, limit changes |

Verification is not the same as retuning. When a result contradicts an
expectation, the contradiction is the finding and is reported as such.

---

## 7. Measured versus sourced — the policy

**Measured** values exist only in `results/raw/`, produced on this machine.

**Sourced** values exist only in `pricing.yaml`: API price cards, GPU rental
rates, hardware prices, the electricity tariff. Each carries a source, a
snapshot date, and a `confidence` of `primary` (read from the vendor's own page)
or `secondary` (from a third-party tracker). Every table and plot that consumes
one labels it *reported, not measured*.

They are never blended. In particular:

- Rental cost lines **do not** add measured electricity, because rental rates
  already include power. The analysis enforces this rather than trusting the
  reader to remember.
- The datacenter reference (InferenceMAX) is cited as prior art. No number from
  it is ever plotted beside a measured WattBench value.
- Where a sourced input is uncertain — the electricity tariff varies by tier and
  zone; the 4090's street price varies about twofold across SKUs — the
  **sensitivity band is carried through the result**, because the conclusion is
  linear in both.

### Cost model

```
energy $/1M output tokens = (J_per_token x 1e6 / 3.6e6) x $/kWh
owned   $/1M output tokens = card_price / (3 yr x 8766 h) / (tok_per_s x 3600 / 1e6)
                             + energy $/1M
rental  $/1M output tokens = $/hr / (tok_per_s x 3600 / 1e6)      [power included]
break-even tokens/day      = amortisation_per_day x 1e6 / (api $/1M - energy $/1M)
```

Two honesty constraints on those formulas:

- The amortisation line assumes the card is **busy at the quoted throughput
  every hour of its three-year life** — the most generous possible assumption
  for owning. At 10 % duty cycle the amortised figure is ten times higher, and
  the README says so next to the number.
- API prices are compared at an **effective output price** that folds in the
  input charge at the benchmarked input:output ratio. Self-hosting pays for
  prefill too; comparing against an API's output price alone would flatter the
  API.
- Amortisation covers the **card only**. The rest of the host is excluded, and
  that exclusion is stated wherever the owned line appears.

---

## 8. What this project does not claim

- **No capability equivalence.** A locally served Qwen2.5-7B is not a substitute
  for a frontier API model. Frontier price rows appear to show the ceiling of
  the market, not an available trade, and are marked as not weight-class
  comparable. The only capability evidence produced here is the E2 GSM8K
  exact-match guard, and it licenses no claim beyond that task.
- **No datacenter comparison.** Single consumer card, cited references only.
- **No measured API latency.** Without keys, no hosted endpoint was timed. Every
  measured latency in this repo is self-hosted, and no measured-vs-reported
  latency comparison is made anywhere.
- **No wall-socket energy.** GPU rail only (§3).
- **Not run means not run.** Anything unmeasured prints as `not run`, never as
  an estimate dressed as a measurement.
