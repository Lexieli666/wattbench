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
9. **Pin the sampling temperature.** vLLM 0.26's benchmark client no longer
   forces greedy decoding, so an unset temperature means each checkpoint's own
   `generation_config.json` default applies — which would silently vary sampling
   across the E3 model ladder.

### Why the throughput CV is not the variance bar

E0's output-throughput CV is **0.01%**, and that number is not used as a
resolution limit, because it does not measure the card.

Below saturation, with a fixed seed, every repeat replays an identical Poisson
arrival schedule and the server keeps up with it. The run therefore ends when
the last request was *scheduled*, not when the card finished working: the three
E0 durations were 202.2316 s, 202.2287 s and 202.2019 s, and throughput is just
`num_prompts / duration`. That CV measures the load generator's determinism.

Throughput only becomes a real measurement of the hardware **at saturation**,
where the queue grows and duration is set by service rate rather than by the
schedule — which is exactly the regime E1's high-rate points probe.
`analyze.py` detects the schedule-pinned condition and says so instead of
quoting the flattering figure.

### The resolution limits this project actually uses

Measured on the pinned configuration — the same one E1–E3 run under:

| Metric | CV | Smallest difference claimed |
|---|---|---|
| Energy per output token | 0.16% | 0.3% |
| Mean GPU power | 0.25% | 0.5% |
| E2E p95 | 0.41% | 0.8% |
| **TTFT p95** | **0.48%** | **1.0%** (binding) |

No E0 run tripped a thermal, clock-sag, sample-gap or power-limit flag; the card
held 2655 MHz at 77 °C, 343 W.

### Within-session and between-session variance are different numbers

The table above is **within-session** variance: three repeats back-to-back on
one server load, sharing thermal state, memory layout and CUDA-graph capture.
That is the right bar for comparing points measured inside a single sweep.

It is *not* the right bar for comparing across sweeps. E1's chat 4 req/s point
is byte-identical in configuration to E0's reference, and ran about an hour
later in a separate session with a separately measured idle baseline. The
result:

| Metric | E0 session | E1 session | change | within-session CV |
|---|---|---|---|---|
| Output throughput | 506.5 tok/s | 506.5 tok/s | −0.00% | 0.01% |
| Mean GPU power | 344.4 W | 343.7 W | −0.19% | 0.25% |
| J per output token | 0.7015 | 0.7021 | +0.09% | 0.16% |
| ITL p50 | 16.73 ms | 16.78 ms | +0.31% | 0.23% |
| TTFT p50 | 87.72 ms | 88.06 ms | +0.39% | 0.54% |
| E2E p95 | 3075 ms | 3103 ms | +0.91% | 0.41% |
| **TTFT p95** | **155.2 ms** | **158.3 ms** | **+2.00%** | **0.48%** |

Throughput, power and energy reproduce across sessions essentially exactly —
those are the project's headline quantities, and they are solid. **Tail latency
is not**: TTFT p95 moved 2.0%, which is 4.2× the within-session CV.

Two consequences, both applied:

1. **Cross-sweep comparisons of tail latency use a ~2% floor, not 0.48%.**
   Anything smaller is session drift. Throughput, power and J/token keep the
   tighter within-session bars, since they demonstrably reproduce.
2. **Controls live in the same session as their treatment.** The
   download-interference test originally compared a download-active point
   against E1's own quiet r8 — different sessions, so it would have charged the
   download for ordinary drift. It now runs a quiet control and a
   download-active treatment back-to-back on one server, where the download is
   the only thing that differs.

This is why the reference configuration is repeated rather than run once: a
single run cannot tell you which of its digits are real.

### A background download destroys latency measurements

Weight downloads were kept off the machine during measured runs on the
assumption that ~3 MB/s of network traffic was negligible against a GPU-bound
workload. That assumption was wrong, and measuring it was worth the twelve
minutes it cost.

Control and treatment ran back-to-back on one server — same session, same
configuration, same offered load — with the download as the only difference:

| Metric | quiet | with download | change |
|---|---|---|---|
| Output throughput | 1001 tok/s | 1010 tok/s | +0.9% |
| **TTFT p50** | 166.3 ms | 395.1 ms | **+138%** |
| **TTFT p95** | 470.8 ms | 4327 ms | **+819%** |
| **ITL p50** | 20.1 ms | 36.2 ms | **+80%** |
| **E2E p95** | 10.6 s | 26.8 s | **+153%** |
| Mean GPU power | 359.3 W | 336.2 W | −6.4% |
| J per output token | 0.375 | 0.349 | −6.9% |

The mechanism is client-side starvation, not server contention. `vllm bench
serve` is an asyncio client on the same host; a saturated link delays both
request dispatch and the reading of streamed tokens, which is why *inter-token*
latency rises 80%. The GPU is left underfed — hence lower power and lower
energy per token — while the fixed arrival schedule still drives throughput to
roughly its usual value.

That combination is the dangerous part: **throughput and energy look almost
normal while latency is off by a factor of nine.** A benchmark that reported
only tokens/s and watts would have published this as a clean run.

Downloads are therefore paused for the duration of every measured series, and
no point is compared across that boundary.

### The temperature pin was measured, not assumed

`ignore_eos` fixes the token count, so pinning temperature "should not" change
throughput or energy. It does, slightly: a control run of the E0 configuration
with `temperature: 0` drew **+0.98% mean power** and ran **−1.64% on ITL p50**
versus the server default — small, but 3–6 CVs outside the unpinned band.

So the pin is not free, and E0 was re-run under it so the variance bar is
measured in the same configuration as the experiments it bounds. The unpinned
runs remain committed as superseded records rather than being deleted.

Pinning also **halved the run-to-run variance** — TTFT p95 CV fell from 1.31% to
0.48%, energy from 0.20% to 0.16% — which is the expected consequence of
removing sampling randomness: identical token streams produce identical compute.
That is a second, independent reason to pin it.

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

#### Known limitation: the idle *mean* is not a robust statistic on this machine

Measured 2026-08-11, across all ten committed baselines. Idle draw here is
bimodal: a stable quiescent floor, plus brief excursions to ~62 W from an
intermittent consumer outside this benchmark (the card also drives the Windows
desktop). **Every** baseline has a median between 18.7 and 22.2 W across three
days. The mean is decided by how many excursions happen to land in the 130 s
window:

| | baselines | samples > 40 W | mean W | median W |
|---|---|---|---|---|
| quiet window | E0E1, E0, E1, E2 ×2 | 0 % | 18.8–20.6 | 18.7–20.6 |
| excursions present | E0-sat, E3, E3-clean, m1-smoke, E1-dltest | 1.6–16.9 % | 21.5–27.9 | 20.3–22.2 |

Two consequences, both stated rather than smoothed:

- Re-measuring does not reliably fix it. The E3 baseline was re-measured
  specifically to replace a suspect one, and came back **worse** — 27.878 W
  against 24.558 W — despite a *lower* median (20.41 vs 22.16 W).
- The 25 % utilisation guard does not catch it, and cannot. On this machine
  `util_gpu_pct` reads 19–41 % at a genuinely quiescent ~20 W with SM clocks
  pinned at their 210 MHz floor, because desktop compositing registers as
  utilisation while costing almost no power. Utilisation is the wrong proxy
  here; power is the signal.

This affects only **incremental** J/token. Raw J/token, the E3 frontier table,
and the E5 cost model are all baseline-independent and unaffected. Whether
idle subtraction should use the median instead of the mean — or require a
window with zero samples above 40 W — is an open decision, not settled here,
because it would change every incremental figure in E0–E5 and the M3 energy
validation along with them.

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

### A failed request is not a fast request

The first version of the goodput calculation counted failures as SLO-meeting.
vLLM records a failed request as `ttft = 0.0`, `output_len = 0` and a non-empty
error string — and `0.0 <= 1.0 s` passes the TTFT SLO, `0.0 <= 10 s` passes the
end-to-end SLO. A connection failure therefore scored as the best possible
response.

It surfaced because goodput came out *above* measured throughput at one E1
point (7.9098 vs 7.9049 req/s), which the sanity checks flag as impossible.
One request in 1600 had died with a transport traceback.

The fix excludes a request from the numerator if it carries an error, produced
zero output tokens, or reported a non-positive TTFT — while keeping it in the
denominator, because a server that sheds load is not thereby faster. Corrected
goodput now matches vLLM's own independently computed `request_goodput` to
seven significant figures at that point.

Points measured before the fix keep their original stored value in
`results/raw/`; `analyze.py` recomputes goodput from the per-request arrays,
which are committed, so every table and plot uses the corrected figure and the
raw files stay an honest record of what the harness computed at the time. A
recomputed row carries `superseded_stored_value` so the difference is visible
rather than silent.

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
