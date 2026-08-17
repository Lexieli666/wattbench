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
| For llama.cpp runs, the **source commit it was built from** | It has no release cadence to cite, so the commit is the version |
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
- **incremental J/token** — with `idle_median_W × window_seconds` subtracted.
  This is the marginal energy of the serving work itself.

Neither is "the" answer; the raw figure drives the cost model, because an owner
pays for the idle floor too.

#### The idle *mean* is not a robust statistic here, so the median is subtracted

**Decided 2026-08-14**, replacing the mean-subtraction used from 2026-08-09 to
2026-08-13. What changed is which statistic is read out of the idle window; no
measurement was repeated and no raw file was rewritten.

Idle draw on this machine is bimodal: a stable quiescent floor, plus brief
excursions from an intermittent consumer outside this benchmark (the card also
drives the Windows desktop). Across **twelve committed baselines spanning six
days**, every single one has a median between 18.7 and 22.2 W, while the mean
ranges 18.8–27.9 W depending on how many excursions land in the 130 s window:

| | baselines | samples > 40 W | mean W | median W |
|---|---|---|---|---|
| quiet window | E0E1, E0, E1, E2 ×2 | 0 % | 18.8–20.6 | 18.7–20.6 |
| excursions present | m1-smoke, E0-sat, E1-dltest, E3, E3-clean, E2-recheck, E4 | 1.6–16.9 % | 21.5–27.9 | 20.0–22.2 |

The excursions have structure, measured from the committed sample traces:
**1.7–4.0 s bursts to 50–66 W, recurring every ~30–50 s**. That is what settles
the choice between the two candidate policies:

- **Requiring an excursion-free window is not a protocol on this machine.** The
  longest excursion-free stretch inside a contaminated window is 44–99 s, always
  shorter than the 130 s measurement, so no amount of re-measuring produces a
  clean baseline while the consumer is active — it is a lottery on which session
  you are in. Re-measuring also demonstrably backfires: the E3 baseline was
  re-measured specifically to replace a suspect one and came back **worse**
  (27.878 vs 24.558 W) despite a *lower* median. And the rule would delete
  evidence: 13 of 47 runs would lose their incremental figure, including all six
  E3 rungs, all three E0-sat repeats and both E2 r16 points.
- **The median is stable across every window measured**, contaminated or not, and
  keeps every measured point reportable.

What the change actually does to the numbers, recomputed across all 47 runs:
**median +0.03 %, maximum +2.56 %**. The 34 runs on excursion-free baselines move
by less than 0.05 %. In the published tables only `e1_sweep.md`'s "J/tok net"
column moves at all, in the third significant figure of two rows.

The evidence that the median is *better*, not merely different: 7B AWQ was
measured twice, an hour apart, against two different baselines. Raw J/token
agrees to **0.02 %**; the idle-subtracted figure disagreed by **1.12 % under the
mean and 0.60 % under the median**. Halving a disagreement that raw energy says
should not exist is the whole case. The residual 0.60 % is not excursion noise —
the E3 session's quiescent floor genuinely sat ~2 W higher (quiescent mean 23.6 W
against ~20.6 W typical) — and no estimator fixes a floor that really moved.

Two limits of the policy, stated rather than smoothed:

- During a measured run the same host consumer is presumably still bursting,
  invisible inside a 400 W trace. The mean assumed it ran at exactly the
  baseline rate; the median assumes it did not run at all. Neither is exactly
  right, and **the gap between the two policies, ≤2.6 % of incremental, is the
  bound on that ambiguity**. It is not resolvable from this machine's telemetry.
- The 25 % utilisation guard does not catch contamination and cannot. On this
  machine `util_gpu_pct` reads 19–41 % at a genuinely quiescent ~20 W with SM
  clocks pinned at their 210 MHz floor, because desktop compositing registers as
  utilisation while costing almost no power. Utilisation is the wrong proxy
  here; power is the signal. Utilisation is now reported but no longer gates.

**Unaffected by any of this:** raw J/token, the E3 frontier table, and the E5
cost model, all of which are baseline-independent by construction.

##### The M3 stability gate had to move with it

M3's baseline check tested `(p95 − min) / mean < 0.35`. On a bimodal idle window
the p95 *is* the excursion, so that statistic read 7–189 % across a corpus whose
quiescent floor never moved by more than 3.4 W. Run against the full twelve
baselines it failed four of them — and failing a baseline blocks *all* J/token
reporting, including the raw figure that the baseline cannot affect.

The gate is now **IQR / median < 0.35**, which reads **2.1–15.8 %** over the same
twelve. All twelve pass, with the worst at less than half the threshold, so the
check still has room to catch a window that is genuinely unstable rather than
merely interrupted. Every baseline additionally records its excursion fraction
above 40 W, so contamination is visible in the file rather than inferred.

Implementation, for anyone auditing: `power_log.py` writes the median as
`idle_power_w` (keeping the mean as `idle_power_w_mean`) for baselines measured
from 2026-08-14; for the eleven earlier ones `analyze.py` reads the median that
was already stored as `power_w.p50` and recomputes the incremental figure at
analysis time. Raw results keep the value the harness computed on the day, the
same treatment goodput got when its definition was corrected — the correction
lives in git and in this document, not in an edited raw file.

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

### FP8 on Ada (SM 8.9) — attempted, and it serves

Plan §3 listed FP8 as exploratory: attempt it, and if the build's support turns
out partial, report that in one sentence rather than fighting it. **It is not
partial.** vLLM 0.26.0 quantized the BF16 checkpoint on the fly
(`--quantization fp8`, no pre-quantized checkpoint and no download) and served
it, and the server's own log names the path taken: `Selected
CutlassFP8ScaledMMLinearKernel for Fp8PerTensorOnlineLinearMethod`, i.e. online
per-tensor W8A8 through the CUTLASS FP8 GEMM. One build caveat is in the same
log and is recorded because it is the only thing that looked partial: the
optional `vllm.third_party.deep_gemm` backend failed to import here
(`AssertionError` in `_find_cuda_home` — no `CUDA_HOME` in this venv), so
whatever that path would have contributed was not available. The CUTLASS path
served the whole run without error.

The point is `configs/e2/fp8/r4.yaml`, tagged `E2-fp8` so it cannot leak into
the E2 tables. It extends the E1 r4 config rather than restating it, so the
traffic is identical to `e1_chat_7b_bf16_r4` by construction and precision is
the only variable. Measured 2026-08-17: 508.1 out tok/s (offered-rate bound, as
every r4 point is), TTFT p95 105.5 ms against BF16's 158.3, ITL p50 11.36 ms
against 16.78, mean power 289.0 W against 343.7, and **0.590 J per output token
against BF16's 0.702 and AWQ's 0.650** — the lowest of the three at this load.
The server was given 11.5 GiB of KV cache where BF16 got 6.06 GiB and AWQ ~15
GiB, so FP8 buys roughly 90% more KV headroom than BF16 and still less than
int4. No throttling (`sw_power_cap_frac` 0.0, clocks 2655 MHz p50), energy
cross-check 0.58%, all eleven sanity checks pass.

Three limits on that paragraph. It is **one point at one load**, not a sweep.
It is a **new arm from a later session** — 2026-08-17 against the series'
2026-08-10/11 — so it carries the ~2% cross-session tail-latency drift on top of
the within-session variance bar, and its idle-subtracted energy is computed
against a baseline ~3.4 W higher than the older arms used (raw J/token, quoted
above, is baseline-independent). And **no GSM8K guard was run for FP8**, so this
project makes no claim whatsoever about its output quality; dynamic
quantization is exactly the kind of change a quality guard exists to catch, and
here there is none.

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
  for owning. What it costs at any other duty is a curve, not a multiplier:
  amortisation is a fixed daily cost spread over the day's actual output, so
  the owned figure is `amort_per_day / (V/1e6) + energy $/1M` at daily volume
  `V`, and duty cycle is *determined* by `V` (`V / (tok_per_s x 86400)`) rather
  than free to vary alongside it. `results/plots/e5_cost_vs_volume.png` and the
  sensitivity rows in `e5_economics.md` are that curve. Note what does **not**
  move with duty: the break-even volumes, because amortisation is daily on both
  sides of the comparison and cancels.
- API prices are compared at an **effective output price** that folds in the
  input charge at the benchmarked input:output ratio. Self-hosting pays for
  prefill too; comparing against an API's output price alone would flatter the
  API.
- Amortisation covers the **card only**. The rest of the host is excluded, and
  that exclusion is stated wherever the owned line appears.

---

## 7b. E4: comparing two serving stacks without pretending they are the same

E4 puts vLLM and llama.cpp side by side at concurrency 1, 8 and 32. One thing
about it cannot be fixed by careful measurement, so it is stated on every table
and every plot rather than in a footnote:

> **GGUF Q4_K_M and AWQ int4 are different quantization formats.** E4 compares
> two stacks each at its own native int4, not one set of weights on two
> servers. A throughput or energy gap includes whatever the formats themselves
> cost.

Serving each stack at its native format is the honest choice available: vLLM
does not serve GGUF well and llama.cpp does not serve AWQ at all, so the
alternative to different formats is not "same weights" but "no comparison".
What a reader gets is the question they actually have — *which of these should I
run on my card* — with the caveat that the answer is a stack-plus-format
package, not a statement about batching algorithms in isolation.

**Everything else is held equal, deliberately:**

| Held equal | How |
|---|---|
| Model family and size | Qwen2.5-7B-Instruct both arms |
| Traffic shape | 512 in / 128 out, `ignore_eos`, temperature 0 |
| Prompts | Same tokenizer (`Qwen/Qwen2.5-7B-Instruct`) and same seed, so both arms receive the same token sequences |
| Load generator | `vllm bench serve` for both; the llama.cpp arm is driven through the OpenAI-compatible path (`--backend openai`) against the same `/v1/completions` endpoint shape |
| Concurrency | Client-side `--max-concurrency`, identical per point |
| Session | Both arms measured back to back against one idle baseline, on one driver and one power limit |
| Energy | Same poller, same integration, same window definition |

**Serving flags are minimal on both sides, on purpose.** Each stack runs at its
own defaults except where the workload forces a choice: how many concurrent
slots, how much KV context, and all layers on the GPU. Tuning one arm and not
the other would measure the tuning. Two consequences worth knowing:

- llama.cpp's `-c` is the **total** KV context divided across `--parallel`
  slots, where vLLM's `--max-model-len` is **per sequence**. Each llama.cpp slot
  is therefore given `4096 × parallel` so that a slot gets the same per-sequence
  budget a vLLM sequence gets.
- vLLM allocates one shared KV pool sized by `gpu_memory_utilization`;
  llama.cpp partitions fixed per-slot budgets. This is an architectural
  difference, not a configuration one, and it is why the two stacks report
  different things about themselves (below).

**Two columns are empty for llama.cpp, and both absences are findings:**

- **Preemptions.** llama.cpp has no such counter because it does not preempt. A
  request that finds no free slot is *deferred before it starts*; vLLM admits
  it and may *evict it after it starts*. Reporting zero for llama.cpp would
  claim a measurement that does not exist.
- **KV utilisation.** The build used here exports no `kv_cache_usage_ratio`, and
  the number would not mean the same thing if it did, per the pool-vs-slots
  difference above.

**SGLang, the plan's optional third stack, was not run.** It would have added a
third quantization format and a third set of serving defaults to a comparison
that is already carrying one large caveat. Recorded as a deliberate omission
rather than an oversight.

### The HTTP transport is part of the measurement, and it had to be pinned

This was the largest single measurement error found in the project, and it was
found by E4 rather than caused by it.

The load generator reuses pooled HTTP connections by default. llama.cpp leaves
cpp-httplib's five-second keep-alive timeout in place, so the server closes
connections the client still holds. Under a **fixed concurrency** that is not a
minor inefficiency:

| with connection reuse | vLLM | llama.cpp |
|---|---|---|
| requests dropped @ c8 | 0 | 87 / 800 (10.9%) |
| requests dropped @ c32 | 0 | 179 / 1600 (11.2%) |
| throughput lost @ c1 | **−59%** | −0.9% |
| throughput lost @ c32 | **−44%** | −1.4% |

The two stacks pay in different currencies — llama.cpp in dropped requests,
vLLM in throughput — which is exactly what makes it dangerous: measured with
reuse on, llama.cpp appears **2.35× faster than vLLM at concurrency 1**, where
the two are in fact within 2.2%.

It is not a client-clock artefact. Mean GPU power, which is server-side
telemetry, moved 185 → 300 W at concurrency 1 on identical work, and integrated
energy for the same 200 requests fell 78.3 → 53.3 kJ. With reuse, the card was
genuinely idle waiting on the client.

**The rule this project now follows:** the transport is stated and set
identically on every arm of a comparison. Both E4 arms send `Connection: close`.
Setting it only on the arm that visibly suffered would have made the transport a
variable in the comparison.

**E0–E3 were checked, not assumed.** They all ran with connection reuse, because
the flag did not exist in the harness then. Re-running `e1_chat_7b_bf16_r4` with
the transport as the only change moved throughput by 0.04%, energy by 1.4%, and
goodput not at all — latency was marginally *worse* without reuse, inside E0's
cross-session drift. The pathology needs a **closed loop**: at a fixed offered
rate the next request is scheduled by a timer, so a reuse stall overlaps waiting
that was going to happen anyway; under a concurrency cap the next request cannot
start until a connection is released, so the stall serialises with generation.
E1 is open-loop, and is unaffected.

Full account and every number: `results/tables/e4_transport.md`.

**Dropped requests are shown, not excluded.** Each one trips the
`all_requests_completed` sanity check, and the completed/issued count is printed
in the E4 table so the reader can weigh it. After the transport was pinned, all
six E4 points completed every request they issued.

### The stack recommendation is quality-guarded, at n=200

A stack recommendation with no quality control is the weakest claim available:
"vLLM is faster" is worth nothing if what it serves is worse. E2's guard covers
BF16, AWQ and GPTQ but not GGUF Q4_K_M, so until this ran the llama.cpp arm had
no quality number at all.

Both E4 arms were run through `gsm8k_guard.py` against the same serving flags
their throughput points used — llama.cpp under `configs/e4/llamacpp_c1.yaml`
(`-ngl 99 -c 4096 --parallel 1 --flash-attn auto`), vLLM under
`configs/e4/vllm_c1.yaml`. Three choices are worth stating:

- **n=200, not E2's n=50.** At n=50 the 95% interval is roughly ±14 points,
  wide enough that a ten-point gap would license nothing. n=200 roughly halves
  it to ±7, and costs a few minutes of greedy decoding. The n=50 and n=200
  subsets are **different draws from the same test split, not nested**, so they
  are separate row sets and are never pooled; every table that shows them
  together states both sample sizes.
- **The served model id was read from the endpoint, not assumed.** llama.cpp
  answers to its `--alias` rather than to an HF repo id: `/v1/models` reports
  `qwen2.5-7b-instruct-q4_k_m` (`ftype: Q4_K - Medium`, `n_ctx 4096`). The guard
  records the GGUF repo separately so the weights still resolve to a commit.
- **The guard's own transport was verified on a socket**, because E4's finding
  is that the client is part of the instrument. `urllib.request` sends
  `Connection: close` and opens one connection per request — confirmed by
  pointing the guard at a local server that echoes its headers, not by reading
  the documentation. That matches the transport both E4 serving arms used.

Result: vLLM/AWQ **91.0%** (182/200), llama.cpp/GGUF **90.0%** (180/200), zero
request errors on either. The difference is **+1.0 point with a 95% interval of
−4.9 to +6.9** (Newcombe score method for the difference of two proportions —
an interval on the *difference*, because two intervals eyeballed for overlap is
the conservative mistake rather than the safe one). **Within noise.**

So E4's throughput and energy result reads as a serving recommendation rather
than a speed-for-accuracy trade. It is not proof of parity: an interval that
contains zero contains everything else inside it too. And the format caveat
above still governs — a quality difference here would belong to the
format-plus-stack pair, not to the batching implementation.

As a check on the guard rather than a second result, the AWQ checkpoint scored
92.0% at n=50 in E2, on a different draw, under different serving flags, in
another session; the two agree to within −7.8 to +10.3 points.

---

## 8. What this project does not claim

- **No capability equivalence.** A locally served Qwen2.5-7B is not a substitute
  for a frontier API model. Frontier price rows appear to show the ceiling of
  the market, not an available trade, and are marked as not weight-class
  comparable. The only capability evidence produced here is the GSM8K
  exact-match guard — n=50 for the E2 quantization arms, n=200 for the two E4
  stack arms — and it licenses no claim beyond that task.
- **No datacenter comparison.** Single consumer card, cited references only.
- **No identical-weights stack comparison.** E4's two arms serve different int4
  formats because that is what each stack natively serves (§7b). It answers
  "which stack should I run", not "which batching algorithm is faster".
- **No measured API latency.** Without keys, no hosted endpoint was timed. Every
  measured latency in this repo is self-hosted, and no measured-vs-reported
  latency comparison is made anywhere.
- **No wall-socket energy.** GPU rail only (§3).
- **Not run means not run.** Anything unmeasured prints as `not run`, never as
  an estimate dressed as a measurement.
