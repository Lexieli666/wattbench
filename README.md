# WattBench

**What can one RTX 4090 actually serve, and at what volume does owning it beat
paying an API per token?**

Raw tokens-per-second tables for consumer GPUs exist in quantity. What is
missing is the layer above them: **goodput under an SLO, dollars per million
tokens, and joules per token as functions of offered load**, measured on
hardware people already own, with the harness and the raw data published so the
numbers can be checked.

That is what this repository is. Every measured value here traces to a
timestamped file in [`results/raw/`](results/raw/) produced on one machine.
Every external value — API price cards, GPU rental rates, the electricity
tariff — lives in [`pricing.yaml`](pricing.yaml) with a source and a date, and
is labelled *reported, not measured* wherever it appears. The two are never
blended.

> **Status.** Experiments are run in order E0 → E3, then the economics memo.
> Sections below are filled in from committed raw data as each completes;
> anything not yet measured says **not run** rather than carrying an estimate.

---

## Prior art, and what is missing from it

- **[InferenceMAX / InferenceX](https://inferencemax.ai/) (SemiAnalysis).**
  Continuous, open-source inference benchmarking with tokens-per-dollar and
  tokens-per-watt metrics, on datacenter hardware (B200/GB200/MI355X-class) and
  large models. It legitimises the economics framing and supplies the
  vocabulary this project borrows. It does not touch consumer cards or
  ownership economics.
- **Consumer-GPU benchmark collections** — `XiongjieDai/GPU-Benchmarks-on-LLM-Inference`,
  `hholtmann/llm-consumer-gpu-benchmark`, hardware-corner's llama-bench tables,
  the various Ollama benchmark posts. These report tokens/s for single requests
  or one fixed concurrency. Almost none sweep offered load, essentially none
  report goodput under a stated SLO, and none measure energy per token under
  batching or derive API break-even volumes.
- **Per-query energy analyses.** Mostly modelled, or measured datacenter-side.
  A measured J/token-versus-load curve on owned hardware is largely absent.

**The niche, in one line:** *load-dependent economics — goodput, $/1M tokens,
J/token as functions of offered load — on hardware people own, reproducibly.*

---

## What was measured

| | |
|---|---|
| Hardware | 1 × NVIDIA GeForce RTX 4090, 24 GB, in a Windows desktop |
| Serving | vLLM 0.26.0 under WSL2 (Ubuntu 26.04), driver 595.95 |
| Models | Qwen2.5-Instruct, 1.5B → 32B, BF16 and int4 |
| Shapes | chat (512 in / 128 out) and RAG (2048 in / 256 out) |
| Arrivals | Poisson, fixed seed, 0.5 → 32 req/s |
| SLO | TTFT ≤ 1 s **and** end-to-end ≤ 10 s |
| Energy | GPU-rail draw polled at ~2 Hz, integrated trapezoidally to joules |

Full protocol and its limits: [`METHODOLOGY.md`](METHODOLOGY.md).
Reproducing it on Windows: [`SETUP-WSL2.md`](SETUP-WSL2.md).
Verified machine inventory: [`results/environment.md`](results/environment.md).

---

## Headline results

*Filled from committed raw data as experiments complete — see `results/tables/`.*

### 1. Cost per 1M output tokens vs offered load

Not run.

### 2. Energy per token vs offered load

Not run.

### 3. Model-size frontier

Not run.

### Break-even volume

Not run.

---

## Run-to-run variance — the bar every other number carries

Before any comparison is claimed, the same reference configuration is run three
times and its coefficient of variation reported. Differences smaller than that
CV are noise, and this project does not claim them.

See [`results/tables/e0_variance.md`](results/tables/e0_variance.md).

---

## Reproducing

```bash
uv venv --python 3.12 ~/wattbench-venv
VIRTUAL_ENV=~/wattbench-venv uv pip install vllm pyyaml matplotlib

./fetch_models.sh                     # pre-fetch weights (hours, at ~3 MB/s)
./baseline.sh E1-chat                 # fresh idle-power baseline, GPU quiet
./sweep.sh --series E1-chat configs/e1/chat/*.yaml
./analyze.py all                      # tables + plots into results/
```

`run.sh` takes one config and produces one raw result; it reuses a running vLLM
server across points that share serving flags, and skips points that already
have a result, so an interrupted sweep resumes rather than restarts.

Two environment variables are set for every run because vLLM 0.26 will not
otherwise start under WSL2 — `VLLM_USE_V2_MODEL_RUNNER=0` and
`VLLM_USE_FLASHINFER_SAMPLER=0`. Both are explained in
[`SETUP-WSL2.md`](SETUP-WSL2.md), and both mean **these throughput numbers
should not be compared against published vLLM figures from a native-Linux
host.**

---

## What this does not claim

- **No capability equivalence.** A locally served Qwen2.5-7B is not a substitute
  for a frontier API model. Frontier prices appear in the comparison to show the
  ceiling of the market, marked as not weight-class comparable. The only
  capability evidence produced here is a 50-item GSM8K exact-match guard, which
  licenses no claim beyond that task — and at n=50 the 95% interval is roughly
  ±14 points, so only large gaps mean anything.
- **No measured API latency.** Without API keys, no hosted endpoint was timed.
  Every measured latency here is self-hosted.
- **No datacenter comparison.** One consumer card; datacenter work is cited, not
  reproduced.
- **GPU rail only.** Host power (CPU, RAM, PSU losses) is not measured, so a
  wall-socket figure would be higher than the J/token reported here.
- **One card, no failover, no ops budget.** The cost model prices silicon and
  electricity. It does not price your time, downtime, or the absence of a
  second machine.

---

## Layout

```
configs/        one YAML per experiment point
run.sh          config in -> one raw result out; resumable
sweep.sh        run a list of points unattended, baseline first
baseline.sh     measure the idle-power baseline for a series
power_log.py    telemetry polling and joule integration
harness.py      config parsing, provenance, result assembly, sanity checks
validate_energy.py  M3 gate: energy numbers are checked before they are used
gsm8k_guard.py  50-item quality guard for the quantization ablation
probe_limits.sh longest servable context per checkpoint
analyze.py      raw -> tables and plots
pricing.yaml    dated external data, all reported-not-measured
results/raw/    committed raw output, append-only
results/idle/   idle-power baselines, one per series
```
