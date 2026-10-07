# Status and remaining work

Updated 2026-10-07. Everything below is recoverable from the repo alone.

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
| M8 E4 stacks | vLLM vs llama.cpp at concurrency 1/8/32, plus 3 transport and scheduler controls |
| v0.2 R1 | duty-cycle cost curve + sensitivity rows (re-analysis only) |
| v0.2 R2 | GSM8K guard for both E4 arms at n=200 |
| v0.2 R3 | FP8 on Ada attempted; it serves, one point recorded |
| M9 third stack (harness) | `pytorch_server.py`, `configs/_base_7b_pytorch.yaml`, `configs/e4/pytorch_*.yaml`, `run_e4_pytorch.sh`; protocol-tested on CPU, measured 2026-10-07 in the E4 session on driver 610.60; its `torch.compile` control not run in this session, see METHODOLOGY §7c |
| M9 memory + GPU time | `gpu_memory` / `gpu_time` / `profile` sections in every new record; `analyze.py memory` fills the peak-memory column for every existing run from its committed power log |

Regenerate every table and plot from raw data, in this order and with the venv
python — system `python3` has no matplotlib and writes the tables before it
fails at the plots, so a partial run looks like a success:

```bash
~/wattbench-venv/bin/python ./validate_energy.py   # M3 gate, must pass first
~/wattbench-venv/bin/python ./analyze.py all
~/wattbench-venv/bin/python ./verify_readme.py     # README numbers trace to raw
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

3. **M8 / E4 is measured and written up**, 2026-08-15. vLLM leads llama.cpp by
   7.9% at concurrency 1, +80% at 8 and +119% at 32, at 47% and 23% less energy
   per token (the table as re-measured in one session on driver 610.60,
   2026-10-07; METHODOLOGY §7b). Plan §4's expected shape, confirmed — but the
   first pass measured the opposite, and why is the more useful finding:

   **HTTP connection reuse moved measured throughput by up to 59%.** The two
   stacks paid in different currencies, which is what made it dangerous:
   llama.cpp dropped ~11% of requests (cpp-httplib's 5 s keep-alive timeout,
   left at its default), while vLLM dropped none and instead lost 59% of its
   concurrency-1 throughput. Measured with reuse on, llama.cpp looks 2.35×
   faster than vLLM at concurrency 1, where the two were within 2.2% in the
   August session. Server-side power confirms it is real, not a client clock:
   185 → 300 W on identical work, 78.3 → 53.3 kJ for the same 200 requests.
   Both arms now send `Connection: close`. Full account:
   `results/tables/e4_transport.md`.

   E0–E3 were *checked*, not assumed: an E1 point re-run with the transport as
   the only change moved 0.04% on throughput and 1.4% on energy. The pathology
   needs a closed loop; E1's Poisson arrivals are open-loop and absorb it.

## Closed since 2026-08-15

1. **Duty cycle is priced as a curve, not disposed of in a sentence**
   (2026-08-17, re-analysis only, no GPU). The old text multiplied the
   amortised figure by a duty factor, which is the wrong shape: at daily output
   volume `V` the card only runs `V/(tput*86400)` of the day, so duty is
   *determined* by `V` and multiplying an already volume-dependent $/1M by it
   counts the same effect twice. What follows:

   - **The break-even volumes do not move with duty cycle.** Amortisation is a
     fixed daily cost on both sides of the comparison and cancels;
     `e5_economics.md`'s break-even table is unchanged, and that is now stated
     rather than left to be rediscovered.
   - **The owned $/1M does move, and that was unreported.** $0.037 at full
     duty, $0.052 at 50%, $0.081 at 25%, $0.169 at 10%, $1.487 at 1%. Below
     ~43k output tokens/day the card's cost per million tokens exceeds even
     Anthropic's Opus list price — a price statement, not a capability one.
   - `results/plots/e5_cost_vs_volume.png` carries the curve with duty cycle as
     a secondary axis (the same axis rescaled, so it cannot be read as an
     independent variable) and the API levels drawn across it. Its crossings
     are read off the plotted samples and agree with the break-even table to
     **0.0004%** — a check on the chart, not a restatement of the algebra.

2. **Both E4 arms now carry a quality guard, at n=200** (2026-08-17). vLLM/AWQ
   **91.0%** (182/200), llama.cpp/GGUF **90.0%** (180/200), zero request errors
   on either. Difference **+1.0 point, 95% Newcombe interval −4.9 to +6.9**:
   within noise. E4's speed and energy result therefore reads as a serving
   recommendation rather than a speed-for-accuracy trade — not as proof of
   parity, since an interval containing zero contains everything else in it.

   n=200 rather than E2's n=50 because ±14 points is wide enough that even a
   ten-point gap licenses nothing. The two subsets are **different draws, not
   nested**, so they are separate row sets, never pooled, and both sample sizes
   are printed wherever they appear together. E2's published n=50 rows are
   untouched. Two things were verified rather than assumed: llama.cpp answers to
   its `--alias` (`/v1/models` → `qwen2.5-7b-instruct-q4_k_m`, ftype
   `Q4_K - Medium`), and the guard's transport is `Connection: close` with one
   connection per request — read off a socket, and matching what both E4 serving
   arms used. Cross-checked against E2's n=50 AWQ guard (92.0%, different draw,
   different session): consistent at −7.8 to +10.3 points.

3. **FP8 on Ada was attempted, and it serves** (2026-08-17). Plan §3 asked for
   the attempt with a one-sentence finding if support turned out partial; the
   item had been dropped without a record, which is the silent omission this
   repo has a rule against. vLLM 0.26.0 quantized the BF16 checkpoint on the fly
   (`--quantization fp8`, no download) and served it through
   `CutlassFP8ScaledMMLinearKernel` / `Fp8PerTensorOnlineLinearMethod`. One
   chat-shape point at r4, extending the E1 r4 config so precision is the only
   variable: TTFT p95 105.5 ms against BF16's 158.3, ITL p50 11.36 against
   16.78, 289.0 W against 343.7, **0.590 J/token against 0.702** — and against
   AWQ int4, slower per token (6.18 ms ITL) but lower energy (0.650 J). KV
   headroom 11.5 GiB, between BF16's 6.06 and AWQ's ~15.

   Tagged `E2-fp8`, not `E2`: it is a new arm measured a week after the series,
   so it carries cross-session drift and a higher idle baseline, and it stays
   out of the E2 tables by construction. **No GSM8K guard was run for it**, so
   it carries no quality claim at all. The one partial-looking thing in the
   build is recorded: the optional `vllm.third_party.deep_gemm` backend failed
   to import (no `CUDA_HOME`); the CUTLASS path served the whole run.

## Open: M9, the PyTorch arm, is harnessed and measured (2026-10-07)

What exists: a third serving stack, HF transformers + plain `model.generate`,
behind the same OpenAI-compatible endpoint, load generator, transport, power
poller and `/metrics` scrape as the other two (METHODOLOGY §7c). Static
batching is left as it is because it is the property under test; the queue it
creates is exported into the same column vLLM's queue fills. It serves BF16 —
plain transformers has no int4 path that is still "just PyTorch" — so the E4
format caveat widens to three formats, and the like-for-like format control is
the E1/E2 vLLM BF16 server. Every new record also carries peak GPU memory from
two instruments and GPU time per token from two instruments, each named for
what it measures (METHODOLOGY §3b); the nvidia-smi memory peak is derived at
read time for all existing runs, no raw file rewritten.

What was verified without a GPU: the wire protocol against a tiny local
checkpoint on CPU (`tests/test_pytorch_server.py` — one SSE chunk per token,
usage in the final chunk, `finish_reason` on the last token chunk, both
`Connection: close` and chunked framing, batching of concurrent requests,
`/metrics` counters, `/wattbench/profile`, context-length rejection, the chat
path the guard uses), `harness.py export-env` for every new config, and
`analyze.py` against the committed E4 records. What was **not** verified: a
real `vllm bench serve` client against the server, CUDA-event timing, the
profiler's kernel accounting, and `torch.compile` with a static cache on
transformers ≥ 5 — all of which need the 4090.

To run it, in one session:

```bash
./baseline.sh E4-pytorch                 # fresh idle baseline first
./run.sh --force configs/e4/vllm_c8.yaml # one vLLM point as a drift check
./run_e4_pytorch.sh                      # c1/c8/c32, compile control, guard n=200
~/wattbench-venv/bin/python ./analyze.py stacks --experiment E4
~/wattbench-venv/bin/python ./analyze.py memory
```

Expect, before believing any of it: `generation_tokens_total` on `/metrics`
equal to the bench's `total_output_tokens`; `all_requests_completed` PASS;
`gpu_memory.torch_max_memory_allocated_gib` ≈ 15.2 GB (14.19 GiB) of weights
plus a KV term that scales with concurrency; and the c1 point within the same
order of magnitude as vLLM's c1 — if it is 10× slower, suspect the SDPA backend
or a CPU-side sync per step before suspecting the card. If the compiled point
recompiles mid-window the server log says so (`torch._dynamo` lines); then
either lengthen `protocol.warmup_s` for it or drop the control, never widen
the window.

## Remaining

Nothing blocking v0.2. Optional next work, in rough order of value:

0. **Measure M9** (above).
1. **Stretch items from plan §6**, untouched: prefix caching on/off for the RAG
   shape, and speculative decoding with a 1.5B draft model.
2. **The keep-alive result deserves a second client.** Everything here is
   measured through `vllm bench serve`; whether the 59% is a property of that
   client's connection pooling or of vLLM's HTTP layer is not established. The
   n=200 guards go through `urllib` instead, but they are serial and unbatched,
   so they exercise the transport without exercising the pathology.

## Rules that bit hard here — do not relax them

- **Never run a download during a measured run.** Measured: TTFT p95 +819%,
  while throughput and energy look almost normal. Corrupted data that looks
  clean is the failure mode.
- **A fresh idle baseline per session, and subtract its median.** Twelve
  baselines: medians 18.7–22.2 W, means 18.8–27.9 W. The gap is an external host
  consumer, not the card. Two E4 windows sixteen minutes apart came back 7.9%
  and 0.0% contaminated, which is why the statistic has to be robust rather than
  the window clean.
- **Never reuse a non-idle server.** A vLLM server carrying another run's
  requests measured 33 ms ITL at 192 W where a fresh one measured 17 ms at
  312 W. `run.sh` now gates on quiescence, for all three stacks.
- **E0's variance bar is regime-specific.** Unsaturated: 0.16% energy, 0.48%
  TTFT p95. Saturated: 1.09% energy, 5.83% TTFT p95. Cross-session tail latency
  drifts ~2%. Do not borrow one regime's bar for another.
- **Compare a new point against an existing measurement of the same config.**
  That reflex caught three defects nothing else did — now five. It is what
  established the E3 sweep was clean despite a visibly suspect idle baseline,
  and it is the only reason the E4 transport artefact was found: E4's c32 point
  disagreed with E2's ladder by 1.9×, and chasing that disagreement was what
  turned up a 59% measurement error.
- **The client is part of the instrument.** HTTP connection reuse changed
  measured throughput by up to 59% and dropped 11% of requests, differently for
  each stack. Any comparison must state and equalise its transport, and a
  measured "stack A is faster" is really "stack A plus this client is faster".
- **Isolate one variable per control, and prefer a control to an argument.**
  Three single-variable runs settled E4: `max_num_seqs` (explained 0.2% of a
  1.9× gap), the transport (explained all of it), and an E1 re-run (proved
  E0–E3 were untouched). Each took minutes; each replaced a plausible story
  with a number.
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
- **Render the chart before believing the code that draws it.** The
  cost-vs-volume chart passed code review and then rendered with six defects:
  matplotlib read the bare dollar signs in a legend label as mathtext and set
  "$2,100 over 3 yr" as italic algebra, the title overran the figure, the
  footnote sat on top of the x-label, and three labels collided. None of them
  are visible in the source. A chart is a build artifact and looking at it is
  the test.
- **The idle floor can move as a whole, not only bimodally.** The fourteenth
  baseline came back with a median of 23.53 W against the 18.73–22.16 W band of
  the other thirteen, its *minimum* sample above every prior median, and only
  3.4% excursions — so the robust gate passed and the documented bimodality was
  not the explanation. Idle temperature was 44–51 °C against 36–49 °C before.
  Raw J/token is baseline-independent and unaffected; net J/token measured in
  such a session carries a systematic offset against older arms, which is a
  thing to state rather than to average away.
- **A checker's assertion can be the defect.** `verify_readme.py` first reported
  the README's "19.7M tok/day" as untraceable; the computed value was 19.66M,
  i.e. the same number at a different rounding, and the string comparison was
  wrong rather than the document. Compare numbers numerically, and read a
  failure as "one of these two is wrong" rather than as "the artifact is wrong".
