# HTTP connection reuse cost vLLM 59% of its throughput, and it is not a model result

Found while measuring E4 on 2026-08-15. It is recorded separately because it is
larger than E4: it changes what the E4 arms measure, it explains a 1.9×
disagreement with an earlier measurement, and it had to be shown *not* to
contaminate E0–E3 before anything else could be trusted.

## What happened

The first E4 pass ran the load generator at its default transport, which reuses
pooled HTTP connections. Two things came out of it that did not make sense:

1. **llama.cpp dropped ~11% of requests** at concurrency 8 and 32 — 87 of 800
   and 179 of 1600 — every one an `aiohttp ServerDisconnectedError`, clustered
   at connection-reuse boundaries (request indices 8, 9, 18–20, 29, 30, …).
2. **vLLM measured 862 out tok/s at concurrency 32**, against 1618 tok/s for the
   same model, shape and concurrency in E2's concurrency ladder — 1.9× apart,
   with 2× the TPOT at a *larger* mean batch and *lower* power.

## Cause, read from the source rather than guessed

`llama.cpp/tools/server/server-http.cpp` sets only read and write timeouts and
leaves cpp-httplib's defaults in place:

```
vendor/cpp-httplib/httplib.h:26  #define CPPHTTPLIB_KEEPALIVE_TIMEOUT_SECOND 5
vendor/cpp-httplib/httplib.h:34  #define CPPHTTPLIB_KEEPALIVE_MAX_COUNT 100
```

The server closes any connection idle for five seconds while the client keeps
it in its pool; the next reuse of that socket dies. Sending `Connection: close`
so the client never reuses a connection is the control, and it settles both
questions at once.

## The measurements

Same configs, same session, same idle baseline; the only change is the
transport. Superseded runs remain in `results/raw/`.

| point | transport | completed | out tok/s | TTFT p50 | ITL p50 | mean W | J/tok |
|---|---|---|---|---|---|---|---|
| vLLM c1 | reuse | 200/200 | 61.4 | 187.1 ms | 12.43 ms | 185.2 | 3.061 |
| vLLM c1 | **close** | 200/200 | **148.9** | 57.9 ms | 6.02 ms | 300.0 | 2.083 |
| vLLM c8 | reuse | 800/800 | 517.4 | 683.5 ms | 8.76 ms | 244.9 | 0.491 |
| vLLM c8 | **close** | 800/800 | **820.2** | 355.1 ms | 6.34 ms | 325.4 | 0.419 |
| vLLM c32 | reuse | 1600/1600 | 862.1 | 931.6 ms | 13.35 ms | 259.8 | 0.312 |
| vLLM c32 | **close** | 1600/1600 | **1542.4** | 469.4 ms | 8.21 ms | 381.1 | 0.262 |
| llama.cpp c1 | reuse | 200/200 | 144.3 | 70.0 ms | 6.21 ms | 332.4 | 2.385 |
| llama.cpp c1 | **close** | 200/200 | 145.6 | 66.4 ms | 6.21 ms | 335.6 | 2.383 |
| llama.cpp c8 | reuse | **713/800** | 471.3 | 307.4 ms | 12.71 ms | 362.7 | 0.799 |
| llama.cpp c8 | **close** | 800/800 | 482.8 | 307.4 ms | 12.49 ms | 367.8 | 0.791 |
| llama.cpp c32 | reuse | **1421/1600** | 745.1 | 221.0 ms | 25.15 ms | 243.6 | 0.341 |
| llama.cpp c32 | **close** | 1600/1600 | 755.6 | 221.5 ms | 24.67 ms | 247.1 | 0.339 |

**The two stacks are affected completely differently.** Connection reuse cost
vLLM **59% of its throughput at concurrency 1** (61.4 → 148.9), 37% at
concurrency 8 and 44% at concurrency 32. It cost llama.cpp essentially no
throughput at all (144.3 → 145.6, +0.9%) — what it cost llama.cpp was
*requests*, 11% of them, which vLLM never dropped.

## It is not a measurement artefact in the client's clock

Mean power is server-side telemetry and moves with it: 185.2 → 300.0 W at
concurrency 1 on identical work. Integrated over each measurement window, the
same 200 requests cost **78.3 kJ with reuse and 53.3 kJ without — 32% less
energy for exactly the same tokens**. With reuse, the GPU was genuinely idle
waiting on the client, not merely mis-timed by it.

## What it explains

**The 1.9× disagreement with E2's ladder was the transport, not a flag.** A
control changing only `max_num_seqs` (256 → 512, the value E2's ladder used)
measured 1542.4 vs 1539.4 out tok/s — **0.2% apart**, so the scheduler setting
explains none of it. With reuse disabled, E4's c32 point (1542.4 tok/s) agrees
with E2's independent measurement of the same concurrency (1618.5) to 4.7%; the
residual is a 133 s steady-state run against a 20 s burst.

## What it does *not* affect: E0–E3

Every E0–E3 point ran with connection reuse, because the flag did not exist in
the harness then. That had to be tested rather than argued, so
`e1_chat_7b_bf16_r4` was re-run changing only the transport:

| e1_chat_7b_bf16_r4 | reuse (committed) | close (control) | difference |
|---|---|---|---|
| out tok/s | 506.5 | 506.3 | −0.04% |
| TTFT p50 | 88.06 ms | 92.18 ms | +4.7% |
| TTFT p95 | 158.3 ms | 169.7 ms | +7.2% |
| ITL p50 | 16.78 ms | 17.38 ms | +3.6% |
| e2e p95 | 3103 ms | 3262 ms | +5.1% |
| mean W | 343.7 | 337.0 | −2.0% |
| J/tok | 0.7021 | 0.6922 | −1.4% |
| goodput / SLO attainment | 3.957 / 1.000 | 3.956 / 1.000 | — |

**No effect, and the sign is against the hypothesis** — the no-reuse control is
marginally *slower* on latency, within E0's ~2% cross-session tail-latency
drift. E1's latency, goodput and energy stand as published.

The reason is the difference between open- and closed-loop load. E1 offers
Poisson arrivals at a fixed rate: the next request is scheduled by a timer, so a
stall on connection reuse overlaps with waiting that was going to happen anyway.
E4 holds a fixed concurrency: the next request cannot start until an in-flight
one finishes and its connection is released, so any per-reuse stall serialises
directly with generation. The pathology needs a closed loop to bite.

## What E4 therefore reports

Both arms run with `Connection: close`, set identically on each side rather than
only on the one that needed it. E4's question is which server decodes faster,
and a transport difference between the arms would have answered a different
question — as the first pass did, when it made llama.cpp look 2.35× faster than
vLLM at concurrency 1 where the two are in fact within 2.2%.

The reuse numbers are kept, not discarded. For anyone driving either server from
a keep-alive client in a closed loop, they are the more relevant column.
