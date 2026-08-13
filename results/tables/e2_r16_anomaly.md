# E2 r16: the outlier was a session artefact, and re-measurement resolves it

**Resolved 2026-08-13.** Both E2 r16 points were re-measured on a quiescent GPU
against a fresh idle baseline, per the procedure this document originally
prescribed. The BF16 outlier does **not** reproduce. What follows is the whole
sequence, because the diagnosis was partly right and partly wrong and both
halves are worth keeping.

## The measurements

Qwen2.5-7B, chat shape (512 in / 128 out), 16 req/s offered, 2400 prompts.
Six BF16 measurements and two AWQ measurements of the same configuration:

| run | date | tok/s | mean W | at 450 W cap | SM clk p50 | TPOT p50 | batch | queue | preempt | KV |
|---|---|---|---|---|---|---|---|---|---|---|
| E0-sat A | 08-10 | 1535.1 | 425.8 | 94.7% | 2505 MHz | 118.3 ms | 167.6 | 250.2 | 245 | 85.5% |
| E0-sat B | 08-10 | 1568.8 | 425.7 | 94.1% | 2490 MHz | 114.9 ms | 169.2 | 228.4 | 254 | 86.4% |
| E0-sat C | 08-10 | 1538.4 | 422.9 | 93.3% | 2490 MHz | 114.2 ms | 165.6 | 265.2 | 262 | 84.4% |
| E1 chat r16 | 08-10 | 1613.7 | 426.7 | 94.7% | 2445 MHz | 111.6 ms | 170.0 | 208.8 | 239 | 86.9% |
| ~~E2 bf16 r16~~ | 08-10 | ~~1262.4~~ | ~~396.6~~ | ~~55.6%~~ | ~~2655 MHz~~ | ~~144.4 ms~~ | 169.5 | 388.4 | 271 | 86.4% |
| **E2 bf16 r16 (re-measured)** | **08-13** | **1605.9** | **426.4** | **94.2%** | **2460 MHz** | **111.9 ms** | 170.8 | 214.3 | 239 | 87.2% |
| ~~E2 awq r16~~ | 08-10 | ~~1537.0~~ | ~~398.4~~ | ~~0.0%~~ | ~~2655 MHz~~ | ~~166.4 ms~~ | 229.3 | 237.0 | 0 | 48.7% |
| **E2 awq r16 (re-measured)** | **08-13** | **1607.1** | **413.4** | **0.3%** | **2655 MHz** | **158.4 ms** | 229.9 | 206.6 | 0 | 47.4% |

Struck rows are superseded, not deleted: they remain in `results/raw/` and
`analyze.py` excludes them automatically as older runs of the same point.

The five valid BF16 measurements now span 1535.1–1613.7 tok/s: mean **1572.4**,
CV **2.33%** across three sessions, against E0-sat's 1.20% within one session.
The withdrawn point sits at −19.7%, **−8.4σ** against that five-run consensus.
It was never noise, and it is not reproducible.

## What actually caused it

The original diagnosis — "an underfed GPU, not a limited one" — was right, and
one field that was not examined at the time proves it outright.

**`sw_power_cap_frac`, the fraction of samples at the 450 W limit**, is 93–95%
in every valid saturated BF16 run and **55.6%** in the outlier. A saturated 7B
BF16 decode on this card is power-limited essentially all the time. For 44% of
that run it was not, meaning work was not arriving fast enough to press the card
against its own limit. Everything else follows from that: higher boost clock
(2655 MHz, because there was thermal and power headroom to spend), lower mean
power, 29% worse TPOT, and a queue that grew to a mean depth of 388 waiting
requests against 209–265 everywhere else — the same arrivals, drained more
slowly.

The re-measurement restores every one of those signatures at once: 94.2% at the
cap, 426.4 W, 2460 MHz, TPOT 111.9 ms, queue 214.3, preemptions 239 — the last
two identical to E1's independent measurement of the same point. Six numbers
moving back together is what makes this a resolved artefact rather than a
lucky rerun.

The host-side cause is not identified and cannot be recovered three days later.
The session is the one in which model downloads were competing for the machine,
and the download-interference control (`download_interference.md`) shows this
machine's serving measurements are sensitive to exactly that. That is a
consistent explanation, not a demonstrated one, and it is labelled as such.

## Where the original diagnosis was wrong

It said: *"Both E2 r16 points (BF16 and AWQ) share the ~397 W signature against
~425 W for every other saturated measurement, so the cause is something about
that session rather than about one point."*

The inference was reasonable and the conclusion was half wrong. **AWQ never
reaches the power cap at all** — `sw_power_cap_frac` is 0.0% in the old run and
0.3% in the clean re-measurement, and its mean power is 413.4 W in the clean
run. Low power in the AWQ arm is intrinsic: int4 decode is memory-bound and this
card cannot spend 450 W on it. Only the BF16 arm's 396.6 W was anomalous. Two
points sharing a number does not make it the same phenomenon, and the check that
separates them — *is this arm ever power-capped when healthy?* — is one this
project now has for free in every run.

AWQ did move as well, from 1537.0 to 1607.1 tok/s (+4.6%) and 398.4 to 413.4 W,
so the session touched both arms; it just hit BF16 four times harder.

## What this changes

**Restored.** The r16 row of `e2_ablation.md` is no longer provisional. Both
arms are re-measured, in the same session, minutes apart, against one baseline.

**The saturated ablation result, now on clean same-session data:**

| | BF16 | AWQ | difference |
|---|---|---|---|
| out tok/s | 1605.9 | 1607.1 | **+0.1%** |
| J/token (raw) | 0.2794 | 0.2708 | −3.1% |
| mean power | 426.4 W | 413.4 W | −3.0% |
| running batch | 170.8 | 229.9 | +35% |
| KV utilisation | 87.2% | 47.4% | −40 pts |
| preemptions | 239 | 0 | — |

At 16 req/s offered, **AWQ and BF16 deliver the same throughput** — +0.1%
against a saturated-regime bar of 1.2%, which is parity by any reading. The two
arms get there differently: AWQ's per-token decode is 42% slower (TPOT p50
158.4 vs 111.9 ms) and it offsets that with a 35% larger running batch, which is
what its smaller weights buy in KV space.

**Still withdrawn:** "AWQ delivers +21.7% throughput over BF16 at saturation."
The clean re-measurement puts the two arms at parity here, so the original claim
was wrong by more than its own magnitude.

**Now measured twice, from two directions:** the concurrency ladder's c256 point
gave AWQ +6.8% (1709.4 vs 1600.3 tok/s) at fixed concurrency, where this
fixed-rate point gives +0.1%. Both are real and they are not in conflict — at a
fixed offered *rate* both arms are already saturated and land at the same
ceiling, while at a fixed high *concurrency* AWQ's KV headroom lets it keep more
requests genuinely resident instead of evicting them. The honest single-sentence
version is that AWQ's saturated throughput advantage on this card is between 0
and 7% depending on how load is applied, not the 22% originally claimed.

**Unaffected, and still the strongest E2 findings:** BF16 exhausts its KV cache
at ~170 concurrent and evicts 239–271 times in every saturated run measured;
AWQ holds ~230 concurrent at 47% KV with zero preemptions. Both re-measurements
reproduced these to within a point.
