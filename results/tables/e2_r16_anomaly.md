# E2 BF16 r16 is an outlier — the +21.7% saturated throughput claim is withdrawn

Five measurements of Qwen2.5-7B BF16, chat shape, 16 req/s offered:

| run | tok/s | mean W | SM clock p50 | temp max | batch | preempt |
|---|---|---|---|---|---|---|
| E0-sat A | 1535.1 | 425.8 | 2505 MHz | 79 °C | 167.6 | 245 |
| E0-sat B | 1568.8 | 425.7 | — | — | 169.2 | 254 |
| E0-sat C | 1538.4 | 422.9 | — | — | 165.6 | 262 |
| E1 chat r16 | 1613.7 | 426.7 | 2445 MHz | 78 °C | 170.0 | 239 |
| **E2 bf16 r16** | **1262.4** | **396.6** | **2655 MHz** | 78 °C | 169.5 | 271 |

The saturated-regime variance bar, measured from the three E0-sat repeats, is
**1.20% CV on throughput**. The E2 point sits 18% below the four-run consensus
of ~1564 tok/s — roughly 15 sigma. It is not noise.

It is also not throttling: the outlier ran at a *higher* SM clock (2655 vs
2445–2505 MHz) with *lower* power and no clock sag, at the same temperature.
Higher clock, less power, less work done is an underfed GPU, not a limited one.
Both E2 r16 points (BF16 and AWQ) share the ~397 W signature against ~425 W for
every other saturated measurement, so the cause is something about that session
rather than about one point.

## What this changes

**Withdrawn:** "AWQ delivers +21.7% throughput over BF16 at saturation." That
figure divided AWQ's r16 by a BF16 denominator that is 18% too low. Against the
four-run BF16 consensus, AWQ's 1537.0 tok/s is parity, not a 22% win.

**Stands, from a same-session comparison:** the concurrency ladder's c256 point
measured both arms minutes apart under identical conditions — AWQ 1709.4 vs
BF16 1600.3 tok/s, **+6.8%**. That is the defensible saturated throughput
number.

**Unaffected:** the preemption and KV-headroom results, which are the strongest
findings in E2 and do not depend on the anomalous point. BF16 exhausts its KV
cache at ~170 concurrent and evicts 239–271 times across every saturated run
measured; AWQ holds 221.9 concurrent at 57% KV with zero preemptions.

## Outstanding

E2's BF16 and AWQ r16 points are queued for re-measurement once the GPU is free
of downloads. Until then the r16 row of the ablation table should be read as
provisional, and the c256 ladder point used in its place.
