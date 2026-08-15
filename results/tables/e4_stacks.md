## E4 — serving-stack comparison (Qwen2.5-7B int4, chat shape)

**GGUF Q4_K_M and AWQ int4 are different quantization formats.** This compares two serving stacks each at its own native int4, not one set of weights on two servers, so a throughput or energy gap here includes whatever the formats themselves cost. Everything else is held equal: same model family and size (Qwen2.5-7B-Instruct), same traffic shape (512 in / 128 out), same tokenizer, same load generator, same idle baseline, same session.

| concurrency | stack     | format      | status | out tok/s | vs vLLM  | TTFT p50 ms | ITL p50 ms | E2E p95 ms | mean W | J/tok | vs vLLM  |
|-------------|-----------|-------------|--------|-----------|----------|-------------|------------|------------|--------|-------|----------|
| 1           | vLLM      | AWQ int4    | served | 148.9     | baseline | 57.9        | 6.02       | 863.1      | 300    | 2.08  | baseline |
| 1           | llama.cpp | GGUF Q4_K_M | served | 145.6     | -2.2%    | 66.44       | 6.21       | 892.3      | 335.6  | 2.38  | +14.4%   |
| 8           | vLLM      | AWQ int4    | served | 820.2     | baseline | 355.1       | 6.34       | 1255       | 325.4  | 0.419 | baseline |
| 8           | llama.cpp | GGUF Q4_K_M | served | 482.8     | -41.1%   | 307.4       | 12.5       | 2167       | 367.8  | 0.791 | +88.5%   |
| 32          | vLLM      | AWQ int4    | served | 1542      | baseline | 469.4       | 8.21       | 2663       | 381.1  | 0.262 | baseline |
| 32          | llama.cpp | GGUF Q4_K_M | served | 755.6     | -51.0%   | 221.5       | 24.7       | 5490       | 247.1  | 0.339 | +29.1%   |

Relative columns compare llama.cpp against vLLM at the same concurrency. Concurrency is held by the load generator (`--max-concurrency`), so it is the number of requests in flight, not an offered rate: there is no queue to grow and no SLO column, because every request is admitted as soon as a slot frees.

### What each stack reports about itself

| concurrency | stack     | requests running (mean) | KV utilisation | preemptions | completed |
|-------------|-----------|-------------------------|----------------|-------------|-----------|
| 1           | vLLM      | 0.933                   | 0.2%           | 0           | 200/200   |
| 1           | llama.cpp | 0.956                   | not run        | not run     | 200/200   |
| 8           | vLLM      | 6.62                    | 1.3%           | 0           | 800/800   |
| 8           | llama.cpp | 7.31                    | not run        | not run     | 800/800   |
| 32          | vLLM      | 26.7                    | 5.4%           | 0           | 1600/1600 |
| 32          | llama.cpp | 30.1                    | not run        | not run     | 1600/1600 |

`not run` in the last two columns is a difference between the stacks rather than a gap in the measurement. llama.cpp reports no preemption counter because it does not preempt: a request that finds no free slot is deferred before it starts rather than evicted after it starts. It also exports no KV-utilisation ratio, and the quantity would not mean the same thing if it did — llama.cpp partitions KV into fixed per-slot budgets where vLLM shares one pool.

**Every point completed every request it issued.** That is worth stating because it was not true on the first attempt: with HTTP connection reuse left on, llama.cpp dropped 10.9% of requests at concurrency 8 and 11.2% at concurrency 32 — see `e4_transport.md`, and the superseded runs in `results/raw/`.
