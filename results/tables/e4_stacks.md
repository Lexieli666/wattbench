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

Relative columns compare each stack against vLLM at the same concurrency. Concurrency is held by the load generator (`--max-concurrency`), so it is the number of requests in flight, not an offered rate: there is no queue to grow and no SLO column, because every request is admitted as soon as a slot frees.

### What each stack reports about itself

| concurrency | stack     | requests running (mean) | queued (mean) | KV utilisation | preemptions | completed |
|-------------|-----------|-------------------------|---------------|----------------|-------------|-----------|
| 1           | vLLM      | 0.933                   | 0             | 0.2%           | 0           | 200/200   |
| 1           | llama.cpp | 0.956                   | 0             | not run        | not run     | 200/200   |
| 8           | vLLM      | 6.62                    | 0             | 1.3%           | 0           | 800/800   |
| 8           | llama.cpp | 7.31                    | 0             | not run        | not run     | 800/800   |
| 32          | vLLM      | 26.7                    | 0.0141        | 5.4%           | 0           | 1600/1600 |
| 32          | llama.cpp | 30.1                    | 0             | not run        | not run     | 1600/1600 |

`not run` in the KV and preemption columns is a difference between the stacks rather than a gap in the measurement. llama.cpp reports no preemption counter because it does not preempt: a request that finds no free slot is deferred before it starts rather than evicted after it starts. It also exports no KV-utilisation ratio, and the quantity would not mean the same thing if it did — llama.cpp partitions KV into fixed per-slot budgets where vLLM shares one pool.

### Memory and GPU time

| concurrency | stack     | nvidia-smi peak | torch peak alloc | weights | GPU ms/tok (stream) | GPU busy | kernel ms/tok (profiler) |
|-------------|-----------|-----------------|------------------|---------|---------------------|----------|--------------------------|
| 1           | vLLM      | 23.40 GiB       | not run          | not run | not run             | not run  | not run                  |
| 1           | llama.cpp | 6.32 GiB        | not run          | not run | not run             | not run  | not run                  |
| 8           | vLLM      | 23.38 GiB       | not run          | not run | not run             | not run  | not run                  |
| 8           | llama.cpp | 7.86 GiB        | not run          | not run | not run             | not run  | not run                  |
| 32          | vLLM      | 23.38 GiB       | not run          | not run | not run             | not run  | not run                  |
| 32          | llama.cpp | 13.11 GiB       | not run          | not run | not run             | not run  | not run                  |

**The two memory columns are different instruments and read differently on purpose.** `nvidia-smi peak` is `memory.used` sampled at ~2 Hz by the power poller, for every stack; it is what the driver handed the process. For vLLM that is the pool it pre-allocates at startup (`gpu_memory_utilization` × the card), so it is nearly flat at every concurrency and says what the server *reserved*, not what it *used*. For llama.cpp it is weights plus the fixed per-slot KV budget. `torch peak alloc` is `torch.cuda.max_memory_allocated` read off the serving process itself — live tensor bytes at the high-water mark, weights and `DynamicCache` included — and only the HF + PyTorch arm can report it, because only there is the server a PyTorch process this harness can ask.

**The two GPU-time columns differ the same way.** `GPU ms/tok (stream)` is CUDA-event elapsed time around every `generate()` call in the window, divided by output tokens: the GPU cost of a token as this stack actually pays it, Python-side gaps between kernels included, with `GPU busy` the same time as a fraction of the window. `kernel ms/tok (profiler)` is self CUDA kernel time from a `torch.profiler` pass over one synthetic batch of the point's shape, run **after** the window closed so its overhead touches no measured number; `b=` is the batch it profiled. The gap between the two columns is the HF decode loop's overhead per token. Neither exists for vLLM or llama.cpp — the first is a counter only `pytorch_server.py` exports, and the second needs the profiler inside the server — so both read `not run` there rather than being estimated.

**Every point completed every request it issued.** That is worth stating because it was not true on the first attempt: with HTTP connection reuse left on, llama.cpp dropped 10.9% of requests at concurrency 8 and 11.2% at concurrency 32 — see `e4_transport.md`, and the superseded runs in `results/raw/`.

### Quality guard — GSM8K exact match, n=200

| stack        | format      | correct | exact match | 95% Wilson interval | request errors |
|--------------|-------------|---------|-------------|---------------------|----------------|
| vLLM         | AWQ int4    | 182/200 | 91.0%       | 86.2% – 94.2%       | 0              |
| llama.cpp    | GGUF Q4_K_M | 180/200 | 90.0%       | 85.1% – 93.4%       | 0              |
| HF + PyTorch | BF16        | not run | not run     | not run             | not run        |

vLLM/AWQ minus llama.cpp/GGUF is **+1.0 points**, 95% interval on the difference -4.9 to +6.9 points (Newcombe score method). The gap is **within noise** at n=200.

Plainly: **this guard finds no quality difference between the two arms**, which is the useful outcome for a reader choosing a stack — it means E4's throughput and energy result can be read as a serving recommendation rather than a speed-for-accuracy trade. It is not proof of parity: an interval that contains zero also contains everything else inside it.

**The caveat that governs this table is the same one that governs the rest of E4: these are different int4 formats.** A quality difference here is a property of the format-plus-stack pair, not of the batching implementation — GGUF Q4_K_M and AWQ quantize different tensors to different group sizes, and nothing in this project separates the format's contribution from the server's.

**Cross-check against the E2 guard.** The same AWQ checkpoint was measured at n=50 in E2 and scored 92.0% (81.2% – 96.8%), against 91.0% here — a different subset draw, different serving flags and a different session, consistent at -7.8 to +10.3 points. That is a check on the guard itself, not a second result: the two are not pooled.

Sample sizes are stated wherever these numbers appear beside E2's, because they are not the same measurement: E2's guard is n=50 and its subset is a different draw from this one, so the two are separate row sets and are never pooled. Both arms here were measured in one session, greedy, through the same client and the same `Connection: close` transport the E4 serving runs used.
