## E4 — serving-stack comparison (Qwen2.5-7B, chat shape)

**GGUF Q4_K_M, AWQ int4 and BF16 are different weight formats.** This compares serving stacks each at its own native format, not one set of weights on several servers, so a throughput or energy gap here includes whatever the formats themselves cost. The HF + PyTorch arm serves BF16 because plain transformers has no int4 path that is still "just PyTorch"; its like-for-like format control is the vLLM BF16 reference server of E1/E2 (same weights, different stack). Everything else is held equal: same model family and size (Qwen2.5-7B-Instruct), same traffic shape (512 in / 128 out), same tokenizer, same load generator, same transport. All three arms were measured in one session, against one idle baseline (`idle__E4-session2__20261007T135656.json`), on driver 610.60: the driver change since the first E4 session moved the stacks by different amounts, so arms from different sessions are not comparable (METHODOLOGY §7b).

| concurrency | stack        | format      | status | out tok/s | vs vLLM  | TTFT p50 ms | ITL p50 ms | E2E p95 ms | mean W | J/tok | vs vLLM  |
|-------------|--------------|-------------|--------|-----------|----------|-------------|------------|------------|--------|-------|----------|
| 1           | vLLM         | AWQ int4    | served | 158.1     | baseline | 51.48       | 5.95       | 811.5      | 312.7  | 2.05  | baseline |
| 1           | llama.cpp    | GGUF Q4_K_M | served | 146.5     | -7.3%    | 84.29       | 6.06       | 907.2      | 336.5  | 2.38  | +15.7%   |
| 1           | HF + PyTorch | BF16        | served | 47.45     | -70.0%   | 57.97       | 20.2       | 3015       | 280.7  | 5.98  | +191.2%  |
| 8           | vLLM         | AWQ int4    | served | 880.4     | baseline | 319.3       | 6.22       | 1164       | 343.7  | 0.414 | baseline |
| 8           | llama.cpp    | GGUF Q4_K_M | served | 489.5     | -44.4%   | 305.5       | 12         | 2212       | 372.1  | 0.787 | +89.9%   |
| 8           | HF + PyTorch | BF16        | served | 167.4     | -81.0%   | 3188        | 21.8       | 6860       | 260.6  | 1.58  | +280.3%  |
| 32          | vLLM         | AWQ int4    | served | 1664      | baseline | 428.7       | 7.75       | 2463       | 401.7  | 0.259 | baseline |
| 32          | llama.cpp    | GGUF Q4_K_M | served | 761.1     | -54.3%   | 221.5       | 24.4       | 5359       | 248.6  | 0.338 | +30.6%   |
| 32          | HF + PyTorch | BF16        | served | 524.7     | -68.5%   | 4230        | 27.4       | 7943       | 302.1  | 0.599 | +131.6%  |

Relative columns compare each stack against vLLM at the same concurrency. Concurrency is held by the load generator (`--max-concurrency`), so it is the number of requests in flight, not an offered rate: there is no queue to grow and no SLO column, because every request is admitted as soon as a slot frees -- except on the HF + PyTorch arm, where a request admitted by the client still waits server-side for the current static batch to finish. That wait is inside its TTFT, and its queue is in the next table.

### What each stack reports about itself

| concurrency | stack        | requests running (mean) | queued (mean) | KV utilisation | preemptions | completed |
|-------------|--------------|-------------------------|---------------|----------------|-------------|-----------|
| 1           | vLLM         | 0.905                   | 0             | 0.2%           | 0           | 200/200   |
| 1           | llama.cpp    | 0.956                   | 0             | not run        | not run     | 200/200   |
| 1           | HF + PyTorch | 0.989                   | 0             | not run        | not run     | 200/200   |
| 8           | vLLM         | 6.87                    | 0             | 1.4%           | 0           | 800/800   |
| 8           | llama.cpp    | 7.24                    | 0             | not run        | not run     | 800/800   |
| 8           | HF + PyTorch | 3.89                    | 3.99          | not run        | not run     | 800/800   |
| 32          | vLLM         | 26.7                    | 0             | 5.4%           | 0           | 1600/1600 |
| 32          | llama.cpp    | 30.1                    | 0             | not run        | not run     | 1600/1600 |
| 32          | HF + PyTorch | 20                      | 10.7          | not run        | not run     | 800/800   |

`not run` in the KV and preemption columns is a difference between the stacks rather than a gap in the measurement. llama.cpp reports no preemption counter because it does not preempt: a request that finds no free slot is deferred before it starts rather than evicted after it starts. It also exports no KV-utilisation ratio, and the quantity would not mean the same thing if it did — llama.cpp partitions KV into fixed per-slot budgets where vLLM shares one pool. The HF + PyTorch arm has neither: a `DynamicCache` has no pool to be a fraction of, and a static batch is never evicted. What it does report is the **queued** column — requests the client has in flight that are waiting for the current `generate()` call to finish, which is the cost of static batching made visible.

### Memory and GPU time

| concurrency | stack        | nvidia-smi peak | torch peak alloc | weights   | GPU ms/tok (stream) | GPU busy | kernel ms/tok (profiler) |
|-------------|--------------|-----------------|------------------|-----------|---------------------|----------|--------------------------|
| 1           | vLLM         | 23.38 GiB       | not run          | not run   | not run             | not run  | not run                  |
| 1           | llama.cpp    | 6.16 GiB        | not run          | not run   | not run             | not run  | not run                  |
| 1           | HF + PyTorch | 16.22 GiB       | 14.33 GiB        | 14.19 GiB | 21                  | 98%      | 16.8 (b=1)               |
| 8           | vLLM         | 23.38 GiB       | not run          | not run   | not run             | not run  | not run                  |
| 8           | llama.cpp    | 7.69 GiB        | not run          | not run   | not run             | not run  | not run                  |
| 8           | HF + PyTorch | 17.01 GiB       | 14.91 GiB        | 14.19 GiB | 5.96                | 98%      | 2.31 (b=8)               |
| 32          | vLLM         | 23.33 GiB       | not run          | not run   | not run             | not run  | not run                  |
| 32          | llama.cpp    | 12.95 GiB       | not run          | not run   | not run             | not run  | not run                  |
| 32          | HF + PyTorch | 19.59 GiB       | 17.20 GiB        | 14.19 GiB | 1.88                | 93%      | 0.628 (b=32)             |

**The two memory columns are different instruments and read differently on purpose.** `nvidia-smi peak` is `memory.used` sampled at ~2 Hz by the power poller, for every stack; it is what the driver handed the process. For vLLM that is the pool it pre-allocates at startup (`gpu_memory_utilization` × the card), so it is nearly flat at every concurrency and says what the server *reserved*, not what it *used*. For llama.cpp it is weights plus the fixed per-slot KV budget. `torch peak alloc` is `torch.cuda.max_memory_allocated` read off the serving process itself — live tensor bytes at the high-water mark, weights and `DynamicCache` included — and only the HF + PyTorch arm can report it, because only there is the server a PyTorch process this harness can ask.

**The two GPU-time columns differ the same way.** `GPU ms/tok (stream)` is CUDA-event elapsed time around every `generate()` call in the window, divided by output tokens: the GPU cost of a token as this stack actually pays it, Python-side gaps between kernels included, with `GPU busy` the same time as a fraction of the window. `kernel ms/tok (profiler)` is self CUDA kernel time from a `torch.profiler` pass over one synthetic batch of the point's shape, run **after** the window closed so its overhead touches no measured number; `b=` is the batch it profiled. The gap between the two columns is the HF decode loop's overhead per token. Neither exists for vLLM or llama.cpp — the first is a counter only `pytorch_server.py` exports, and the second needs the profiler inside the server — so both read `not run` there rather than being estimated.

Where the HF + PyTorch arm's kernel time went in the profiled decode (self CUDA time, coarse name-based grouping): matmul 92%, elementwise 6%, reduce/index/copy 2%, norm/softmax/activation 0%, other 0%. The per-kernel top 25 is in each record's `profile.top_kernels_generate`.

**Every point completed every request it issued.** That is worth stating because it was not true on the first attempt: with HTTP connection reuse left on, llama.cpp dropped 10.9% of requests at concurrency 8 and 11.2% at concurrency 32 — see `e4_transport.md`, and the superseded runs in `results/raw/`.

### Quality guard — GSM8K exact match, n=200

| stack        | format      | correct | exact match | 95% Wilson interval | request errors |
|--------------|-------------|---------|-------------|---------------------|----------------|
| vLLM         | AWQ int4    | 182/200 | 91.0%       | 86.2% – 94.2%       | 0              |
| llama.cpp    | GGUF Q4_K_M | 180/200 | 90.0%       | 85.1% – 93.4%       | 0              |
| HF + PyTorch | BF16        | 180/200 | 90.0%       | 85.1% – 93.4%       | 0              |

vLLM/AWQ minus llama.cpp/GGUF is **+1.0 points**, 95% interval on the difference -4.9 to +6.9 points (Newcombe score method). The gap is **within noise** at n=200.

vLLM/AWQ minus HF + PyTorch/BF16 is **+1.0 points**, 95% interval on the difference -4.9 to +6.9 points (Newcombe score method). The gap is **within noise** at n=200.

Plainly: **this guard finds no quality difference between the arms**, which is the useful outcome for a reader choosing a stack — it means E4's throughput and energy result can be read as a serving recommendation rather than a speed-for-accuracy trade. It is not proof of parity: an interval that contains zero also contains everything else inside it.

**The caveat that governs this table is the same one that governs the rest of E4: these are different weight formats.** A quality difference here is a property of the format-plus-stack pair, not of the batching implementation — GGUF Q4_K_M and AWQ quantize different tensors to different group sizes, and BF16 quantizes nothing at all, and nothing in this project separates the format's contribution from the server's.

**Cross-check against the E2 guard.** The same AWQ checkpoint was measured at n=50 in E2 and scored 92.0% (81.2% – 96.8%), against 91.0% here — a different subset draw, different serving flags and a different session, consistent at -7.8 to +10.3 points. That is a check on the guard itself, not a second result: the two are not pooled.

Sample sizes are stated wherever these numbers appear beside E2's, because they are not the same measurement: E2's guard is n=50 and its subset is a different draw from this one, so the two are separate row sets and are never pooled. The vLLM and llama.cpp arms were measured in one session and the HF + PyTorch arm in a later one; all three greedy, through the same client and the same `Connection: close` transport the E4 serving runs used.
