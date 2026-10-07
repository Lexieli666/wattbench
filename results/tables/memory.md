## Peak GPU memory, every served run

`nvidia-smi peak` is `memory.used` over the measurement window, sampled at ~2 Hz by the power poller (`mem_used_mib_max` in each run's `.power.json`), so it exists for every run here whether or not the run's record carries a `gpu_memory` section — runs before 2026-10 do not, and the column is derived at read time rather than by rewriting them (`derived` marks those). **For vLLM it is the pool the server reserved at startup, not live use**: vLLM allocates `gpu_memory_utilization` × the card up front and fills its KV cache inside that, so the figure is flat across load and says nothing about how much KV a point actually used — `KV utilisation` in the experiment tables is that number. `torch peak alloc` is `torch.cuda.max_memory_allocated` from the serving process, live tensor bytes at the high-water mark; only the HF + PyTorch arm reports it.

### E0

| point                    | stack | model               | precision | max ctx | nvidia-smi peak GiB | torch peak alloc GiB | source  |
|--------------------------|-------|---------------------|-----------|---------|---------------------|----------------------|---------|
| e0_ref_7b_bf16_chat_r4_a | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.33               | not run              | derived |
| e0_ref_7b_bf16_chat_r4_b | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.32               | not run              | derived |
| e0_ref_7b_bf16_chat_r4_c | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.32               | not run              | derived |

### E0-greedy

| point                         | stack | model               | precision | max ctx | nvidia-smi peak GiB | torch peak alloc GiB | source  |
|-------------------------------|-------|---------------------|-----------|---------|---------------------|----------------------|---------|
| e0_ref_7b_bf16_chat_r4_greedy | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.32               | not run              | derived |

### E0-sat

| point                     | stack | model               | precision | max ctx | nvidia-smi peak GiB | torch peak alloc GiB | source  |
|---------------------------|-------|---------------------|-----------|---------|---------------------|----------------------|---------|
| e0_sat_7b_bf16_chat_r16_a | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.34               | not run              | derived |
| e0_sat_7b_bf16_chat_r16_b | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.41               | not run              | derived |
| e0_sat_7b_bf16_chat_r16_c | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.37               | not run              | derived |

### E1

| point                | stack | model               | precision | max ctx | nvidia-smi peak GiB | torch peak alloc GiB | source  |
|----------------------|-------|---------------------|-----------|---------|---------------------|----------------------|---------|
| e1_chat_7b_bf16_r0p5 | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.38               | not run              | derived |
| e1_chat_7b_bf16_r1   | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.35               | not run              | derived |
| e1_chat_7b_bf16_r12  | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.35               | not run              | derived |
| e1_chat_7b_bf16_r16  | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.35               | not run              | derived |
| e1_chat_7b_bf16_r2   | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.35               | not run              | derived |
| e1_chat_7b_bf16_r24  | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.39               | not run              | derived |
| e1_chat_7b_bf16_r32  | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.36               | not run              | derived |
| e1_chat_7b_bf16_r4   | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.35               | not run              | derived |
| e1_chat_7b_bf16_r8   | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.35               | not run              | derived |
| e1_rag_7b_bf16_r0p5  | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.20               | not run              | derived |
| e1_rag_7b_bf16_r1    | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.15               | not run              | derived |
| e1_rag_7b_bf16_r2    | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.15               | not run              | derived |
| e1_rag_7b_bf16_r3    | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.15               | not run              | derived |
| e1_rag_7b_bf16_r4    | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.19               | not run              | derived |
| e1_rag_7b_bf16_r6    | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.26               | not run              | derived |
| e1_rag_7b_bf16_r8    | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.28               | not run              | derived |

### E1-dltest

| point                     | stack | model               | precision | max ctx | nvidia-smi peak GiB | torch peak alloc GiB | source  |
|---------------------------|-------|---------------------|-----------|---------|---------------------|----------------------|---------|
| e1_chat_7b_bf16_r8_dltest | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.42               | not run              | derived |
| e1_chat_7b_bf16_r8_quiet  | vLLM  | Qwen2.5-7B-Instruct | bf16      | 4096    | 23.32               | not run              | derived |

### E2

| point               | stack | model                         | precision | max ctx | nvidia-smi peak GiB | torch peak alloc GiB | source  |
|---------------------|-------|-------------------------------|-----------|---------|---------------------|----------------------|---------|
| e2_awq_7b_chat_r16  | vLLM  | Qwen2.5-7B-Instruct-AWQ       | awq       | 4096    | 23.31               | not run              | derived |
| e2_awq_7b_chat_r2   | vLLM  | Qwen2.5-7B-Instruct-AWQ       | awq       | 4096    | 23.41               | not run              | derived |
| e2_awq_7b_chat_r8   | vLLM  | Qwen2.5-7B-Instruct-AWQ       | awq       | 4096    | 22.78               | not run              | derived |
| e2_bf16_7b_chat_r16 | vLLM  | Qwen2.5-7B-Instruct           | bf16      | 4096    | 23.25               | not run              | derived |
| e2_bf16_7b_chat_r2  | vLLM  | Qwen2.5-7B-Instruct           | bf16      | 4096    | 23.41               | not run              | derived |
| e2_bf16_7b_chat_r8  | vLLM  | Qwen2.5-7B-Instruct           | bf16      | 4096    | 23.45               | not run              | derived |
| e2_gptq_7b_chat_r8  | vLLM  | Qwen2.5-7B-Instruct-GPTQ-Int4 | gptq      | 4096    | 23.07               | not run              | derived |

### E2-concurrency

| point             | stack | model                   | precision | max ctx | nvidia-smi peak GiB | torch peak alloc GiB | source  |
|-------------------|-------|-------------------------|-----------|---------|---------------------|----------------------|---------|
| e2_conc_awq_c128  | vLLM  | Qwen2.5-7B-Instruct-AWQ | awq       | 4096    | 23.17               | not run              | derived |
| e2_conc_awq_c256  | vLLM  | Qwen2.5-7B-Instruct-AWQ | awq       | 4096    | 23.34               | not run              | derived |
| e2_conc_awq_c32   | vLLM  | Qwen2.5-7B-Instruct-AWQ | awq       | 4096    | 23.35               | not run              | derived |
| e2_conc_awq_c64   | vLLM  | Qwen2.5-7B-Instruct-AWQ | awq       | 4096    | 23.30               | not run              | derived |
| e2_conc_bf16_c128 | vLLM  | Qwen2.5-7B-Instruct     | bf16      | 4096    | 23.32               | not run              | derived |
| e2_conc_bf16_c256 | vLLM  | Qwen2.5-7B-Instruct     | bf16      | 4096    | 23.28               | not run              | derived |
| e2_conc_bf16_c32  | vLLM  | Qwen2.5-7B-Instruct     | bf16      | 4096    | 23.18               | not run              | derived |
| e2_conc_bf16_c64  | vLLM  | Qwen2.5-7B-Instruct     | bf16      | 4096    | 23.18               | not run              | derived |

### E2-fp8

| point          | stack | model               | precision | max ctx | nvidia-smi peak GiB | torch peak alloc GiB | source  |
|----------------|-------|---------------------|-----------|---------|---------------------|----------------------|---------|
| fp8_7b_chat_r4 | vLLM  | Qwen2.5-7B-Instruct | fp8       | 4096    | 23.40               | not run              | derived |

### E3

| point                      | stack | model                     | precision | max ctx | nvidia-smi peak GiB | torch peak alloc GiB | source  |
|----------------------------|-------|---------------------------|-----------|---------|---------------------|----------------------|---------|
| e3_14b_awq_chat_r4         | vLLM  | Qwen2.5-14B-Instruct-AWQ  | awq       | 4096    | 22.68               | not run              | derived |
| e3_1p5b_awq_chat_r4        | vLLM  | Qwen2.5-1.5B-Instruct-AWQ | awq       | 4096    | 23.38               | not run              | derived |
| e3_32b_awq_chat_r4_ctx1024 | vLLM  | Qwen2.5-32B-Instruct-AWQ  | awq       | 1024    | 22.05               | not run              | derived |
| e3_3b_awq_chat_r4          | vLLM  | Qwen2.5-3B-Instruct-AWQ   | awq       | 4096    | 23.39               | not run              | derived |
| e3_7b_awq_chat_r4          | vLLM  | Qwen2.5-7B-Instruct-AWQ   | awq       | 4096    | 23.34               | not run              | derived |

### E3-recheck

| point                     | stack | model                   | precision | max ctx | nvidia-smi peak GiB | torch peak alloc GiB | source  |
|---------------------------|-------|-------------------------|-----------|---------|---------------------|----------------------|---------|
| e3_7b_awq_chat_r4_recheck | vLLM  | Qwen2.5-7B-Instruct-AWQ | awq       | 4096    | 22.99               | not run              | derived |

### E4

| point                | stack        | model                    | precision   | max ctx | nvidia-smi peak GiB | torch peak alloc GiB | source   |
|----------------------|--------------|--------------------------|-------------|---------|---------------------|----------------------|----------|
| e4_llamacpp_gguf_c1  | llama.cpp    | Qwen2.5-7B-Instruct-GGUF | GGUF Q4_K_M | 4096    | 6.16                | not run              | recorded |
| e4_llamacpp_gguf_c32 | llama.cpp    | Qwen2.5-7B-Instruct-GGUF | GGUF Q4_K_M | 4096    | 12.95               | not run              | recorded |
| e4_llamacpp_gguf_c8  | llama.cpp    | Qwen2.5-7B-Instruct-GGUF | GGUF Q4_K_M | 4096    | 7.69                | not run              | recorded |
| e4_pytorch_bf16_c1   | HF + PyTorch | Qwen2.5-7B-Instruct      | BF16        | 4096    | 16.22               | 14.33                | recorded |
| e4_pytorch_bf16_c32  | HF + PyTorch | Qwen2.5-7B-Instruct      | BF16        | 4096    | 19.59               | 17.20                | recorded |
| e4_pytorch_bf16_c8   | HF + PyTorch | Qwen2.5-7B-Instruct      | BF16        | 4096    | 17.01               | 14.91                | recorded |
| e4_vllm_awq_c1       | vLLM         | Qwen2.5-7B-Instruct-AWQ  | awq         | 4096    | 23.38               | not run              | recorded |
| e4_vllm_awq_c32      | vLLM         | Qwen2.5-7B-Instruct-AWQ  | awq         | 4096    | 23.33               | not run              | recorded |
| e4_vllm_awq_c8       | vLLM         | Qwen2.5-7B-Instruct-AWQ  | awq         | 4096    | 23.38               | not run              | recorded |

### E4-control

| point                       | stack     | model                    | precision   | max ctx | nvidia-smi peak GiB | torch peak alloc GiB | source  |
|-----------------------------|-----------|--------------------------|-------------|---------|---------------------|----------------------|---------|
| e1_chat_7b_bf16_r4_noreuse  | vLLM      | Qwen2.5-7B-Instruct      | bf16        | 4096    | 23.40               | not run              | derived |
| e4_llamacpp_gguf_c8_noreuse | llama.cpp | Qwen2.5-7B-Instruct-GGUF | GGUF Q4_K_M | 4096    | 7.88                | not run              | derived |
| e4_vllm_awq_c32_seqs512     | vLLM      | Qwen2.5-7B-Instruct-AWQ  | awq         | 4096    | 23.37               | not run              | derived |

### smoke

| point                  | stack     | model                    | precision   | max ctx | nvidia-smi peak GiB | torch peak alloc GiB | source  |
|------------------------|-----------|--------------------------|-------------|---------|---------------------|----------------------|---------|
| smoke_llamacpp_gguf_c4 | llama.cpp | Qwen2.5-7B-Instruct-GGUF | GGUF Q4_K_M | 4096    | 7.12                | not run              | derived |
| smoke_qwen1_5b_chat_r4 | vLLM      | Qwen2.5-1.5B-Instruct    | bf16        | 4096    | 16.34               | not run              | derived |

61 run(s) — the newest run of every point that served (`--include-superseded` adds the earlier runs of re-run points) — 52 of them with the figure derived at read time from the committed power log; none rewritten.
