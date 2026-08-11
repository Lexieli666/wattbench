## E2 — quantization ablation (Qwen2.5-7B, chat shape)

| req/s | arm  | out tok/s | vs BF16  | TTFT p95 ms | J/tok | vs BF16  | mean W | SLO attain |
|-------|------|-----------|----------|-------------|-------|----------|--------|------------|
| 2     | bf16 | 253.2     | baseline | 129.8       | 1.28  | baseline | 312    | 1.000      |
| 2     | awq  | 254.9     | +0.7%    | 109.4       | 1.08  | -15.6%   | 265.6  | 1.000      |
| 8     | bf16 | 1011      | baseline | 386.3       | 0.383 | baseline | 370.9  | 1.000      |
| 8     | awq  | 1019      | +0.8%    | 191.2       | 0.374 | -2.2%    | 364.6  | 1.000      |
| 8     | gptq | 1019      | +0.8%    | 191.6       | 0.369 | -3.6%    | 358.6  | 1.000      |
| 16    | bf16 | 1262      | baseline | 8.295e+04   | 0.328 | baseline | 396.6  | 0.000      |
| 16    | awq  | 1537      | +21.7%   | 4.432e+04   | 0.272 | -17.0%   | 398.4  | 0.008      |

Relative columns compare against the BF16 arm at the same offered rate. Per E0, throughput differences below ~0.4% and energy differences below ~0.4% are inside run-to-run noise.

### Quality guard — GSM8K exact match

| arm  | correct | exact match | 95% Wilson interval | request errors |
|------|---------|-------------|---------------------|----------------|
| bf16 | 48/50   | 96.0%       | 86.5% – 98.9%       | 0              |
| awq  | 46/50   | 92.0%       | 81.2% – 96.8%       | 0              |
| gptq | 45/50   | 90.0%       | 78.6% – 95.7%       | 0              |

**50 items. The intervals overlap unless the gap is large**, so this guard can only rule out a big quality collapse from quantization — it cannot certify parity. It says nothing about any capability other than grade-school arithmetic word problems.

### Longest servable context

| arm      | longest servable context | KV cache tokens there | why the next rung failed                                                                       |
|----------|--------------------------|-----------------------|------------------------------------------------------------------------------------------------|
| 14b      | 3.28e+04                 | 53,632                | server exited during startup                                                                   |
| 32b      | none served              | -                     | RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {} |
| 32b_fine | 512                      | 4,432                 | RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {} |
| awq      | 3.28e+04                 | 273,840               | server exited during startup                                                                   |
| bf16     | 3.28e+04                 | 105,328               | server exited during startup                                                                   |
