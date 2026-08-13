| point                         | exp            | status                 | rate    | req/s   | out tok/s | TTFT p95 ms | J/tok   | sanity | flags |
|-------------------------------|----------------|------------------------|---------|---------|-----------|-------------|---------|--------|-------|
| e0_ref_7b_bf16_chat_r4_a      | E0             | ok                     | 4       | 3.96    | 506.5     | 155.2       | 0.701   | PASS   | -     |
| e0_ref_7b_bf16_chat_r4_b      | E0             | ok                     | 4       | 3.96    | 506.5     | 155.9       | 0.701   | PASS   | -     |
| e0_ref_7b_bf16_chat_r4_c      | E0             | ok                     | 4       | 3.96    | 506.6     | 156.7       | 0.703   | PASS   | -     |
| e0_ref_7b_bf16_chat_r4_greedy | E0-greedy      | ok                     | 4       | 3.96    | 506.5     | 156.6       | 0.701   | PASS   | -     |
| e0_sat_7b_bf16_chat_r16_a     | E0-sat         | ok                     | 16      | 12      | 1535      | 4.372e+04   | 0.292   | PASS   | -     |
| e0_sat_7b_bf16_chat_r16_b     | E0-sat         | ok                     | 16      | 12.3    | 1569      | 3.927e+04   | 0.286   | PASS   | -     |
| e0_sat_7b_bf16_chat_r16_c     | E0-sat         | ok                     | 16      | 12      | 1538      | 4.328e+04   | 0.29    | PASS   | -     |
| e1_chat_7b_bf16_r0p5          | E1             | ok                     | 0.5     | 0.497   | 63.66     | 291         | 3.67    | PASS   | -     |
| e1_chat_7b_bf16_r12           | E1             | ok                     | 12      | 11.8    | 1515      | 432.3       | 0.293   | PASS   | -     |
| e1_chat_7b_bf16_r16           | E1             | ok                     | 16      | 12.6    | 1614      | 3.463e+04   | 0.277   | PASS   | -     |
| e1_chat_7b_bf16_r1            | E1             | ok                     | 1       | 0.991   | 126.9     | 124.2       | 2.28    | PASS   | -     |
| e1_chat_7b_bf16_r24           | E1             | ok                     | 24      | 12.7    | 1619      | 8.097e+04   | 0.277   | PASS   | -     |
| e1_chat_7b_bf16_r2            | E1             | ok                     | 2       | 1.98    | 253.2     | 127.1       | 1.28    | PASS   | -     |
| e1_chat_7b_bf16_r32           | E1             | ok                     | 32      | 12.4    | 1591      | 1.075e+05   | 0.283   | PASS   | -     |
| e1_chat_7b_bf16_r4            | E1             | ok                     | 4       | 3.96    | 506.5     | 158.3       | 0.702   | PASS   | -     |
| e1_chat_7b_bf16_r8            | E1             | ok                     | 8       | 7.9     | 1012      | 247.2       | 0.399   | FAIL   | -     |
| e1_chat_7b_bf16_r8_dltest     | E1-dltest      | ok                     | 8       | 7.89    | 1010      | 4327        | 0.349   | PASS   | -     |
| e1_chat_7b_bf16_r8_quiet      | E1-dltest      | ok                     | 8       | 7.82    | 1001      | 470.8       | 0.375   | PASS   | -     |
| e1_rag_7b_bf16_r0p5           | E1             | ok                     | 0.5     | 0.494   | 126.5     | 419.5       | 2.33    | PASS   | -     |
| e1_rag_7b_bf16_r1             | E1             | ok                     | 1       | 0.977   | 250.1     | 429.5       | 1.34    | PASS   | -     |
| e1_rag_7b_bf16_r2             | E1             | ok                     | 2       | 1.95    | 498.9     | 725         | 0.772   | PASS   | -     |
| e1_rag_7b_bf16_r3             | E1             | ok                     | 3       | 2.76    | 707.7     | 1.395e+04   | 0.599   | PASS   | -     |
| e1_rag_7b_bf16_r4             | E1             | ok                     | 4       | 2.81    | 720.2     | 7.452e+04   | 0.591   | PASS   | -     |
| e1_rag_7b_bf16_r6             | E1             | ok                     | 6       | 2.85    | 728.6     | 2.069e+05   | 0.586   | PASS   | -     |
| e1_rag_7b_bf16_r8             | E1             | ok                     | 8       | 2.91    | 745.6     | 3.265e+05   | 0.572   | PASS   | -     |
| e2_awq_7b_chat_r16            | E2             | ok                     | 16      | 12.6    | 1607      | 3.646e+04   | 0.271   | PASS   | -     |
| e2_awq_7b_chat_r2             | E2             | ok                     | 2       | 1.99    | 254.9     | 109.4       | 1.08    | PASS   | -     |
| e2_awq_7b_chat_r8             | E2             | ok                     | 8       | 7.96    | 1019      | 191.2       | 0.374   | PASS   | -     |
| e2_bf16_7b_chat_r16           | E2             | ok                     | 16      | 12.5    | 1606      | 3.544e+04   | 0.279   | PASS   | -     |
| e2_bf16_7b_chat_r2            | E2             | ok                     | 2       | 1.98    | 253.2     | 129.8       | 1.28    | PASS   | -     |
| e2_bf16_7b_chat_r8            | E2             | ok                     | 8       | 7.9     | 1011      | 386.3       | 0.383   | PASS   | -     |
| e2_conc_awq_c128              | E2-concurrency | ok                     | inf     | 14.7    | 1886      | 4113        | 0.23    | PASS   | -     |
| e2_conc_awq_c256              | E2-concurrency | ok                     | inf     | 13.4    | 1709      | 7983        | 0.255   | PASS   | -     |
| e2_conc_awq_c32               | E2-concurrency | ok                     | inf     | 12.6    | 1618      | 1277        | 0.251   | PASS   | -     |
| e2_conc_awq_c64               | E2-concurrency | ok                     | inf     | 13.6    | 1743      | 2397        | 0.234   | PASS   | -     |
| e2_conc_bf16_c128             | E2-concurrency | ok                     | inf     | 13.3    | 1699      | 4309        | 0.262   | PASS   | -     |
| e2_conc_bf16_c256             | E2-concurrency | ok                     | inf     | 12.5    | 1600      | 1.033e+04   | 0.281   | PASS   | -     |
| e2_conc_bf16_c32              | E2-concurrency | ok                     | inf     | 8.22    | 1052      | 1203        | 0.369   | PASS   | -     |
| e2_conc_bf16_c64              | E2-concurrency | ok                     | inf     | 11.4    | 1465      | 2263        | 0.282   | PASS   | -     |
| e2_gptq_7b_chat_r8            | E2             | ok                     | 8       | 7.96    | 1019      | 191.6       | 0.369   | PASS   | -     |
| e3_14b_awq_chat_r4            | E3             | ok                     | 4       | 3.97    | 507.7     | 368.4       | 0.752   | PASS   | -     |
| e3_1p5b_awq_chat_r4           | E3             | ok                     | 4       | 3.99    | 511.1     | 42.93       | 0.371   | PASS   | -     |
| e3_32b_awq_chat_r4            | E3             | server_failed_to_start | 4       | not run | not run   | not run     | not run | ?      | -     |
| e3_32b_awq_chat_r4_ctx1024    | E3             | ok                     | 4       | 0.859   | 109.9     | 6.93e+05    | 3.4     | PASS   | -     |
| e3_3b_awq_chat_r4             | E3             | ok                     | 4       | 3.99    | 510.6     | 58.88       | 0.461   | PASS   | -     |
| e3_7b_awq_chat_r4             | E3             | ok                     | 4       | 3.98    | 509.8     | 111.3       | 0.65    | PASS   | -     |
| e3_7b_awq_chat_r4_recheck     | E3-recheck     | ok                     | 4       | 3.98    | 509.8     | 113.1       | 0.65    | PASS   | -     |
| gsm8k_awq                     | E2-guard       | ok                     | not run | not run | not run   | not run     | not run | ?      | -     |
| gsm8k_bf16                    | E2-guard       | ok                     | not run | not run | not run   | not run     | not run | ?      | -     |
| gsm8k_gptq                    | E2-guard       | ok                     | not run | not run | not run   | not run     | not run | ?      | -     |
| limits_context_14b            | E2-limits      | ok                     | not run | not run | not run   | not run     | not run | ?      | -     |
| limits_context_32b            | E2-limits      | no_context_served      | not run | not run | not run   | not run     | not run | ?      | -     |
| limits_context_32b_fine       | E2-limits      | ok                     | not run | not run | not run   | not run     | not run | ?      | -     |
| limits_context_awq            | E2-limits      | ok                     | not run | not run | not run   | not run     | not run | ?      | -     |
| limits_context_bf16           | E2-limits      | ok                     | not run | not run | not run   | not run     | not run | ?      | -     |
| smoke_qwen1_5b_chat_r4        | smoke          | ok                     | 4       | 3.92    | 501.5     | 3139        | 0.428   | PASS   | -     |
