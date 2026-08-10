## E5 — serving economics

Quoted at the best SLO-meeting load point measured: **e1_chat_7b_bf16_r12** (512in/128out, 12 req/s offered), which sustained **1515 output tok/s**.

### Measured inputs (this machine)

| Quantity                                  | Value      | Source                                    |
|-------------------------------------------|------------|-------------------------------------------|
| Output throughput                         | 1515 tok/s | e1_chat_7b_bf16_r12__20260810T014939.json |
| Energy per output token (raw)             | 0.293 J    | e1_chat_7b_bf16_r12__20260810T014939.json |
| Energy per output token (idle-subtracted) | 0.28 J     | e1_chat_7b_bf16_r12__20260810T014939.json |
| Mean GPU power                            | 423.1 W    | e1_chat_7b_bf16_r12__20260810T014939.json |

### Sourced inputs (reported, not measured)

| Quantity              | Value              | Source                            | Dated      |
|-----------------------|--------------------|-----------------------------------|------------|
| Electricity           | $0.277/kWh         | LADWP Standard Residential (R-1A) | 2026-08-09 |
| RTX 4090 street price | $2,100             | retail/used listing trackers      | 2026-08-09 |
| Amortisation          | 3 years, card only | assumption                        | -          |

### Cost per 1M output tokens

| Line                                                                                           | $/1M output tokens | sensitivity band |
|------------------------------------------------------------------------------------------------|--------------------|------------------|
| Electricity only (marginal cost of running the card)                                           | $0.023             | $0.020 – $0.027  |
| Card amortisation at 1515 tok/s sustained                                                      | $0.015             | $0.013 – $0.022  |
| **Owned hardware, total**                                                                      | **$0.037**         | -                |
| Hypothetical rental — RunPod RTX 4090 (24GB), Community Cloud ($0.34/hr, electricity included) | $0.062             | -                |
| Hypothetical rental — RunPod RTX 4090 (24GB), Secure Cloud ($0.69/hr, electricity included)    | $0.127             | -                |

The amortisation line assumes the card is **busy at this throughput every hour of its three-year life**. That is the most generous possible assumption for owning; at 10% duty cycle the amortised figure is 10x higher ($0.15/1M).

### Break-even volume against API list prices

Formula, with $V$ the daily output-token volume:

```
owned daily cost  =  amortisation_per_day + energy_per_1M * V/1e6
api daily cost    =  api_price_per_1M     * V/1e6
break-even V*     =  amortisation_per_day * 1e6 / (api_price_per_1M - energy_per_1M)
```

Amortisation is $1.916/day; the marginal (electricity) cost of self-hosting is $0.023/1M output tokens. API prices are blended to an effective output price at the measured 512in/128out shape, because self-hosting pays for prefill too.

| API (list price, reported not measured)               | comparable weight class | effective $/1M out | break-even     | feasibility                     |
|-------------------------------------------------------|-------------------------|--------------------|----------------|---------------------------------|
| DeepInfra meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo | yes                     | $0.120             | 19.66M tok/day | 15% of the card's daily ceiling |
| DeepInfra Qwen3-14B                                   | yes                     | $0.720             | 2.75M tok/day  | 2% of the card's daily ceiling  |
| DeepInfra Qwen3-32B                                   | yes                     | $0.600             | 3.32M tok/day  | 3% of the card's daily ceiling  |
| DeepInfra Qwen2.5-72B-Instruct                        | **no**                  | $1.840             | 1.05M tok/day  | <1% of the card's daily ceiling |
| Together AI Qwen2.5-7B-Instruct-Turbo                 | yes                     | $1.500             | 1.30M tok/day  | <1% of the card's daily ceiling |
| Together AI Llama-3.1-8B-Instruct-Lite                | yes                     | $0.700             | 2.83M tok/day  | 2% of the card's daily ceiling  |
| Anthropic claude-opus-5                               | **no**                  | $45.000            | 43k tok/day    | <1% of the card's daily ceiling |
| Anthropic claude-sonnet-5                             | **no**                  | $18.000            | 107k tok/day   | <1% of the card's daily ceiling |
| Anthropic claude-haiku-4-5                            | **no**                  | $9.000             | 213k tok/day   | <1% of the card's daily ceiling |
| OpenAI gpt-5.2                                        | **no**                  | $10.500            | 183k tok/day   | <1% of the card's daily ceiling |

The *comparable weight class* column is the one that decides whether a row means anything. A frontier model is not a substitute for a locally served 7B; those rows show the price ceiling of the market, not an available trade. The only capability evidence in this project is the E2 GSM8K guard, and it licenses no claim beyond that task.
