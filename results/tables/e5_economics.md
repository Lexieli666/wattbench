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

Those rows assume the card is **busy at this throughput every hour of its three-year life** — the most generous possible assumption for owning. The next section prices every other case.

### Sensitivity to duty cycle

Nobody runs a 4090 flat out for three years, so the $/1M above is a best case. Duty cycle is **not a free parameter to multiply by**: at a daily output volume $V$ the card only has to run $V / (1515 \times 86400)$ of the day, so duty cycle is *determined* by $V$. Amortisation is a fixed daily cost spread over whatever the day produced:

```
owned $/1M at volume V  =  amortisation_per_day / (V/1e6)  +  electricity_per_1M
                        =  $1.916 / (V/1e6)  +  $0.023
```

Two consequences, and the second is the one that matters. The **break-even volumes below do not move** with duty cycle — they already are the statement about it, because amortisation is daily and cancels the same way on both sides. What does move is the **owned $/1M**, which is only $0.037 at full duty and rises without limit as $V$ falls:

| duty cycle | daily output volume | amortisation $/1M | electricity $/1M | owned total $/1M | vs full duty |
|------------|---------------------|-------------------|------------------|------------------|--------------|
| 100%       | 130.89M tok/day     | $0.015            | $0.023           | **$0.037**       | 1.0×         |
| 50%        | 65.45M tok/day      | $0.029            | $0.023           | **$0.052**       | 1.4×         |
| 25%        | 32.72M tok/day      | $0.059            | $0.023           | **$0.081**       | 2.2×         |
| 10%        | 13.09M tok/day      | $0.146            | $0.023           | **$0.169**       | 4.5×         |
| 1%         | 1.31M tok/day       | $1.464            | $0.023           | **$1.487**       | 40.0×        |

Electricity is the flat column: it is a marginal cost, so an idle hour costs no tokens and no dollars of it. Everything that moves is amortisation, and it moves inversely with volume — which is why a card used lightly is expensive per token no matter how efficient it is while running.

The full curve, with the API price levels drawn across it, is `results/plots/e5_cost_vs_volume.png`.

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

**Self-check.** The cost-vs-volume chart draws the same owned-cost curve and the same price levels, so where it crosses each level must be the break-even volume in this table. Read off the plotted samples by interpolation rather than from the formula, the two agree to within **0.0004%** across the 3 comparable-weight-class levels drawn — DeepInfra Meta-Llama-3.1-8B $0.12: chart 19.6636M vs table 19.6635M tok/day; DeepInfra Qwen3-32B $0.60: chart 3.3188M vs table 3.3188M tok/day; Together AI Qwen2.5-7B $1.50: chart 1.2972M vs table 1.2972M tok/day. The residual is the chart's grid resolution, not a disagreement.

The *comparable weight class* column is the one that decides whether a row means anything. A frontier model is not a substitute for a locally served 7B; those rows show the price ceiling of the market, not an available trade. The only capability evidence in this project is the E2 GSM8K guard, and it licenses no claim beyond that task.
