#!/usr/bin/env python3
"""Check that the numbers quoted in the README trace back to raw data.

  ~/wattbench-venv/bin/python ./verify_readme.py

Every claim here is checked against a raw result in `results/raw/`, a value in
`pricing.yaml`, or a quantity recomputed from those — never against another
sentence in the README. A claim that cannot be traced is a FAIL, and the exit
code is non-zero, so this can gate a release the way `validate_energy.py` gates
J/token.

Scope: the two sections where a wrong number would do the most damage — §3, the
cost model and break-even table, and §6, the stack comparison and its quality
guard — plus the FP8 block in §4. It is a spot-check, not a proof that the whole
document is sound: prose claims and anything outside those sections are still
read by a human.
"""

from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import analyze  # noqa: E402

FAILS: list[str] = []
CHECKS = 0


def chk(name: str, ok: bool, detail: str) -> None:
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILS.append(f"{name}: {detail}")
    print(("  ok   " if ok else "  FAIL ") + f"{name}: {detail}")


def near(a: float, b: float, tol: float = 0.005) -> bool:
    return abs(a - b) <= tol * max(abs(b), 1e-12)


def section(text: str, start: str, end: str) -> str:
    return text.split(start)[1].split(end)[0]


def main() -> int:
    readme = open(os.path.join(REPO, "README.md")).read()
    s3 = section(readme, "### 3. Cost per 1M", "### 4.")
    s4 = section(readme, "### 4. What int4 buys", "### 5.")
    s6 = section(readme, "### 6. Two serving stacks", "## Run-to-run variance")

    rows = {r["point_id"]: analyze.row_view(r) for r in analyze.load_results()}
    every = [analyze.row_view(r)
             for r in analyze.load_results(include_superseded=True)]
    pricing = analyze.load_pricing()

    # --- §3: the cost model ------------------------------------------------
    print("--- README §3: cost model, against raw + pricing.yaml ---")
    p = rows["e1_chat_7b_bf16_r12"]
    tput = p["out_tok_throughput"]
    chk("quoted at 1515 tok/s", "1515" in s3 and near(tput, 1515, 0.001),
        f"raw={tput:.1f} tok/s from {p['file']}")

    rate = pricing["electricity"]["primary"]["usd_per_kwh"]
    card = pricing["hardware"]["gpu"]["street_price_usd"]
    years = pricing["hardware"]["amortization_years"]
    e_cost = analyze.energy_cost_per_1m_out(p["j_per_out_tok"], rate)
    amort_day = analyze.owned_amortization_usd_per_hour(card, years) * 24
    cap = tput * 86400

    chk("electricity $0.023/1M", "$0.023" in s3 and near(round(e_cost, 3), 0.023),
        f"{p['j_per_out_tok']:.3f} J/tok at ${rate}/kWh -> ${e_cost:.4f}")
    chk("amortisation $0.015/1M", "$0.015" in s3
        and near(round(amort_day * 1e6 / cap, 3), 0.015),
        f"${card:,} over {years}yr at full duty -> ${amort_day * 1e6 / cap:.4f}")
    chk("owned total $0.037/1M", "$0.037" in s3
        and near(round(amort_day * 1e6 / cap + e_cost, 3), 0.037),
        f"${amort_day * 1e6 / cap + e_cost:.4f}")
    rent = pricing["rental"]["options"][0]
    r_cost = analyze.usd_per_1m_out(rent["usd_per_hour"], tput)
    chk("rental $0.062/1M", "$0.062" in s3 and near(round(r_cost, 3), 0.062),
        f"${rent['usd_per_hour']}/hr -> ${r_cost:.4f}")

    for duty, vol, amo, tot in ((1.0, "130.9M", 0.015, 0.037),
                                (0.5, "65.5M", 0.029, 0.052),
                                (0.25, "32.7M", 0.059, 0.081),
                                (0.10, "13.1M", 0.146, 0.169),
                                (0.01, "1.31M", 1.464, 1.487)):
        v = cap * duty
        a = amort_day * 1e6 / v
        t = analyze.owned_usd_per_1m_at_volume(amort_day, e_cost, v)
        chk(f"duty-cycle row {duty:.0%}",
            vol in s3 and f"${amo:.3f}" in s3 and f"${tot:.3f}" in s3
            and near(round(a, 3), amo) and near(round(t, 3), tot),
            f"V={v / 1e6:.2f}M/day, amort ${a:.3f}, total ${t:.3f}")

    in_tok, out_tok = p["total_input_tokens"], p["total_output_tokens"]

    def api_row(suffix: str) -> dict:
        return next(a for g in ("open_weight_hosted", "frontier")
                    for a in pricing["api_prices"][g] if a["model"].endswith(suffix))

    for model, eff_s, be_s in (("Meta-Llama-3.1-8B-Instruct-Turbo", "$0.120", "19.7M"),
                               ("Qwen3-14B", "$0.720", "2.75M"),
                               ("Qwen2.5-7B-Instruct-Turbo", "$1.500", "1.30M"),
                               ("claude-haiku-4-5", "$9.000", "213k")):
        api = api_row(model)
        eff = analyze.api_effective_out_price(api["usd_per_1m_input"],
                                              api["usd_per_1m_output"], in_tok, out_tok)
        be = analyze.breakeven_tokens_per_day(amort_day, e_cost, eff)
        # Compare the README's figure numerically, not as a string: it is
        # quoted to three significant figures and the table it came from
        # rounds to two decimals, so "19.7M" and "19.66M" are the same number
        # written twice, and a string match would call that a defect.
        shown = float(be_s.rstrip("Mk")) * (1e6 if be_s.endswith("M") else 1e3)
        chk(f"break-even {model}",
            eff_s in s3 and be_s in s3 and f"${eff:.3f}" == eff_s
            and near(shown, be, 0.01),
            f"effective ${eff:.3f}/1M -> {be:,.0f} tok/day "
            f"(README: {be_s}, {abs(shown - be) / be:.2%} off)")

    opus = api_row("claude-opus-5")
    be_opus = analyze.breakeven_tokens_per_day(
        amort_day, e_cost,
        analyze.api_effective_out_price(opus["usd_per_1m_input"],
                                        opus["usd_per_1m_output"], in_tok, out_tok))
    chk("owning costs more than Opus below ~43k/day",
        "43k" in s3 and 42_000 <= be_opus <= 44_000, f"crossing at {be_opus:,.0f} tok/day")
    be_cheap = analyze.breakeven_tokens_per_day(
        amort_day, e_cost,
        analyze.api_effective_out_price(api_row("Meta-Llama-3.1-8B-Instruct-Turbo")["usd_per_1m_input"],
                                        api_row("Meta-Llama-3.1-8B-Instruct-Turbo")["usd_per_1m_output"],
                                        in_tok, out_tok))
    chk("cheapest hosted model breaks even at 15% duty",
        "15%" in s3 and near(round(100 * be_cheap / cap), 15, 0.05),
        f"{100 * be_cheap / cap:.1f}% of the {cap / 1e6:.1f}M/day ceiling")

    # --- §6: the stack comparison -----------------------------------------
    print("--- README §6: stacks, guard and transport, against raw ---")
    for pid, tps, jtok in (("e4_vllm_awq_c1", 148.9, 2.08),
                           ("e4_llamacpp_gguf_c1", 145.6, 2.38),
                           ("e4_vllm_awq_c8", 820.2, 0.419),
                           ("e4_llamacpp_gguf_c8", 482.8, 0.791),
                           ("e4_vllm_awq_c32", 1542, 0.262),
                           ("e4_llamacpp_gguf_c32", 755.6, 0.339)):
        r = rows[pid]
        chk(f"{pid}", str(tps) in s6 and str(jtok) in s6
            and near(r["out_tok_throughput"], tps, 0.002)
            and near(r["j_per_out_tok"], jtok, 0.005),
            f"raw {r['out_tok_throughput']:.1f} tok/s, {r['j_per_out_tok']:.3f} J/tok")

    guards = analyze.load_gsm8k()
    g_v, g_l = guards["awq_n200"]["metrics"], guards["gguf_q4km_n200"]["metrics"]
    chk("guard vLLM 91.0% (182/200)", "91.0% (182/200)" in s6
        and (g_v["n_correct"], g_v["n_items"]) == (182, 200), f"raw {g_v['n_correct']}/{g_v['n_items']}")
    chk("guard llama.cpp 90.0% (180/200)", "90.0% (180/200)" in s6
        and (g_l["n_correct"], g_l["n_items"]) == (180, 200), f"raw {g_l['n_correct']}/{g_l['n_items']}")
    lo, hi = analyze.newcombe_difference(g_v["n_correct"], g_v["n_items"],
                                         g_l["n_correct"], g_l["n_items"])
    chk("difference interval −4.9 to +6.9", "−4.9 to +6.9" in s6
        and near(round(lo * 100, 1), -4.9, 0.02) and near(round(hi * 100, 1), 6.9, 0.02),
        f"Newcombe {lo * 100:+.1f} to {hi * 100:+.1f} points")
    chk("E2's AWQ guard was 92.0% at n=50", "92.0% at" in s6
        and near(guards["awq"]["metrics"]["exact_match"], 0.92)
        and guards["awq"]["metrics"]["n_items"] == 50,
        f"raw {guards['awq']['metrics']['exact_match']:.1%} at n={guards['awq']['metrics']['n_items']}")

    # The transport claims quote the superseded connection-reuse runs, which are
    # kept in results/raw/ precisely so this comparison can be recomputed.
    def newest(pid: str, when: str) -> dict:
        return max((r for r in every if r["point_id"] == pid and r["file"].startswith(f"{pid}__{when}")),
                   key=lambda r: r["file"])

    reuse_c1, close_c1 = newest("e4_vllm_awq_c1", "20260815T0027"), rows["e4_vllm_awq_c1"]
    loss_c1 = 1 - reuse_c1["out_tok_throughput"] / close_c1["out_tok_throughput"]
    chk("vLLM lost 59% at c1 with connection reuse",
        "−59%" in s6 and "148.9 → 61.4" in s6 and near(round(loss_c1 * 100), 59, 0.02),
        f"{reuse_c1['out_tok_throughput']:.1f} -> {close_c1['out_tok_throughput']:.1f} tok/s "
        f"= {loss_c1:.1%}")

    reuse_c32, close_c32 = newest("e4_vllm_awq_c32", "20260815T0040"), rows["e4_vllm_awq_c32"]
    loss_c32 = 1 - reuse_c32["out_tok_throughput"] / close_c32["out_tok_throughput"]
    chk("vLLM lost 44% at c32 with connection reuse",
        "−44%" in s6 and "1542 → 862" in s6 and near(round(loss_c32 * 100), 44, 0.02),
        f"{reuse_c32['out_tok_throughput']:.1f} -> {close_c32['out_tok_throughput']:.1f} tok/s "
        f"= {loss_c32:.1%}")

    reuse_l8 = newest("e4_llamacpp_gguf_c8", "20260815T0050")
    issued = 800
    dropped = issued - reuse_l8["completed"]
    chk("llama.cpp dropped 87/800 at c8 with connection reuse",
        "87 / 800 (10.9%)" in s6 and dropped == 87
        and near(round(100 * dropped / issued, 1), 10.9, 0.01),
        f"{reuse_l8['completed']}/{issued} completed, {dropped} dropped "
        f"({100 * dropped / issued:.1f}%)")

    # --- §4: the FP8 block -------------------------------------------------
    print("--- README §4: FP8 arm, against raw ---")
    f8, bf, awq = rows["fp8_7b_chat_r4"], rows["e1_chat_7b_bf16_r4"], rows["e3_7b_awq_chat_r4"]
    for label, shown, got in (("FP8 out tok/s", "508.1", f8["out_tok_throughput"]),
                              ("FP8 TTFT p95", "105.5", f8["ttft_p95_ms"]),
                              ("FP8 ITL p50", "11.36", f8["itl_p50_ms"]),
                              ("FP8 mean power", "289.0", f8["mean_power_w"]),
                              ("FP8 J/token", "0.590", f8["j_per_out_tok"]),
                              ("BF16 out tok/s", "506.5", bf["out_tok_throughput"]),
                              ("BF16 TTFT p95", "158.3", bf["ttft_p95_ms"]),
                              ("BF16 ITL p50", "16.78", bf["itl_p50_ms"]),
                              ("BF16 mean power", "343.7", bf["mean_power_w"]),
                              ("BF16 J/token", "0.702", bf["j_per_out_tok"]),
                              ("AWQ out tok/s", "509.8", awq["out_tok_throughput"]),
                              ("AWQ TTFT p95", "111.3", awq["ttft_p95_ms"]),
                              ("AWQ ITL p50", "6.18", awq["itl_p50_ms"]),
                              ("AWQ mean power", "319.7", awq["mean_power_w"]),
                              ("AWQ J/token", "0.650", awq["j_per_out_tok"])):
        chk(label, shown in s4 and near(float(shown), got, 0.002), f"raw={got:.4g}")

    print(f"\n{CHECKS - len(FAILS)}/{CHECKS} checks pass")
    for f_ in FAILS:
        print("  FAILED:", f_)
    if FAILS:
        print("A README number does not trace to raw data. Fix the number, not this file.")
        return 1
    print("Every checked README number traces to results/raw/ or pricing.yaml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
