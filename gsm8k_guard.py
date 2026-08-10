#!/usr/bin/env python3
"""GSM8K exact-match guard for the E2 quantization ablation.

The speed and energy wins from int4 are only interesting if the model still
answers correctly. This runs a fixed 50-item GSM8K subset through a served
endpoint and reports exact-match accuracy, so the quantization tradeoff is
*stated* rather than assumed.

  ./gsm8k_guard.py --label bf16 --model Qwen/Qwen2.5-7B-Instruct

Scope, stated plainly: this measures grade-school arithmetic word problems on
50 items. It licenses no claim about any other capability, and no comparison
against any model not run through this same script.

The subset is chosen by a fixed seed, so every arm sees identical items.
Decoding is greedy, so a rerun of the same arm reproduces.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from datetime import datetime

REPO = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(REPO, "results", ".cache")
GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)

PROMPT = (
    "Solve this grade school math problem. Reason step by step, then give the "
    "final numeric answer on its own last line in the form:\n"
    "#### <number>\n\n"
    "Problem: {question}"
)


def load_gsm8k(n: int, seed: int) -> list[dict]:
    """Fetch the GSM8K test split once, cache it, and take a fixed subset."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "gsm8k_test.jsonl")
    if not os.path.isfile(path):
        print(f"[gsm8k] downloading test split -> {path}")
        urllib.request.urlretrieve(GSM8K_TEST_URL, path)
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(rows)), n))
    return [{"index": i, **rows[i]} for i in idx]


NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text: str) -> str | None:
    """Take the number after the last '####', else the last number in the text."""
    if not text:
        return None
    tail = text.rsplit("####", 1)[-1] if "####" in text else text
    nums = NUM_RE.findall(tail)
    if not nums and "####" in text:
        nums = NUM_RE.findall(text)
    if not nums:
        return None
    return normalise(nums[-1])


def normalise(s: str) -> str:
    s = s.replace(",", "").rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return s
    # 42.0 and 42 are the same answer.
    return str(int(f)) if f == int(f) else str(f)


def gold_answer(row: dict) -> str:
    return normalise(row["answer"].rsplit("####", 1)[-1].strip())


def ask(host: str, port: int, model: str, question: str,
        max_tokens: int, timeout: float) -> tuple[str | None, str | None]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(question=question)}],
        "temperature": 0.0,          # greedy: the guard must be reproducible
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
        return payload["choices"][0]["message"]["content"], None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model id as served")
    ap.add_argument("--label", required=True, help="arm name, e.g. bf16 / awq / gptq")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "results", "raw"))
    args = ap.parse_args()

    items = load_gsm8k(args.n, args.seed)
    print(f"[gsm8k] {len(items)} items, seed {args.seed}, model {args.model}")

    sys.path.insert(0, REPO)
    import harness  # provenance, so the guard is as traceable as a benchmark run

    started = datetime.now().astimezone().isoformat()
    t0 = time.time()
    records, correct, errors = [], 0, 0
    for k, row in enumerate(items, 1):
        text, err = ask(args.host, args.port, args.model, row["question"],
                        args.max_tokens, args.timeout)
        gold = gold_answer(row)
        pred = extract_answer(text) if text else None
        ok = pred is not None and pred == gold
        correct += int(ok)
        errors += int(err is not None)
        records.append({
            "index": row["index"],
            "gold": gold,
            "predicted": pred,
            "correct": ok,
            "error": err,
            # Kept so a disagreement between arms can be inspected rather than
            # argued about. Truncated: this is a guard, not a transcript archive.
            "response_tail": (text or "")[-400:],
        })
        if k % 10 == 0:
            print(f"[gsm8k]   {k}/{len(items)}  running accuracy {correct / k:.1%}")

    elapsed = time.time() - t0
    result = {
        "schema_version": 1,
        "point_id": f"gsm8k_{args.label}",
        "experiment": "E2-guard",
        "status": "ok" if errors == 0 else "partial",
        "description": (
            f"GSM8K exact-match guard, {len(items)} items, greedy decoding, "
            f"arm '{args.label}'"
        ),
        "run_started_at": started,
        "run_finished_at": datetime.now().astimezone().isoformat(),
        "config": {
            "model": args.model,
            "label": args.label,
            "n_items": len(items),
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "prompt_template": PROMPT,
            "endpoint": "/v1/chat/completions",
        },
        "model_revision": harness.resolve_hf_revision(args.model, None),
        "provenance": harness.provenance(),
        "metrics": {
            "n_items": len(items),
            "n_correct": correct,
            "n_errors": errors,
            "exact_match": round(correct / len(items), 4),
            "wall_s": round(elapsed, 1),
        },
        "items": records,
        "scope_note": (
            "50 grade-school arithmetic word problems. This is the only "
            "capability evidence in this project and licenses no claim beyond "
            "this task. With n=50 the 95% binomial interval is roughly +/-14 "
            "points, so only large accuracy gaps are meaningful."
        ),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = os.path.join(args.out_dir, f"gsm8k_{args.label}__{stamp}.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"[gsm8k] {args.label}: {correct}/{len(items)} = {correct / len(items):.1%} "
          f"exact match ({errors} request errors) in {elapsed:.0f}s")
    print(f"[gsm8k] wrote {os.path.relpath(out, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
