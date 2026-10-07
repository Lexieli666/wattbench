#!/usr/bin/env python3
"""Protocol test for pytorch_server.py against a tiny local checkpoint.

Drives the server the way `vllm bench serve --backend openai` does -- SSE on
/v1/completions with `Connection: close`, usage in the final chunk -- and the
way gsm8k_guard.py does (/v1/chat/completions, non-streaming). Also exercises
/metrics, /wattbench/profile and the context-length rejection. CPU is fine: it
tests the wire protocol and the scheduler, not speed.

  python3 tests/test_pytorch_server.py --model /path/to/tiny-checkpoint
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
import urllib.request
import http.client

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wait_health(port: int, timeout: float = 120) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001
            time.sleep(0.3)
    raise SystemExit("server never became healthy")


def stream_completion(port: int, model: str, prompt: str, max_tokens: int,
                      close: bool = True, ignore_eos: bool = True) -> dict:
    """Same parsing as vLLM's async_request_openai_completions, synchronous."""
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "repetition_penalty": 1.0, "logprobs": None,
               "stream": True, "stream_options": {"include_usage": True}}
    if ignore_eos:
        payload["ignore_eos"] = True
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    headers = {"Content-Type": "application/json"}
    if close:
        headers["Connection"] = "close"
    st = time.perf_counter()
    conn.request("POST", "/v1/completions", body=json.dumps(payload), headers=headers)
    resp = conn.getresponse()
    assert resp.status == 200, (resp.status, resp.read()[:300])
    ttft, itl, text, n_chunks, usage, most_recent = None, [], "", 0, None, st
    buf = b""
    while True:
        chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            msg, buf = buf.split(b"\n\n", 1)
            msg = msg.strip().decode()
            if not msg or msg.startswith(":"):
                continue
            data = msg.removeprefix("data: ")
            if data == "[DONE]":
                continue
            d = json.loads(data)
            if choices := d.get("choices"):
                n_chunks += 1
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - st
                else:
                    itl.append(now - most_recent)
                most_recent = now
                text += choices[0].get("text") or ""
                fr = choices[0].get("finish_reason")
                if fr:
                    finish = fr
            elif u := d.get("usage"):
                usage = u
    conn.close()
    return {"ttft": ttft, "n_itl": len(itl), "n_chunks": n_chunks, "text": text,
            "usage": usage, "finish": finish}


def chat(port: int, model: str, content: str, max_tokens: int = 32) -> dict:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}],
                       "temperature": 0.0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def metrics(port: int) -> dict[str, float]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
        out = {}
        for line in r.read().decode().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            k, v = line.rsplit(" ", 1)
            out[k] = float(v)
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=18123)
    ap.add_argument("--served-name", default="tiny")
    args = ap.parse_args()

    proc = subprocess.Popen(
        [sys.executable, os.path.join(REPO, "pytorch_server.py"), "--model", args.model,
         "--port", str(args.port), "--device", "cpu", "--dtype", "float32",
         "--max-batch-size", "8", "--max-model-len", "256",
         "--served-model-name", args.served_name, "--attn-implementation", "sdpa"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    failures = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(("  ok   " if ok else "  FAIL ") + name + (f": {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    try:
        wait_health(args.port)
        m0 = metrics(args.port)
        check("metrics has pytorch:num_requests_waiting", "pytorch:num_requests_waiting" in m0)
        check("metrics has model_weights_bytes", m0.get("pytorch:model_weights_bytes", 0) > 0)

        # 1. single streamed completion, Connection: close, ignore_eos
        r = stream_completion(args.port, args.served_name, "The quick brown fox", 24)
        check("stream: one chunk per token", r["n_chunks"] == 24, f"{r['n_chunks']} chunks")
        check("stream: usage.completion_tokens == max_tokens",
              r["usage"] and r["usage"]["completion_tokens"] == 24, str(r["usage"]))
        check("stream: finish_reason=length on last token chunk", r["finish"] == "length")
        check("stream: ttft measured", r["ttft"] is not None and r["ttft"] > 0)

        # 2. keep-alive (chunked) framing path
        r2 = stream_completion(args.port, args.served_name, "The quick brown fox", 8, close=False)
        check("stream chunked: 8 chunks", r2["n_chunks"] == 8, f"{r2['n_chunks']}")

        # 3. concurrency: 8 requests at once -> should form one batch
        with cf.ThreadPoolExecutor(8) as ex:
            futs = [ex.submit(stream_completion, args.port, args.served_name,
                              f"Problem number {i}: Janet has apples", 16) for i in range(8)]
            rs = [f.result() for f in futs]
        check("batch: all 8 complete with 16 tokens",
              all(x["usage"]["completion_tokens"] == 16 for x in rs))
        m1 = metrics(args.port)
        check("metrics: generate_calls counted", m1["pytorch:generate_calls_total"] >= 3,
              str(m1["pytorch:generate_calls_total"]))
        check("metrics: batch_size_last <= 8 and >= 2",
              2 <= m1["pytorch:batch_size_last"] <= 8, str(m1["pytorch:batch_size_last"]))
        check("metrics: generation_tokens_total == 24+8+128",
              m1["pytorch:generation_tokens_total"] == 24 + 8 + 128,
              str(m1["pytorch:generation_tokens_total"]))
        check("metrics: cuda_time_total_ms grows", m1["pytorch:cuda_time_total_ms"] > 0)

        # 4. chat completions (the GSM8K guard path)
        c = chat(args.port, args.served_name, "What is 2+2? Answer with #### 4", 12)
        check("chat: has message content", isinstance(c["choices"][0]["message"]["content"], str))
        check("chat: usage present", c["usage"]["completion_tokens"] >= 1, str(c["usage"]))

        # 5. context rejection
        body = json.dumps({"model": args.served_name, "prompt": "x " * 200,
                           "max_tokens": 200}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=10)
            check("context: over-length rejected", False, "got 200")
        except urllib.error.HTTPError as e:
            check("context: over-length rejected with 400", e.code == 400, str(e.code))

        # 6. profile endpoint
        body = json.dumps({"batch_size": 2, "input_len": 16, "output_len": 4}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{args.port}/wattbench/profile",
                                     data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            prof = json.load(r)
        check("profile: available", prof.get("available") is True, str(prof)[:200])
        check("profile: has wall_ms.generate", prof["wall_ms"]["generate"] > 0)
        print("  profile summary:", json.dumps({k: prof[k] for k in
              ("cuda_kernel_ms", "cuda_kernel_ms_per_output_token", "n_kernel_launches_generate")}))

        # 7. non-stream completion honours EOS when ignore_eos is off
        r3 = stream_completion(args.port, args.served_name, "hello", 50, ignore_eos=False)
        check("ignore_eos=false: finish is length or stop", r3["finish"] in ("length", "stop"),
              r3["finish"])
    finally:
        proc.terminate()
        try:
            out = proc.communicate(timeout=10)[0]
        except subprocess.TimeoutExpired:
            proc.kill()
            out = proc.communicate()[0]
        print("--- server log tail ---")
        print("\n".join(out.splitlines()[-15:]))

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
