#!/usr/bin/env python3
"""The third serving stack: HF transformers + plain PyTorch, behind the same
OpenAI-compatible endpoint the other two arms are measured through.

  pytorch_server.py --model Qwen/Qwen2.5-7B-Instruct --port 8000 \
      --dtype bfloat16 --max-batch-size 32 [--compile]

What this is, stated plainly: `AutoModelForCausalLM.from_pretrained(...)` and
`model.generate(...)`, nothing else. No paged KV cache, no continuous batching,
no custom kernels beyond what `attn_implementation` selects inside transformers.
It is the baseline every serving framework is implicitly compared against and
almost never measured, which is why it is here.

Batching is **static**: requests that arrive while a batch is decoding wait for
the next one, and a batch runs until its longest request finishes. That is not
a shortcut to be fixed later, it is the property of the PyTorch-native path that
the stack comparison exists to put a number on. The queue depth it causes is
exported on /metrics, in the same columns the other stacks fill.

Why an HTTP server at all, rather than calling generate() from a script: E4
established that the client is part of the instrument -- a transport setting
moved measured throughput by 59%. A third arm driven by a different client would
not be comparable with the first two. So this arm is driven by the same
`vllm bench serve --backend openai`, over the same `Connection: close` transport,
through the same power poller and the same /metrics scrape as vLLM and
llama.cpp.

Endpoints:
  GET  /health                     200 once weights are loaded
  GET  /metrics                    Prometheus text; `pytorch:*` names, aliased
                                   onto vLLM's by kv_log.py
  GET  /v1/models
  POST /v1/completions             streaming (SSE) and non-streaming
  POST /v1/chat/completions        non-streaming and streaming; the GSM8K guard
  POST /wattbench/profile          one synthetic batch under torch.profiler,
                                   run on the scheduler thread, outside any
                                   measured window; kernel time per token
  POST /wattbench/reset_peak_memory_stats

Dependencies: torch, transformers. Nothing from vLLM is imported.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

LOG_PREFIX = "[pytorch_server]"


def log(msg: str) -> None:
    print(f"{LOG_PREFIX} {time.strftime('%H:%M:%S')} {msg}", flush=True)


# --------------------------------------------------------------------------
# requests
# --------------------------------------------------------------------------


class Request:
    """One generation request, from HTTP thread to scheduler thread and back."""

    __slots__ = ("id", "prompt_ids", "max_tokens", "ignore_eos", "sampling",
                 "events", "created", "n_generated", "generated_ids", "done",
                 "cancelled", "finish_reason")

    def __init__(self, prompt_ids: list[int], max_tokens: int, ignore_eos: bool,
                 sampling: tuple) -> None:
        self.id = f"cmpl-{uuid.uuid4().hex[:24]}"
        self.prompt_ids = prompt_ids
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        # (do_sample, temperature, top_p, top_k): requests in one batch must
        # share these, because generate() applies them to the whole batch.
        self.sampling = sampling
        self.events: Queue = Queue()   # int token ids, then None on finish
        self.created = time.time()
        self.n_generated = 0
        self.generated_ids: list[int] = []
        self.done = False
        self.cancelled = False
        self.finish_reason: str | None = None


class ProfileJob:
    """A torch.profiler pass, queued like a request so it never overlaps one."""

    def __init__(self, batch_size: int, input_len: int, output_len: int) -> None:
        self.batch_size = batch_size
        self.input_len = input_len
        self.output_len = output_len
        self.result: Queue = Queue()


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------


class StreamTap(StoppingCriteria):
    """A StoppingCriteria that hands each new token to its request as it is
    sampled, and stops rows individually.

    generate() calls the stopping criteria once per decode step with the full
    `input_ids`, so the last column is the token just chosen for every row.
    That is the only hook transformers offers for per-step, per-row streaming
    without rewriting the decode loop -- and rewriting the decode loop would
    stop this being "model.generate", which is the thing being measured.

    The criteria also does ALL stopping. generate()'s own EOS criterion is
    disabled (eos_token_id=None) so that an `ignore_eos` request keeps going
    past EOS the way vLLM's does, while a request that wants EOS honoured gets
    it here, row by row.
    """

    def __init__(self, engine: "Engine", batch: list[Request]) -> None:
        super().__init__()
        self.engine = engine
        self.batch = batch
        self.first_step_event: torch.cuda.Event | None = None
        self.steps = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
                 **kwargs) -> torch.BoolTensor:
        if self.steps == 0 and self.engine.cuda:
            self.first_step_event = torch.cuda.Event(enable_timing=True)
            self.first_step_event.record()
        self.steps += 1
        last = input_ids[:, -1].tolist()
        done = []
        eos = self.engine.eos_ids
        for req, tok in zip(self.batch, last):
            if req.done:
                done.append(True)
                continue
            req.n_generated += 1
            req.generated_ids.append(tok)
            self.engine.generation_tokens_total += 1
            if req.n_generated >= req.max_tokens:
                req.done, req.finish_reason = True, "length"
            elif not req.ignore_eos and tok in eos:
                req.done, req.finish_reason = True, "stop"
            # The finish reason rides on the last token's event, so the HTTP
            # thread can put it on that token's chunk rather than on an extra
            # empty chunk the client would count as one more inter-token gap.
            if not req.cancelled:
                req.events.put((tok, req.finish_reason if req.done else None))
            if req.done:
                req.events.put(None)
            done.append(req.done)
        return torch.tensor(done, dtype=torch.bool, device=input_ids.device)


class Engine:
    def __init__(self, args: argparse.Namespace) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.args = args
        self.cuda = torch.cuda.is_available() and args.device.startswith("cuda")
        self.device = torch.device(args.device if self.cuda else "cpu")
        if not self.cuda:
            log(f"WARNING: CUDA not available; serving on {self.device}. "
                "Only useful for protocol testing.")

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[args.dtype]

        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer or args.model, revision=args.revision)
        # Left padding: every row's last token must be its own newest token
        # for the batch to decode in lockstep.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = int(self.tokenizer.pad_token_id)

        load_kwargs: dict = {"revision": args.revision}
        if args.attn_implementation:
            load_kwargs["attn_implementation"] = args.attn_implementation
        try:   # transformers >= 5 names it `dtype`; 4.x only knows `torch_dtype`
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=dtype, **load_kwargs)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=dtype, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.model_load_s = time.time() - t0

        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        self.eos_ids: set[int] = set(eos if isinstance(eos, (list, tuple)) else [eos])
        if self.tokenizer.eos_token_id is not None:
            self.eos_ids.add(int(self.tokenizer.eos_token_id))

        self.weights_bytes = sum(p.numel() * p.element_size()
                                 for p in self.model.parameters())
        self.weights_bytes += sum(b.numel() * b.element_size()
                                  for b in self.model.buffers())

        self.compiled = False
        if args.compile:
            # transformers' documented recipe: a static KV cache so shapes are
            # fixed per (batch, length), and the forward compiled with CUDA
            # graphs. Every new (batch size, cache length) pair recompiles,
            # which the warmup pass absorbs for a fixed-concurrency point and
            # does not for a Poisson one -- so compiled points in this project
            # are fixed-concurrency by convention (see configs/README.md).
            import torch._dynamo as _dynamo
            _dynamo.config.cache_size_limit = max(64, _dynamo.config.cache_size_limit)
            self.model.generation_config.cache_implementation = "static"
            self.model.forward = torch.compile(
                self.model.forward, mode=args.compile_mode, fullgraph=False)
            self.compiled = True

        # Served name: what the client puts in "model". vLLM serves under the
        # repo id; this does the same unless told otherwise, so the load
        # generator sends identical request bodies to every arm.
        self.served_name = args.served_model_name or args.model
        self.max_model_len = args.max_model_len

        # Scheduler state. `waiting` is the queue the other stacks also expose:
        # a request sits here until a batch forms around it.
        self.lock = threading.Condition()
        self.waiting: list[Request | ProfileJob] = []
        self.running: list[Request] = []

        # Counters, exported on /metrics. The time counters are CUDA-event
        # elapsed times around generate(), so they are time the GPU stream was
        # inside a generate() call -- kernels plus whatever gaps the HF decode
        # loop leaves between them. That is the honest unit: "GPU time per
        # token as this stack actually spends it", not kernel-only time, which
        # /wattbench/profile measures separately.
        self.prompt_tokens_total = 0
        self.generation_tokens_total = 0
        self.generate_calls_total = 0
        self.cuda_time_total_ms = 0.0
        self.prefill_time_total_ms = 0.0
        self.batch_size_last = 0
        self.requests_total = 0
        self.requests_rejected_total = 0

        log(f"Model loading took {self.model_load_s:.1f} s: {args.model}")
        log(f"weights {self.weights_bytes / 2**30:.2f} GiB in {args.dtype}, "
            f"attn_implementation={getattr(self.model.config, '_attn_implementation', None)}, "
            f"torch {torch.__version__}, compile={'on:' + args.compile_mode if self.compiled else 'off'}")
        if self.cuda:
            free, total = torch.cuda.mem_get_info()
            log(f"device {torch.cuda.get_device_name(self.device)}: "
                f"{free / 2**30:.2f} GiB free of {total / 2**30:.2f} GiB after load; "
                f"allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

        # Kineto (torch.profiler's CUDA backend) wants its first registration
        # on the thread that imported it. /wattbench/profile runs on the
        # scheduler thread, so touch the profiler once here, on the main
        # thread, before that thread exists. Cheap, and outside any window.
        try:
            from torch.profiler import ProfilerActivity, profile
            acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if self.cuda else [])
            with profile(activities=acts):
                torch.ones(1, device=self.device).sum()
        except Exception as exc:  # noqa: BLE001
            log(f"WARNING: profiler warm-up failed ({exc}); /wattbench/profile may be degraded")

        self.thread = threading.Thread(target=self._loop, name="scheduler", daemon=True)
        self.thread.start()

    # ---- admission -------------------------------------------------------

    def submit(self, item: Request | ProfileJob) -> None:
        with self.lock:
            self.waiting.append(item)
            if isinstance(item, Request):
                self.requests_total += 1
            self.lock.notify()

    def validate(self, prompt_len: int, max_tokens: int) -> str | None:
        if prompt_len + max_tokens > self.max_model_len:
            # vLLM's behaviour for the same input: refuse, do not truncate.
            return (f"This model's maximum context length is {self.max_model_len} "
                    f"tokens. However, you requested {prompt_len + max_tokens} "
                    f"tokens ({prompt_len} in the prompt; {max_tokens} in the "
                    f"completion).")
        return None

    # ---- scheduler -------------------------------------------------------

    def _take_batch(self) -> list[Request] | ProfileJob:
        """Form a batch from the waiting list. Called with the lock held."""
        first = self.waiting.pop(0)
        if isinstance(first, ProfileJob):
            return first
        batch = [first]
        keep: list[Request | ProfileJob] = []
        for item in self.waiting:
            if (len(batch) < self.args.max_batch_size and isinstance(item, Request)
                    and item.sampling == first.sampling):
                batch.append(item)
            else:
                keep.append(item)
        self.waiting = keep
        return batch

    def _loop(self) -> None:
        wait_s = self.args.batch_wait_ms / 1000.0
        while True:
            with self.lock:
                while not self.waiting:
                    self.lock.wait()
                if wait_s > 0 and len(self.waiting) < self.args.max_batch_size:
                    # Hold the door briefly so near-simultaneous arrivals share
                    # a batch. Zero by default: the other stacks are not given
                    # a batching delay either.
                    self.lock.wait(timeout=wait_s)
                batch = self._take_batch()
                if isinstance(batch, list):
                    self.running = [r for r in batch]
            try:
                if isinstance(batch, ProfileJob):
                    try:
                        batch.result.put(self._profile(batch))
                    except Exception as exc:  # noqa: BLE001
                        batch.result.put({"error": f"{type(exc).__name__}: {exc}",
                                          "traceback": traceback.format_exc()})
                else:
                    self._run_batch(batch)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                if isinstance(batch, list):
                    for r in batch:
                        if not r.done:
                            r.done, r.finish_reason = True, "error"
                            r.events.put(None)
            finally:
                with self.lock:
                    self.running = []

    def _generation_kwargs(self, sampling: tuple) -> dict:
        do_sample, temperature, top_p, top_k = sampling
        kw: dict = {"do_sample": do_sample}
        if do_sample:
            kw["temperature"] = temperature
            if top_p is not None:
                kw["top_p"] = top_p
            if top_k is not None:
                kw["top_k"] = top_k
        else:
            # Silence the "temperature is set but do_sample=False" warnings
            # the checkpoint's generation_config would otherwise trigger.
            kw.update(temperature=None, top_p=None, top_k=None)
        return kw

    @torch.inference_mode()
    def _run_batch(self, batch: list[Request]) -> None:
        self.batch_size_last = len(batch)
        max_len = max(len(r.prompt_ids) for r in batch)
        ids = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, r in enumerate(batch):
            n = len(r.prompt_ids)
            ids[i, max_len - n:] = torch.tensor(r.prompt_ids, dtype=torch.long)
            mask[i, max_len - n:] = 1
            self.prompt_tokens_total += n
        ids = ids.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)

        tap = StreamTap(self, batch)
        max_new = max(r.max_tokens for r in batch)

        start = end = None
        if self.cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        t0 = time.perf_counter()
        self.model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=max_new,
            # All stopping is done by the tap, per row (see StreamTap).
            eos_token_id=None,
            pad_token_id=self.pad_id,
            stopping_criteria=StoppingCriteriaList([tap]),
            use_cache=True,
            **self._generation_kwargs(batch[0].sampling),
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if self.cuda:
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)
            if tap.first_step_event is not None:
                self.prefill_time_total_ms += start.elapsed_time(tap.first_step_event)
        else:
            elapsed = wall_ms
        self.cuda_time_total_ms += elapsed
        self.generate_calls_total += 1

        # Anything the tap did not close (it closes every row on its own stop
        # condition; this is belt and braces for max_new_tokens).
        for r in batch:
            if not r.done:
                r.done, r.finish_reason = True, "length"
                r.events.put(None)

    # ---- profiler --------------------------------------------------------

    @torch.inference_mode()
    def _profile(self, job: ProfileJob) -> dict:
        """Kernel time for one synthetic batch of the configured shape.

        Two profiled passes: a prefill-only forward, then a full generate() of
        the same prompt. Their difference is the decode share, and decode
        kernel time divided by (batch x output_len) is the kernel-level cost
        of an output token at this batch size -- the number that sits beside
        J/token and ITL in the stack table.

        Run on the scheduler thread, so it never overlaps a measured request,
        and only ever invoked by run.sh after the measurement window closes.
        The profiler's overhead is therefore outside every number that
        depends on timing.
        """
        from torch.profiler import ProfilerActivity, profile

        vocab = int(getattr(self.model.config, "vocab_size", 32000))
        gen = torch.Generator(device="cpu").manual_seed(0)
        ids = torch.randint(0, vocab, (job.batch_size, job.input_len), generator=gen)
        ids = ids.to(self.device)
        mask = torch.ones_like(ids)
        activities = [ProfilerActivity.CPU]
        if self.cuda:
            activities.append(ProfilerActivity.CUDA)

        def sync() -> None:
            if self.cuda:
                torch.cuda.synchronize()

        # Untimed pass so one-off costs (lazy init, compile) stay out.
        self.model(input_ids=ids, attention_mask=mask, use_cache=True)
        sync()

        t0 = time.perf_counter()
        with profile(activities=activities) as p_prefill:
            self.model(input_ids=ids, attention_mask=mask, use_cache=True)
            sync()
        prefill_wall_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        with profile(activities=activities) as p_full:
            self.model.generate(
                input_ids=ids, attention_mask=mask,
                max_new_tokens=job.output_len, min_new_tokens=job.output_len,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                eos_token_id=None, pad_token_id=self.pad_id, use_cache=True)
            sync()
        full_wall_ms = (time.perf_counter() - t0) * 1000.0

        pre = _summarise_profile(p_prefill)
        full = _summarise_profile(p_full)
        n_out = job.batch_size * job.output_len
        decode_kernel_ms = max(0.0, full["cuda_kernel_ms"] - pre["cuda_kernel_ms"])
        return {
            "available": True,
            "batch_size": job.batch_size,
            "input_len": job.input_len,
            "output_len": job.output_len,
            "wall_ms": {"prefill_forward": round(prefill_wall_ms, 2),
                        "generate": round(full_wall_ms, 2)},
            "cuda_kernel_ms": {
                "prefill_forward": round(pre["cuda_kernel_ms"], 2),
                "generate": round(full["cuda_kernel_ms"], 2),
                "decode_derived": round(decode_kernel_ms, 2),
            },
            "cuda_kernel_ms_per_output_token": (
                round(decode_kernel_ms / n_out, 4) if n_out else None),
            "cuda_kernel_ms_per_decode_step": (
                round(decode_kernel_ms / max(job.output_len - 1, 1), 3)),
            # Kernel time over wall time inside generate(): how much of the
            # call the GPU was actually executing kernels.
            "gpu_busy_frac_generate": (
                round(full["cuda_kernel_ms"] / full_wall_ms, 4) if full_wall_ms else None),
            "categories_generate": full["categories"],
            "top_kernels_generate": full["top"],
            "top_kernels_prefill": pre["top"][:10],
            "n_kernel_launches_generate": full["n_launches"],
            "compiled": self.compiled,
            "attn_implementation": getattr(self.model.config, "_attn_implementation", None),
            "torch": torch.__version__,
            "note": ("synthetic prompt of random token ids; greedy; EOS disabled so "
                     "every row decodes exactly output_len tokens. Measured after "
                     "the benchmark window, never inside it."),
        }

    # ---- metrics ---------------------------------------------------------

    def metrics_text(self) -> str:
        with self.lock:
            waiting = sum(1 for w in self.waiting if isinstance(w, Request))
            running = sum(1 for r in self.running if not r.done)
        lines = [
            "# HELP pytorch:num_requests_running rows in the batch currently in generate()",
            f"pytorch:num_requests_running {running}",
            "# HELP pytorch:num_requests_waiting requests queued for the next static batch",
            f"pytorch:num_requests_waiting {waiting}",
            f"pytorch:prompt_tokens_total {self.prompt_tokens_total}",
            f"pytorch:generation_tokens_total {self.generation_tokens_total}",
            f"pytorch:generate_calls_total {self.generate_calls_total}",
            f"pytorch:requests_total {self.requests_total}",
            f"pytorch:requests_rejected_total {self.requests_rejected_total}",
            f"pytorch:batch_size_last {self.batch_size_last}",
            "# HELP pytorch:cuda_time_total_ms CUDA-event time the stream spent inside generate()",
            f"pytorch:cuda_time_total_ms {self.cuda_time_total_ms:.3f}",
            f"pytorch:prefill_time_total_ms {self.prefill_time_total_ms:.3f}",
            f"pytorch:model_weights_bytes {self.weights_bytes}",
        ]
        if self.cuda:
            lines += [
                f"pytorch:memory_allocated_bytes {torch.cuda.memory_allocated(self.device)}",
                f"pytorch:max_memory_allocated_bytes {torch.cuda.max_memory_allocated(self.device)}",
                f"pytorch:memory_reserved_bytes {torch.cuda.memory_reserved(self.device)}",
                f"pytorch:max_memory_reserved_bytes {torch.cuda.max_memory_reserved(self.device)}",
            ]
        return "\n".join(lines) + "\n"

    def memory_json(self) -> dict:
        out: dict = {"model_weights_bytes": self.weights_bytes, "device": str(self.device)}
        if self.cuda:
            free, total = torch.cuda.mem_get_info(self.device)
            out.update({
                "memory_allocated_bytes": torch.cuda.memory_allocated(self.device),
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "memory_reserved_bytes": torch.cuda.memory_reserved(self.device),
                "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
                "device_free_bytes": free, "device_total_bytes": total,
            })
        return out

    def reset_peak_memory_stats(self) -> None:
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)


# Kernel names grouped coarsely, so the table can say "x% matmul, y% attention"
# without a reader having to know what `nvjet_tst_...` is.
_CATEGORIES = [
    ("matmul", re.compile(r"gemm|gemv|cutlass|nvjet|matmul|sm\d\d_xmma|wgmma|ampere_|hopper_", re.I)),
    ("attention", re.compile(r"flash|fmha|attention|attn|sdpa|mem_eff|fused_sdp", re.I)),
    ("norm/softmax/activation", re.compile(r"norm|softmax|silu|gelu|swiglu|rms", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized|unrolled|binary|unary|where|mul|add", re.I)),
    ("reduce/index/copy", re.compile(r"reduce|index|gather|scatter|copy|memcpy|memset|fill|cat|embedding|cumsum|sort|topk|argmax|multinomial", re.I)),
]


def _categorise(name: str) -> str:
    for cat, rx in _CATEGORIES:
        if rx.search(name):
            return cat
    return "other"


def _summarise_profile(prof) -> dict:
    """Self CUDA kernel time by kernel, from a torch.profiler trace."""
    try:
        from torch.autograd import DeviceType
        cuda_type = DeviceType.CUDA
    except Exception:  # noqa: BLE001
        cuda_type = None
    rows = []
    for evt in prof.key_averages():
        dev = getattr(evt, "device_type", None)
        is_cuda = (dev == cuda_type) if cuda_type is not None else False
        # Attribute renamed across torch versions; take whichever exists.
        self_us = getattr(evt, "self_device_time_total", None)
        if self_us is None:
            self_us = getattr(evt, "self_cuda_time_total", 0.0)
        if is_cuda and self_us and self_us > 0:
            rows.append((evt.key, float(self_us) / 1000.0, int(evt.count)))
    total_ms = sum(ms for _, ms, _ in rows)
    rows.sort(key=lambda x: -x[1])
    cats: dict[str, float] = {}
    for name, ms, _ in rows:
        cats[_categorise(name)] = cats.get(_categorise(name), 0.0) + ms
    return {
        "cuda_kernel_ms": total_ms,
        "n_launches": sum(c for _, _, c in rows),
        "top": [{"kernel": name[:160], "self_cuda_ms": round(ms, 3), "calls": calls,
                 "share": round(ms / total_ms, 4) if total_ms else None}
                for name, ms, calls in rows[:25]],
        "categories": {k: {"ms": round(v, 2),
                           "share": round(v / total_ms, 4) if total_ms else None}
                       for k, v in sorted(cats.items(), key=lambda kv: -kv[1])},
    }


# --------------------------------------------------------------------------
# detokenisation
# --------------------------------------------------------------------------


def detokenize_step(tokenizer, ids: list[int], prefix_offset: int,
                    read_offset: int) -> tuple[str, int, int]:
    """Incremental decode that never splits a multi-byte character.

    Decodes a sliding window rather than the whole sequence, and holds text
    back while it ends in U+FFFD (an incomplete UTF-8 sequence). Same scheme
    vLLM uses; the token-per-chunk cadence is preserved either way because a
    chunk is still sent when the delta is empty.
    """
    prefix_text = tokenizer.decode(ids[prefix_offset:read_offset], skip_special_tokens=True)
    new_text = tokenizer.decode(ids[prefix_offset:], skip_special_tokens=True)
    if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
        return new_text[len(prefix_text):], read_offset, len(ids)
    return "", prefix_offset, read_offset


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def make_handler(engine: Engine):
    tok = engine.tokenizer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "wattbench-pytorch/0.1"

        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            # Per-request access logging is noise in a 2000-line server log;
            # errors are logged explicitly where they happen.
            return

        # ---- helpers -----------------------------------------------------

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if self._client_wants_close():
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code: int, message: str, etype: str = "invalid_request_error") -> None:
            self._json(code, {"error": {"message": message, "type": etype,
                                        "param": None, "code": code}})

        def _client_wants_close(self) -> bool:
            return self.headers.get("Connection", "").lower() == "close"

        def _read_body(self) -> dict | None:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            if not raw:
                return {}
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                self._error(400, "request body is not valid JSON")
                return None

        def _begin_stream(self) -> bool:
            """Open an SSE response. Returns whether chunked framing is in use."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            chunked = not self._client_wants_close()
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
            else:
                # The client asked for one connection per request (every arm in
                # this project does): plain writes, then close, no framing.
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            return chunked

        def _sse(self, chunked: bool, data: str) -> None:
            payload = f"data: {data}\n\n".encode()
            if chunked:
                self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
            else:
                self.wfile.write(payload)
            self.wfile.flush()

        def _end_stream(self, chunked: bool) -> None:
            if chunked:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

        # ---- GET ---------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/health":
                self._json(200, {"status": "ok"})
            elif path == "/metrics":
                body = engine.metrics_text().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": engine.served_name, "object": "model", "created": 0,
                     "owned_by": "wattbench", "max_model_len": engine.max_model_len}]})
            elif path == "/wattbench/memory":
                self._json(200, engine.memory_json())
            else:
                self._error(404, f"no route for GET {path}")

        # ---- POST --------------------------------------------------------

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            body = self._read_body()
            if body is None:
                return
            try:
                if path == "/v1/completions":
                    self._completions(body, chat=False)
                elif path == "/v1/chat/completions":
                    self._completions(body, chat=True)
                elif path == "/wattbench/profile":
                    self._profile(body)
                elif path == "/wattbench/reset_peak_memory_stats":
                    engine.reset_peak_memory_stats()
                    self._json(200, {"ok": True})
                else:
                    self._error(404, f"no route for POST {path}")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                try:
                    self._error(500, f"{type(exc).__name__}: {exc}", "server_error")
                except Exception:  # noqa: BLE001
                    pass

        def _profile(self, body: dict) -> None:
            job = ProfileJob(int(body.get("batch_size") or 1),
                             int(body.get("input_len") or 512),
                             int(body.get("output_len") or 128))
            engine.submit(job)
            result = job.result.get()
            self._json(200 if "error" not in result else 500, result)

        def _completions(self, body: dict, chat: bool) -> None:
            # --- prompt -> token ids ---
            if chat:
                messages = body.get("messages")
                if not isinstance(messages, list) or not messages:
                    return self._error(400, "'messages' must be a non-empty list")
                prompt_ids = tok.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True)
                if hasattr(prompt_ids, "input_ids"):   # BatchEncoding on some versions
                    prompt_ids = prompt_ids["input_ids"]
                if hasattr(prompt_ids, "tolist"):      # a tensor on others
                    prompt_ids = prompt_ids.tolist()
                prompt_ids = list(prompt_ids)
                if prompt_ids and isinstance(prompt_ids[0], list):   # batched shape
                    prompt_ids = prompt_ids[0]
            else:
                prompt = body.get("prompt")
                if isinstance(prompt, list) and prompt and isinstance(prompt[0], str):
                    if len(prompt) != 1:
                        return self._error(400, "only one prompt per request is supported")
                    prompt = prompt[0]
                if isinstance(prompt, str):
                    prompt_ids = tok.encode(prompt, add_special_tokens=False)
                elif isinstance(prompt, list) and all(isinstance(x, int) for x in prompt):
                    prompt_ids = list(prompt)
                else:
                    return self._error(400, "'prompt' must be a string or a list of token ids")

            max_tokens = body.get("max_tokens")
            if max_tokens is None:
                max_tokens = body.get("max_completion_tokens")
            if max_tokens is None:
                max_tokens = 16
            max_tokens = int(max_tokens)
            if max_tokens < 1:
                return self._error(400, "max_tokens must be >= 1")
            problem = engine.validate(len(prompt_ids), max_tokens)
            if problem:
                engine.requests_rejected_total += 1
                return self._error(400, problem)

            temperature = body.get("temperature")
            if temperature is None:
                temperature = float(engine.model.generation_config.temperature or 1.0)
                do_sample = bool(engine.model.generation_config.do_sample)
            else:
                temperature = float(temperature)
                do_sample = temperature > 0.0
            top_p = body.get("top_p")
            top_k = body.get("top_k")
            sampling = (do_sample, temperature if do_sample else 0.0,
                        float(top_p) if (do_sample and top_p is not None) else None,
                        int(top_k) if (do_sample and top_k is not None) else None)
            stream = bool(body.get("stream", False))
            include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))
            ignore_eos = bool(body.get("ignore_eos", False))
            model_name = body.get("model") or engine.served_name

            req = Request(prompt_ids, max_tokens, ignore_eos, sampling)
            engine.submit(req)
            created = int(req.created)
            obj = "chat.completion.chunk" if chat else "text_completion"

            if not stream:
                while True:
                    ev = req.events.get()
                    if ev is None:
                        break
                text = tok.decode(req.generated_ids, skip_special_tokens=True)
                usage = {"prompt_tokens": len(prompt_ids),
                         "completion_tokens": req.n_generated,
                         "total_tokens": len(prompt_ids) + req.n_generated}
                if chat:
                    choice = {"index": 0, "finish_reason": req.finish_reason,
                              "message": {"role": "assistant", "content": text}}
                    return self._json(200, {"id": req.id.replace("cmpl", "chatcmpl"),
                                            "object": "chat.completion", "created": created,
                                            "model": model_name, "choices": [choice],
                                            "usage": usage})
                choice = {"index": 0, "text": text, "logprobs": None,
                          "finish_reason": req.finish_reason}
                return self._json(200, {"id": req.id, "object": "text_completion",
                                        "created": created, "model": model_name,
                                        "choices": [choice], "usage": usage})

            # --- streaming: one SSE chunk per generated token ---
            chunked = self._begin_stream()
            prefix_off = read_off = 0
            first = True
            ids_seen: list[int] = []
            try:
                while True:
                    ev = req.events.get()
                    if ev is None:
                        break
                    tok_id, finish = ev
                    ids_seen.append(tok_id)
                    delta, prefix_off, read_off = detokenize_step(
                        tok, ids_seen, prefix_off, read_off)
                    if chat:
                        d = {"content": delta}
                        if first:
                            d["role"] = "assistant"
                        choice = {"index": 0, "delta": d, "logprobs": None,
                                  "finish_reason": finish}
                    else:
                        choice = {"index": 0, "text": delta, "logprobs": None,
                                  "finish_reason": finish}
                    first = False
                    self._sse(chunked, json.dumps({
                        "id": req.id, "object": obj, "created": created,
                        "model": model_name, "choices": [choice]}))
                if include_usage:
                    self._sse(chunked, json.dumps({
                        "id": req.id, "object": obj, "created": created,
                        "model": model_name, "choices": [],
                        "usage": {"prompt_tokens": len(prompt_ids),
                                  "completion_tokens": req.n_generated,
                                  "total_tokens": len(prompt_ids) + req.n_generated}}))
                self._sse(chunked, "[DONE]")
                self._end_stream(chunked)
            except (BrokenPipeError, ConnectionResetError):
                # Client went away mid-stream. The row keeps decoding (a static
                # batch cannot drop one row), but nothing more is queued for it.
                req.cancelled = True

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 1024


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--served-model-name", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-batch-size", type=int, default=32,
                    help="rows per static batch; the analogue of --max-num-seqs")
    ap.add_argument("--batch-wait-ms", type=float, default=0.0,
                    help="how long to hold a batch open for more arrivals (0: none)")
    ap.add_argument("--attn-implementation", default="sdpa",
                    help="sdpa | eager | flash_attention_2 | '' for the checkpoint default")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the forward with a static KV cache")
    ap.add_argument("--compile-mode", default="reduce-overhead")
    args = ap.parse_args()
    if args.attn_implementation == "":
        args.attn_implementation = None

    try:
        engine = Engine(args)
    except Exception:  # noqa: BLE001
        # The traceback's last line is what harness.read_failure lifts out as
        # the root cause, so let it print whole.
        traceback.print_exc()
        return 1

    httpd = Server((args.host, args.port), make_handler(engine))
    log(f"serving {engine.served_name} on http://{args.host}:{args.port} "
        f"(max_batch_size={args.max_batch_size}, batch_wait_ms={args.batch_wait_ms}, "
        f"max_model_len={args.max_model_len})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
