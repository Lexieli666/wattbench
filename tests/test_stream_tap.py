#!/usr/bin/env python3
"""StreamTap must copy what it keeps out of generate()'s buffers.

  ~/wattbench-venv/bin/python tests/test_stream_tap.py

Under `mode=reduce-overhead` the tensors generate() hands its stopping
criteria can live in a CUDA-graph pool that the next replay overwrites in
place. Anything the tap keeps across steps -- the token ids it streams and
records -- must therefore be a copy, not a view. This feeds the tap a buffer,
overwrites that buffer in place after each call the way a replay would, and
checks nothing the tap retained moved. CPU only.

(The 2026-10-07 compile crash was not here -- the overwritten tensor was the
static KV cache, allocated inside a compiled prefill -- but this is the one
place in our code that holds per-step data across steps, so it is pinned.)
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pytorch_server import Request, StreamTap  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("  ok   " if ok else "  FAIL ") + name + (f": {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def drain(req: Request) -> list:
    out = []
    while not req.events.empty():
        out.append(req.events.get_nowait())
    return out


def main() -> int:
    engine = SimpleNamespace(cuda=False, eos_ids={2}, generation_tokens_total=0)
    greedy = (False, None, None, None)
    batch = [Request([1, 1, 1], max_tokens=3, ignore_eos=True, sampling=greedy),
             Request([1, 1, 1], max_tokens=3, ignore_eos=True, sampling=greedy)]
    tap = StreamTap(engine, batch)

    # One buffer for the whole run, rewritten in place between steps, which
    # is what a CUDA-graph replay does to its static output.
    buf = torch.tensor([[1, 1, 1, 7], [1, 1, 1, 10]])
    scores = torch.zeros(2, 16)
    retained, events = [], [[], []]
    for step, nxt in enumerate(([7, 10], [11, 12], [13, 14])):
        buf[:, -1] = torch.tensor(nxt)
        done = tap(buf, scores)
        retained.append([list(r.generated_ids) for r in batch])
        for i, r in enumerate(batch):
            events[i] += drain(r)
        # The replay: overwrite the buffer, the scores and the returned mask.
        buf.fill_(-1)
        scores.fill_(float("nan"))
        done.fill_(False)
        check(f"step {step + 1}: retained ids unchanged after the buffer is overwritten",
              [list(r.generated_ids) for r in batch] == retained[-1],
              str([r.generated_ids for r in batch]))

    check("generated_ids hold every step's token",
          [r.generated_ids for r in batch] == [[7, 11, 13], [10, 12, 14]],
          str([r.generated_ids for r in batch]))
    check("retained ids are Python ints, not tensors",
          all(type(t) is int for r in batch for t in r.generated_ids))
    check("streamed events carry the same tokens, finish on the last",
          events[0] == [(7, None), (11, None), (13, "length"), None]
          and events[1] == [(10, None), (12, None), (14, "length"), None], str(events))
    check("event tokens are Python ints",
          all(type(e[0]) is int for ev in events for e in ev if e is not None))
    held = [k for k, v in vars(tap).items() if isinstance(v, torch.Tensor)]
    check("the tap holds no tensor across steps", not held, str(held))
    check("every row closed at max_tokens", all(r.done for r in batch))
    check("generation_tokens_total counts 2 rows x 3 steps",
          engine.generation_tokens_total == 6, str(engine.generation_tokens_total))

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
