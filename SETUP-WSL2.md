# Reproducing this on Windows + WSL2

Written for the configuration half this benchmark's audience actually has: an
NVIDIA GPU in a Windows desktop, with vLLM run under WSL2 because vLLM and
SGLang are Linux-only.

Everything below was executed on this machine. Where something did **not** work,
that is stated with the actual error, because the three genuinely hard parts of
this setup all fail with messages that do not obviously point at WSL2.

**The three things that will cost you an evening if nobody tells you:**

1. vLLM 0.26's default model runner cannot start under WSL2 at all → §4
2. FlashInfer's JIT sampler needs `nvcc`, which you probably do not have → §4
3. You cannot set the GPU power limit from inside WSL2, or from Windows without
   an elevated shell → §5

---

## 1. Versions this was verified on

| Component | Version |
|---|---|
| Windows | 10.0.26200 |
| WSL | 2.7.3.0, kernel 6.6.114.1 |
| Distro | Ubuntu 26.04 LTS |
| GPU | NVIDIA GeForce RTX 4090, 24 GB |
| Driver (Windows host) | 595.95 |
| CUDA reported by driver | 13.2 |
| Python | 3.12.13 (venv via `uv`) |
| vLLM | 0.26.0 |
| PyTorch | 2.11.0 (built against CUDA 13.0) |

You do **not** install an NVIDIA driver inside WSL2. The Windows driver is
projected in; `nvidia-smi` lives at `/usr/lib/wsl/lib/nvidia-smi`. Installing a
Linux driver in the distro breaks this.

You also do not need the CUDA toolkit. The PyTorch and vLLM wheels ship their
own kernels. (One consequence bites in §4.)

---

## 2. Filesystem: keep weights off `/mnt/*`

NTFS passthrough (`/mnt/c`, `/mnt/d`) is slow enough to dominate model-load
time. Weights, the venv, and scratch output all belong on the WSL ext4
filesystem.

```bash
export HF_HOME="$HOME/.cache/huggingface"   # ext4 — NOT /mnt/anything
```

This repository's `run.sh` hard-fails if `HF_HOME` points under `/mnt/`, because
the failure mode otherwise is "everything works but is mysteriously slow."

Budget ~50 GB for a 1.5B→32B int4 ladder. Check with `df -h /`.

The repo itself can live on `/mnt/d` — it is small text and JSON, and git
handles it fine. Only the weights matter.

---

## 3. Environment

```bash
# uv is not required, but Ubuntu 26.04 ships Python 3.14, which vLLM does not
# support yet. A pinned 3.12 venv avoids the whole problem.
uv venv --python 3.12 ~/wattbench-venv
VIRTUAL_ENV=~/wattbench-venv uv pip install vllm
```

Verify the GPU is visible before going further:

```bash
~/wattbench-venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# True 13.0
```

A CUDA 13.0 build against a 13.2 driver is the normal forward-compatible
direction and is fine.

### RAM

Weights stream through host memory during load. WSL defaults to about half of
host RAM (31 GiB of 64 GiB here), which was enough for everything in this
project. If a large model dies with a host-memory error, create
`C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=48GB
```

then `wsl --shutdown` from Windows. Treat that as a mid-project environment
change: rerun anything already measured.

---

## 4. The two WSL2 failures that actually stop you

### `RuntimeError: UVA is not available`

```
vllm/v1/worker/gpu/buffer_utils.py:47  raise RuntimeError("UVA is not available")
RuntimeError: Engine core initialization failed.
```

vLLM 0.26 defaults to a new **V2 GPU model runner** that allocates unified
virtual addressing host buffers. WSL2's CUDA passthrough does not expose UVA, so
the engine cannot construct its request state. This happens before any weights
load, and the traceback points at vLLM internals rather than at WSL2.

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
```

The V1 runner has no UVA dependency. This is not free — V2 is 0.26's default and
presumably its faster path — so **throughput measured this way should not be
compared against published vLLM 0.26 numbers from a native-Linux host.**

### `RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`

```
flashinfer/jit/cpp_ext.py:61  in get_cuda_path
```

FlashInfer JIT-compiles its top-k/top-p sampling kernel on first use, which
needs `nvcc`. The wheels ship prebuilt kernels for everything else, so most
people never install a CUDA toolkit — and then hit this at the first sampled
token.

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

vLLM's own error message recommends this. It falls back to the PyTorch-native
sampler; sampling is a negligible share of decode time.

Installing `nvidia-cuda-toolkit` from apt is the other option, but Ubuntu's
package is CUDA 12.4 against a torch built for 13.0 — not worth the risk for a
sampling kernel.

### Together

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HOME="$HOME/.cache/huggingface"
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --max-model-len 4096 \
  --gpu-memory-utilization 0.90 --max-num-seqs 256 \
  --dtype bfloat16 --no-enable-prefix-caching
```

---

## 5. Power telemetry and the power limit

### Telemetry works — this was the biggest risk and it is fine

```bash
nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu \
           --format=csv,noheader --loop-ms=500
```

returns live, varying values under WSL2. No Windows-side poller is needed. All
of these fields are available, including the throttle reasons the driver now
names `clocks_event_reasons.*`:

```
timestamp, power.draw, power.limit, clocks.sm, clocks.mem, temperature.gpu,
utilization.gpu, utilization.memory, memory.used, pstate,
clocks_throttle_reasons.{sw_power_cap,hw_thermal_slowdown,sw_thermal_slowdown,hw_slowdown}
```

**Two caveats that matter if you integrate energy:**

- The delivered cadence is **~620 ms for a requested 500 ms** — NVML queries pay
  passthrough overhead.
- The cadence is **not uniform**: queries return faster at idle than under load.

So integrate against the driver's own timestamps, not an assumed interval. An
unweighted mean of samples is not the time-weighted mean here, and using it
biases energy toward whatever the card was doing when NVML answered fastest.

### The power limit cannot be set from WSL2

```
$ nvidia-smi -pl 450
Failed to set power management limit for GPU 00000000:01:00.0: Insufficient Permissions
```

`sudo` inside WSL2 does not help — this is the passthrough stub, not a real
management interface. The Windows-side binary is reachable through interop at
`/mnt/c/Windows/System32/nvidia-smi.exe` and fails the same way from a
non-elevated context.

Two options:

1. **Set it once from an Administrator Windows terminal:** `nvidia-smi -pl 450`
2. **Disclose and verify instead of controlling** — what this project does. The
   card runs at its stock limit (450 W, which is also its maximum and its
   power-on default); `power.limit` is read on every telemetry sample and any
   mid-run change invalidates the point.

Either way, **state the limit in your results.** A benchmark that does not is
not comparable with one that does.

---

## 6. Downloads

Measured here on the same 3.1 GB file, 60 s each:

| Backend | Downloaded | Rate |
|---|---|---|
| Xet (huggingface_hub default) | 76 MB | 1.27 MB/s |
| `HF_HUB_DISABLE_XET=1` (plain HTTPS) | 171 MB | 2.85 MB/s |
| `HF_XET_HIGH_PERFORMANCE=1` | 1 MB | 0.02 MB/s |

```bash
export HF_HUB_DISABLE_XET=1
```

`hf_transfer` is no longer a lever: `huggingface_hub` has deprecated
`HF_HUB_ENABLE_HF_TRANSFER` and ignores it.

Your connection may differ — but *measure* before assuming the default is
fastest, because here it was 2× slower than plain HTTPS.

**Pre-fetch the whole ladder before benchmarking.** At ~3 MB/s a 60 GB ladder is
a multi-hour job, and you do not want a sweep stalling mid-run on a download —
or, worse, a download competing with a measured run. `fetch_models.sh` does this
in the order the experiments need.

One CLI trap: `hf download REPO --exclude "*.pth" "*.msgpack"` silently
downloads **nothing** — the extra patterns are parsed as filenames, and the
command exits 0. Watch the cache size, not the exit code.

---

## 7. Before every measured run

- **Close other GPU consumers**, browser hardware acceleration included. The
  desktop's residual draw belongs in your idle baseline; a browser decoding
  video mid-run does not.
- **Measure a fresh idle baseline.** Idle draw moves between sessions with
  desktop activity, driver state and ambient temperature. This repo refuses to
  run against a baseline older than 12 hours rather than silently subtracting a
  stale number.
- **Check nothing else is on the card:** `nvidia-smi` should show only Xwayland
  and your desktop's ~1.5 GB.

---

## 8. Quick checklist

```bash
uname -r                                    # ...-microsoft-standard-WSL2
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv
df -h /                                     # ext4 space for weights
free -g                                     # RAM visible to WSL
echo $HF_HOME                               # must NOT be under /mnt/
~/wattbench-venv/bin/python -c "import torch;print(torch.cuda.is_available())"
timeout 6 nvidia-smi --query-gpu=timestamp,power.draw --format=csv,noheader --loop-ms=500
```

If the last command prints varying wattages, the energy half of this
methodology will work on your machine.
