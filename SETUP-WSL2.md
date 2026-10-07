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

## 6b. Building llama.cpp with CUDA when you have no `sudo` and no CUDA toolkit

E4 needs a second serving stack, and llama.cpp's CUDA backend has to be
compiled — the project's Linux release archives are CPU-only. That normally
means `apt install nvidia-cuda-toolkit`, which needs root. It is not required:
**the CUDA toolkit is installable from PyPI**, and the resulting `nvcc` compiles
and links a working CUDA build.

The one trap: torch and vLLM leave a *mixed* toolkit behind in their venv —
here `nvcc` 13.3.73 alongside `cuda-runtime` 13.0.96 — and CCCL's headers refuse
it outright:

```
cuda_toolkit.h:41: error: "CUDA compiler and CUDA toolkit headers are incompatible,
                           please check your include paths"
```

So build the toolchain in its **own** venv, where the versions resolve
together, and leave the measurement venv untouched:

```bash
uv venv ~/wattbench-tools-venv --python 3.12
uv pip install --python ~/wattbench-tools-venv/bin/python \
  cmake ninja nvidia-cuda-nvcc nvidia-cuda-runtime nvidia-cuda-cccl \
  nvidia-cublas nvidia-nvvm nvidia-cuda-crt
```

The pip layout is not the layout CMake expects, so point a small shim tree at
it: `bin`, `include` and `nvvm` symlinked to the package, plus a real `lib`
holding the unversioned `libcublas.so` / `libcudart.so` aliases that the linker
looks for and a `stubs/libcuda.so` pointing at WSL's own driver library:

```bash
CU=~/wattbench-tools-venv/lib/python3.12/site-packages/nvidia/cu13
SHIM=~/src/cuda-shim
mkdir -p "$SHIM/lib/stubs"
ln -sfn "$CU/bin" "$SHIM/bin"; ln -sfn "$CU/include" "$SHIM/include"
ln -sfn "$CU/nvvm" "$SHIM/nvvm"; ln -sfn lib "$SHIM/lib64"
for f in "$CU"/lib/*; do ln -sf "$f" "$SHIM/lib/$(basename "$f")"; done
(cd "$SHIM/lib" && for l in libcublas libcublasLt libcudart libnvrtc; do
   ln -sf "$(ls $l.so.* | head -1)" "$l.so"; done)
ln -sf /usr/lib/wsl/lib/libcuda.so.1 "$SHIM/lib/stubs/libcuda.so"
```

Then configure and build. `-rpath-link` is the flag people miss: without it the
final link cannot resolve `libcudart.so.13` through the intermediate
`libggml-cuda.so` and fails with a screen of undefined CUDA symbols even though
every library is present.

```bash
export PATH=~/wattbench-tools-venv/bin:$PATH
TCU=~/wattbench-tools-venv/lib/python3.12/site-packages/nvidia/cu13/lib
LDF="-Wl,-rpath-link,$HOME/src/cuda-shim/lib -Wl,-rpath-link,$TCU -Wl,-rpath,$TCU"
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=$HOME/src/cuda-shim/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCUDAToolkit_ROOT=$HOME/src/cuda-shim \
  -DCMAKE_EXE_LINKER_FLAGS="$LDF" -DCMAKE_SHARED_LINKER_FLAGS="$LDF" \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --target llama-server -j 24
```

`89` is Ada (RTX 40-series); use `86` for Ampere consumer cards. GCC 15.2 is
accepted by nvcc 13.3 here — no host-compiler downgrade needed.

Two notes for anyone benchmarking llama.cpp against vLLM afterwards:

- **`-c` is the total KV context, divided across `--parallel` slots**, where
  vLLM's `--max-model-len` is per sequence. To give each of 32 slots the same
  4096-token budget a vLLM sequence gets, pass `-c 131072`.
- **Build it before you measure, not between points.** The build saturates the
  CPU for several minutes; this project's rule against downloads during a
  measured run applies to compiles for the same reason.

---

## 6c. The PyTorch stack needs nothing extra — except for `torch.compile`

`pytorch_server.py` runs on the same venv: torch and transformers are already
there because vLLM depends on them, and recording one torch version for both
arms is the point. Check once:

```bash
~/wattbench-venv/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
```

The compiled control (`configs/e4/pytorch_c8_compile.yaml`) is the one place
that can fail here. `mode=reduce-overhead` goes through Inductor, which
generates Triton kernels — Triton ships its own `ptxas`, so no CUDA toolkit is
needed, same as the FlashInfer note in §4 — but Inductor also needs a C/C++
compiler on `PATH` for its wrapper code. If the server log shows
`CppCompileError` or `No such file or directory: 'g++'`:

```bash
sudo apt install -y build-essential      # needs sudo; if you have none, skip the control
```

With no `sudo`, run `./run_e4_pytorch.sh --skip-compile`: the three plain
points and the guard do not touch Inductor. Set `TORCHINDUCTOR_CACHE_DIR` to an
ext4 path, not `/mnt/*`, for the same reason as `HF_HOME` (§2); it is recorded
in provenance when set.

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
