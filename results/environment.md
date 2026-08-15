# Environment — verified from the machine

Everything below was read off this machine on **2026-08-09**, not assumed. Commands
are given so each line can be re-checked. Where the machine disagrees with the
project plan, the disagreement is stated rather than smoothed over.

## Verdict

The environment supports the benchmark as designed, with **three deviations**,
all recorded below and all constant across every experiment so comparability
holds:

1. The GPU power limit cannot be set from this session, so it stays at the
   stock 450 W rather than being explicitly locked ("Power limit").
2. vLLM 0.26.0's default GPU model runner cannot start under WSL2 at all; the
   project runs the V1 runner instead ("The UVA wall").
3. Model downloads run at ~2.9 MB/s, which makes fetching the ladder a
   multi-hour job that has to be scheduled around GPU work ("Download
   bandwidth").

## OS layer

| Item | Value | Source |
|---|---|---|
| Shell environment | WSL2 (Ubuntu), not Windows-native | `uname -r` → `6.6.114.1-microsoft-standard-WSL2`; `WSL_INTEROP` set |
| Distro | Ubuntu 26.04 LTS (Resolute Raccoon) | `/etc/os-release` |
| WSL version | 2.7.3.0, kernel 6.6.114.1-1, WSLg 1.0.73 | `wsl.exe --version` |
| Windows host | 10.0.26200.8875 | `cmd.exe /c ver` |
| CPU | Intel Core i9-14900KF (32 threads visible) | `/proc/cpuinfo`, `nproc` |
| RAM visible to WSL | 31 GiB (+8 GiB swap) | `free -g` |

**Deviation from plan §2 (WSL RAM):** there is no `.wslconfig` on the Windows
side, so WSL is running on the default allocation of ~50% of host RAM (31 GiB
of ~64 GiB). That is enough to stream the largest planned checkpoint
(Qwen2.5-32B int4, ~19 GB) through host memory, so nothing is raised for now.
If a 32B load fails with a host-memory error in E3, the fix is a `.wslconfig`
with `memory=48GB` followed by `wsl --shutdown`, and that would be recorded as
a mid-project environment change.

## GPU

| Item | Value | Source |
|---|---|---|
| Card | NVIDIA GeForce RTX 4090 | `nvidia-smi --query-gpu=name` |
| VRAM | 24564 MiB (24 GB) | `nvidia-smi --query-gpu=memory.total` |
| Driver (Windows host) | 595.95 | `nvidia-smi` |
| NVIDIA-SMI (WSL stub) | 595.61, at `/usr/lib/wsl/lib/nvidia-smi` | `which nvidia-smi` |
| CUDA version reported by driver | 13.2 | `nvidia-smi` header |
| Persistence mode | Enabled | `nvidia-smi --query-gpu=persistence_mode` |
| Idle VRAM already in use | ~1.4–1.9 GiB (Xwayland / Windows desktop) | `nvidia-smi` process table |

The card is confirmed as an actual 4090 with 24 GB; it was not assumed from the
plan. The ~1.5 GiB of desktop VRAM in use is why serving configs use
`gpu_memory_utilization: 0.90` rather than something closer to 1.0.

## Power telemetry — the WSL2 risk, resolved

This was the single biggest risk to the energy half of the project, so it was
tested before anything else was built.

`nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu --loop-ms=500`
**works under WSL2 and returns live, varying values.** A 12-sample capture at
idle moved between 20.70 W and 22.39 W with the clock and temperature fields
populated, so the readings are real rather than a frozen cached value. No
Windows-host-side fallback poller is needed; the fallback described in the
project brief is not used.

Fields confirmed available (all used by `power_log.py`):
`timestamp, power.draw, power.limit, clocks.sm, clocks.mem, temperature.gpu,
utilization.gpu, utilization.memory, memory.used, pstate`, plus the four
`clocks_throttle_reasons.*` fields the driver now names
`clocks_event_reasons.*` (`sw_power_cap`, `hw_thermal_slowdown`,
`sw_thermal_slowdown`, `hw_slowdown`).

**Actual sample cadence is ~620 ms, not the requested 500 ms**, because each
NVML query pays WSL2 passthrough overhead. This does not bias energy: joules
are integrated trapezoidally against the timestamps the driver reports, not
against an assumed interval. `power_log.py` records the observed p50/p95/max
sample interval in every run so the cadence is auditable.

**Idle baseline, measured:** 25.72 W over 130 s (150 s capture, first 20 s
discarded as settle) → `results/idle_baseline.json`. This includes the Windows
desktop's own GPU use; it is subtracted to produce *incremental* energy, and
both raw and incremental J/token are reported.

## Power limit — the one deviation

`nvidia-smi -pl 450` **fails from WSL2** with `Insufficient Permissions`, and
passwordless `sudo` is not configured here. The Windows-side binary
(`/mnt/c/Windows/System32/nvidia-smi.exe`, reachable through WSL interop) fails
the same way from a non-elevated context.

State of the card as it will be run:

```
Current Power Limit : 450.00 W
Requested           : 450.00 W
Default             : 450.00 W
Min / Max           : 150.00 W / 450.00 W
```

The limit is therefore at the card's default, which is also its maximum, and it
is the value the card will hold across a reboot without anyone setting it. The
plan's requirement is control and disclosure of the power limit; what is
delivered instead is **disclosure plus per-run verification**: `power_log.py`
records `power.limit` on every sample and raises a
`power_limit_changed_mid_run` flag if it ever differs, and the observed limit is
stored in every raw result. If the limit ever moves, affected points are
invalidated rather than blended.

To lock it explicitly, the owner would run this once from an **Administrator**
Windows terminal (not required for v0.1):

```
nvidia-smi -pl 450
```

## Storage

| Item | Value |
|---|---|
| WSL ext4 root (`/dev/sdd`) | 1007 GB total, 879 GB free |
| Repo location | `/mnt/d/...` — a Windows NTFS mount |
| HF cache | `~/.cache/huggingface` on **ext4** (1.3 GB used at start) |

**Deviation worth knowing:** the repository itself lives on `/mnt/d` (NTFS
passthrough), which plan §2 warns against. This is fine for the repo — the files
are small text and JSON — but would be ruinous for model weights, so the harness
hard-fails if `HF_HOME` points anywhere under `/mnt/`. Weights, the venv, and
all scratch output stay on ext4. 879 GB free comfortably covers the ~50 GB
ladder.

## Python and serving stack

| Item | Value |
|---|---|
| System Python | 3.14.4 — **too new for vLLM**, not used |
| Project Python | 3.12.13, in `~/wattbench-venv` (ext4), created with `uv` |
| vLLM | **0.26.0** |
| PyTorch | 2.11.0, built against CUDA 13.0 |
| transformers | 5.14.1 |
| flashinfer-python | 0.6.14 |
| xformers | not installed |
| `torch.cuda.is_available()` | `True` |

The CUDA 13.0 build runs against the 13.2 driver, which is the normal
forward-compatible direction.

**Load generator:** the installed vLLM exposes `vllm bench serve` (the modern
subcommand, not the older `benchmarks/benchmark_serving.py` script). Verified
present and used: `--backend vllm`, `--dataset-name random`,
`--random-input-len/--random-output-len/--random-range-ratio`, `--request-rate`,
`--burstiness`, `--max-concurrency`, `--num-prompts`, `--ignore-eos`,
`--percentile-metrics`, `--metric-percentiles`, `--goodput`, `--save-result`,
`--save-detailed`, `--seed`.

**Two flags from the plan-era vLLM no longer exist in 0.26.0** and were removed
from the harness after checking `--help=all`:

- `--swap-space` — gone; CPU KV swapping is no longer a server flag.
- `--disable-log-requests` — gone; per-request logging is already off by default.

`--enable-prefix-caching` / `--no-enable-prefix-caching` both exist; the harness
passes the negative form explicitly for E0–E3 so results are cache-free by
construction rather than by default.

## The UVA wall — vLLM 0.26 will not start under WSL2 out of the box

The first end-to-end smoke test failed at engine init, before any weights were
touched:

```
vllm/v1/worker/gpu/model_runner.py:216  self.req_states = RequestState(
vllm/v1/worker/gpu/states.py:33         self.all_token_ids = StagedWriteTensor(
vllm/v1/worker/gpu/buffer_utils.py:141  self._uva_buf = UvaBuffer(size, dtype)
vllm/v1/worker/gpu/buffer_utils.py:47   raise RuntimeError("UVA is not available")
RuntimeError: Engine core initialization failed.
```

vLLM 0.26.0 defaults to a new **V2 GPU model runner**, which allocates unified
virtual addressing (UVA) host buffers via `get_accelerator_view_from_cpu_tensor`.
WSL2's CUDA passthrough does not expose UVA, so `is_uva_available()` is false and
the engine cannot construct its request state. This is not a configuration
mistake; it is a hard capability gap in the WSL2 CUDA layer.

**Resolution:** run the V1 model runner, which has no UVA dependency:

```
VLLM_USE_V2_MODEL_RUNNER=0
```

`run.sh` exports this for every run, so the setting is identical across E0–E5
and no comparison is affected. It is not a free choice — the V2 runner is
0.26.0's default and presumably its faster path — so **every throughput number
in this project is a V1-runner number** and should not be compared against
published vLLM 0.26 figures that used the default. Downgrading vLLM was the
alternative; staying on 0.26.0 with the V1 runner was preferred because it keeps
one stack version across the whole project.

Anyone reproducing this on WSL2 will hit the same wall. It is the single most
useful line in `SETUP-WSL2.md`.

## Download bandwidth

Weight downloads run at roughly **2.9 MB/s** from this connection, which makes
the ~60 GB ladder a multi-hour unattended fetch rather than a background detail.
Three transfer backends were measured on the same 3.1 GB file, 60 s each:

| Backend | Downloaded in 60 s | Rate |
|---|---|---|
| Xet (huggingface_hub default) | 76 MB | 1.27 MB/s |
| `HF_HUB_DISABLE_XET=1` (plain HTTPS) | 171 MB | 2.85 MB/s |
| `HF_XET_HIGH_PERFORMANCE=1` | 1 MB | 0.02 MB/s |

Plain HTTPS wins by a factor of ~2 over the default, so `HF_HUB_DISABLE_XET=1`
is set in both `run.sh` and `fetch_models.sh`. `hf_transfer` is no longer a
lever: `huggingface_hub` has deprecated `HF_HUB_ENABLE_HF_TRANSFER` and ignores
it.

Practical consequence: `fetch_models.sh` pre-fetches the ladder in the order the
experiments need it, so no measured run stalls on a download, and downloads are
kept off the machine during measured runs.

## Repo state at start

Empty except for `CLAUDE.md`; no prior work to preserve. `git init` was run here
(`main` branch). `results/raw/` is explicitly **not** ignored.

## Addendum, 2026-08-13 — re-verified at the start of the E4 session

A new session on the same machine. Everything that governs comparability with
the E0–E3 series was re-read from the machine before any measurement, because a
silent driver update or a changed power limit would invalidate cross-series
comparison rather than merely add noise.

| Item | E0–E3 (2026-08-09) | Now (2026-08-13) | Comparable |
|---|---|---|---|
| Card / VRAM | RTX 4090, 24564 MiB | same | yes |
| Driver (host) | 595.95 | 595.95 | yes |
| NVIDIA-SMI (WSL stub) | 595.61 | 595.61 | yes |
| Power limit (current/default/max) | 450 / 450 / 450 W | same | yes |
| Persistence mode | Enabled | Enabled | yes |
| Kernel | 6.6.114.1-microsoft-standard-WSL2 | same | yes |
| Python / vLLM / torch | 3.12.13 / 0.26.0 / 2.11.0+cu130 | same | yes |
| RAM visible to WSL | 31 GiB | 31 GiB | yes |

**Nothing that affects comparability changed**, so the E0–E3 points and anything
measured in this session sit in one series.

Two non-comparability facts did move, both recorded for completeness: ext4 free
space is now 813 GB (was 879 GB) with the HF cache at 61 GB, which is the model
ladder on disk; and the idle GPU state at the moment of checking was 20.0–20.8 W
with SM clocks at their 210 MHz floor and no compute processes, i.e. quiescent
by the power criterion that this project trusts (`util_gpu_pct` read 16–40% at
the same time and, as established earlier, means nothing here).

Session baseline: `results/idle/idle__E2-recheck__20260813T104658.json`, mean
23.268 W. It reproduces the bimodality documented in METHODOLOGY: median
19.95 W against a mean of 23.27 W, p95 58.19 W, max 64.51 W. The intermittent
~62 W host consumer is still present on this machine and is still the reason the
idle *mean* is a fragile statistic.

## Addendum, 2026-08-14 — re-verified before the E4 measurements

Same machine, next day, new session. Re-read before touching the GPU, for the
same reason as the 2026-08-13 check: comparability with E0–E3 is an assumption
that has to be re-earned each session, not inherited.

| Item | E0–E3 (2026-08-09) | Now (2026-08-14) | Comparable |
|---|---|---|---|
| Card / VRAM | RTX 4090, 24564 MiB | same | yes |
| Driver (host) | 595.95 | 595.95 | yes |
| Power limit (current/default/max) | 450 / 450 / 450 W | same | yes |
| Kernel | 6.6.114.1-microsoft-standard-WSL2 | same | yes |
| Python / vLLM / torch | 3.12.13 / 0.26.0 / 2.11.0+cu130 | same | yes |
| matplotlib (analysis) | 3.11.1 | 3.11.1 | yes |
| RAM visible to WSL | 31 GiB | 31 GiB | yes |
| ext4 free | 879 GB | 813 GB (HF cache 61 GB) | n/a |

**Nothing that affects comparability changed.** E4 therefore joins the E0–E3
series rather than starting a new one.

GPU state at the check: 19.2–20.4 W, SM clocks pinned at the 210 MHz floor,
1.95 GiB VRAM held by the desktop, and **no compute processes**
(`--query-compute-apps` empty). Utilisation read 0–27% over the same six
samples, which is again why this project judges quiescence by power.

Session baseline: `results/idle/idle__E4__20260814T234054.json`, mean
**22.657 W**, median 20.10 W, p95 50.91 W, max 65.90 W. That is the **twelfth**
committed baseline, and it reproduces the documented bimodality: its median sits
inside the 18.7–22.2 W band that all twelve share, while 7.9% of its samples are
excursions above 40 W from a host consumer this project does not control, which
is what pulls the mean up.

That baseline is also what settled the open baseline-statistic question the same
evening (METHODOLOGY, "the idle *mean* is not a robust statistic here"): the
policy is now median subtraction. A second E4 baseline was measured 16 minutes
later under the new code — `idle__E4__20260814T235651.json`, median 20.21 W,
mean 20.50 W — and that window came back **completely excursion-free**. Two
windows on the same idle machine a quarter of an hour apart, one 7.9 % dirty and
one 0 %, is the clearest single demonstration of why "re-measure until the window
is clean" is a lottery rather than a protocol. E4 runs subtract the later one.

One field-name note for anyone reproducing: this driver's WSL stub rejects
`nvidia-smi --query-gpu=... -l N -c M` for the field set used here; loop in the
shell instead. `power_log.py` polls per-sample and is unaffected.

## Things to keep true across runs

- Close other GPU consumers before a measured run (plan §2). The desktop's
  ~1.5 GiB and low-single-digit utilisation are part of the measured idle
  baseline and are subtracted; a browser doing video decode mid-run is not, and
  would corrupt the point.
- Never move `HF_HOME` onto `/mnt/*`.
- Do not change power limit, driver, or serving flags inside an experiment
  series without rerunning the affected points.
