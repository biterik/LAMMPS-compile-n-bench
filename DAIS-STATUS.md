# DAIS GRACE benchmark — status & handoff

Last updated: 2026-07-07. Goal: run the GRACE benchmark on DAIS's NVIDIA GPUs
(H200, B200, RTX PRO 6000), same as the other machines. **H200 RESOLVED &
benchmarked** (1L 114.26, 2L 19.80 katom-step/s — ~2× A100). B200 / RTX PRO 6000
still pending a run.

## RESOLUTION (2026-07-07) — what unblocked H200

The CPU fallback was a chain of runtime issues, all now fixed:
1. **Driver not found** → TF logged `Could not find cuda drivers`; the driver
   `libcuda.so.1` lives in `/usr/lib64` (node-local, not in the nvidia wheels).
   Fix: prepend `/usr/lib64` to `LD_LIBRARY_PATH`.
2. **No CUDA wheels in the venv** → `tf-cuda` had plain `tensorflow==2.18.0`.
   Fix: `pip install 'tensorflow[and-cuda]==2.18.0'` (PTMP cache/tmp), then add
   all `nvidia/*/lib` dirs to `LD_LIBRARY_PATH`. After this, Python TF saw the H200.
3. **XLA GPU platform not registered** → the binary linked the *downloaded C-API
   tarball* `libtensorflow.so.2.18.0`, which lacks the XLA GPU platform the GRACE
   saved-models need → `NOT_FOUND: could not find registered platform with id`.
   Fix (strategy #2, matches raven): **rebuild against the venv's full
   `libtensorflow_cc.so.2`** — build with `PYTHON=.../tf-cuda/bin/python` and
   **no** `TF_LIB_FILE` (Python's `tf.sysconfig` supplies both the lib and the C
   headers; setting `TF_LIB_FILE` broke the header path). At runtime the venv's
   `site-packages/tensorflow` must be on `LD_LIBRARY_PATH` (it's the only TF copy
   now, so no registry split), plus the gcc/14 `libstdc++` `LD_PRELOAD` for CXXABI.

The working submit script is `bench/run-dais-grace.slurm` (self-contained, with a
pre-flight that verifies binary/input/models/CUDA/GPU and aborts loudly).

## Where the GRACE benchmark stands (all machines, fcc-Cu 16k)

| Machine | GPU/CPU | 1L katom-step/s | 2L katom-step/s | State |
|---|---|--:|--:|---|
| cmmg | 256-core EPYC (CPU) | 10.03 | 3.61 | ✅ done |
| raven | 1× A100 (TF-CUDA) | 57.53 | 12.82 | ✅ done |
| viper | 1× MI300A (TF-ROCm) | 4.33 | 0.94 | ✅ done (slow; TF-ROCm) |
| **DAIS** | 1× H200 (TF-CUDA) | 114.26 | 19.80 | ✅ done (~2× A100) |
| **DAIS** | B200 / RTX PRO 6000 | — | — | ⏳ pending run (Blackwell — may hit TF 2.18/CUDA limit) |

## DAIS facts (confirmed 2026-07-06)

- User `biterik`, `HOME=/u/biterik`. DAIS has its own `/dais/fs/scratch/biterik`,
  **but DAIS also mounts Viper's `/viper/ptmp2`** — we use that so the repo and
  models are already present (no re-transfer).
- **Work dir:** `/viper/ptmp2/biterik/gracework-dais` (DAIS build + venv here).
- **Models (shared w/ Viper):** `/viper/ptmp2/biterik/gracework/grace-cache/`
  → `GRACE-1L-SMAX-OMAT-large`, `GRACE-2L-SMAX-OMAT-medium` (TF saved_models).
- **Repo (shared):** `/viper/ptmp2/biterik/gracework/LAMMPS-compile-n-bench`.
- **Partitions:** `gpu` (exclusive whole 8-GPU node, 24h) · `gpu1` (**shared,
  <4 GPU — use this for 1-GPU jobs**) · `gpudev` (debug 15min; `srun` to it gave
  "Invalid qos specification" — needs a `--qos`, unresolved) · `small` (CPU).
- **GPU gres tokens:** `gpu:h200:1` (daisg101-117), `gpu:b200:1` (daisg201-210),
  `gpu:rtx_pro_6000:1` (daisg301-302).
- **`gpu1` requires an explicit `--mem`** (≤375000 MB/GPU). We use `--mem=120000`.
- **Modules:** `gcc/12,14,15`; `cuda/12.8`, `cuda/13.0`; `python-waterboa/2024.06`
  (Python 3.12.4), `python-waterboa/2025.06`; `cmake/3.30` (assumed present).
  MPI is hierarchical (not needed here — serial build).

## What was built (works)

- `build-lammps-dais-fork.sh`: thermoatoms fork (pinned `24da74cd…`), **serial
  (`BUILD_MPI=off`), no Kokkos, `GRACE_TF=on` (TF-CUDA)**. GRACE runs on the GPU
  via TensorFlow, so no MPI and no GPU-arch compilation are needed — one binary
  for all three GPU types. Build is fast (no nvcc/hipcc). **It compiles.**
- **Binary:** `/viper/ptmp2/biterik/gracework-dais/lammps-fork/build-dais-fork/lmp_dais_fork`
- **venv:** `/viper/ptmp2/biterik/gracework-dais/tf-cuda` (python-waterboa/2024.06, Py 3.12).
- **Runner:** `/viper/ptmp2/biterik/gracework-dais/run_dais.sh` (self-contained,
  see below). Submit examples use `sbatch --mem=120000 --partition=gpu1
  --gres=gpu:<type>:1 --export=GPU=<name> ... run_dais.sh`.
- A repo `bench/submit-dais-grace.slurm` exists but was flaky on DAIS (stale copy
  / env issues); the standalone `run_dais.sh` is what we've been iterating.

## The runtime problem chain (what we hit and fixed, in order)

1. **Models "not found"** — job looked in `$HOME/.cache/grace`. The submit
   script's `GRACE_CACHE` default didn't apply (env not propagated / stale copy).
   **Fix:** pass models dir explicitly; standalone `run_dais.sh` hardcodes
   `GC=/viper/ptmp2/biterik/gracework/grace-cache`.
2. **`gpu1` rejected job** — "Memory limit must be provided for shared jobs".
   **Fix:** `--mem=120000`.
3. **`libstdc++` `CXXABI_1.3.15` not found** — binary built with gcc/14 but the
   loader picked python-waterboa's older `libstdc++` (pulled in via TF's runpath;
   `RPATH` beats `LD_LIBRARY_PATH`). **Fix:** `LD_PRELOAD` gcc/14's
   `libstdc++.so.6` **and run the binary directly (not via `srun`)** — `srun`
   reset the environment so the preload was lost.
4. **`undefined symbol … absl::lts_20230802 / descriptor_table_tsl…`** — TF
   version mismatch. **Root cause (from the fork's `cmake/Modules/Packages/
   ML-PACE.cmake`):** GRACE uses cppflow + the **C-API `libtensorflow.so.2`**.
   The cmake first tries the venv's `libtensorflow_cc.so.2`; if absent it
   **downloads the official `libtensorflow-gpu-linux 2.18.0`** and links that.
   The pip wheels we used don't ship `libtensorflow_cc.so.2`, so the binary is
   linked to **downloaded TF 2.18.0** (its `libtensorflow.so.2.18.0` +
   `libtensorflow_framework.so.2.18.0` sit in `build-dais-fork/lib/`). Mixing a
   different-version framework from a venv (2.19/2.21) on `LD_LIBRARY_PATH`
   caused the undefined symbols.
5. **`Cannot dlopen some GPU libraries` → `platform Host` (CPU)** — the current
   blocker. The downloaded `libtensorflow-gpu 2.18.0` needs the CUDA/cuDNN it was
   built against (**CUDA 12.3 / cuDNN 9.x** per TF's build config). Findings:
   - `pip install tensorflow==2.18.0` (plain) installs **no CUDA wheels**
     (`nvidia/*/lib` did not exist) — confirmed by a verbose `srun` check.
   - `pip install 'tensorflow[and-cuda]==2.18.0'` was then run, but the job
     **still** reported "Cannot dlopen some GPU libraries" and fell back to CPU.
   - The exact missing `.so` is **not yet captured** (batch run used
     `TF_CPP_MIN_LOG_LEVEL=3`, which hides the "Could not load dynamic library
     'X'" lines).

## Immediate next step (do this first on a fresh start)

Get the **exact missing library** with a verbose run on `gpu1`, and list what
CUDA wheels are actually installed:

```bash
srun --partition=gpu1 --gres=gpu:h200:1 --mem=120000 --cpus-per-task=8 --time=00:10:00 bash -c '
  module purge; module load gcc/14
  SUB=/viper/ptmp2/biterik/gracework-dais
  echo "== nvidia wheel lib dirs =="; ls -d $SUB/tf-cuda/lib/python*/site-packages/nvidia/*/lib 2>&1
  echo "== cudnn libs =="; ls $SUB/tf-cuda/lib/python*/site-packages/nvidia/cudnn/lib/ 2>&1 | head
  export LD_PRELOAD=$(realpath "$(g++ -print-file-name=libstdc++.so.6)")
  export LD_LIBRARY_PATH=$(dirname "$LD_PRELOAD"):$(ls -d $SUB/tf-cuda/lib/python*/site-packages/nvidia/*/lib 2>/dev/null|paste -sd: -):$(ls -d $SUB/tf-cuda/lib/python*/site-packages/tensorflow):$LD_LIBRARY_PATH
  export TF_CPP_MIN_LOG_LEVEL=0
  cd $SUB
  $SUB/lammps-fork/build-dais-fork/lmp_dais_fork -in in.grace_bench -var pstyle grace \
    -var model /viper/ptmp2/biterik/gracework/grace-cache/GRACE-1L-SMAX-OMAT-large -var nsteps 0 2>&1 \
    | grep -iE "could not load|dlerror|cannot open|dlopen" | head'
```

Then satisfy that one library from: the `nvidia-*-cu12` wheel it belongs to, the
DAIS **`cuda/12.8` module**, or a **`cudnn` module** (`module avail cudnn`), and
add its dir to `LD_LIBRARY_PATH`.

## Candidate strategies if the wheel route keeps fighting

1. **System CUDA instead of wheels:** `module load cuda/12.8` (+ a cuDNN module)
   and put their `lib64` on `LD_LIBRARY_PATH` — provides standard SONAMEs the
   2.18 C-API expects. Simplest if DAIS ships a matching cuDNN.
2. **Make the fork use the venv TF (avoid the 2.18 download):** if a pip TF that
   *does* ship `libtensorflow_cc.so.2` is used, cmake links the venv TF directly
   (fully consistent, and could be a newer, Blackwell-capable version). Check
   whether any available `tensorflow`/`tensorflow[and-cuda]` wheel contains
   `libtensorflow_cc.so.2`; if so, rebuild against it.
3. **Accept H200 only:** TF 2.18's CUDA is 12.x/Hopper-era — good for **H200**,
   but **B200 / RTX PRO 6000 are Blackwell (sm_100)** needing CUDA 12.8+. So even
   once H200 works, Blackwell likely needs strategy #2 (newer TF). That "the
   silicon is ready but the TF/CUDA stack isn't" is itself a legitimate result.

## Reusable `run_dais.sh` (current form)

Self-contained: sources module init, `module load gcc/14`, `LD_PRELOAD` gcc
libstdc++, `LD_LIBRARY_PATH` = gcc-lib : TF_PYLIB : nvidia-wheel-libs, runs the
binary **directly** (no `srun`), 1L then 2L, logs `log.grace_{1l,2l}_dais_$GPU`.
Submit with `sbatch --mem=120000 --partition=gpu1 --gres=gpu:<type>:1
--export=GPU=<name> ... run_dais.sh`. (Recreate from the chat if lost; the key
ingredients are the preload + direct run + matching-version TF+CUDA.)
