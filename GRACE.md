# GRACE builds & benchmark

This adds **GRACE** (Graph Atomic Cluster Expansion) builds and a GRACE
throughput benchmark alongside the existing PACE/ACE benchmark, built from the
**thermoatoms LAMMPS fork** (which also carries the efficient ACE MC/MD). It is
kept as a **separate set of scripts** (`*-fork.sh`, `*-grace.slurm`) and a
**separate results table** from the 256k-atom PACE benchmark.

> Why the fork: GRACE is **not in upstream LAMMPS** and not on a stable tag.
> The thermoatoms fork (base `patch_11Feb2026`, pinned commit
> `24da74cd73…`) ships the GRACE pair styles *and* the fast ACE MC. See
> `FORK-ANALYSIS.md`.

---

## 1. Which GRACE potential — SMAX-OMAT (recommended general-purpose)

GRACE foundation models are **universal** (all elements), so they run on the
*same fcc-Cu system* as the PACE benchmark — just point at a different model
file. We use the maker's recommended general-purpose family, **SMAX-OMAT**
(maximum-entropy prior + OMat24 accuracy), in two architectures:

| Role | Model (Full Name) | Arch | Notes |
|---|---|---|---|
| **1-layer** (local, fast) | `GRACE-1L-SMAX-OMAT-large` | single-layer | also exported to **FS** (no TensorFlow) for the headline cross-machine point |
| **2-layer** (semi-local, accurate) | `GRACE-2L-SMAX-OMAT-medium` | two-layer (message-passing) | TensorFlow only; the heavy/accurate tier |

SMAX models use an element-dependent cutoff (5.0–7.5 Å); nothing to set for Cu.
(Alternatives if you prefer: the `-OAM` or base `-OMAT` families — same scripts,
just change the model names.)

### Getting the models

```bash
pip install tensorpotential            # provides the grace_models / gracemaker CLI
grace_models list                      # see all models + their CHECKPOINT field
grace_models download GRACE-1L-SMAX-OMAT-large
grace_models download GRACE-2L-SMAX-OMAT-medium
# default cache: $HOME/.cache/grace/<Full-Name>/   (override with $GRACE_CACHE)
```

### FS export does NOT work for these foundation models → use TensorFlow

We tried exporting the 1-layer model to GRACE/FS (`grace_utils … export -sf`) to
get a fast, TensorFlow-free CPU path. It **fails** for the SMAX-OMAT foundation
models with `KeyError: 'RadialBasis'` — their radial basis is not the linear
"FS" form the C++ exporter needs (FS export only works for models fitted with the
FS architecture, not the full GRACE foundation models). **Confirmed on cmmg,
2026-07-01.**

So we run the 1-layer model through **TensorFlow** too, via `grace/1layer/chunk`,
straight from the downloaded `saved_model` (no export step):

```bash
grace_models download GRACE-1L-SMAX-OMAT-large   # saved_model dir in $GRACE_CACHE
grace_models download GRACE-2L-SMAX-OMAT-medium
# then the submit scripts point pair_coeff at $GRACE_CACHE/<model> — nothing to export.
```

Consequence: every GRACE run here needs a **TF-enabled** LAMMPS (`GRACE_TF=on`),
and the CPU runs pay the CPU-TensorFlow performance tax (see §6).

---

## 2. GRACE pair styles in this fork (and the GPU limitation)

| `pair_style` | model | TensorFlow? | MPI | GPU |
|---|---|---|---|---|
| `grace` | 1L / 2L | **yes** | 1L yes / 2L single-proc | via TF (CUDA / ROCm) |
| `grace/1layer/chunk` | 1L | yes | yes | via TF |
| `grace/2layer/chunk` | 2L | yes | yes | via TF |
| `grace/2layer/parallel` | 2L | yes | yes | via TF |
| `grace/fs` | FS | **no** | yes | **CPU only** |

**There is NO `grace/fs/kk`** (Kokkos FS) in this fork — so the only way to put
GRACE *on a GPU* is the TensorFlow path with a GPU TensorFlow build. This is why:

- **NVIDIA (raven):** GRACE on the A100 works through **TensorFlow-CUDA**.
- **AMD MI300A (viper):** needs **TensorFlow-ROCm** for gfx942 — **experimental**,
  may not work; no Kokkos fallback. (PACE/ACE on the APU is unaffected.)

If you later get `grace/fs/kk` added to the fork (ask Sarath — they pull GRACE
from Yury), the GPU story becomes a clean Kokkos path on both vendors.

---

## 3. Per-machine plan

| Machine | Build script | GRACE on GPU? | What runs |
|---|---|---|---|
| **cmmg** (EPYC, CPU) | `build-lammps-cmmg-fork.sh` | n/a | `grace/fs` (1L) + TF `grace/2layer/chunk` (2L) |
| **viper-cpu** (Genoa, CPU) | `build-lammps-viper-cpu-fork.sh` | n/a | same as cmmg |
| **raven** (A100) | `build-lammps-raven-fork.sh` | ✅ TF-CUDA | `grace` (1L) + `grace/2layer/chunk` (2L) on GPU |
| **viper** (MI300A) | `build-lammps-viper-fork.sh` | ⚠️ TF-ROCm (experimental) | `grace` (1L/2L) if TF-ROCm works |

The **cross-machine-comparable point is the 1-layer model**: `grace/fs` on the
CPU machines vs the TF `grace` on the GPUs — *same model, different evaluator*,
so katom-step/s is comparable as "the 1L model on that machine". The **2-layer**
model is the accuracy tier (CPU + raven-GPU; viper-GPU only if TF-ROCm works).

---

## 4. TensorFlow setup (per machine)

The fork's `cmake/Modules/Packages/ML-PACE.cmake` finds TensorFlow by importing
it from a Python env. Provide the right wheel and pass `PYTHON=` to the build
(and `TF_PYLIB=` to the submit script for the runtime libs).

> **Put the venv, pip cache, and model cache in PTMP — never `$HOME`.** The TF
> stack pulls multi-GB `nvidia-cu12` wheels and the GRACE models are large; the
> cluster `$HOME` quota is small and *will* overflow. Before installing:
> ```bash
> export PTMP=/u/$USER/PTMP                 # cmmg: /u/biterik/PTMP
> export PIP_CACHE_DIR=$PTMP/.pipcache TMPDIR=$PTMP/.tmp
> export GRACE_CACHE=$PTMP/gracework/grace-cache   # grace_models downloads here
> mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$GRACE_CACHE"
> python -m venv $PTMP/gracework/tf-cpu     # venv in PTMP, not ~/tf-cpu
> ```
> The `~/tf-*` paths in the table below are shorthand — use the PTMP venv path.
> The submit scripts read `$GRACE_CACHE` for the model dirs (default `$HOME/.cache/grace`).

| Machine | TensorFlow wheel | Build | Run |
|---|---|---|---|
| cmmg / viper-cpu | `pip install tensorflow-cpu` | `PYTHON=~/tf-cpu/bin/python ./build-…-fork.sh` | `TF_PYLIB=~/tf-cpu/.../tensorflow` |
| raven | `pip install tensorflow` (CUDA) | `PYTHON=~/tf-gpu/bin/python ./build-lammps-raven-fork.sh` | `TF_PYLIB=~/tf-gpu/.../tensorflow` |
| viper (APU) | `pip install tensorflow-rocm` | `GRACE_TF=on PYTHON=~/tf-rocm/bin/python ./build-lammps-viper-fork.sh` | `TF_PYLIB=~/tf-rocm/.../tensorflow` |

To build **without** TensorFlow (FS + ACE only — e.g. the FS-only CPU point, or
viper with no TF-ROCm), pass `GRACE_TF=off` (sets `-D NO_GRACE_TF=ON`).
Alternatively set `TF_LIB_FILE=/path/libtensorflow_cc.so.2` directly.

> **raven GRACE-TF fallback.** Linking a CUDA libtensorflow into the
> nvcc_wrapper/Kokkos-CUDA binary can hit CUDA-runtime version skew. If the
> combined build fails, build raven with `GRACE_TF=off` (keeps `pace/kk` +
> `grace/fs` CPU) and make a *separate* plain-g++ + TF-CUDA build just for the
> GRACE-GPU runs (same cmake, drop the Kokkos flags, keep the GRACE_TF block).

---

## 5. Build → run → compare

```bash
# on each machine's LOGIN node, from a /ptmp/$USER work dir:
PYTHON=~/tf-*/bin/python ./build-lammps-<machine>-fork.sh     # -> lammps-fork/build-<machine>-fork/lmp_<machine>_fork

# put the inputs + models in the work dir:
cp mpcdf-lammps/bench/in.grace_bench .
cp seed/1/FS_model.yaml .                                     # 1L FS export
#   (2L/1L TF saved_models are read from ~/.cache/grace/... by the submit scripts)

sbatch mpcdf-lammps/bench/submit-cmmg-grace.slurm            # FS 1L (+ optional 2L TF)
sbatch mpcdf-lammps/bench/submit-viper-cpu-grace.slurm
sbatch mpcdf-lammps/bench/submit-raven-grace.slurm           # TF-CUDA 1L + 2L on the A100
sbatch mpcdf-lammps/bench/submit-viper-grace.slurm           # TF-ROCm, experimental

# gather the log.grace_* onto one host, then:
bash mpcdf-lammps/bench/compare-grace.sh
```

**System size:** `in.grace_bench` defaults to **16³ fcc = 16,384 atoms** (a
common size that fits even a 2-layer model on a 40 GB A100), `nsteps 100`. This
is intentionally smaller than the 256k PACE benchmark — see the header of
`in.grace_bench`. Keep nx/ny/nz and nsteps identical across machines for a fair
GRACE-vs-GRACE comparison (1L vs 2L may use different nsteps on purpose; the
submit scripts lower nsteps for 2L).

---

## 6. Results (fcc-Cu, 16,384 atoms)

For reference, ACE/PACE on a full cmmg node (256k atoms) runs **393** katom-step/s;
CPU-TensorFlow GRACE is ~40× (1L) to ~110× (2L) slower per atom — the CPU-TF tax.
**One A100 (TF-CUDA) is 5.7× (1L) and 3.5× (2L) faster than the full 256-core
cmmg node** — GRACE belongs on the GPU. The 2L GPU run measured 64.5 µs/atom,
right in the published A100 range (~27–120 µs/atom), vs 64,514 µs/atom on cmmg CPU.

| Machine | Model | pair_style | TF? | atoms | nsteps | katom-step/s | Pair% | Notes |
|---|---|---|:--:|--:|--:|--:|--:|---|
| cmmg | 1L-SMAX-OMAT-L | grace/1layer/chunk | yes | 16384 | 100 | **10.03** | 81.4% | full node, 256 ranks (2026-07-01) |
| cmmg | 2L-SMAX-OMAT-M | grace/2layer/chunk | yes | 16384 | 20 | **3.61** | 97.2% | full node, CPU TF (slow) |
| **raven** | 1L-SMAX-OMAT-L | grace | yes | 16384 | 100 | **57.53** | 98.1% | 1 A100, TF-CUDA (2026-07-01) |
| **raven** | 2L-SMAX-OMAT-M | grace/2layer/chunk | yes | 16384 | 50 | **12.82** | 99.7% | 1 A100, TF-CUDA (2026-07-01) |
| viper-cpu | 1L-SMAX-OMAT-L | grace/1layer/chunk | yes | 16384 | 100 | _pending_ | | full node, 128 ranks |
| viper-cpu | 2L-SMAX-OMAT-M | grace/2layer/chunk | yes | 16384 | 20 | _pending_ | | CPU TF (slow) |
| viper | 1L-SMAX-OMAT-L | grace | yes | 16384 | 100 | _pending_ | | 1 MI300A, TF-ROCm (experimental) |
| viper | 2L-SMAX-OMAT-M | grace/2layer/chunk | yes | 16384 | 50 | _pending_ | | 1 MI300A, TF-ROCm (experimental) |

---

## 7. Caveats

- **viper-GPU GRACE is experimental** (TF-ROCm for gfx942). If it doesn't run,
  report the CPU GRACE numbers (cmmg / viper-cpu) and note the APU as N/A.
- **CPU TensorFlow is slow** (the GRACE docs say so); the 2L CPU rows are a
  data point, not a showcase — keep their nsteps small.
- **FS vs TF for the 1L model** are the *same model*, different evaluators; the
  FS C++ path is faster and TF-free, which is exactly why it's the CPU headline.
- **Don't cross-read absolute throughput** between this 16k GRACE table and the
  256k PACE table blindly — they're different sizes. katom-step/s is roughly
  per-atom comparable for compute-bound runs, but treat them as separate.
