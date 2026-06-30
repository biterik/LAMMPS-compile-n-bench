# LAMMPS-on-MPCDF — project status

Goal: build the latest **stable** LAMMPS (`stable_22Jul2025_update4`, the `stable`
git branch) on three MPCDF machines with the full package set from Erik's
`lammps_250722_config.txt` dump (minus the Intel-only package), then run a single
**PACE** benchmark on all three and compare.

Everything lives in `mpcdf-lammps/` (copy this folder to each cluster; launch the
build scripts from a `/ptmp/$USER` work dir — they clone + build into `$PWD/lammps`).

## Where each machine stands

| Machine | HW / backend | Build status | Benchmark |
|---|---|---|---|
| **cmmg** | AMD EPYC 9754 (Zen4c), CPU, Kokkos/OpenMP | ✅ **builds, `lmp` produced** | ✅ **run (full node, 256 ranks)** |
| **cmti** | Intel Xeon Gold 6230 (Cascade Lake), CPU, oneAPI + INTEL pkg | ✅ **builds, `lmp_cmti`** | ✅ **run (full node, 40 cores)** |
| **viper** | AMD MI300A APU (gfx942 APU), Kokkos/HIP | ✅ **builds, `lmp_viper`** | ✅ **run (1 APU)** |
| **viper-cpu** | AMD EPYC 9554 (Zen4 Genoa), CPU, Kokkos/OpenMP | ✅ **builds, `lmp_viper_cpu`** | ✅ **run (full node, 128 cores)** |
| **raven** | NVIDIA A100 (AMPERE80), Kokkos/CUDA | ✅ **builds, `lmp_raven`** (external MKL linalg; KIM pre-built; conda-free) | ✅ **run (1 A100)** |
| **raven-cpu** | Intel Xeon IceLake-SP, CPU, oneAPI + INTEL pkg | ✅ **builds, `lmp_raven_cpu`** (benign KIM warnings, gotcha 15) | ✅ **run (full node, 72 cores)** |

> All builds now emit a distinctly-named binary `lmp_<machine>` (LAMMPS_MACHINE)
> in `lammps/build-<machine>/`, so nothing overwrites anything.

## Benchmark results (PACE, fcc-Cu, 256k atoms, 500 steps, `timer full`)

Reference = **full-node cmmg** (256 cores). Only full-node CPU + single-device GPU
runs are reported; the old half-node (128-core) cmmg run is excluded (contended).

| Machine | config | procs | katom-step/s | Pair% | Comm% | wall | speedup |
|---|---|---|---|---|---|---|---|
| **cmmg** | full node, 256 EPYC 9754 cores | 256 | 393 | 95.4% | 4.5% | 325 s | 1.00× |
| **cmti** | full node, 40 Xeon Gold 6230 cores (INTEL pkg) | 40 | 62 | 98.1% | 1.8% | 2049 s | 0.16× |
| **raven-cpu** | full node, 72 Xeon IceLake cores (INTEL pkg) | 72 | 129 | 96.2% | 3.7% | 990 s | 0.33× |
| **viper-cpu** | full node, 128 EPYC 9554 cores | 128 | 402 | 96.4% | 3.5% | 318 s | 1.02× |
| **raven** | 1 A100 40GB | 1 | 360 | 99.9% | 0.1% | 355 s | 0.92× |
| **viper** | 1 MI300A APU | 1 | 509 | 99.8% | 0.2% | 251 s | 1.29× |

All runs are **compute-bound** (Pair ≥ 95%) → the benchmark measures the ACE force
eval, not MPI. For this kernel: **MI300A ≈ 1.4× one A100** (509 vs 360), **1.29× a
full 256-core EPYC node**. Among CPU nodes, a **128-core Genoa node ≈ a 256-core
Bergamo node** (Genoa Zen4 cores ~2× the Zen4c cores); the **72-core Xeon IceLake
node** (129) and the **40-core Xeon Cascade-Lake cmti node** (62) trail, cmti
slowest overall. Per-core: Genoa ≈ 3.1, IceLake ≈ 1.8, Cascade-Lake ≈ 1.6,
Zen4c ≈ 1.5 — cmti's low total is its small core count, not weak cores.

**Caveat (Erik's rule: full nodes only).** The earlier 1-socket cmmg run gave 113
katom-step/s — 3.5× slower than the full node for half the cores, i.e. super-linear,
the signature of a co-scheduled job on the shared half-node. Excluded. The GPU runs
above used `--gres=gpu:1` on a shared node; the submit scripts **now request
`#SBATCH --exclusive`**, so re-running viper/raven will refresh those numbers
contention-free (ACE is compute-bound, so expect only a small shift).

## GRACE track (thermoatoms fork) — separate builds + benchmark

Built from the **thermoatoms fork** (pinned `24da74cd…`, base `patch_11Feb2026`)
which adds the GRACE pair styles + the fast ACE MC. Kept separate from the
stable PACE builds: `build-lammps-<machine>-fork.sh` → `lmp_<machine>_fork`,
`bench/in.grace_bench` (common ~16k fcc-Cu), `bench/submit-<machine>-grace.slurm`,
`bench/compare-grace.sh`. Model: **SMAX-OMAT** (1L-large + 2L-medium). See
**GRACE.md** for the full plan, model download/export, and TF setup.

| Machine | Build script | GRACE on GPU | Status |
|---|---|---|---|
| cmmg | `build-lammps-cmmg-fork.sh` | n/a (CPU) | scripts ready — build + run |
| viper-cpu | `build-lammps-viper-cpu-fork.sh` | n/a (CPU) | scripts ready — build + run |
| raven | `build-lammps-raven-fork.sh` | TF-CUDA | scripts ready — build + run |
| viper | `build-lammps-viper-fork.sh` | TF-ROCm (experimental) | scripts ready — needs tensorflow-rocm |

**Key constraint:** the fork has **no `grace/fs/kk`** (no Kokkos FS), so GPU
GRACE is TensorFlow-only. The cross-machine-comparable point is the **1-layer**
model: `grace/fs` (CPU, no TF) vs TF `grace` (GPU). cmti and raven-cpu are
**not** part of the GRACE track (per scope). Open: ask Sarath to add
`grace/fs/kk` for a clean Kokkos GPU path on both vendors.

## Immediate next steps

All three machines now build **and** have a benchmark result (table above). Builds
and the first comparison are done. Remaining / optional:

1. **Raven CPU build + benchmark (INTEL package).** New
   `build-lammps-raven-cpu.sh` + `submit-raven-cpu.slurm`, modules pinned
   (intel/2025.3 + impi/2021.17 + mkl/2025.3). Run `./build-lammps-raven-cpu.sh`
   on a login node, then `sbatch bench/submit-raven-cpu.slurm`; fill the
   `_pending_` raven-cpu row when the log lands. (pace has no intel variant → it's
   a fair Xeon CPU data point.) **Likewise `build-lammps-viper-cpu.sh` +
   `submit-viper-cpu.slurm`** for the Viper EPYC-Genoa CPU node (128 ranks);
   modules confirmed (gcc/14 + openmpi/5.0).
2. **Exclusive GPU re-runs (for rigour).** `submit-viper.slurm` /
   `submit-raven.slurm` now request `#SBATCH --exclusive`. Re-run both to refresh
   the single-GPU numbers contention-free (ACE is compute-bound, so the shift
   should be small), then update the table if it moves.
3. **Re-clone build sanity (only if you wipe `lammps/`).** Raven builds clean now;
   if you ever `rm -rf lammps/build-raven`, just re-run
   `bash mpcdf-lammps/build-lammps-raven.sh`. `ML-UF3` is ON and built fine —
   disable with `-D PKG_ML-UF3=off` only if a re-clone hits the CUDA ScatterView
   assertion (gotcha 10).
4. **Longer / larger runs.** To stress the GPUs more, raise `-var nsteps` (or
   `nx/ny/nz`); keep atoms×steps identical across machines for a fair compare,
   and re-run `bash mpcdf-lammps/bench/compare-pace.sh` to regenerate the table.

<details><summary>How to re-run the benchmark (reference)</summary>

   ```bash
   cp mpcdf-lammps/bench/in.pace_bench .
   cp lammps/potentials/Cu-PBE-core-rep.ace .
   sbatch mpcdf-lammps/bench/submit-cmmg.slurm
   sbatch mpcdf-lammps/bench/submit-raven.slurm
   sbatch mpcdf-lammps/bench/submit-viper.slurm
   ```
   Then **compare**: gather the `log.pace_*` onto one host and
   `bash mpcdf-lammps/bench/compare-pace.sh` (reports katom-step/s + Pair%/Comm%).
   Keep atoms×steps identical across machines for a fair comparison.

</details>

## Per-machine build config (confirmed working values)

- **cmmg**: `module load gcc/13 impi/2021.16 cmake/3.30 mkl/2025.2 gsl/2.7`.
  MPI wrappers are **`mpigcc` / `mpig++`** (NOT mpicc/mpicxx), found under
  `$I_MPI_ROOT/bin`. Kokkos OpenMP + `Kokkos_ARCH_ZEN4`, `-march=znver4`.
- **cmti** (Intel Xeon Gold 6230 nodes of the *same* Sustainable-Materials cluster
  as cmmg): **Intel oneAPI build, mirrors raven-cpu** — `module load intel/2025.2
  impi/2021.16 mkl/2025.2 cmake/3.30` (⚠ confirm impi version with
  `module load intel/2025.2 && module avail impi`), `icpx`/`icx` via `mpiicpx`,
  INTEL package on (`INTEL_ARCH=cpu`), no Kokkos, external MKL + `FFT=MKL`,
  `-xCORE-AVX512 -qopt-zmm-usage=high`, PLUMED off. Partition `p.cmfe` (full node =
  40 cores), binary `lmp_cmti`, run with `-pk intel 0 omp 1 -sf intel`.
- **raven**: `module load gcc/13 cuda/12.6 openmpi_gpu/5.0 cmake mkl/2025.2`.
  Kokkos CUDA, `Kokkos_ARCH_AMPERE80`, via `nvcc_wrapper` (set as `OMPI_CXX`,
  `CXX=mpicxx`). Uses **external MKL** linalg (`USE_INTERNAL_LINALG=off`, gotcha
  12), **pre-built KIM-API** + a **conda-free** environment (gotcha 13), and
  `-diag-suppress` for the benign nvcc diagnostics (gotcha 14).
- **viper**: `module load gcc/14 rocm/6.3 openmpi_gpu/5.0 cmake`. Kokkos HIP,
  `Kokkos_ARCH_AMD_GFX942_APU`, `-munsafe-fp-atomics`. **`CXX` = a generated
  `hipcc-cxx17` wrapper** (not bare hipcc) that strips CMake's `-std=c++98` probe
  flag, appends `-std=c++17`, and pins `--offload-arch=gfx942` — see gotchas 8–9.
  `ML-UF3` is **off** (gotcha 10). Runtime needs **`HSA_XNACK=1`** (gotcha 11).
- **viper-cpu** (AMD EPYC Genoa nodes, no GPU): `module load gcc/14 openmpi/5.0
  cmake` (confirmed on Viper/RHEL 9; openmpi is hierarchical → appears under gcc,
  gcc/14 offers openmpi/4.1 + 5.0). Mirrors cmmg: Kokkos OpenMP + `Kokkos_ARCH_ZEN4`,
  `-march=znver4`, plain MPI `pair_style pace` at run time. INTEL/PLUMED off.
- **raven-cpu** (Intel Xeon nodes, no GPU): `module load intel/2025.3
  impi/2021.17 mkl/2025.3 cmake` (confirmed on Raven; impi is hierarchical — it
  resolves only after the intel module, gotcha 2). Intel oneAPI via `mpiicpx`/`mpiicpc`
  with `I_MPI_CXX=icpx`. **INTEL package on** (`-D PKG_INTEL=on -D
  INTEL_ARCH=cpu`), no Kokkos, external MKL (`USE_INTERNAL_LINALG=off`,
  `FFT=MKL`), `-xCORE-AVX512 -qopt-zmm-usage=high`. Benchmark runs one full node
  (72 ranks) with `-pk intel 0 omp 1 -sf intel`; `pair_style pace` has no intel
  variant so it falls back to the standard CPU ACE kernel (fair Xeon CPU point).

## Packages: PLUMED and VORONOI are special

- **PLUMED**: needs BLAS/LAPACK + GSL. **ON only on cmmg** (MKL 2025.2 + gsl 2.7 +
  `-D BLA_VENDOR=Intel10_64lp_seq`). **OFF on Raven/Viper** (Erik's decision; it's
  CPU-only and unused by the benchmark). Set via `-D PKG_PLUMED=on/off` per script;
  not in the shared preset.
- **VORONOI**: kept **on all three**, but voro++ cannot be compiled by
  `nvcc`/`hipcc`. The GPU scripts **pre-build voro++ with `g++`** and pass it via
  `-D DOWNLOAD_VORO=off -D VORO_LIBRARY=… -D VORO_INCLUDE_DIR=…`. cmmg builds it
  normally (g++ anyway). Validated end-to-end.

## Hard-won gotchas (so we don't rediscover them)

1. **`#!/bin/bash -l` makes scripts "do nothing"** on this account: the login
   startup files `exit` for non-interactive shells. All scripts use `#!/bin/bash`
   plus a guard that sources the modules init only if `module` isn't already
   defined. Run as `bash script` or `./script`; both work.
2. **No default module versions** — every module is version-pinned. `module avail X`
   only shows hierarchical modules (mpi/mkl/gsl) **after** loading the compiler.
3. **`set -e` + `VAR=$(cmd)` where cmd returns non-zero aborts silently.** The
   wrapper-detection `pick()` always `return 0`. Module loads happen *before*
   `set -euo pipefail` for the same reason.
4. **An MPI binary won't run on a login node** (`MPI_Init`→`PMI2_Job_GetId`
   abort). The build scripts' `lmp -h` smoke test runs singleton
   (`unset I_MPI_PMI_LIBRARY`) and is non-fatal. Real runs go through `srun`.
5. A **pyiron conda env** on `$PATH` was shadowing `mpicxx`; we resolve wrappers by
   absolute path from `$I_MPI_ROOT`. `conda deactivate` before building is safest.
6. **`-D` overrides a `FORCE`'d cache var** when it comes *after* `-C` on the
   command line (verified) — that's how per-machine package overrides work.
7. **PACE on GPU requires the `product` algorithm** (+ newton on, half neigh
   lists); `recursive` is GPU-unsupported. The benchmark input uses
   `pair_style pace product`, which also runs on CPU → one input for all machines.
8. **Viper hipcc + CMake 4.2 = broken-compiler error.** CMake's `project()`
   compiler probe compiles a test with `-std=c++98`; ROCm 6.3's HIP headers
   (`__clang_hip_math.h` → libstdc++ `<type_traits>`) don't compile under C++98.
   `HIPCC_COMPILE_FLAGS_APPEND` can't fix it (hipcc inserts it *before* CMake's
   flags; clang takes the **last** `-std`). `CMAKE_CXX_COMPILER_WORKS=ON` only
   silences the top-level probe — **every nested ExternalProject (KIM, PACE, …)
   re-runs its own probe** and fails. Fix: a `hipcc-cxx17` wrapper that drops
   `-std=c++98/03` and appends `-std=c++17` last; inherited by all sub-projects.
9. **Viper login nodes have no GPU**, so hipcc auto-detects arch as **gfx906**
   (wrong) — the wrapper also pins `--offload-arch=gfx942`. (Banner even warns:
   "specify at least --offload-arch=gfx942".)
10. **`ML-UF3` won't compile on HIP**: `pair_uf3_kokkos.cpp` builds a `HostSpace`
    `ScatterView` from a `HIPSpace` view (illegal cross-space copy) → static-assert
    failure. Disabled via `-D PKG_ML-UF3=off` on Viper; unused by the benchmark.
    The CUDA backend may hit the same — disable on Raven too if it appears.
11. **APU build needs `HSA_XNACK=1` at runtime.** `Kokkos_ARCH_AMD_GFX942_APU`
    uses unified memory (GPU reads host allocations via HMM); without XNACK the
    first device access faults and the run dies silently right after Kokkos init
    (`log` stops at the LAMMPS banner). Set in `submit-viper.slurm`.
12. **Internal linalg won't compile under nvcc (Raven).** The shared preset sets
    `USE_INTERNAL_LINALG=ON` (bundled f2c LAPACK). On Raven everything is built
    via `nvcc_wrapper`, and nvcc force-includes `crt/math_functions.h`, whose
    `log` collides with the f2c `double log(doublereal)` decl in
    `lib/linalg/dbdsdc.cpp` → *"linkage specification is incompatible with
    previous log"*. Fix: link external MKL instead — `module load mkl/2025.2`
    plus `-D USE_INTERNAL_LINALG=off -D BLA_VENDOR=Intel10_64lp_seq` (these `-D`
    come after `-C`, so they override the FORCED preset value, per gotcha 6).
    cmmg (g++) and viper (hipcc/clang) don't hit this — it's nvcc-only.
13. **An active conda env poisons the Raven build (libgfortran clash).** A child
    `bash` inherits conda's `PATH`/`LD_LIBRARY_PATH` and `CONDA_PREFIX` even
    though the `conda` *function* isn't defined non-interactively (so you can't
    just `conda deactivate` inside the script). Conda ships an old
    **libgfortran.so.4**; when LAMMPS auto-builds KIM-API its Fortran links that,
    and the final link warns *"libgfortran.so.4 … may conflict with
    libgfortran.so.5"* (gcc/13's). Two-part fix in `build-lammps-raven.sh`:
    (a) strip every `conda` element from `PATH`/`LD_LIBRARY_PATH`/`LIBRARY_PATH`
    and unset the `CONDA_*` vars at the top; (b) **pre-build KIM-API separately**
    with the module `gcc/g++/gfortran` (mirrors the voro++ pattern) and pass it
    via `-D DOWNLOAD_KIM=off` + `CMAKE_PREFIX_PATH`/`PKG_CONFIG_PATH`. The KIM
    version is read from LAMMPS' own `cmake/Modules/Packages/KIM.cmake` to stay
    in lockstep. The KIM lib dir is added to the `lmp` rpath so a KIM potential
    resolves at runtime. (Only matters if you actually use KIM — the PACE
    benchmark doesn't — but the build is now clean.)
14. **Cosmetic nvcc diagnostics are silenced, not fixed.** The Raven build emits
    many benign `#177-D`/`#550-D` (unused var), `#611-D` (partially-overridden
    virtual), `#186-D` (unsigned-vs-zero), and `#20011-D` (host dtor from
    host/device fn, in unused KOKKOS files) warnings — all in third-party or
    unused code, none affecting the binary. They're quieted with
    `-D CMAKE_CXX_FLAGS="-diag-suppress 177,550,611,186,20011"` so genuine
    warnings stand out. Remove that flag if you want to see them again.
15. **raven-cpu (Intel) build emits a flood of benign KIM warnings.** The
    auto-downloaded KIM-API builds its Fortran with the **system GCC-7**
    binutils/gfortran (`/usr/lib64/gcc/x86_64-suse-linux/7/...ld`) rather than the
    loaded oneAPI toolchain, producing hundreds of linker warnings
    *"alignment 4 of normal symbol `KIM_…` is smaller than 8 … alignment
    discrepancies can cause real problems. Investigation is advised."* These are a
    long-standing, harmless KIM C↔Fortran enum-alignment quirk (the symbols are
    integer name-constants); the "investigation advised" text is generic binutils
    boilerplate. Plus cosmetic `ifx` name-too-long (`#5462`) in KIM *example*
    models, and icpx `-Wvla-cxx-extension` / `-Wnontrivial-memcall` /
    `loop not vectorized` in the INTEL & PACE code. **None affect the binary**, and
    KIM is unused by the PACE benchmark. To silence the flood, either build with
    `-D PKG_KIM=off` (simplest; KIM not needed for the benchmark) or pre-build
    KIM with a single consistent toolchain (the raven-GPU pattern). Left ON by
    default for package parity with the other machines.

## File map (`mpcdf-lammps/`)

- `cmake/lammps-packages-mpcdf.cmake` — shared package set (no PLUMED; VORONOI on).
- `build-lammps-{cmmg,raven,viper}.sh` — per-machine build scripts. Each now
  **tees all output to `build-<machine>-<timestamp>.log`**. The viper script also
  generates the `hipcc-cxx17` wrapper in the run dir (see gotchas 8–9).
- `build-lammps-cmti.sh` — **cmti** build: Intel Xeon Gold 6230 (Cascade Lake) nodes
  of the cmmg cluster. **Intel oneAPI build, mirrors raven-cpu** (`icpx` + Intel MPI +
  MKL, INTEL package on, no Kokkos), pinned to intel/2025.2 + mkl/2025.2 + impi/2021.16,
  `-xCORE-AVX512`, binary `lmp_cmti`. Matching `bench/submit-cmti.slurm` (1 full node,
  40 ranks, `-sf intel`).
- `build-lammps-raven-cpu.sh` — Raven **CPU** build: Intel oneAPI (`icpx`) + Intel
  MPI + MKL, **INTEL package on**, no Kokkos. For the Xeon nodes, not the A100s.
- `build-lammps-viper-cpu.sh` — Viper **CPU** build: gcc + OpenMPI, Kokkos/OpenMP,
  `Kokkos_ARCH_ZEN4`. For the EPYC Genoa nodes, not the MI300A APU.
- All build scripts set `-D LAMMPS_MACHINE=<machine>` → binary `lmp_<machine>`.
- `bench/in.pace_bench` — fcc-Cu, 256k atoms, `pair_style pace product`,
  `nsteps 500` (default; sized so cmmg finishes in the wall limit), `timer full`.
- `bench/submit-{cmmg,raven,viper}.slurm` — cmmg = full node (256 ranks),
  raven/viper = single GPU (now `--exclusive`). viper sets `HSA_XNACK=1`.
- `bench/submit-raven-cpu.slurm` — one full Raven CPU node (72 ranks, `--exclusive`),
  INTEL package engaged (`-pk intel 0 omp 1 -sf intel`).
- `bench/submit-viper-cpu.slurm` — one full Viper CPU node (128 ranks, `--exclusive`),
  plain MPI `pair_style pace`.
- `bench/compare-pace.sh` — parses the logs into a throughput table (now also
  reports `pair%` / `comm%` from the `timer full` breakdown).
- `README.md` — usage; this `STATUS.md` — current state + next steps.

(Also at the repo root: `dump-lammps-config.sh` — the original package-probe
script; `kokkos-hip-mi300a.cmake` — an early standalone preset, now superseded by
the inline flags in `build-lammps-viper.sh`.)
