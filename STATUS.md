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
| **cmmg** | AMD EPYC 9754 (Zen4c), CPU, Kokkos/OpenMP | ✅ **builds, `lmp` produced** | ✅ **run (1 socket / 128 ranks)** |
| **viper** | AMD MI300A APU (gfx942 APU), Kokkos/HIP | ✅ **builds, `lmp` produced** | ✅ **run (1 APU)** |
| **raven** | NVIDIA A100 (AMPERE80), Kokkos/CUDA | ✅ **builds, `lmp` produced** (external MKL linalg; KIM pre-built; conda-free) | not run yet |

## Benchmark results so far (PACE, fcc-Cu, 256k atoms, 500 steps, `timer full`)

| Machine | procs | katom-step/s | Pair% | Comm% | speedup |
|---|---|---|---|---|---|
| **viper** (1 MI300A APU) | 1 | 509 | 99.8% | 0.2% | 4.49× |
| **cmmg** (1 socket, 128 cores) | 128 | 113 | 96.3% | 3.6% | 1.00× |

Both runs are **compute-bound** (Comm ≤ 4%) → the benchmark measures the ACE force
eval, not MPI. One MI300A APU ≈ **4.5× one 128-core EPYC socket**. `submit-cmmg.slurm`
is now set to **`--ntasks=256`** (full node); the full-node cmmg number is pending
(expect ~225 katom-step/s → APU ≈ 2.3× a full node).

## Immediate next steps

1. **cmmg full node** — re-run now that `submit-cmmg.slurm` uses `--ntasks=256`
   (1000 atoms/core; Comm% will rise a little). ~10 min wall.
2. **Raven** — ✅ builds (external MKL linalg, pre-built KIM-API, conda-free,
   benign diagnostics suppressed). Just **run the benchmark** (step 3). If you
   ever re-clone, re-run clean with:
   ```bash
   cd ~/PTMP          # dir that holds lammps/
   rm -rf lammps/build-raven
   bash mpcdf-lammps/build-lammps-raven.sh 2>&1 | tee /tmp/raven.log
   ```
   `ML-UF3` is still ON on Raven and built fine; disable with `-D PKG_ML-UF3=off`
   only if a future re-clone hits the CUDA ScatterView assertion (gotcha 10).
3. **Run the benchmark** on each (from the work dir where `lammps/` lives):
   ```bash
   cp mpcdf-lammps/bench/in.pace_bench .
   cp lammps/potentials/Cu-PBE-core-rep.ace .
   sbatch mpcdf-lammps/bench/submit-cmmg.slurm
   sbatch mpcdf-lammps/bench/submit-raven.slurm
   sbatch mpcdf-lammps/bench/submit-viper.slurm
   ```
4. **Compare**: gather the three `log.pace_*` onto one host, then
   `bash mpcdf-lammps/bench/compare-pace.sh` (reports katom-step/s + Pair%/Comm%).
   Tune `-var nsteps` so a single GPU run is ~10 min; keep atoms×steps identical
   across machines for a fair comparison.

## Per-machine build config (confirmed working values)

- **cmmg**: `module load gcc/13 impi/2021.16 cmake/3.30 mkl/2025.2 gsl/2.7`.
  MPI wrappers are **`mpigcc` / `mpig++`** (NOT mpicc/mpicxx), found under
  `$I_MPI_ROOT/bin`. Kokkos OpenMP + `Kokkos_ARCH_ZEN4`, `-march=znver4`.
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

## File map (`mpcdf-lammps/`)

- `cmake/lammps-packages-mpcdf.cmake` — shared package set (no PLUMED; VORONOI on).
- `build-lammps-{cmmg,raven,viper}.sh` — per-machine build scripts. Each now
  **tees all output to `build-<machine>-<timestamp>.log`**. The viper script also
  generates the `hipcc-cxx17` wrapper in the run dir (see gotchas 8–9).
- `bench/in.pace_bench` — fcc-Cu, 256k atoms, `pair_style pace product`,
  `nsteps 500` (default; sized so cmmg finishes in the wall limit), `timer full`.
- `bench/submit-{cmmg,raven,viper}.slurm` — cmmg = full node (256 ranks),
  raven/viper = single GPU. viper sets `HSA_XNACK=1`.
- `bench/compare-pace.sh` — parses the logs into a throughput table (now also
  reports `pair%` / `comm%` from the `timer full` breakdown).
- `README.md` — usage; this `STATUS.md` — current state + next steps.

(Also at the repo root: `dump-lammps-config.sh` — the original package-probe
script; `kokkos-hip-mi300a.cmake` — an early standalone preset, now superseded by
the inline flags in `build-lammps-viper.sh`.)
