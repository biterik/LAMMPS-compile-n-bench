# LAMMPS-compile-n-bench

Reproducible build recipes and a single portable **PACE / ACE** benchmark for
[LAMMPS](https://www.lammps.org/) on three [MPCDF](https://www.mpcdf.mpg.de/)
machines — an AMD EPYC CPU cluster, an AMD MI300A APU system, and an NVIDIA A100
GPU system. One benchmark input, one potential, three architectures, directly
comparable throughput numbers.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![LAMMPS stable_22Jul2025_update4](https://img.shields.io/badge/LAMMPS-stable__22Jul2025__update4-blue)

---

## What this is

Three self-contained build scripts compile the latest **stable** LAMMPS
(`stable_22Jul2025_update4`, the `stable` git branch) with a shared package set
on three different hardware/backends, plus a single `pair_style pace` benchmark
that runs *unmodified* on all of them so the results are apples-to-apples.

| Build (binary) | Hardware | Kokkos backend | Kokkos arch | Compiler |
|---|---|---|---|---|
| **cmmg** (`lmp_cmmg`) | 2× AMD EPYC 9754 (Zen4c), CPU | OpenMP | `ZEN4` | `g++` (`-march=znver4`) |
| **cmti** (`lmp_cmti`) | 2× Intel Xeon Gold 6230 (Cascade Lake), CPU | — (MPI + INTEL/OpenMP) | — | `icpx` (oneAPI, `-xCORE-AVX512`) |
| **viper** (`lmp_viper`) | AMD Instinct MI300A APU (gfx942) | HIP | `AMD_GFX942_APU` | `hipcc` (wrapper) |
| **viper-cpu** (`lmp_viper_cpu`) | 2× AMD EPYC 9554 (Zen4 Genoa), CPU | OpenMP | `ZEN4` | `g++` (`-march=znver4`) |
| **raven** (`lmp_raven`) | NVIDIA A100 40 GB (Ampere) | CUDA | `AMPERE80` | `nvcc_wrapper` / g++13 |
| **raven-cpu** (`lmp_raven_cpu`) | 2× Intel Xeon IceLake-SP (AVX-512), CPU | — (MPI + INTEL/OpenMP) | — | `icpx` (oneAPI, `-xCORE-AVX512`) |

Each build produces a **distinctly named binary** `lmp_<machine>` (via the LAMMPS
`LAMMPS_MACHINE` option) in its own `lammps/build-<machine>/` directory, so no two
builds ever overwrite each other — see [Binaries](#binaries). The **raven-cpu**
build is the odd one out: it targets Raven's Intel Xeon CPU nodes (not the A100s)
with the Intel oneAPI toolchain and the LAMMPS **INTEL** package enabled — see
[the package table below](#why-some-packages-are-built-on-some-machines-but-not-others).

### The target machines

- **Viper-GPU** (AMD MI300A APU) — Max Planck Computing and Data Facility.
  [System overview](https://www.mpcdf.mpg.de/services/supercomputing/viper) ·
  [Viper-GPU user guide](https://docs.mpcdf.mpg.de/doc/computing/viper-gpu-user-guide.html).
  Each APU couples 24 CPU cores and one GPU with 128 GB shared HBM3 (unified
  host/device memory). Viper also has a large **CPU** partition — 2× AMD EPYC
  9554 "Genoa" (128 Zen4 cores/node),
  [Viper-CPU user guide](https://docs.mpcdf.mpg.de/doc/computing/viper-user-guide.html) —
  targeted by the `viper-cpu` build/benchmark.
- **Raven** (NVIDIA A100) — Max Planck Computing and Data Facility.
  [System overview](https://www.mpcdf.mpg.de/services/supercomputing/raven) ·
  [User guide](https://docs.mpcdf.mpg.de/doc/computing/raven-user-guide.html) ·
  [Hardware details](https://docs.mpcdf.mpg.de/doc/computing/raven-details.html).
  GPU nodes carry 4× A100 40 GB SXM (NVLink); this benchmark uses a single A100.
  Raven's **CPU** nodes have 2× Intel Xeon IceLake-SP Platinum 8360Y (72 cores,
  AVX-512); the `raven-cpu` build/benchmark targets one full CPU node.
- **cmmg / cmti** (AMD EPYC 9754 / Intel Xeon Gold 6230) — two partitions of the
  same compute cluster of the
  [Max Planck Institute for Sustainable Materials (MPIE)](https://www.mpie.de/4065158/Hardware),
  operated with MPCDF
  ([cluster docs](https://docs.mpcdf.mpg.de/doc/computing/clusters/systems/Sustainable_Materials.html)).
  The **cmmg** nodes have 2× 128-core EPYC 9754 (256 cores, Zen4c) and 768 GB RAM
  (partition `p.cmmg`); the older **cmti** nodes have 2× 20-core Xeon Gold 6230
  (40 cores, Cascade Lake, AVX-512) (partition `p.cmfe`). Each benchmark uses one
  full node.

General MPCDF HPC documentation: <https://docs.mpcdf.mpg.de/doc/computing/>.

---

## Layout

```
LAMMPS-compile-n-bench/
  cmake/lammps-packages-mpcdf.cmake   shared package set (used by all builds)
  build-lammps-viper.sh               Viper-GPU: KOKKOS + HIP   (MI300A) -> lmp_viper
  build-lammps-viper-cpu.sh           Viper-CPU: KOKKOS/OpenMP  (EPYC Genoa) -> lmp_viper_cpu
  build-lammps-raven.sh               Raven-GPU: KOKKOS + CUDA  (A100)  -> lmp_raven
  build-lammps-raven-cpu.sh           Raven-CPU: Intel oneAPI + INTEL pkg -> lmp_raven_cpu
  build-lammps-cmmg.sh                cmmg:      CPU, KOKKOS/OpenMP (EPYC) -> lmp_cmmg
  build-lammps-cmti.sh                cmti:      CPU, Intel oneAPI + INTEL pkg (Xeon CLX) -> lmp_cmti
  bench/in.pace_bench                 PACE fcc-Cu benchmark input (CPU + GPU)
  bench/submit-viper.slurm            1× MI300A APU (exclusive node)
  bench/submit-viper-cpu.slurm        1 full Viper CPU node (128 ranks)
  bench/submit-raven.slurm            1× A100 (exclusive node)
  bench/submit-raven-cpu.slurm        1 full Raven CPU node (72 ranks, INTEL pkg)
  bench/submit-cmmg.slurm             1 full EPYC node (256 ranks)
  bench/submit-cmti.slurm             1 full Xeon node (40 ranks)
  bench/compare-pace.sh               parses logs into a throughput table
  README.md                           this file
  PACKAGES.md                         full list of compiled packages (+ per-machine matrix)
  STATUS.md                           live build/benchmark status + build gotchas
  LICENSE                             MIT
```

> **Potentials are not redistributed here.** The benchmark uses
> `Cu-PBE-core-rep.ace`, which ships with LAMMPS in `potentials/`. Copy it into
> `bench/` (or your run dir) at run time — see below.

---

## Building

Run each build script **on a login node** of the corresponding machine — the
package downloads (PACE, KIM, Eigen3, voro++, PLUMED) and the `git clone` need
internet, which compute nodes don't have.

```bash
# copy this folder to the cluster, then, from a /ptmp/$USER work dir:
./build-lammps-viper.sh       # on viper login nodes (MI300A GPU)
./build-lammps-viper-cpu.sh   # on viper login nodes (EPYC Genoa CPU)
./build-lammps-raven.sh       # on raven login nodes (A100 GPU)
./build-lammps-raven-cpu.sh   # on raven login nodes (Xeon CPU + INTEL package)
./build-lammps-cmmg.sh        # on cmmg login nodes (EPYC CPU)
./build-lammps-cmti.sh        # on cmti login nodes (Xeon Cascade-Lake CPU; same cluster as cmmg)
```

The clone and build land in the directory you launch from:
`<launch-dir>/lammps/build-<machine>/lmp_<machine>` (override with `SRC=` /
`BUILD=`). Launch from `/ptmp/$USER`, **not** `$HOME`. The GPU builds compile every
source through `hipcc` / `nvcc_wrapper` and pull in the external packages, so
expect a long compile (tens of minutes). Each script tees its output to
`build-<machine>-<timestamp>.log`.

### Binaries

Every build sets the LAMMPS `LAMMPS_MACHINE` option, so the executable is named
**`lmp_<machine>`** rather than the default `lmp`:

| Build | Directory | Binary |
|---|---|---|
| cmmg | `lammps/build-cmmg/` | `lmp_cmmg` |
| cmti | `lammps/build-cmti/` | `lmp_cmti` |
| viper | `lammps/build-viper/` | `lmp_viper` |
| viper-cpu | `lammps/build-viper-cpu/` | `lmp_viper_cpu` |
| raven | `lammps/build-raven/` | `lmp_raven` |
| raven-cpu | `lammps/build-raven-cpu/` | `lmp_raven_cpu` |

Builds already live in separate `build-<machine>/` directories, so they never
clobber each other; the distinct *names* mean you can also copy several binaries
into one work dir (or `$PATH`) without collision. Each `submit-*.slurm` already
points `LMP` at the matching `lmp_<machine>`.

Per-machine module stacks and the exact, confirmed-working CMake flags are
documented in **[STATUS.md](STATUS.md)**, along with 14 hard-won build gotchas
(the `#!/bin/bash -l` trap, the `hipcc-cxx17` wrapper, `HSA_XNACK=1` at runtime,
external MKL under nvcc, conda/libgfortran, etc.). Read it before your first build.

---

## Why some packages are built on some machines but not others

The package set is the "Installed packages" list from the central
`lammps/250722` module, with a few deliberate, hardware-driven exceptions. The
shared list lives in `cmake/lammps-packages-mpcdf.cmake`; the exceptions are set
per machine in the build scripts. The **full list of compiled packages** is in
[PACKAGES.md](PACKAGES.md); the table below covers only the per-machine
exceptions.

| Package | cmmg (CPU) | viper (HIP) | raven (CUDA) | raven-cpu (Intel) | Reason |
|---|:--:|:--:|:--:|:--:|---|
| **INTEL** | ✗ | ✗ | ✗ | ✓ | Intel-only AVX-512 optimizations. Useless on the AMD CPU/GPU and the CUDA build — but **on for the raven-cpu** Xeon build (`INTEL_ARCH=cpu`, compiled with `icpx`). |
| **PLUMED** | ✓ | ✗ | ✗ | ✗ | CPU-only and **unused by the benchmark**; pulls in BLAS/LAPACK + GSL. On only on cmmg (MKL + GSL); off elsewhere to avoid the dependency. |
| **VORONOI** | ✓ | ✓ | ✓ | ✓ | Kept everywhere, **but** voro++ can't be compiled by `nvcc`/`hipcc`. The GPU scripts pre-build voro++ with `g++`; cmmg and raven-cpu build it normally (`icpx` compiles it fine). |
| **ML-UF3** | ✓ | ✗ | ⚠️ | ✓ | `pair_uf3_kokkos.cpp` builds a `HostSpace` `ScatterView` from a device view — illegal on HIP. Disabled on viper; disable on raven (CUDA) too if the assertion appears. Non-Kokkos CPU builds are fine. |
| **KOKKOS** | ✓ (OpenMP) | ✓ (HIP) | ✓ (CUDA) | ✗ | The raven-cpu build uses INTEL/OPENMP/OPT acceleration instead of Kokkos. |

The raven-cpu build also links **external MKL** (`USE_INTERNAL_LINALG=off`,
`FFT=MKL`) rather than the bundled linalg/KISS FFT, since the Intel toolchain
ships MKL anyway.

**cmti** follows the **same recipe as raven-cpu** (Intel oneAPI `icpx` + Intel MPI +
MKL, **INTEL package on**, no Kokkos, external MKL linalg + `FFT=MKL`, PLUMED off),
just pinned to cmti's `intel/2025.2 + impi/2021.16 + mkl/2025.2` and the
Cascade-Lake arch (`-xCORE-AVX512`), on partition `p.cmfe`. Like raven-cpu, the
INTEL package doesn't speed up `pair_style pace` (no intel variant) — it's a fair
Intel-Xeon CPU data point on the older Cascade-Lake silicon.

Two more architecture facts worth knowing (full detail in STATUS.md):

- **PACE on GPU requires the `product` algorithm** (`recursive` is
  GPU-unsupported). `product` also runs on the CPU, so the *same* input file
  benchmarks all three machines.
- **MI300A must use `Kokkos_ARCH_AMD_GFX942_APU`** (not plain `GFX942`) — the
  `_APU` target enables the unified host/device memory, and the runtime needs
  `HSA_XNACK=1` or the first device access faults silently.

The external-library packages (ML-PACE, KIM, VORONOI, MACHDYN/Eigen3) download
and build extra sources at configure time, which is why builds must run on a
login node. None of them are required for the benchmark — if a heavy package
breaks a build, comment it out in `cmake/lammps-packages-mpcdf.cmake` (clearly
marked block) and continue.

---

## Running the benchmark

`bench/in.pace_bench`: fcc Cu, **256,000 atoms** (40³ cells), 500 steps,
`pair_style pace product`, `timer full`. Sizes are `variable index` knobs you
can override on the command line.

Run from the **same work dir where you built** (the submit scripts default `LMP`
to `$SLURM_SUBMIT_DIR/lammps/build-<machine>/lmp`). Put the input, the potential,
and the submit script there:

```bash
# in your work dir, e.g. /ptmp/$USER/lammpswork  (where ./lammps/build-* exists)
cp LAMMPS-compile-n-bench/bench/in.pace_bench .
cp lammps/potentials/Cu-PBE-core-rep.ace .

sbatch LAMMPS-compile-n-bench/bench/submit-cmmg.slurm       # 1 full EPYC node (256 ranks)
sbatch LAMMPS-compile-n-bench/bench/submit-cmti.slurm       # 1 full Xeon node (40 ranks)
sbatch LAMMPS-compile-n-bench/bench/submit-raven.slurm      # 1 A100 (exclusive node)
sbatch LAMMPS-compile-n-bench/bench/submit-raven-cpu.slurm  # 1 full Raven CPU node (72 ranks, INTEL pkg)
sbatch LAMMPS-compile-n-bench/bench/submit-viper.slurm      # 1 MI300A APU (sets HSA_XNACK=1)
sbatch LAMMPS-compile-n-bench/bench/submit-viper-cpu.slurm  # 1 full Viper CPU node (128 ranks)
```

The `raven-cpu` run engages the INTEL package (`-pk intel 0 omp 1 -sf intel`).
`pair_style pace` has no INTEL-accelerated variant, so the ACE kernel runs as the
standard CPU style — the run is a fair Intel-Xeon CPU data point, not a showcase
of INTEL-package speedup (which applies to styles like `eam`, `sw`, `lj/*`).

(Or set `LMP=/full/path/to/lmp` when submitting, and `cd` wherever you keep the
input.) The dev partitions (`apudev`, `gpudev`) cap at 15 min, which is fine for
a benchmark; for longer runs use the regular partitions and raise `--time`.

### Comparing the results

The work (atoms × steps) is fixed and identical everywhere, so compare the
**throughput** LAMMPS prints at the end of each `log.pace_*`:

```
Performance: ... ns/day, ... timesteps/s, ... katom-step/s
```

`katom-step/s` (or `ns/day`) is the apples-to-apples number across CPU and GPU.
Wall time will differ per machine — that *is* the result. Keep `nx/ny/nz` and
`nsteps` identical across machines for a fair comparison; only then are the
throughputs directly comparable. To land each run near ~10 min, scale `nsteps`
on the command line: `-var nsteps 5000`.

Gather the three logs onto one host and run:

```bash
bash LAMMPS-compile-n-bench/bench/compare-pace.sh
```

which prints a throughput table including the `Pair%` / `Comm%` breakdown from
`timer full`.

---

## Results so far

PACE, fcc-Cu, **256,000 atoms, 500 steps**, `timer full`. The **full-node cmmg run
is the reference** (speedup = 1.00×). "Run mode" is how the binary is actually
driven: cmmg runs as plain MPI with the standard (non-Kokkos) `pair_style pace`;
the GPU runs use Kokkos. Only **full-node** CPU and **single-device** GPU runs are
reported — half-node CPU runs are excluded (see the note below).

| Machine | Config | Run mode | Procs | katom-step/s | Pair % | Comm % | Speedup |
|---|---|---|--:|--:|--:|--:|--:|
| **cmmg** | full node, 2× EPYC 9754 (256 cores) | MPI 256×1, non-Kokkos `pace` | 256 | 393 | 95.4 | 4.5 | 1.00× |
| **viper-cpu** | full node, 2× EPYC 9554 (128 cores) | MPI 128×1, non-Kokkos `pace` | 128 | 402 | 96.4 | 3.5 | 1.02× |
| **raven-cpu** | full node, 2× Xeon IceLake (72 cores) | MPI 72×1, INTEL pkg | 72 | 129 | 96.2 | 3.7 | 0.33× |
| **cmti** | full node, 2× Xeon Gold 6230 (40 cores) | MPI 40×1, non-Kokkos `pace` | 40 | _pending_ | — | — | _pending_ |
| **viper** | 1× MI300A APU | Kokkos HIP | 1 | **509** | 99.8 | 0.2 | **1.29×** |
| **raven** | 1× A100 (40 GB) | Kokkos CUDA | 1 | 360 | 99.9 | 0.1 | 0.92× |

Wall time for the 500 steps: cmmg 325 s, viper-cpu 318 s, raven-cpu 990 s,
viper 251 s, raven 355 s.

**Reading the numbers.** For this ACE `product` kernel a single **MI300A APU is
~1.4× a single A100** (509 vs 360 katom-step/s) and **1.29× a full 256-core EPYC
node**. Among the CPU nodes, a full **Viper EPYC-Genoa node (128 cores) ≈ a full
cmmg EPYC-Bergamo node (256 cores)** — the Genoa cores (full Zen4, higher clock)
are roughly twice as fast each as the denser Zen4c cores, so half as many keep
pace. The **Raven Xeon-IceLake node** is the slowest (129; older, 72 cores), ~3×
behind the EPYC nodes and ~4× behind one A100. All runs are **compute-bound**
(Pair ≥ 95 %, Comm ≤ 4.5 %), so the benchmark measures the ACE force evaluation,
not MPI.

> **Why only full-node CPU runs?** An earlier one-socket (128-core) cmmg run
> reported just 113 katom-step/s — 3.5× *slower* than the full node for half the
> cores (strongly super-linear), the fingerprint of a co-scheduled job contending
> for shared resources on a half-empty node. The full-node run is the trustworthy
> CPU number. For the same reason the GPU submit scripts now request
> `#SBATCH --exclusive` (reserve the whole node, even though only one device is
> used); the single-GPU numbers above predate that change and will be refreshed on
> the next run. See [STATUS.md](STATUS.md) for live status.

---

## Reproducing / adapting

- **Different potential or element:** change `pair_coeff` and `mass` in
  `bench/in.pace_bench` and point `-var pot` at your `.ace` file.
- **Different problem size:** override `-var nx/ny/nz` (atoms = 4·nx·ny·nz) and
  `-var nsteps`. Keep them identical across machines to compare.
- **Different cluster:** copy the closest build script, adjust the module loads
  and `Kokkos_ARCH_*`, and reuse `cmake/lammps-packages-mpcdf.cmake`.

## License

[MIT](LICENSE) © 2026 Erik Bitzek. LAMMPS itself is GPL-2.0 and is fetched from
its own repository at build time — it is not redistributed here.
