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

| Machine | Hardware | Kokkos backend | Kokkos arch | Compiler |
|---|---|---|---|---|
| **cmmg** | 2× AMD EPYC 9754 (Zen4c), CPU | OpenMP | `ZEN4` | `g++` (`-march=znver4`) |
| **viper** | AMD Instinct MI300A APU (gfx942) | HIP | `AMD_GFX942_APU` | `hipcc` (wrapper) |
| **raven** | NVIDIA A100 40 GB (Ampere) | CUDA | `AMPERE80` | `nvcc_wrapper` / g++13 |

### The target machines

- **Viper-GPU** (AMD MI300A APU) — Max Planck Computing and Data Facility.
  [System overview](https://www.mpcdf.mpg.de/services/supercomputing/viper) ·
  [Viper-GPU user guide](https://docs.mpcdf.mpg.de/doc/computing/viper-gpu-user-guide.html).
  Each APU couples 24 CPU cores and one GPU with 128 GB shared HBM3 (unified
  host/device memory).
- **Raven** (NVIDIA A100) — Max Planck Computing and Data Facility.
  [System overview](https://www.mpcdf.mpg.de/services/supercomputing/raven) ·
  [User guide](https://docs.mpcdf.mpg.de/doc/computing/raven-user-guide.html) ·
  [Hardware details](https://docs.mpcdf.mpg.de/doc/computing/raven-details.html).
  GPU nodes carry 4× A100 40 GB SXM (NVLink); this benchmark uses a single A100.
- **cmmg** (AMD EPYC 9754) — compute cluster of the
  [Max Planck Institute for Sustainable Materials (MPIE)](https://www.mpie.de/4065158/Hardware),
  operated with MPCDF. Nodes have 2× 128-core EPYC 9754 (256 cores) and 768 GB
  RAM. This benchmark uses one full node.

General MPCDF HPC documentation: <https://docs.mpcdf.mpg.de/doc/computing/>.

---

## Layout

```
LAMMPS-compile-n-bench/
  cmake/lammps-packages-mpcdf.cmake   shared package set (used by all builds)
  build-lammps-viper.sh               Viper-GPU: KOKKOS + HIP   (MI300A, gfx942 APU)
  build-lammps-raven.sh               Raven:     KOKKOS + CUDA  (A100, AMPERE80)
  build-lammps-cmmg.sh                cmmg:      CPU, KOKKOS/OpenMP (EPYC Zen4)
  bench/in.pace_bench                 PACE fcc-Cu benchmark input (CPU + GPU)
  bench/submit-viper.slurm            1× MI300A APU
  bench/submit-raven.slurm            1× A100
  bench/submit-cmmg.slurm             1 full EPYC node (256 ranks)
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
./build-lammps-viper.sh     # on viper login nodes
./build-lammps-raven.sh     # on raven login nodes
./build-lammps-cmmg.sh      # on cmmg login nodes
```

The clone and build land in the directory you launch from:
`<launch-dir>/lammps/build-<machine>/lmp` (override with `SRC=` / `BUILD=`).
Launch from `/ptmp/$USER`, **not** `$HOME`. The GPU builds compile every source
through `hipcc` / `nvcc_wrapper` and pull in the external packages, so expect a
long compile (tens of minutes). Each script tees its output to
`build-<machine>-<timestamp>.log`.

Per-machine module stacks and the exact, confirmed-working CMake flags are
documented in **[STATUS.md](STATUS.md)**, along with 11 hard-won build gotchas
(the `#!/bin/bash -l` trap, the `hipcc-cxx17` wrapper, `HSA_XNACK=1` at runtime,
etc.). Read it before your first build.

---

## Why some packages are built on some machines but not others

The package set is the "Installed packages" list from the central
`lammps/250722` module, with a few deliberate, hardware-driven exceptions. The
shared list lives in `cmake/lammps-packages-mpcdf.cmake`; the exceptions are set
per machine in the build scripts. The **full list of compiled packages** is in
[PACKAGES.md](PACKAGES.md); the table below covers only the per-machine
exceptions.

| Package | cmmg (CPU) | viper (HIP) | raven (CUDA) | Reason |
|---|:--:|:--:|:--:|---|
| **INTEL** | ✗ | ✗ | ✗ | Intel-only optimizations. Both AMD machines and the CUDA build can't use it, so it's dropped everywhere. |
| **PLUMED** | ✓ | ✗ | ✗ | CPU-only and **unused by the benchmark**. It also pulls in BLAS/LAPACK + GSL. Enabled only on cmmg (MKL 2025.2 + GSL 2.7); omitted on the GPU machines to avoid the dependency. |
| **VORONOI** | ✓ | ✓ | ✓ | Kept on all three, **but** voro++ can't be compiled by `nvcc`/`hipcc`. The GPU scripts pre-build voro++ with `g++` and pass it via `-D DOWNLOAD_VORO=off -D VORO_LIBRARY=… -D VORO_INCLUDE_DIR=…`. cmmg builds it normally. |
| **ML-UF3** | ✓ | ✗ | ⚠️ | `pair_uf3_kokkos.cpp` builds a `HostSpace` `ScatterView` from a device view — an illegal cross-memory-space copy that fails a static assert on HIP. Disabled on viper; disable on raven too if the same CUDA assertion appears. Unused by the benchmark. |

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

sbatch LAMMPS-compile-n-bench/bench/submit-cmmg.slurm     # 1 full EPYC node (256 ranks)
sbatch LAMMPS-compile-n-bench/bench/submit-raven.slurm    # 1 A100
sbatch LAMMPS-compile-n-bench/bench/submit-viper.slurm    # 1 MI300A APU (sets HSA_XNACK=1)
```

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
| **viper** | 1× MI300A APU | Kokkos HIP | 1 | **509** | 99.8 | 0.2 | **1.29×** |
| **raven** | 1× A100 (40 GB) | Kokkos CUDA | 1 | 360 | 99.9 | 0.1 | 0.92× |

Wall time for the 500 steps: cmmg 325 s, viper 251 s, raven 355 s.

**Reading the numbers.** For this ACE `product` kernel a single **MI300A APU is
~1.4× a single A100** (509 vs 360 katom-step/s) and **1.29× a full 256-core EPYC
node**; a full EPYC node is itself within ~8 % of one A100. All runs are
**compute-bound** (Pair ≥ 95 %, Comm ≤ 4.5 %), so the benchmark measures the ACE
force evaluation, not MPI.

> **Why only full-node CPU runs?** An earlier one-socket (128-core) cmmg run
> reported just 113 katom-step/s — 3.5× *slower* than the full node for half the
> cores (strongly super-linear), the fingerprint of a co-scheduled job contending
> for shared resources on a half-empty node. The full-node run is the trustworthy
> CPU number. For the same reason the single-GPU runs are best taken on an
> **exclusive** node allocation (`#SBATCH --exclusive`) to rule out interference
> from jobs sharing the node. See [STATUS.md](STATUS.md) for live status.

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
