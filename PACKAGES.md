# Compiled LAMMPS packages

Full package set built by these scripts on top of LAMMPS
`stable_22Jul2025_update4`. The shared list lives in
[`cmake/lammps-packages-mpcdf.cmake`](cmake/lammps-packages-mpcdf.cmake); the
per-machine exceptions are applied in the `build-lammps-*.sh` scripts. This
mirrors the "Installed packages" block of the central `lammps/250722` module,
**minus the Intel-only `INTEL` package**.

See the [README](README.md#why-some-packages-are-built-on-some-machines-but-not-others)
for the rationale behind the per-machine differences.

## Per-machine differences (the only ones)

| Package | cmmg (CPU) | viper (HIP) | raven (CUDA) | raven-cpu (Intel) | Note |
|---|:--:|:--:|:--:|:--:|---|
| `INTEL`   | ✗ | ✗ | ✗ | ✓ | Intel AVX-512 optimizations; useless on AMD/CUDA, **on** for the raven-cpu Xeon build (`INTEL_ARCH=cpu`, `icpx`). |
| `PLUMED`  | ✓ | ✗ | ✗ | ✗ | CPU-only, unused by the benchmark; pulls in BLAS/LAPACK + GSL. On only on cmmg (MKL + GSL). |
| `VORONOI` | ✓ (downloaded) | ✓ (voro++ pre-built, g++) | ✓ (voro++ pre-built, g++) | ✓ (downloaded) | voro++ can't compile under hipcc/nvcc; GPU builds pre-build it with g++. cmmg/raven-cpu build it normally. |
| `ML-UF3`  | ✓ | ✗ | ✓ (off if CUDA ScatterView assert) | ✓ | Illegal cross-memory-space `ScatterView` copy fails on HIP; non-Kokkos CPU builds are fine. |
| `KOKKOS`  | ✓ (OpenMP) | ✓ (HIP) | ✓ (CUDA) | ✗ | raven-cpu uses INTEL/OPENMP/OPT instead of Kokkos. |
| `KIM`     | ✓ (downloaded) | ✓ (downloaded) | ✓ (pre-built, g++) | ✓ (downloaded) | raven (CUDA) pre-builds KIM with g++ in a conda-free env to avoid a libgfortran clash (gotcha 13). |

Everything below is **built identically on all builds** (modulo the rows above).

## Packages with no external library

```
AMOEBA        ASPHERE       BOCS          BODY          BPM
BROWNIAN      CG-DNA        CG-SPICA      CLASS2        COLLOID
CORESHELL     DIELECTRIC    DIFFRACTION   DIPOLE        DPD-BASIC
DPD-MESO      DPD-REACT     DPD-SMOOTH    DRUDE         EFF
ELECTRODE     EXTRA-COMMAND EXTRA-COMPUTE EXTRA-DUMP    EXTRA-FIX
EXTRA-MOLECULE EXTRA-PAIR   FEP           GRANULAR      INTERLAYER
KSPACE        LEPTON        MANYBODY      MC            MEAM
MESONT        MISC          ML-IAP        ML-POD        ML-RANN
ML-SNAP       MOFFF         MOLECULE      OPENMP        OPT
ORIENT        PERI          PHONON        PLUGIN        POEMS
QEQ           REACTION      REAXFF        REPLICA       RHEO
RIGID         SHOCK         SPH           SPIN          SRD
TALLY         UEF           YAFF
```

(`ML-UF3` is in the common list too, but is turned **off** on viper — see the
table above.)

## Packages with a bundled library (no download)

```
COLVARS       (lib/colvars)
COMPRESS      (system zlib — gzip/zstd dump compression)
```

## Packages that download + build an external library at configure time

These need internet, which is why the build scripts must run on a **login node**
(compute nodes have no network). None are required for the PACE benchmark.

| Package | External library | CMake download var |
|---|---|---|
| `ML-PACE` | pacelib (the ACE evaluator — **the benchmark potential**) | `DOWNLOAD_PACELIB=on` |
| `KIM` | kim-api | `DOWNLOAD_KIM=on` |
| `VORONOI` | voro++ 0.4.6 | `DOWNLOAD_VORO=on` (CPU) / pre-built with g++ (GPU) |
| `MACHDYN` | Eigen3 headers | `DOWNLOAD_EIGEN3=on` |
| `PLUMED` *(cmmg only)* | PLUMED (static) | `DOWNLOAD_PLUMED=on` |

For an offline login node, point the `*_URL` cache variables at pre-fetched
tarballs (see the note at the bottom of
[`cmake/lammps-packages-mpcdf.cmake`](cmake/lammps-packages-mpcdf.cmake)).

## Build-wide settings

```
LAMMPS_SIZES        = smallbig        (-DLAMMPS_SMALLBIG, matches the reference build)
BUILD_MPI           = ON
USE_INTERNAL_LINALG = ON   (preset default; OFF + MKL on raven-CUDA & raven-cpu)
PKG_KOKKOS          = ON   (OpenMP cmmg / HIP viper / CUDA raven; OFF on raven-cpu)
FFT                 = KISS (cmmg/viper) / CUFFT (raven) / MKL (raven-cpu)
```

## Not included

```
INTEL    — excluded on cmmg / viper / raven (AMD CPU, AMD GPU, NVIDIA CUDA);
           ENABLED on the raven-cpu build (Intel Xeon + oneAPI).
```
