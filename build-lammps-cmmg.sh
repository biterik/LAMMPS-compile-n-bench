#!/bin/bash
# ===========================================================================
# Build LAMMPS (stable branch) on the cmmg partition (Sustainable Materials).
# Hardware: dual-socket AMD EPYC 9754 "Bergamo" (Zen4c, 256 physical cores/node).
# CPU-only cluster  ->  KOKKOS with the OpenMP host backend, arch ZEN4.
#
# The PACE benchmark on cmmg runs MPI-parallel with the plain CPU `pair_style
# pace` (one rank per physical core); KOKKOS is still built so the package set
# matches the GPU machines.
#
# RUN THIS ON A LOGIN NODE (cmti001/002): internet only on login nodes.
# Builds and git-clones into the directory you launch it from (not $HOME).
#
#   ./build-lammps-cmmg.sh
# ===========================================================================

# ---------------------------------------------------------------------------
# Module setup FIRST, *before* `set -euo pipefail`. The Lmod `module` function
# evals code that references unset variables and can return non-zero; under
# `set -e`/`set -u` that makes the script abort silently ("does nothing").
# This cluster has NO default module versions: `module load impi` fails with
# "Loading impi/.noversion ... ERROR". Pin versions (see `module avail`).
# ---------------------------------------------------------------------------
GCC_VER="${GCC_VER:-13}"
IMPI_VER="${IMPI_VER:-2021.16}"
CMAKE_VER="${CMAKE_VER:-3.30}"
MKL_VER="${MKL_VER:-2025.2}"      # supplies BLAS/LAPACK for the PLUMED package
GSL_VER="${GSL_VER:-2.7}"         # GSL: the 3rd (and last) required PLUMED dependency

# Make the 'module' function available without a LOGIN shell. (We deliberately
# do NOT use '#!/bin/bash -l': on this account the login startup files exit for
# non-interactive shells, so '-l' made the script terminate with no output.)
if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh \
              "${MODULESHOME:+$MODULESHOME/init/bash}" \
              /mpcdf/soft/SLE_15/packages/x86_64/Modules/5.4.0/init/bash; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
module load gcc/${GCC_VER}
module load impi/${IMPI_VER}
module load cmake/${CMAKE_VER}
module load mkl/${MKL_VER}       # BLAS/LAPACK for PLUMED (sets MKLROOT)
module load gsl/${GSL_VER}       # GSL for PLUMED (sets GSL_HOME)

# now safe to be strict
set -euo pipefail

export I_MPI_CXX=g++             # make the impi C++ wrapper use g++, not icpx
export I_MPI_CC=gcc

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(pwd)"

# --- Capture EVERYTHING (stdout + stderr) to a timestamped log file while still
#     echoing to the screen. Every warning/error from the module loads, wrapper
#     detection, the cmake configure and the build itself lands in $LOG.
#     Override the path with:  LOG=/ptmp/$USER/cmmg.log ./build-lammps-cmmg.sh
LOG="${LOG:-$RUN_DIR/build-cmmg-$(date +%Y%m%d-%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
echo ">> logging this run to: $LOG"

SRC="${SRC:-$RUN_DIR/lammps}"
BUILD="${BUILD:-$SRC/build-cmmg}"
JOBS="${JOBS:-16}"

# --- guard against a conda env on PATH hijacking the toolchain --------------
# (a pyiron mambaforge env was shadowing 'mpicxx' with a non-Intel MPI).
if command -v python >/dev/null && python -c 'import sys; sys.exit(0 if "conda" in sys.prefix or "mambaforge" in sys.prefix or "envs" in sys.prefix else 1)' 2>/dev/null; then
    echo "WARNING: a conda/mambaforge env is active ($(command -v python))." >&2
    echo "         Run 'conda deactivate' until it's gone, then re-run — it can" >&2
    echo "         shadow mpicxx/cmake/zlib/fftw and corrupt the build." >&2
fi

# --- locate the REAL Intel MPI wrappers (they live under I_MPI_ROOT, not PATH) ---
: "${I_MPI_ROOT:?I_MPI_ROOT not set — is the impi/${IMPI_VER} module loaded?}"
MPI_BIN=""
for d in "$I_MPI_ROOT/bin" "$I_MPI_ROOT/intel64/bin"; do
    [ -d "$d" ] && MPI_BIN="$d" && break
done
# IMPORTANT: 'pick' must NEVER return non-zero. Under 'set -e', a non-zero
# command substitution in an assignment ( MPICC="$(pick ...)" ) aborts the whole
# script *silently* — that was the "does nothing" bug.
pick(){
    local n
    for n in "$@"; do
        [ -n "${MPI_BIN:-}" ] && [ -x "$MPI_BIN/$n" ] && { echo "$MPI_BIN/$n"; return 0; }
    done
    return 0
}
# GNU wrappers first. On this oneAPI install the GNU wrappers are named
# 'mpigcc' / 'mpig++' (NOT mpicxx/mpigxx). Intel-LLVM (mpiicx/mpiicpx) last resort.
MPICC="${MPICC:-$(pick mpigcc mpicc mpiicx)}"
MPICXX="${MPICXX:-$(pick mpig++ mpicxx mpigxx mpiicpx)}"

echo ">> module list:"; module list 2>&1 | sed 's/^/   /'
echo ">> I_MPI_ROOT = $I_MPI_ROOT"
echo ">> MPI_BIN    = $MPI_BIN"
echo ">> wrappers in MPI_BIN:"; ls "$MPI_BIN" 2>/dev/null | grep -i '^mpi' | sed 's/^/     /' || echo "     (none)"
echo ">> MPICC  = ${MPICC:-NOT FOUND}"
echo ">> MPICXX = ${MPICXX:-NOT FOUND}"
if [ -z "$MPICXX" ] || [ -z "$MPICC" ]; then
    echo "ERROR: no usable MPI compiler wrapper found in $MPI_BIN" >&2
    echo "       Set them by hand, e.g.  MPICC=/path/mpicc MPICXX=/path/mpicxx $0" >&2
    exit 1
fi
echo ">> g++ version: $(g++ -dumpfullversion 2>/dev/null || g++ -dumpversion)"
echo ">> build dir  : $BUILD"

# --- source: 'stable' tracks the newest stable release incl. its updates -----
if [ ! -d "$SRC/.git" ]; then
    git clone -b stable https://github.com/lammps/lammps.git "$SRC"
fi
cd "$SRC"

# A stale cache from an earlier (failed) configure pins the wrong compiler and
# makes cmake error/no-op — start clean if the compiler doesn't match.
if [ -f "$BUILD/CMakeCache.txt" ] && ! grep -q "CMAKE_CXX_COMPILER:.*=${MPICXX}$" "$BUILD/CMakeCache.txt"; then
    echo ">> removing stale build dir $BUILD"
    rm -rf "$BUILD"
fi

cmake -S "$SRC/cmake" -B "$BUILD" \
    -C "$SCRIPT_DIR/cmake/lammps-packages-mpcdf.cmake" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=17 \
    -D BUILD_MPI=on -D BUILD_OMP=on \
    -D CMAKE_C_COMPILER="$MPICC" \
    -D CMAKE_CXX_COMPILER="$MPICXX" \
    -D CMAKE_CXX_FLAGS="-O3 -march=znver4" \
    -D BLA_VENDOR=Intel10_64lp_seq \
    -D PKG_PLUMED=on -D DOWNLOAD_PLUMED=on -D PLUMED_MODE=static \
    -D PKG_KOKKOS=on \
    -D Kokkos_ENABLE_SERIAL=on \
    -D Kokkos_ENABLE_OPENMP=on \
    -D Kokkos_ARCH_ZEN4=on \
    -D FFT=KISS \
    -D CMAKE_EXE_LINKER_FLAGS="-Wl,-rpath=\$ORIGIN/../lib64"

cmake --build "$BUILD" -j "$JOBS"
echo
echo "DONE: $BUILD/lmp"

# Smoke test: list installed packages. The binary is Intel-MPI-linked, so a bare
# run on the login node hits SLURM's PMI (I_MPI_PMI_LIBRARY) and aborts in
# MPI_Init. Force a true singleton init by clearing the PMI lib, and never let
# this optional check abort the (already successful) build.
echo ">> installed packages:"
( unset I_MPI_PMI_LIBRARY; I_MPI_FABRICS=shm "$BUILD/lmp" -h 2>/dev/null \
    | sed -n '/Installed packages/,/^$/p' | head -20 ) \
  || echo "   (could not run lmp on the login node — verify in a job: srun ... $BUILD/lmp -h)"
