#!/bin/bash
# ===========================================================================
# Build LAMMPS + MC-SITES (compute sites/voronoi + fix mc/sites) on cmmg (CPU).
# Base: thermoatoms/lammps @ 24da74cd (patch_11Feb2026) + feature/mc-sites patches.
# Hardware: 2x AMD EPYC 9754 "Bergamo" (Zen4c, 256 cores/node) -> KOKKOS/OpenMP, ZEN4.
#
# Full compile-n-bench package set (shared cmake/lammps-packages-mpcdf.cmake) PLUS
# the MC-SITES contribution. PKG_MC + PKG_ML-PACE are required on the fork (its
# MC/fix_atom_swap.cpp #includes pair_pace.h); VORONOI carries compute sites/voronoi.
#
# Binary:  lmp_mcsites_fork24da74_cmmg
#
# GRACE/TensorFlow is OFF by default (GRACE_TF=off -> NO_GRACE_TF). fix mc/sites
# works fully with EAM/MEAM/ACE/PACE without TF. To also run GRACE ML potentials
# with the energy-only fast path, build with:  GRACE_TF=on PYTHON=~/tf-cpu/bin/python
# (a CPU TensorFlow venv; see README "GRACE / TensorFlow").
#
# RUN THIS ON A cmmg LOGIN NODE (cmti001/002): internet only on login nodes.
# cd into your PTMP work dir first (never $HOME), e.g. /u/$USER/PTMP/mcsites, then:
#   ./build-mcsites-cmmg.sh
# ===========================================================================

GCC_VER="${GCC_VER:-13}"
IMPI_VER="${IMPI_VER:-2021.16}"
CMAKE_VER="${CMAKE_VER:-3.30}"
MKL_VER="${MKL_VER:-2025.2}"      # supplies BLAS/LAPACK for the PLUMED package
GSL_VER="${GSL_VER:-2.7}"         # GSL: the last required PLUMED dependency

# --- MC-SITES source: thermoatoms fork (pinned) + patches -------------------
FORK_URL="${FORK_URL:-https://github.com/thermoatoms/lammps.git}"
FORK_BRANCH="${FORK_BRANCH:-develop}"
FORK_COMMIT="${FORK_COMMIT:-24da74cd73323f5e7415fdd9a9670b88535464d3}"
MCSITES_BRANCH="${MCSITES_BRANCH:-feature/mc-sites}"
GRACE_TF="${GRACE_TF:-off}"       # off = NO_GRACE_TF (no TensorFlow needed). on = GRACE.
PYMODULE="${PYMODULE:-}"          # Python module for TF discovery when GRACE_TF=on

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
module load mkl/${MKL_VER}        # BLAS/LAPACK for PLUMED (sets MKLROOT)
module load gsl/${GSL_VER}        # GSL for PLUMED (sets GSL_HOME)
if [ "$GRACE_TF" != "off" ] && [ -n "${PYMODULE:-}" ]; then
    module load "$PYMODULE" && echo ">> loaded Python module for TF discovery: $PYMODULE"
fi

set -euo pipefail

export I_MPI_CXX=g++
export I_MPI_CC=gcc

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(pwd)"
PATCHES_DIR="${PATCHES_DIR:-$SCRIPT_DIR/patches}"

LOG="${LOG:-$RUN_DIR/build-mcsites-cmmg-$(date +%Y%m%d-%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
echo ">> logging this run to: $LOG"

SRC="${SRC:-$RUN_DIR/lammps-mcsites}"
BUILD="${BUILD:-$SRC/build-mcsites-cmmg}"
JOBS="${JOBS:-16}"

# --- guard against a conda env hijacking the toolchain ----------------------
if command -v python >/dev/null && python -c 'import sys; sys.exit(0 if "conda" in sys.prefix or "mambaforge" in sys.prefix or "envs" in sys.prefix else 1)' 2>/dev/null; then
    echo "WARNING: a conda/mambaforge env is active ($(command -v python))." >&2
    echo "         Run 'conda deactivate' until it's gone, then re-run." >&2
fi

# --- locate the REAL Intel MPI wrappers (under I_MPI_ROOT, not PATH) ---------
: "${I_MPI_ROOT:?I_MPI_ROOT not set — is the impi/${IMPI_VER} module loaded?}"
MPI_BIN=""
for d in "$I_MPI_ROOT/bin" "$I_MPI_ROOT/intel64/bin"; do
    [ -d "$d" ] && MPI_BIN="$d" && break
done
pick(){ local n; for n in "$@"; do [ -n "${MPI_BIN:-}" ] && [ -x "$MPI_BIN/$n" ] && { echo "$MPI_BIN/$n"; return 0; }; done; return 0; }
MPICC="${MPICC:-$(pick mpigcc mpicc mpiicx)}"
MPICXX="${MPICXX:-$(pick mpig++ mpicxx mpigxx mpiicpx)}"
echo ">> MPICC=${MPICC:-NONE}  MPICXX=${MPICXX:-NONE}  GRACE_TF=$GRACE_TF"
[ -n "${MPICXX:-}" ] && [ -n "${MPICC:-}" ] || { echo "ERROR: MPI wrappers not found in $MPI_BIN" >&2; exit 1; }
echo ">> g++ version: $(g++ -dumpfullversion 2>/dev/null || g++ -dumpversion)"

# --- obtain source: fork + mc-sites patches (idempotent) --------------------
if [ ! -d "$SRC/.git" ]; then
    git clone -b "$FORK_BRANCH" "$FORK_URL" "$SRC"
fi
cd "$SRC"
git fetch --all -q || true
if git rev-parse -q --verify "$MCSITES_BRANCH" >/dev/null; then
    echo ">> reusing branch $MCSITES_BRANCH: $(git log -1 --format='%h %s' "$MCSITES_BRANCH")"
else
    echo ">> creating $MCSITES_BRANCH from $FORK_COMMIT and applying mc-sites patches"
    git checkout -q "$FORK_COMMIT"
    git checkout -q -b "$MCSITES_BRANCH"
    [ -d "$PATCHES_DIR" ] || { echo "ERROR: PATCHES_DIR not found: $PATCHES_DIR (ship the patches/ dir next to this script or set PATCHES_DIR=)" >&2; exit 1; }
    git am "$PATCHES_DIR"/00*.patch
fi
git checkout -q "$MCSITES_BRANCH"
echo ">> source at: $(git log -1 --format='%H %s')"

# --- GRACE/TensorFlow flags -------------------------------------------------
GRACE_FLAGS=()
if [ "$GRACE_TF" = "off" ]; then
    GRACE_FLAGS+=( -D NO_GRACE_TF=ON )
else
    PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
    [ -n "${PYTHON:-}" ] && GRACE_FLAGS+=( -D PACE_PYTHON_EXEC="$PYTHON" -D Python_EXECUTABLE="$PYTHON" )
    [ -n "${TF_LIB_FILE:-}" ] && GRACE_FLAGS+=( -D TF_LIB_FILE="$TF_LIB_FILE" )
    echo ">> GRACE_TF=on: TensorFlow discovery via ${PYTHON:-<none>}"
fi

if [ -f "$BUILD/CMakeCache.txt" ] && ! grep -q "CMAKE_CXX_COMPILER:.*=${MPICXX}$" "$BUILD/CMakeCache.txt"; then
    echo ">> removing stale build dir $BUILD"; rm -rf "$BUILD"
fi

cmake -S "$SRC/cmake" -B "$BUILD" \
    -C "$SCRIPT_DIR/cmake/lammps-packages-mpcdf.cmake" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=17 \
    -D BUILD_MPI=on -D BUILD_OMP=on \
    -D LAMMPS_MACHINE=mcsites_fork24da74_cmmg \
    -D CMAKE_C_COMPILER="$MPICC" \
    -D CMAKE_CXX_COMPILER="$MPICXX" \
    -D CMAKE_CXX_FLAGS="-O3 -march=znver4" \
    -D BLA_VENDOR=Intel10_64lp_seq \
    -D PKG_PLUMED=on -D DOWNLOAD_PLUMED=on -D PLUMED_MODE=static \
    -D PKG_MC=on -D PKG_ML-PACE=on -D PKG_VORONOI=on \
    -D PKG_KOKKOS=on \
    -D Kokkos_ENABLE_SERIAL=on \
    -D Kokkos_ENABLE_OPENMP=on \
    -D Kokkos_ARCH_ZEN4=on \
    -D FFT=KISS \
    "${GRACE_FLAGS[@]}" \
    -D CMAKE_EXE_LINKER_FLAGS="-Wl,-rpath=\$ORIGIN/../lib64"

cmake --build "$BUILD" -j "$JOBS"
echo
echo "DONE: $BUILD/lmp_mcsites_fork24da74_cmmg"
echo ">> mc-sites styles present?"
( unset I_MPI_PMI_LIBRARY; I_MPI_FABRICS=shm "$BUILD/lmp_mcsites_fork24da74_cmmg" -h 2>/dev/null \
    | grep -iE 'sites/voronoi|mc/sites' ) \
  || echo "   (could not run on the login node — verify in a job: srun ... $BUILD/lmp_mcsites_fork24da74_cmmg -h | grep -iE 'sites/voronoi|mc/sites')"
