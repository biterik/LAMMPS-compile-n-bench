#!/bin/bash
# ===========================================================================
# Build LAMMPS from the thermoatoms FORK on the cmmg partition (CPU).
# Hardware: 2x AMD EPYC 9754 "Bergamo" (Zen4c, 256 cores/node), CPU-only.
#
# WHY THE FORK (not lammps/lammps stable):
#   thermoatoms/lammps adds (a) the efficient MC on `fix atom/swap`
#   (noforce / localE keywords, fast ACE evaluator) and (b) the GRACE pair
#   styles. It is based on upstream patch_11Feb2026 (newer than the current
#   stable). We pin a commit for reproducibility (override with FORK_COMMIT=).
#
# This is the CPU build: it gives both GRACE tracks ->
#   * grace/fs            (1-layer FS model, NO TensorFlow, fast)         and
#   * grace / grace/2layer/chunk  (TensorFlow models, incl. the 2-layer)  .
# So we build WITH TensorFlow (GRACE_TF=on, default). To skip TF and build only
# the FS + ACE styles, run with GRACE_TF=off (sets NO_GRACE_TF -> no TF needed).
#
# TensorFlow: the fork's cmake/Modules/Packages/ML-PACE.cmake discovers TF from
# a Python install (`import tensorflow`). Provide a CPU TensorFlow, e.g.
#     python -m venv ~/tf-cpu && ~/tf-cpu/bin/pip install tensorflow-cpu
#     PYTHON=~/tf-cpu/bin/python ./build-lammps-cmmg-fork.sh
# or point TF_LIB_FILE directly at a libtensorflow_cc.so.2. See GRACE.md.
#
# RUN THIS ON A LOGIN NODE (cmti001/002): internet only on login nodes.
# Builds + git-clones into the cmmg scratch by default — WORK=/u/$USER/PTMP/gracework
# (Erik's cmmg PTMP is /u/biterik/PTMP), NOT $HOME. Override with WORK=/some/path,
# or WORK="$(pwd)" to build in the current dir.
#
#   ./build-lammps-cmmg-fork.sh
# ===========================================================================

GCC_VER="${GCC_VER:-13}"
IMPI_VER="${IMPI_VER:-2021.16}"
CMAKE_VER="${CMAKE_VER:-3.30}"
MKL_VER="${MKL_VER:-2025.2}"      # BLAS/LAPACK for the PLUMED package
GSL_VER="${GSL_VER:-2.7}"         # GSL: last required PLUMED dependency

# --- the fork (pinned) ------------------------------------------------------
FORK_URL="${FORK_URL:-https://github.com/thermoatoms/lammps.git}"
FORK_BRANCH="${FORK_BRANCH:-develop}"
# Pin = tip of develop on 2026-06-22 ("MCnoforce-localE", base patch_11Feb2026).
FORK_COMMIT="${FORK_COMMIT:-24da74cd73323f5e7415fdd9a9670b88535464d3}"

# GRACE / TensorFlow: on by default so the 2-layer TF model can run here.
GRACE_TF="${GRACE_TF:-on}"

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

set -euo pipefail

export I_MPI_CXX=g++             # make the impi C++ wrapper use g++, not icpx
export I_MPI_CC=gcc

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cmmg scratch (not $HOME). Erik's cmmg PTMP = /u/biterik/PTMP; generic form below.
WORK="${WORK:-/u/$USER/PTMP/gracework}"
mkdir -p "$WORK" || { echo "ERROR: cannot create WORK=$WORK — set WORK=/path you can write" >&2; exit 1; }
RUN_DIR="$WORK"
cd "$RUN_DIR"
echo ">> work dir (build + clone land here): $RUN_DIR"

LOG="${LOG:-$RUN_DIR/build-cmmg-fork-$(date +%Y%m%d-%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
echo ">> logging this run to: $LOG"

SRC="${SRC:-$RUN_DIR/lammps-fork}"
BUILD="${BUILD:-$SRC/build-cmmg-fork}"
JOBS="${JOBS:-16}"
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"

# --- locate the REAL Intel MPI wrappers (under I_MPI_ROOT, not PATH) ---------
: "${I_MPI_ROOT:?I_MPI_ROOT not set — is the impi/${IMPI_VER} module loaded?}"
MPI_BIN=""
for d in "$I_MPI_ROOT/bin" "$I_MPI_ROOT/intel64/bin"; do
    [ -d "$d" ] && MPI_BIN="$d" && break
done
pick(){ local n; for n in "$@"; do [ -n "${MPI_BIN:-}" ] && [ -x "$MPI_BIN/$n" ] && { echo "$MPI_BIN/$n"; return 0; }; done; return 0; }
MPICC="${MPICC:-$(pick mpigcc mpicc mpiicx)}"
MPICXX="${MPICXX:-$(pick mpig++ mpicxx mpigxx mpiicpx)}"
echo ">> MPICC=$MPICC  MPICXX=$MPICXX  PYTHON=${PYTHON:-<none>}  GRACE_TF=$GRACE_TF"
[ -n "${MPICXX:-}" ] && [ -n "${MPICC:-}" ] || { echo "ERROR: MPI wrappers not found in $MPI_BIN" >&2; exit 1; }

# --- clone the fork at the pinned commit ------------------------------------
if [ ! -d "$SRC/.git" ]; then
    git clone -b "$FORK_BRANCH" "$FORK_URL" "$SRC"
fi
cd "$SRC"
git fetch --all -q || true
git checkout -q "$FORK_COMMIT" || { echo "ERROR: cannot checkout $FORK_COMMIT" >&2; exit 1; }
echo ">> fork at: $(git log -1 --format='%H %s')"

# --- GRACE/TensorFlow flags -------------------------------------------------
GRACE_FLAGS=()
if [ "$GRACE_TF" = "off" ]; then
    GRACE_FLAGS+=( -D NO_GRACE_TF=ON )                 # FS + ACE only, no TensorFlow
else
    [ -n "${PYTHON:-}" ] && GRACE_FLAGS+=( -D PACE_PYTHON_EXEC="$PYTHON" -D Python_EXECUTABLE="$PYTHON" )
    [ -n "${TF_LIB_FILE:-}" ] && GRACE_FLAGS+=( -D TF_LIB_FILE="$TF_LIB_FILE" )
fi

if [ -f "$BUILD/CMakeCache.txt" ] && ! grep -q "CMAKE_CXX_COMPILER:.*=${MPICXX}$" "$BUILD/CMakeCache.txt"; then
    echo ">> removing stale build dir $BUILD"; rm -rf "$BUILD"
fi

cmake -S "$SRC/cmake" -B "$BUILD" \
    -C "$SCRIPT_DIR/cmake/lammps-packages-mpcdf.cmake" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=17 \
    -D BUILD_MPI=on -D BUILD_OMP=on \
    -D LAMMPS_MACHINE=cmmg_fork \
    -D CMAKE_C_COMPILER="$MPICC" \
    -D CMAKE_CXX_COMPILER="$MPICXX" \
    -D CMAKE_CXX_FLAGS="-O3 -march=znver4" \
    -D BLA_VENDOR=Intel10_64lp_seq \
    -D PKG_PLUMED=on -D DOWNLOAD_PLUMED=on -D PLUMED_MODE=static \
    -D PKG_MC=on -D PKG_ML-PACE=on \
    -D PKG_KOKKOS=on \
    -D Kokkos_ENABLE_SERIAL=on \
    -D Kokkos_ENABLE_OPENMP=on \
    -D Kokkos_ARCH_ZEN4=on \
    -D FFT=KISS \
    "${GRACE_FLAGS[@]}" \
    -D CMAKE_EXE_LINKER_FLAGS="-Wl,-rpath=\$ORIGIN/../lib64"

cmake --build "$BUILD" -j "$JOBS"
echo
echo "DONE: $BUILD/lmp_cmmg_fork"
echo ">> styles (grace/pace/atom_swap):"
( unset I_MPI_PMI_LIBRARY; I_MPI_FABRICS=shm "$BUILD/lmp_cmmg_fork" -h 2>/dev/null \
    | grep -iE 'grace|pace|atom/swap' | head -20 ) \
  || echo "   (could not run on the login node — verify in a job: srun ... $BUILD/lmp_cmmg_fork -h)"
