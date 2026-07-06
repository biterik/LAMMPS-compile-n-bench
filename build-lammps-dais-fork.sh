#!/bin/bash
# ===========================================================================
# Build LAMMPS from the thermoatoms FORK on DAIS (NVIDIA H200 / B200 / RTX PRO
# 6000). GRACE runs on the GPU through **TensorFlow-CUDA**, NOT Kokkos — so this
# is a plain serial gcc + GRACE-TF build with NO MPI and NO GPU-arch compilation.
# ONE binary (lmp_dais_fork) therefore runs on ALL THREE DAIS GPU types; TF picks
# up whichever GPU the job is placed on.
#   (GPU-accelerated ACE `pace/kk` would need a separate Kokkos/CUDA build per
#    arch — Hopper sm_90 / Blackwell sm_100 — and is NOT needed for GRACE.)
#
# DAIS shares /viper/ptmp2 with Viper, so the GRACE models already downloaded to
# /viper/ptmp2/$USER/gracework/grace-cache are reused (no re-download).
#
# PREREQUISITE — a TensorFlow-CUDA venv (create once on a login node):
#   module purge && module load python-waterboa/2024.06
#   export PIP_CACHE_DIR=/viper/ptmp2/$USER/.pipcache TMPDIR=/viper/ptmp2/$USER/.tmp
#   python -m venv /viper/ptmp2/$USER/gracework-dais/tf-cuda
#   /viper/ptmp2/$USER/gracework-dais/tf-cuda/bin/pip install -U pip tensorflow
#     (the CUDA build; brings its own CUDA/cuDNN via nvidia-cu12 wheels)
#
# THEN build (login node):
#   CMAKE_VER=<ver> PYTHON=/viper/ptmp2/$USER/gracework-dais/tf-cuda/bin/python \
#     ./build-lammps-dais-fork.sh
# Find CMAKE_VER first:  module load gcc/14 && module avail cmake
# ===========================================================================

GCC_VER="${GCC_VER:-14}"
CMAKE_VER="${CMAKE_VER:-3.30}"     # adjust to what `module avail cmake` shows on DAIS

FORK_URL="${FORK_URL:-https://github.com/thermoatoms/lammps.git}"
FORK_BRANCH="${FORK_BRANCH:-develop}"
FORK_COMMIT="${FORK_COMMIT:-24da74cd73323f5e7415fdd9a9670b88535464d3}"
GRACE_TF="${GRACE_TF:-on}"
# Python module kept loaded so the TF-CUDA venv interpreter runs during configure.
PYMODULE="${PYMODULE:-python-waterboa/2024.06}"

if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}" \
              /mpcdf/soft/SLE_15/packages/x86_64/Modules/*/init/bash; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
module load gcc/${GCC_VER}
module load cmake/${CMAKE_VER}
[ "$GRACE_TF" != "off" ] && [ -n "${PYMODULE:-}" ] && module load "$PYMODULE"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Build in the DAIS scratch (shared /viper/ptmp2), never $HOME.
WORK="${WORK:-/viper/ptmp2/$USER/gracework-dais}"
mkdir -p "$WORK" || { echo "ERROR: cannot create WORK=$WORK" >&2; exit 1; }
cd "$WORK"
echo ">> work dir: $WORK"

LOG="${LOG:-$WORK/build-dais-fork-$(date +%Y%m%d-%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
echo ">> logging to: $LOG"

SRC="${SRC:-$WORK/lammps-fork}"
BUILD="${BUILD:-$SRC/build-dais-fork}"
JOBS="${JOBS:-16}"
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
CXX="${CXX:-$(command -v g++)}"
CC="${CC:-$(command -v gcc)}"
echo ">> g++: $CXX   python: ${PYTHON:-<none>}   GRACE_TF=$GRACE_TF"

if [ ! -d "$SRC/.git" ]; then
    git clone -b "$FORK_BRANCH" "$FORK_URL" "$SRC"
fi
cd "$SRC"
git fetch --all -q || true
git checkout -q "$FORK_COMMIT" || { echo "ERROR: cannot checkout $FORK_COMMIT" >&2; exit 1; }
echo ">> fork at: $(git log -1 --format='%H %s')"

GRACE_FLAGS=()
if [ "$GRACE_TF" = "off" ]; then
    GRACE_FLAGS+=( -D NO_GRACE_TF=ON )
else
    [ -n "${PYTHON:-}" ] && GRACE_FLAGS+=( -D PACE_PYTHON_EXEC="$PYTHON" -D Python_EXECUTABLE="$PYTHON" )
    [ -n "${TF_LIB_FILE:-}" ] && GRACE_FLAGS+=( -D TF_LIB_FILE="$TF_LIB_FILE" )
fi

cmake -S "$SRC/cmake" -B "$BUILD" \
    -C "$SCRIPT_DIR/cmake/lammps-packages-mpcdf.cmake" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=17 \
    -D BUILD_MPI=off -D BUILD_OMP=on \
    -D LAMMPS_MACHINE=dais_fork \
    -D CMAKE_C_COMPILER="$CC" \
    -D CMAKE_CXX_COMPILER="$CXX" \
    -D CMAKE_CXX_FLAGS="-O3 -march=native" \
    -D PKG_PLUMED=off \
    -D PKG_MC=on -D PKG_ML-PACE=on \
    -D PKG_KOKKOS=off \
    -D PKG_VORONOI=on -D DOWNLOAD_VORO=on \
    -D FFT=KISS \
    "${GRACE_FLAGS[@]}" \
    -D CMAKE_EXE_LINKER_FLAGS="-Wl,-rpath=\$ORIGIN/../lib64"

cmake --build "$BUILD" -j "$JOBS"
echo
echo "DONE: $BUILD/lmp_dais_fork"
echo ">> styles (grace/pace/atom_swap):"
( "$BUILD/lmp_dais_fork" -h 2>/dev/null | grep -iE 'grace|pace|atom/swap' | head -20 ) \
  || echo "   (could not run on the login node — verify in a job)"
