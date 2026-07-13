#!/bin/bash
# ===========================================================================
# Build LAMMPS + MC-SITES on VIPER-CPU (2x AMD EPYC 9554 "Genoa", 128 cores).
# Base: thermoatoms/lammps @ 24da74cd + feature/mc-sites patches.
# CPU-only -> KOKKOS/OpenMP, arch ZEN4 (mirrors cmmg, minus PLUMED).
#
# Full compile-n-bench package set + MC-SITES (PKG_MC + PKG_ML-PACE + PKG_VORONOI).
#
# Binary:  lmp_mcsites_fork24da74_viper_cpu
#
# GRACE/TensorFlow OFF by default. GRACE_TF=on PYTHON=~/tf-cpu/bin/python to enable
# the 2-layer TF GRACE model + energy-only fast path (CPU TensorFlow).
#
# RUN THIS ON A VIPER LOGIN NODE (viper11i/12i/13i). cd into /viper/ptmp/$USER/mcsites first.
#   ./build-mcsites-viper-cpu.sh
# ===========================================================================

GCC_VER="${GCC_VER:-14}"
OMPI_VER="${OMPI_VER:-5.0}"
CMAKE_VER="${CMAKE_VER:-}"         # empty -> unversioned 'cmake'

FORK_URL="${FORK_URL:-https://github.com/thermoatoms/lammps.git}"
FORK_BRANCH="${FORK_BRANCH:-develop}"
FORK_COMMIT="${FORK_COMMIT:-24da74cd73323f5e7415fdd9a9670b88535464d3}"
MCSITES_BRANCH="${MCSITES_BRANCH:-feature/mc-sites}"
GRACE_TF="${GRACE_TF:-off}"
PYMODULE="${PYMODULE:-}"

if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh \
              "${MODULESHOME:+$MODULESHOME/init/bash}" \
              /mpcdf/soft/RHEL_9/packages/x86_64/Modules/*/init/bash \
              /mpcdf/soft/SLE_15/packages/x86_64/Modules/5.4.0/init/bash; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
module load gcc/${GCC_VER}
module load openmpi/${OMPI_VER}
module load cmake${CMAKE_VER:+/$CMAKE_VER}
if [ "$GRACE_TF" != "off" ] && [ -n "${PYMODULE:-}" ]; then
    module load "$PYMODULE" && echo ">> loaded Python module for TF discovery: $PYMODULE"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(pwd)"
PATCHES_DIR="${PATCHES_DIR:-$SCRIPT_DIR/patches}"

LOG="${LOG:-$RUN_DIR/build-mcsites-viper-cpu-$(date +%Y%m%d-%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
echo ">> logging this run to: $LOG"

# --- detach from any active conda env ---------------------------------------
_drop_conda() { printf '%s' "${1:-}" | tr ':' '\n' | grep -viE 'conda' | paste -sd: ; true; }
if [ -n "${CONDA_PREFIX:-}" ] || printf '%s' "${PATH}" | grep -qiE 'conda'; then
    echo ">> conda detected — removing it from PATH/LD_LIBRARY_PATH for a clean build"
    export PATH="$(_drop_conda "$PATH")"
    export LD_LIBRARY_PATH="$(_drop_conda "${LD_LIBRARY_PATH:-}")"
    export LIBRARY_PATH="$(_drop_conda "${LIBRARY_PATH:-}")"
    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE CONDA_SHLVL || true
fi

SRC="${SRC:-$RUN_DIR/lammps-mcsites}"
BUILD="${BUILD:-$SRC/build-mcsites-viper-cpu}"
JOBS="${JOBS:-16}"

MPICXX="${MPICXX:-$(command -v mpic++ || command -v mpicxx || true)}"
MPICC="${MPICC:-$(command -v mpicc || true)}"
echo ">> MPICXX=${MPICXX:-NONE}  MPICC=${MPICC:-NONE}  GRACE_TF=$GRACE_TF"
[ -n "${MPICXX:-}" ] || { echo "ERROR: no MPI C++ wrapper (mpic++/mpicxx) on PATH" >&2; exit 1; }
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
    [ -d "$PATCHES_DIR" ] || { echo "ERROR: PATCHES_DIR not found: $PATCHES_DIR" >&2; exit 1; }
    git am "$PATCHES_DIR"/00*.patch
fi
git checkout -q "$MCSITES_BRANCH"
echo ">> source at: $(git log -1 --format='%H %s')"

GRACE_FLAGS=()
if [ "$GRACE_TF" = "off" ]; then
    GRACE_FLAGS+=( -D NO_GRACE_TF=ON )
else
    PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
    [ -n "${PYTHON:-}" ] && GRACE_FLAGS+=( -D PACE_PYTHON_EXEC="$PYTHON" -D Python_EXECUTABLE="$PYTHON" )
    [ -n "${TF_LIB_FILE:-}" ] && GRACE_FLAGS+=( -D TF_LIB_FILE="$TF_LIB_FILE" )
    echo ">> GRACE_TF=on: CPU TensorFlow via ${PYTHON:-<none>}"
fi

cmake -S "$SRC/cmake" -B "$BUILD" \
    -C "$SCRIPT_DIR/cmake/lammps-packages-mpcdf.cmake" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=17 \
    -D BUILD_MPI=on -D BUILD_OMP=on \
    -D LAMMPS_MACHINE=mcsites_fork24da74_viper_cpu \
    -D CMAKE_C_COMPILER="$MPICC" \
    -D CMAKE_CXX_COMPILER="$MPICXX" \
    -D CMAKE_CXX_FLAGS="-O3 -march=znver4" \
    -D PKG_PLUMED=off \
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
echo "DONE: $BUILD/lmp_mcsites_fork24da74_viper_cpu"
echo ">> mc-sites styles present?"
( "$BUILD/lmp_mcsites_fork24da74_viper_cpu" -h 2>/dev/null | grep -iE 'sites/voronoi|mc/sites' ) \
  || echo "   (could not run on the login node — verify in a job: srun ... -h | grep -iE 'sites/voronoi|mc/sites')"
