#!/bin/bash
# ===========================================================================
# Build LAMMPS from the thermoatoms FORK on VIPER-CPU (no GPU).
# Hardware: 2x AMD EPYC 9554 "Genoa" (Zen4, 128 cores/node), CPU-only.
#
# CPU counterpart of build-lammps-viper-fork.sh. Mirrors build-lammps-cmmg-fork.sh
# (fork pinned, both GRACE tracks) but with Viper's gcc/14 + openmpi/5.0 stack
# and PLUMED off (kept on only on cmmg). Binary: lmp_viper_cpu_fork.
#
# GRACE_TF=on (default) builds with TensorFlow so the 2-layer TF model can run
# here; GRACE_TF=off builds FS + ACE only (no TF). Provide a CPU TensorFlow via
# a venv and PYTHON=, or TF_LIB_FILE=, exactly as for cmmg (see GRACE.md).
#
# RUN THIS ON A VIPER LOGIN NODE (viper11i/12i/13i): internet only on login nodes.
#   ./build-lammps-viper-cpu-fork.sh
# ===========================================================================

GCC_VER="${GCC_VER:-14}"
OMPI_VER="${OMPI_VER:-5.0}"
CMAKE_VER="${CMAKE_VER:-}"        # empty -> unversioned 'cmake'

FORK_URL="${FORK_URL:-https://github.com/thermoatoms/lammps.git}"
FORK_BRANCH="${FORK_BRANCH:-develop}"
FORK_COMMIT="${FORK_COMMIT:-24da74cd73323f5e7415fdd9a9670b88535464d3}"
GRACE_TF="${GRACE_TF:-on}"

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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(pwd)"

LOG="${LOG:-$RUN_DIR/build-viper-cpu-fork-$(date +%Y%m%d-%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
echo ">> logging this run to: $LOG"

# --- detach from any active conda env (it can shadow mpic++ / libstdc++) -----
_drop_conda() { printf '%s' "${1:-}" | tr ':' '\n' | grep -viE 'conda' | paste -sd: ; true; }
if [ -n "${CONDA_PREFIX:-}" ] || printf '%s' "${PATH}" | grep -qiE 'conda'; then
    echo ">> conda detected — removing it from PATH/LD_LIBRARY_PATH for a clean build"
    export PATH="$(_drop_conda "$PATH")"
    export LD_LIBRARY_PATH="$(_drop_conda "${LD_LIBRARY_PATH:-}")"
    export LIBRARY_PATH="$(_drop_conda "${LIBRARY_PATH:-}")"
    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE CONDA_SHLVL || true
fi

SRC="${SRC:-$RUN_DIR/lammps-fork}"
BUILD="${BUILD:-$SRC/build-viper-cpu-fork}"
JOBS="${JOBS:-16}"
LMP_NAME="lmp_viper_cpu_fork"
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"

MPICXX="${MPICXX:-$(command -v mpic++ || command -v mpicxx || true)}"
MPICC="${MPICC:-$(command -v mpicc || true)}"
echo ">> MPICXX=${MPICXX:-NONE}  MPICC=${MPICC:-NONE}  PYTHON=${PYTHON:-<none>}  GRACE_TF=$GRACE_TF"
[ -n "${MPICXX:-}" ] || { echo "ERROR: no MPI C++ wrapper (mpic++/mpicxx) on PATH" >&2; exit 1; }

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
    -D BUILD_MPI=on -D BUILD_OMP=on \
    -D LAMMPS_MACHINE=viper_cpu_fork \
    -D CMAKE_C_COMPILER="$MPICC" \
    -D CMAKE_CXX_COMPILER="$MPICXX" \
    -D CMAKE_CXX_FLAGS="-O3 -march=znver4" \
    -D PKG_PLUMED=off \
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
echo "DONE: $BUILD/$LMP_NAME"
echo ">> styles (grace/pace/atom_swap):"
( "$BUILD/$LMP_NAME" -h 2>/dev/null | grep -iE 'grace|pace|atom/swap' | head -20 ) \
  || echo "   (could not run $LMP_NAME on the login node — verify in a job)"
