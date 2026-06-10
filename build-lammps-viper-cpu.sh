#!/bin/bash
# ===========================================================================
# Build LAMMPS (stable branch) on VIPER-CPU (no GPU).
# Hardware: 2x AMD EPYC 9554 "Genoa" (Zen4, 128 physical cores/node). This is the
# CPU counterpart to build-lammps-viper.sh (which targets the MI300A APU GPU).
#
# CPU-only -> KOKKOS with the OpenMP host backend, arch ZEN4 (same family as the
# cmmg EPYC build). The PACE benchmark runs MPI-parallel with the plain CPU
# `pair_style pace` (one rank per physical core); Kokkos is still built so the
# package set matches the other machines.
#
# Binary is named  lmp_viper_cpu  (LAMMPS_MACHINE) so it never collides with the
# GPU build's  lmp_viper.
#
# RUN THIS ON A VIPER LOGIN NODE (viper11i/12i/13i): internet only on login nodes.
# Builds and git-clones into the directory you launch it from (not $HOME).
#
#   ./build-lammps-viper-cpu.sh
# ===========================================================================

# NO default module versions on MPCDF (gotcha 2): pin them. Confirmed on Viper
# (RHEL 9, Jun 2026): openmpi is hierarchical — it appears only after the gcc
# module, and gcc/14 offers openmpi/4.1 and openmpi/5.0.
GCC_VER="${GCC_VER:-14}"
OMPI_VER="${OMPI_VER:-5.0}"        # openmpi/5.0 under gcc/14 (confirmed)
CMAKE_VER="${CMAKE_VER:-}"         # empty -> load unversioned 'cmake' (adjust if pinned)

# Make the 'module' function available without a LOGIN shell. (We deliberately do
# NOT use '#!/bin/bash -l': the login startup files exit for non-interactive
# shells, which made '-l' scripts terminate with no output.)
if ! command -v module >/dev/null 2>&1; then
    # Viper is RHEL 9 (Raven/cmmg are SLE 15); the first two entries are
    # distro-agnostic and normally suffice — the `module` function is usually
    # already active in an interactive shell, so this fallback rarely fires.
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

# --- Capture EVERYTHING (stdout + stderr) to a timestamped log file -----------
LOG="${LOG:-$RUN_DIR/build-viper-cpu-$(date +%Y%m%d-%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
echo ">> logging this run to: $LOG"

# --- Detach from any active conda environment (gotcha 13) -------------------
_drop_conda() { printf '%s' "${1:-}" | tr ':' '\n' | grep -viE 'conda' | paste -sd: ; true; }
if [ -n "${CONDA_PREFIX:-}" ] || printf '%s' "${PATH}" | grep -qiE 'conda'; then
    echo ">> conda detected — removing it from PATH/LD_LIBRARY_PATH for a clean build"
    export PATH="$(_drop_conda "$PATH")"
    export LD_LIBRARY_PATH="$(_drop_conda "${LD_LIBRARY_PATH:-}")"
    export LIBRARY_PATH="$(_drop_conda "${LIBRARY_PATH:-}")"
    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE CONDA_SHLVL || true
fi

SRC="${SRC:-$RUN_DIR/lammps}"
BUILD="${BUILD:-$SRC/build-viper-cpu}"
JOBS="${JOBS:-16}"
LMP_NAME="lmp_viper_cpu"          # distinct binary name (LAMMPS_MACHINE)

MPICXX="${MPICXX:-$(command -v mpic++ || command -v mpicxx || true)}"
MPICC="${MPICC:-$(command -v mpicc || true)}"
echo ">> module list:"; module list 2>&1 | sed 's/^/   /'
echo ">> MPICXX = ${MPICXX:-NOT FOUND}"
echo ">> MPICC  = ${MPICC:-NOT FOUND}"
[ -n "${MPICXX:-}" ] || { echo "ERROR: no MPI C++ wrapper (mpic++/mpicxx) on PATH" >&2; exit 1; }
echo ">> g++ version: $(g++ -dumpfullversion 2>/dev/null || g++ -dumpversion)"

# --- source: 'stable' tracks the newest stable release incl. its updates -----
if [ ! -d "$SRC/.git" ]; then
    git clone -b stable https://github.com/lammps/lammps.git "$SRC"
fi
cd "$SRC"

cmake -S "$SRC/cmake" -B "$BUILD" \
    -C "$SCRIPT_DIR/cmake/lammps-packages-mpcdf.cmake" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=17 \
    -D BUILD_MPI=on -D BUILD_OMP=on \
    -D LAMMPS_MACHINE=viper_cpu \
    -D CMAKE_C_COMPILER="$MPICC" \
    -D CMAKE_CXX_COMPILER="$MPICXX" \
    -D CMAKE_CXX_FLAGS="-O3 -march=znver4" \
    -D PKG_PLUMED=off \
    -D PKG_KOKKOS=on \
    -D Kokkos_ENABLE_SERIAL=on \
    -D Kokkos_ENABLE_OPENMP=on \
    -D Kokkos_ARCH_ZEN4=on \
    -D FFT=KISS \
    -D CMAKE_EXE_LINKER_FLAGS="-Wl,-rpath=\$ORIGIN/../lib64"

cmake --build "$BUILD" -j "$JOBS"
echo
echo "DONE: $BUILD/$LMP_NAME"
echo ">> installed packages:"
( "$BUILD/$LMP_NAME" -h 2>/dev/null | sed -n '/Installed packages/,/^$/p' | head -25 ) \
  || echo "   (could not run $LMP_NAME on the login node — verify in a job: srun ... $BUILD/$LMP_NAME -h)"
