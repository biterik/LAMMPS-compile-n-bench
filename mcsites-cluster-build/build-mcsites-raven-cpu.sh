#!/bin/bash
# ===========================================================================
# Build LAMMPS + MC-SITES on RAVEN-CPU (2x Intel Xeon IceLake-SP 8360Y, 72 cores).
# Base: thermoatoms/lammps @ 24da74cd + feature/mc-sites patches.
# Toolchain: Intel oneAPI (icpx/icx) + Intel MPI + MKL, LAMMPS INTEL package (cpu).
#
# This is a NEW variant (there was no raven-cpu fork build before): it takes the
# proven compile-n-bench raven-cpu Intel recipe and adds MC-SITES + the fork's
# required PKG_ML-PACE coupling. `pair_style pace` has no INTEL variant, so ACE
# runs as the standard CPU kernel — a fair Intel-Xeon data point.
#
# Binary:  lmp_mcsites_fork24da74_raven_cpu
#
# GRACE/TensorFlow OFF by default (recommended on the Intel toolchain). fix mc/sites
# works with EAM/MEAM/ACE/PACE. GRACE_TF=on PYTHON=~/tf-cpu/bin/python to enable TF.
#
# RUN THIS ON A RAVEN LOGIN NODE (raven01i..04i). cd into /ptmp/$USER/mcsites first.
#   ./build-mcsites-raven-cpu.sh
# ===========================================================================

INTEL_VER="${INTEL_VER:-2025.3}"   # oneAPI compilers (icpx/icx)
IMPI_VER="${IMPI_VER:-2021.17}"    # Intel MPI (hierarchical, under intel/2025.3)
MKL_VER="${MKL_VER:-2025.3}"       # BLAS/LAPACK + FFT
CMAKE_VER="${CMAKE_VER:-3.30}"

FORK_URL="${FORK_URL:-https://github.com/thermoatoms/lammps.git}"
FORK_BRANCH="${FORK_BRANCH:-develop}"
FORK_COMMIT="${FORK_COMMIT:-24da74cd73323f5e7415fdd9a9670b88535464d3}"
MCSITES_BRANCH="${MCSITES_BRANCH:-feature/mc-sites}"
GRACE_TF="${GRACE_TF:-off}"
PYMODULE="${PYMODULE:-}"

if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh \
              "${MODULESHOME:+$MODULESHOME/init/bash}" \
              /mpcdf/soft/SLE_15/packages/x86_64/Modules/5.4.0/init/bash; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
module load intel/${INTEL_VER}
module load impi/${IMPI_VER}
module load mkl/${MKL_VER}
module load cmake/${CMAKE_VER}
if [ "$GRACE_TF" != "off" ] && [ -n "${PYMODULE:-}" ]; then
    module load "$PYMODULE" && echo ">> loaded Python module for TF discovery: $PYMODULE"
fi

set -euo pipefail

export I_MPI_CXX=icpx
export I_MPI_CC=icx

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(pwd)"
PATCHES_DIR="${PATCHES_DIR:-$SCRIPT_DIR/patches}"

LOG="${LOG:-$RUN_DIR/build-mcsites-raven-cpu-$(date +%Y%m%d-%H%M%S).log}"
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
BUILD="${BUILD:-$SRC/build-mcsites-raven-cpu}"
JOBS="${JOBS:-16}"

# --- locate the REAL Intel MPI wrappers -------------------------------------
: "${I_MPI_ROOT:?I_MPI_ROOT not set — is the impi/${IMPI_VER} module loaded?}"
MPI_BIN=""
for d in "$I_MPI_ROOT/bin" "$I_MPI_ROOT/intel64/bin"; do
    [ -d "$d" ] && MPI_BIN="$d" && break
done
pick(){ local n; for n in "$@"; do [ -n "${MPI_BIN:-}" ] && [ -x "$MPI_BIN/$n" ] && { echo "$MPI_BIN/$n"; return 0; }; done; return 0; }
MPICXX="${MPICXX:-$(pick mpiicpx mpiicpc mpicxx)}"
MPICC="${MPICC:-$(pick mpiicx mpiicc mpicc)}"
echo ">> MPICXX=${MPICXX:-NONE} (I_MPI_CXX=$I_MPI_CXX)  MPICC=${MPICC:-NONE}  GRACE_TF=$GRACE_TF"
[ -n "${MPICXX:-}" ] && [ -n "${MPICC:-}" ] || { echo "ERROR: no usable Intel MPI wrapper in $MPI_BIN" >&2; exit 1; }
echo ">> icpx version: $(icpx --version 2>/dev/null | head -1 || echo '?')"

# --- obtain source: fork + mc-sites patches (idempotent) --------------------
if [ ! -d "$SRC/.git" ]; then
    git clone -b "$FORK_BRANCH" "$FORK_URL" "$SRC"
fi
cd "$SRC"
git fetch --all -q || true
if git rev-parse -q --verify "$MCSITES_BRANCH" >/dev/null && \
   git cat-file -e "$MCSITES_BRANCH:src/MC/fix_mc_sites.cpp" 2>/dev/null; then
    echo ">> reusing branch $MCSITES_BRANCH: $(git log -1 --format='%h %s' "$MCSITES_BRANCH")"
else
    echo ">> (re)creating $MCSITES_BRANCH from $FORK_COMMIT and applying mc-sites patches"
    git am --abort 2>/dev/null || true
    git checkout -q "$FORK_COMMIT"
    git branch -D "$MCSITES_BRANCH" 2>/dev/null || true
    git checkout -q -b "$MCSITES_BRANCH"
    [ -d "$PATCHES_DIR" ] || { echo "ERROR: PATCHES_DIR not found: $PATCHES_DIR" >&2; exit 1; }
    git -c user.email=mcsites@localhost -c user.name='mc-sites build' am "$PATCHES_DIR"/00*.patch
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

if [ -f "$BUILD/CMakeCache.txt" ] && ! grep -q "CMAKE_CXX_COMPILER:.*=${MPICXX}$" "$BUILD/CMakeCache.txt"; then
    echo ">> removing stale build dir $BUILD"; rm -rf "$BUILD"
fi

cmake -S "$SRC/cmake" -B "$BUILD" \
    -C "$SCRIPT_DIR/cmake/lammps-packages-mpcdf.cmake" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=17 \
    -D BUILD_MPI=on -D BUILD_OMP=on \
    -D LAMMPS_MACHINE=mcsites_fork24da74_raven_cpu \
    -D CMAKE_C_COMPILER="$MPICC" \
    -D CMAKE_CXX_COMPILER="$MPICXX" \
    -D CMAKE_CXX_FLAGS="-O3 -xCORE-AVX512 -qopt-zmm-usage=high" \
    -D PKG_INTEL=on -D INTEL_ARCH=cpu \
    -D PKG_OPENMP=on -D PKG_OPT=on \
    -D PKG_MC=on -D PKG_ML-PACE=on -D PKG_VORONOI=on \
    -D USE_INTERNAL_LINALG=off -D BLA_VENDOR=Intel10_64lp_seq \
    -D FFT=MKL -D FFT_KOKKOS=KISS \
    -D PKG_PLUMED=off \
    -D PKG_KOKKOS=off \
    "${GRACE_FLAGS[@]}" \
    -D CMAKE_EXE_LINKER_FLAGS="-Wl,-rpath=\$ORIGIN/../lib64"

# NOTE: KIM downloads + builds its Fortran with the oneAPI toolchain. If that step
# fails, re-run with  -D PKG_KIM=off appended (KIM is unused by fix mc/sites):
#     PKG_KIM_OFF=1 ...   -> add '-D PKG_KIM=off' to the cmake line above.

cmake --build "$BUILD" -j "$JOBS"
echo
echo "DONE: $BUILD/lmp_mcsites_fork24da74_raven_cpu"
echo ">> mc-sites styles present?"
( unset I_MPI_PMI_LIBRARY; I_MPI_FABRICS=shm "$BUILD/lmp_mcsites_fork24da74_raven_cpu" -h 2>/dev/null \
    | grep -iE 'sites/voronoi|mc/sites' ) \
  || echo "   (could not run on the login node — verify in a job: srun ... -h | grep -iE 'sites/voronoi|mc/sites')"
