#!/bin/bash
# ===========================================================================
# Build LAMMPS + MC-SITES on RAVEN-GPU (NVIDIA A100, Ampere CC 8.0).
# Base: thermoatoms/lammps @ 24da74cd + feature/mc-sites patches.
# KOKKOS + CUDA, arch AMPERE80.
#
# Full compile-n-bench package set + MC-SITES (PKG_MC + PKG_ML-PACE + PKG_VORONOI).
# NOTE: fix mc/sites has no Kokkos variant (v1) — its bookkeeping runs host-side
# while MD/pair styles use the GPU; this is expected and fine.
#
# Binary:  lmp_mcsites_fork24da74_raven_gpu
#
# GRACE/TensorFlow OFF by default. For GRACE on the A100 you need a *CUDA*
# TensorFlow ABI-compatible with cuda/12.6:  GRACE_TF=on PYTHON=~/tf-gpu/bin/python
# (see README; CUDA-TF/nvcc skew is possible — fall back to GRACE_TF=off).
#
# RUN THIS ON A RAVEN LOGIN NODE (raven01i..04i). cd into /ptmp/$USER/mcsites first.
#   ./build-mcsites-raven-gpu.sh
# ===========================================================================

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
# Pin CMake 3.30: CMake 4.x's compiler probe feeds nvcc_wrapper a stray arg.
module load gcc/13 cuda/12.6 openmpi_gpu/5.0 cmake/3.30
module load mkl/2025.2            # external LAPACK (internal f2c linalg won't compile under nvcc)
if [ "$GRACE_TF" != "off" ] && [ -n "${PYMODULE:-}" ]; then
    module load "$PYMODULE" && echo ">> loaded Python module for TF discovery: $PYMODULE"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(pwd)"
PATCHES_DIR="${PATCHES_DIR:-$SCRIPT_DIR/patches}"

LOG="${LOG:-$RUN_DIR/build-mcsites-raven-gpu-$(date +%Y%m%d-%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
echo ">> logging this run to: $LOG"

# --- detach from conda (gotcha 13: libgfortran.so.4 leaks into the KIM link) -
_drop_conda() { printf '%s' "${1:-}" | tr ':' '\n' | grep -viE 'conda' | paste -sd: ; true; }
if [ -n "${CONDA_PREFIX:-}" ] || printf '%s' "${PATH}" | grep -qiE 'conda'; then
    echo ">> conda detected — removing it from PATH/LD_LIBRARY_PATH for a clean GNU build"
    export PATH="$(_drop_conda "$PATH")"
    export LD_LIBRARY_PATH="$(_drop_conda "${LD_LIBRARY_PATH:-}")"
    export LIBRARY_PATH="$(_drop_conda "${LIBRARY_PATH:-}")"
    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE CONDA_SHLVL || true
fi
echo ">> g++: $(command -v g++)   gfortran: $(command -v gfortran)"

SRC="${SRC:-$RUN_DIR/lammps-mcsites}"
BUILD="${BUILD:-$SRC/build-mcsites-raven-gpu}"
JOBS="${JOBS:-8}"                 # nvcc is memory-hungry; keep -j modest

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

# --- pre-build voro++ with host g++ (nvcc_wrapper can't compile it) ----------
VORO_VER="${VORO_VER:-0.4.6}"
VORO_DIR="$RUN_DIR/voro++-$VORO_VER"
VORO_LIB="$VORO_DIR/src/libvoro++.a"; VORO_INC="$VORO_DIR/src"
if [ ! -f "$VORO_LIB" ]; then
    echo ">> building voro++ $VORO_VER with g++"
    ( cd "$RUN_DIR"
      if [ ! -d "$VORO_DIR" ]; then
          curl -fL -o "voro++-$VORO_VER.tar.gz" "https://download.lammps.org/thirdparty/voro++-$VORO_VER.tar.gz"
          tar xzf "voro++-$VORO_VER.tar.gz"
          VPATCH="$SRC/cmake/patches/voro-make.patch"
          [ -f "$VPATCH" ] || VPATCH="$SRC/lib/voronoi/voro-make.patch"
          ( cd "$VORO_DIR" && patch -b -p0 < "$VPATCH" )
      fi
      make -C "$VORO_DIR" CXX=g++ CFLAGS="-O3 -fPIC" )
fi
[ -f "$VORO_LIB" ] || { echo "ERROR: voro++ build failed" >&2; exit 1; }

# --- pre-build KIM-API with the GNU toolchain (conda-free) -------------------
KIM_VER="${KIM_VER:-$(grep -oE 'kim-api-[0-9]+\.[0-9]+\.[0-9]+' "$SRC/cmake/Modules/Packages/KIM.cmake" 2>/dev/null | head -1 | sed 's/kim-api-//')}"
KIM_VER="${KIM_VER:-2.4.1}"
KIM_PREFIX="$RUN_DIR/kim-api-$KIM_VER-install"
if [ ! -f "$KIM_PREFIX/lib64/libkim-api.so" ] && [ ! -f "$KIM_PREFIX/lib/libkim-api.so" ]; then
    echo ">> building KIM-API $KIM_VER with gcc/g++/gfortran"
    ( cd "$RUN_DIR"
      if [ ! -d "kim-api-$KIM_VER" ]; then
          curl -fL -o "kim-api-$KIM_VER.txz" "https://s3.openkim.org/kim-api/kim-api-$KIM_VER.txz"
          tar xf "kim-api-$KIM_VER.txz"
      fi
      cmake -S "kim-api-$KIM_VER" -B "kim-api-$KIM_VER/build" \
            -D CMAKE_BUILD_TYPE=Release -D CMAKE_C_COMPILER=gcc \
            -D CMAKE_CXX_COMPILER=g++ -D CMAKE_Fortran_COMPILER=gfortran \
            -D CMAKE_INSTALL_PREFIX="$KIM_PREFIX"
      cmake --build "kim-api-$KIM_VER/build" -j "$JOBS"
      cmake --install "kim-api-$KIM_VER/build" )
fi
{ [ -f "$KIM_PREFIX/lib64/libkim-api.so" ] || [ -f "$KIM_PREFIX/lib/libkim-api.so" ]; } \
    || { echo "ERROR: KIM-API build failed" >&2; exit 1; }
export CMAKE_PREFIX_PATH="$KIM_PREFIX:${CMAKE_PREFIX_PATH:-}"
export PKG_CONFIG_PATH="$KIM_PREFIX/lib64/pkgconfig:$KIM_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

# Device code through Kokkos' nvcc_wrapper; OpenMPI C++ wrapper calls it.
export NVCC_WRAPPER_DEFAULT_COMPILER=g++
export OMPI_CXX="$SRC/lib/kokkos/bin/nvcc_wrapper"

GRACE_FLAGS=()
if [ "$GRACE_TF" = "off" ]; then
    GRACE_FLAGS+=( -D NO_GRACE_TF=ON )
else
    PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"
    [ -n "${PYTHON:-}" ] && GRACE_FLAGS+=( -D PACE_PYTHON_EXEC="$PYTHON" -D Python_EXECUTABLE="$PYTHON" )
    [ -n "${TF_LIB_FILE:-}" ] && GRACE_FLAGS+=( -D TF_LIB_FILE="$TF_LIB_FILE" )
    echo ">> GRACE_TF=on: CUDA TensorFlow via ${PYTHON:-<none>} (watch for CUDA-runtime skew)"
fi

cmake -S "$SRC/cmake" -B "$BUILD" \
    -C "$SCRIPT_DIR/cmake/lammps-packages-mpcdf.cmake" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=17 \
    -D BUILD_MPI=on -D BUILD_OMP=on \
    -D LAMMPS_MACHINE=mcsites_fork24da74_raven_gpu \
    -D CMAKE_CXX_COMPILER=mpicxx \
    -D USE_INTERNAL_LINALG=off -D BLA_VENDOR=Intel10_64lp_seq \
    -D PKG_PLUMED=off \
    -D PKG_MC=on -D PKG_ML-PACE=on \
    -D PKG_VORONOI=on -D DOWNLOAD_VORO=off \
    -D VORO_LIBRARY="$VORO_LIB" -D VORO_INCLUDE_DIR="$VORO_INC" \
    -D PKG_KIM=on -D DOWNLOAD_KIM=off \
    -D PKG_KOKKOS=on \
    -D Kokkos_ENABLE_SERIAL=on \
    -D Kokkos_ENABLE_OPENMP=off \
    -D Kokkos_ENABLE_CUDA=on \
    -D Kokkos_ARCH_AMPERE80=on \
    -D FFT=KISS -D FFT_KOKKOS=CUFFT \
    "${GRACE_FLAGS[@]}" \
    -D CMAKE_EXE_LINKER_FLAGS="-Wl,-rpath=\$ORIGIN/../lib64 -Wl,-rpath=$KIM_PREFIX/lib64 -Wl,-rpath=$KIM_PREFIX/lib"

cmake --build "$BUILD" -j "$JOBS"
echo
echo "DONE: $BUILD/lmp_mcsites_fork24da74_raven_gpu"
echo ">> mc-sites styles present?"
( "$BUILD/lmp_mcsites_fork24da74_raven_gpu" -h 2>/dev/null | grep -iE 'sites/voronoi|mc/sites' ) \
  || echo "   (could not run on the login node — verify in a job: srun ... -h | grep -iE 'sites/voronoi|mc/sites')"
