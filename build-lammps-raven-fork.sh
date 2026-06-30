#!/bin/bash
# ===========================================================================
# Build LAMMPS from the thermoatoms FORK on RAVEN (GPU build).
# Hardware: NVIDIA A100 40GB (Ampere, CC 8.0) -> KOKKOS + CUDA, arch AMPERE80.
#
# This binary serves TWO benchmark roles:
#   * ACE / PACE on the GPU      via pair_style pace/kk   (Kokkos/CUDA)
#   * GRACE on the GPU           via pair_style grace / grace/2layer/chunk
#                                (TensorFlow-CUDA; TF does the GPU work itself)
#
# So we build WITH TensorFlow (GRACE_TF=on, default). TF must be the *CUDA*
# build and ABI-compatible with the loaded CUDA. Provide it from a Python env:
#     python -m venv ~/tf-gpu && ~/tf-gpu/bin/pip install tensorflow   # CUDA wheel
#     PYTHON=~/tf-gpu/bin/python ./build-lammps-raven-fork.sh
# or set TF_LIB_FILE=/path/libtensorflow_cc.so.2. See GRACE.md.
#
# RISK: linking a CUDA libtensorflow into an nvcc_wrapper/Kokkos-CUDA binary can
# hit CUDA-runtime version skew (TF's bundled CUDA vs module cuda/12.6). If the
# combined build/link fails, build with GRACE_TF=off (pace/kk + grace/fs only)
# and run the GRACE-TF GPU benchmark from a SEPARATE plain-g++ + TF-CUDA build
# (see GRACE.md "raven GRACE-TF fallback").
#
# RUN THIS ON A RAVEN LOGIN NODE (raven01i..04i): internet only on login nodes.
#   ./build-lammps-raven-fork.sh
# ===========================================================================

FORK_URL="${FORK_URL:-https://github.com/thermoatoms/lammps.git}"
FORK_BRANCH="${FORK_BRANCH:-develop}"
FORK_COMMIT="${FORK_COMMIT:-24da74cd73323f5e7415fdd9a9670b88535464d3}"
GRACE_TF="${GRACE_TF:-on}"

if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh \
              "${MODULESHOME:+$MODULESHOME/init/bash}" \
              /mpcdf/soft/SLE_15/packages/x86_64/Modules/5.4.0/init/bash; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
module load gcc/13 cuda/12.6 openmpi_gpu/5.0 cmake
module load mkl/2025.2            # external LAPACK (internal f2c linalg won't compile under nvcc)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(pwd)"

LOG="${LOG:-$RUN_DIR/build-raven-fork-$(date +%Y%m%d-%H%M%S).log}"
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
# NOTE: if GRACE_TF=on, your TF Python env is the one exception — pass it
# explicitly via PYTHON= (above) so it is used ONLY for TF discovery, not the
# build toolchain.
echo ">> g++: $(command -v g++)   gfortran: $(command -v gfortran)"

SRC="${SRC:-$RUN_DIR/lammps-fork}"
BUILD="${BUILD:-$SRC/build-raven-fork}"
JOBS="${JOBS:-8}"
PYTHON="${PYTHON:-$(command -v python3 || command -v python || true)}"

if [ ! -d "$SRC/.git" ]; then
    git clone -b "$FORK_BRANCH" "$FORK_URL" "$SRC"
fi
cd "$SRC"
git fetch --all -q || true
git checkout -q "$FORK_COMMIT" || { echo "ERROR: cannot checkout $FORK_COMMIT" >&2; exit 1; }
echo ">> fork at: $(git log -1 --format='%H %s')"

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
          ( cd "$VORO_DIR" && patch -b -p0 < "$SRC/lib/voronoi/voro-make.patch" )
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
    [ -n "${PYTHON:-}" ] && GRACE_FLAGS+=( -D PACE_PYTHON_EXEC="$PYTHON" -D Python_EXECUTABLE="$PYTHON" )
    [ -n "${TF_LIB_FILE:-}" ] && GRACE_FLAGS+=( -D TF_LIB_FILE="$TF_LIB_FILE" )
fi

cmake -S "$SRC/cmake" -B "$BUILD" \
    -C "$SCRIPT_DIR/cmake/lammps-packages-mpcdf.cmake" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=17 \
    -D BUILD_MPI=on -D BUILD_OMP=on \
    -D LAMMPS_MACHINE=raven_fork \
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
    -D CMAKE_CXX_FLAGS="-diag-suppress 177,550,611,186,20011" \
    -D CMAKE_EXE_LINKER_FLAGS="-Wl,-rpath=\$ORIGIN/../lib64 -Wl,-rpath=$KIM_PREFIX/lib64 -Wl,-rpath=$KIM_PREFIX/lib"

cmake --build "$BUILD" -j "$JOBS"
echo
echo "DONE: $BUILD/lmp_raven_fork"
echo ">> styles (grace/pace/atom_swap):"
( "$BUILD/lmp_raven_fork" -h 2>/dev/null | grep -iE 'grace|pace|atom/swap' | head -20 ) \
  || echo "   (could not run on the login node — verify in a job)"
