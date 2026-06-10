#!/bin/bash
# ===========================================================================
# Build LAMMPS (stable branch) on RAVEN, GPU build.
# Hardware: NVIDIA A100 40GB (Ampere, CC 8.0)  ->  KOKKOS + CUDA, arch AMPERE80.
#
# NOTE: this is a *GPU* build (CUDA+Kokkos), which is what the single-GPU PACE
# benchmark needs. It is intentionally different from the Intel/oneAPI *CPU*
# build snippet you pasted — that one does not use the A100s.
#
# RUN THIS ON A RAVEN LOGIN NODE (raven01i..04i): internet only on login nodes.
#
#   ./build-lammps-raven.sh
# ===========================================================================
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

# --- Module setup FIRST, before `set -euo pipefail`: the Lmod `module` function
#     can return non-zero / reference unset vars and would abort the script
#     silently under strict mode. ---
module purge
module load gcc/13 cuda/12.6 openmpi_gpu/5.0 cmake
# External LAPACK for the GPU build: the bundled (internal) linalg is f2c C++ and
# cannot be compiled by nvcc_wrapper (nvcc force-includes CUDA's math_functions.h,
# whose `log` clashes with the f2c `log` decl in lib/linalg/dbdsdc.cpp). We link
# MKL instead and turn USE_INTERNAL_LINALG off below. If this exact version isn't
# present, run `module avail mkl` and pin an available one.
module load mkl/2025.2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Build (and git-clone) in the directory you launch this script from, not $HOME.
RUN_DIR="$(pwd)"

# --- Capture EVERYTHING (stdout + stderr) to a timestamped log file while still
#     echoing to the screen. Every warning/error from the module loads, voro++,
#     the cmake configure and the build itself lands in $LOG.
#     Override the path with:  LOG=/ptmp/$USER/raven.log ./build-lammps-raven.sh
LOG="${LOG:-$RUN_DIR/build-raven-$(date +%Y%m%d-%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
echo ">> logging this run to: $LOG"

# --- Detach from any active conda environment (gotcha 13) -------------------
# A child shell inherits conda's PATH / LD_LIBRARY_PATH entries (and CONDA_PREFIX)
# even though the `conda` shell function isn't defined non-interactively. Conda
# ships an old libgfortran (.so.4) that otherwise leaks into the KIM link and
# clashes with gcc/13's libgfortran.so.5. Strip every conda element so the build
# uses ONLY the module GNU toolchain. (Equivalent to `conda deactivate` first.)
_drop_conda() { printf '%s' "${1:-}" | tr ':' '\n' | grep -viE 'conda' | paste -sd: ; true; }
if [ -n "${CONDA_PREFIX:-}" ] || printf '%s' "${PATH}" | grep -qiE 'conda'; then
    echo ">> conda detected — removing it from PATH/LD_LIBRARY_PATH for a clean GNU build"
    export PATH="$(_drop_conda "$PATH")"
    export LD_LIBRARY_PATH="$(_drop_conda "${LD_LIBRARY_PATH:-}")"
    export LIBRARY_PATH="$(_drop_conda "${LIBRARY_PATH:-}")"
    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE CONDA_SHLVL || true
fi
echo ">> g++:      $(command -v g++)"
echo ">> gfortran: $(command -v gfortran)"

SRC="${SRC:-$RUN_DIR/lammps}"
BUILD="${BUILD:-$SRC/build-raven}"
JOBS="${JOBS:-8}"                 # nvcc is memory-hungry; keep -j modest

if [ ! -d "$SRC/.git" ]; then
    git clone -b stable https://github.com/lammps/lammps.git "$SRC"
fi
cd "$SRC"

# --- pre-build voro++ with the host g++ (VORONOI) --------------------------
# nvcc_wrapper cannot compile voro++ (nvcc's frontend chokes on it), and LAMMPS
# would build it with CMAKE_CXX_COMPILER (=mpicxx->nvcc_wrapper). So we build
# voro++ ourselves with plain g++ and hand LAMMPS the prebuilt static library.
VORO_VER="${VORO_VER:-0.4.6}"
VORO_DIR="$RUN_DIR/voro++-$VORO_VER"
VORO_LIB="$VORO_DIR/src/libvoro++.a"
VORO_INC="$VORO_DIR/src"
if [ ! -f "$VORO_LIB" ]; then
    echo ">> building voro++ $VORO_VER with g++ (host compiler, not nvcc)"
    ( cd "$RUN_DIR"
      if [ ! -d "$VORO_DIR" ]; then
          curl -fL -o "voro++-$VORO_VER.tar.gz" \
               "https://download.lammps.org/thirdparty/voro++-$VORO_VER.tar.gz"
          tar xzf "voro++-$VORO_VER.tar.gz"
          ( cd "$VORO_DIR" && patch -b -p0 < "$SRC/lib/voronoi/voro-make.patch" )
      fi
      make -C "$VORO_DIR" CXX=g++ CFLAGS="-O3 -fPIC" )
fi
[ -f "$VORO_LIB" ] || { echo "ERROR: voro++ build failed; $VORO_LIB missing" >&2; exit 1; }
echo ">> voro++ lib: $VORO_LIB"

# --- pre-build KIM-API with the host GNU toolchain (KIM) --------------------
# Under the auto-download path LAMMPS builds kim-api during configure, and its
# Fortran links against whatever gfortran is first on PATH — an active conda env
# pulls in libgfortran.so.4 and clashes with gcc/13's .so.5 (the
# "libgfortran.so.4 ... may conflict with libgfortran.so.5" linker warning).
# We build kim-api ourselves with the module gcc/gfortran (conda already stripped
# above) and hand it to LAMMPS via DOWNLOAD_KIM=off + find_package. Version is
# read from LAMMPS' own KIM.cmake so we stay in lockstep with the release.
# Override with KIM_VER=... if needed.
KIM_VER="${KIM_VER:-$(grep -oE 'kim-api-[0-9]+\.[0-9]+\.[0-9]+' "$SRC/cmake/Modules/Packages/KIM.cmake" 2>/dev/null | head -1 | sed 's/kim-api-//')}"
KIM_VER="${KIM_VER:-2.4.1}"
KIM_PREFIX="$RUN_DIR/kim-api-$KIM_VER-install"
if [ ! -f "$KIM_PREFIX/lib64/libkim-api.so" ] && [ ! -f "$KIM_PREFIX/lib/libkim-api.so" ]; then
    echo ">> building KIM-API $KIM_VER with the GNU toolchain (gcc/g++/gfortran), conda-free"
    ( cd "$RUN_DIR"
      if [ ! -d "kim-api-$KIM_VER" ]; then
          curl -fL -o "kim-api-$KIM_VER.txz" \
               "https://s3.openkim.org/kim-api/kim-api-$KIM_VER.txz"
          tar xf "kim-api-$KIM_VER.txz"
      fi
      cmake -S "kim-api-$KIM_VER" -B "kim-api-$KIM_VER/build" \
            -D CMAKE_BUILD_TYPE=Release \
            -D CMAKE_C_COMPILER=gcc \
            -D CMAKE_CXX_COMPILER=g++ \
            -D CMAKE_Fortran_COMPILER=gfortran \
            -D CMAKE_INSTALL_PREFIX="$KIM_PREFIX"
      cmake --build "kim-api-$KIM_VER/build" -j "$JOBS"
      cmake --install "kim-api-$KIM_VER/build" )
fi
{ [ -f "$KIM_PREFIX/lib64/libkim-api.so" ] || [ -f "$KIM_PREFIX/lib/libkim-api.so" ]; } \
    || { echo "ERROR: KIM-API build failed; libkim-api.so missing under $KIM_PREFIX" >&2; exit 1; }
echo ">> KIM-API prefix: $KIM_PREFIX"
# Make the prebuilt KIM discoverable by LAMMPS' find_package / pkg-config.
export CMAKE_PREFIX_PATH="$KIM_PREFIX:${CMAKE_PREFIX_PATH:-}"
export PKG_CONFIG_PATH="$KIM_PREFIX/lib64/pkgconfig:$KIM_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

# Compile through Kokkos' nvcc_wrapper; make the OpenMPI C++ wrapper call it so
# MPI include/link flags are handled while device code still goes through nvcc.
export NVCC_WRAPPER_DEFAULT_COMPILER=g++
export OMPI_CXX="$SRC/lib/kokkos/bin/nvcc_wrapper"

cmake -S "$SRC/cmake" -B "$BUILD" \
    -C "$SCRIPT_DIR/cmake/lammps-packages-mpcdf.cmake" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=17 \
    -D BUILD_MPI=on -D BUILD_OMP=on \
    -D LAMMPS_MACHINE=raven \
    -D CMAKE_CXX_COMPILER=mpicxx \
    -D USE_INTERNAL_LINALG=off -D BLA_VENDOR=Intel10_64lp_seq \
    -D PKG_PLUMED=off \
    -D PKG_VORONOI=on -D DOWNLOAD_VORO=off \
    -D VORO_LIBRARY="$VORO_LIB" -D VORO_INCLUDE_DIR="$VORO_INC" \
    -D PKG_KIM=on -D DOWNLOAD_KIM=off \
    -D PKG_KOKKOS=on \
    -D Kokkos_ENABLE_SERIAL=on \
    -D Kokkos_ENABLE_OPENMP=off \
    -D Kokkos_ENABLE_CUDA=on \
    -D Kokkos_ARCH_AMPERE80=on \
    -D FFT=KISS -D FFT_KOKKOS=CUFFT \
    -D CMAKE_CXX_FLAGS="-diag-suppress 177,550,611,186,20011" \
    -D CMAKE_EXE_LINKER_FLAGS="-Wl,-rpath=\$ORIGIN/../lib64 -Wl,-rpath=$KIM_PREFIX/lib64 -Wl,-rpath=$KIM_PREFIX/lib"

cmake --build "$BUILD" -j "$JOBS"
echo
echo "DONE: $BUILD/lmp_raven"
# Optional smoke test; never abort the (successful) build if the binary can't
# run on the login node (no GPU / MPI launch differences).
echo ">> installed packages:"
( "$BUILD/lmp_raven" -h 2>/dev/null | sed -n '/Installed packages/,/^$/p' | head -20 ) \
  || echo "   (could not run lmp_raven on the login node — verify in a job: srun ... $BUILD/lmp_raven -h)"
