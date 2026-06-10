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

# Compile through Kokkos' nvcc_wrapper; make the OpenMPI C++ wrapper call it so
# MPI include/link flags are handled while device code still goes through nvcc.
export NVCC_WRAPPER_DEFAULT_COMPILER=g++
export OMPI_CXX="$SRC/lib/kokkos/bin/nvcc_wrapper"

cmake -S "$SRC/cmake" -B "$BUILD" \
    -C "$SCRIPT_DIR/cmake/lammps-packages-mpcdf.cmake" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=17 \
    -D BUILD_MPI=on -D BUILD_OMP=on \
    -D CMAKE_CXX_COMPILER=mpicxx \
    -D PKG_PLUMED=off \
    -D PKG_VORONOI=on -D DOWNLOAD_VORO=off \
    -D VORO_LIBRARY="$VORO_LIB" -D VORO_INCLUDE_DIR="$VORO_INC" \
    -D PKG_KOKKOS=on \
    -D Kokkos_ENABLE_SERIAL=on \
    -D Kokkos_ENABLE_OPENMP=off \
    -D Kokkos_ENABLE_CUDA=on \
    -D Kokkos_ARCH_AMPERE80=on \
    -D FFT=KISS -D FFT_KOKKOS=CUFFT \
    -D CMAKE_EXE_LINKER_FLAGS="-Wl,-rpath=\$ORIGIN/../lib64"

cmake --build "$BUILD" -j "$JOBS"
echo
echo "DONE: $BUILD/lmp"
# Optional smoke test; never abort the (successful) build if the binary can't
# run on the login node (no GPU / MPI launch differences).
echo ">> installed packages:"
( "$BUILD/lmp" -h 2>/dev/null | sed -n '/Installed packages/,/^$/p' | head -20 ) \
  || echo "   (could not run lmp on the login node — verify in a job: srun ... $BUILD/lmp -h)"
