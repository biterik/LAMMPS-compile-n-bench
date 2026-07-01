#!/bin/bash
# ===========================================================================
# Build LAMMPS from the thermoatoms FORK on VIPER-GPU (AMD MI300A APU).
# KOKKOS + HIP, arch gfx942 (APU variant). Binary: lmp_viper_fork.
#
# Roles:
#   * ACE / PACE on the GPU   via pair_style pace/kk        (Kokkos/HIP)  -> works
#   * GRACE on the GPU        via the TensorFlow styles + TensorFlow-ROCm -> BEST-EFFORT
#
# ## IMPORTANT — GRACE on the MI300A is experimental ##
# This fork has NO grace/fs/kk (Kokkos FS) style, so the ONLY way to run GRACE on
# this GPU is the TensorFlow path with a *ROCm* TensorFlow build for gfx942. That
# is not guaranteed to exist/work for this arch. You must supply tensorflow-rocm
# via a Python env and pass it explicitly:
#     python -m venv ~/tf-rocm && ~/tf-rocm/bin/pip install tensorflow-rocm
#     PYTHON=~/tf-rocm/bin/python ./build-lammps-viper-fork.sh
# (or TF_LIB_FILE=/path/libtensorflow_cc.so.2 from a ROCm TF install).
# If you do NOT have tensorflow-rocm, build with GRACE_TF=off — you then get
# pace/kk (GPU ACE) + grace/fs (CPU only); GRACE will not run on the APU GPU.
# Do NOT let the fork cmake auto-download the default (CUDA) TensorFlow here.
#
# RUN THIS ON A VIPER LOGIN NODE (viper11i/12i/13i): internet only on login nodes.
#   PYTHON=~/tf-rocm/bin/python ./build-lammps-viper-fork.sh
# ===========================================================================

FORK_URL="${FORK_URL:-https://github.com/thermoatoms/lammps.git}"
FORK_BRANCH="${FORK_BRANCH:-develop}"
FORK_COMMIT="${FORK_COMMIT:-24da74cd73323f5e7415fdd9a9670b88535464d3}"
# Default OFF here: TF-ROCm must be set up deliberately. Set GRACE_TF=on + PYTHON=.
GRACE_TF="${GRACE_TF:-off}"

if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh \
              "${MODULESHOME:+$MODULESHOME/init/bash}" \
              /mpcdf/soft/SLE_15/packages/x86_64/Modules/5.4.0/init/bash; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
module load gcc/14 rocm/6.3 openmpi_gpu/5.0 cmake

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(pwd)"

LOG="${LOG:-$RUN_DIR/build-viper-fork-$(date +%Y%m%d-%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
echo ">> logging this run to: $LOG"

SRC="${SRC:-$RUN_DIR/lammps-fork}"
BUILD="${BUILD:-$SRC/build-viper-fork}"
JOBS="${JOBS:-8}"
PYTHON="${PYTHON:-}"             # only set this to a tensorflow-rocm python if GRACE_TF=on

if [ ! -d "$SRC/.git" ]; then
    git clone -b "$FORK_BRANCH" "$FORK_URL" "$SRC"
fi
cd "$SRC"
git fetch --all -q || true
git checkout -q "$FORK_COMMIT" || { echo "ERROR: cannot checkout $FORK_COMMIT" >&2; exit 1; }
echo ">> fork at: $(git log -1 --format='%H %s')   GRACE_TF=$GRACE_TF"

ROCM="${ROCM_PATH:-${ROCM_HOME:-/opt/rocm}}"

# --- pre-build voro++ with host g++ (hipcc can't compile it) -----------------
VORO_VER="${VORO_VER:-0.4.6}"
VORO_DIR="$RUN_DIR/voro++-$VORO_VER"
VORO_LIB="$VORO_DIR/src/libvoro++.a"; VORO_INC="$VORO_DIR/src"
if [ ! -f "$VORO_LIB" ]; then
    echo ">> building voro++ $VORO_VER with g++"
    ( cd "$RUN_DIR"
      if [ ! -d "$VORO_DIR" ]; then
          curl -fL -o "voro++-$VORO_VER.tar.gz" "https://download.lammps.org/thirdparty/voro++-$VORO_VER.tar.gz"
          tar xzf "voro++-$VORO_VER.tar.gz"
          # patch moved cmake/patches/ (newer LAMMPS) <- lib/voronoi/ (older); try both.
          VPATCH="$SRC/cmake/patches/voro-make.patch"
          [ -f "$VPATCH" ] || VPATCH="$SRC/lib/voronoi/voro-make.patch"
          ( cd "$VORO_DIR" && patch -b -p0 < "$VPATCH" )
      fi
      make -C "$VORO_DIR" CXX=g++ CFLAGS="-O3 -fPIC" )
fi
[ -f "$VORO_LIB" ] || { echo "ERROR: voro++ build failed" >&2; exit 1; }

# --- hipcc wrapper: force C++17 + gfx942 on every hipcc invocation -----------
REAL_HIPCC="$(command -v hipcc)"
HIPCC_WRAP="$RUN_DIR/hipcc-cxx17"
cat > "$HIPCC_WRAP" <<EOF
#!/bin/bash
args=()
for a in "\$@"; do
  case "\$a" in
    -std=c++98|-std=gnu++98|-std=c++03|-std=gnu++03) ;;
    *) args+=("\$a") ;;
  esac
done
exec "$REAL_HIPCC" --offload-arch=gfx942 "\${args[@]}" -std=c++17
EOF
chmod +x "$HIPCC_WRAP"
echo ">> hipcc wrapper: $HIPCC_WRAP  (real: $REAL_HIPCC)"

if [ -f "$BUILD/CMakeCache.txt" ] && ! grep -q "CMAKE_CXX_COMPILER:.*=${HIPCC_WRAP}\$" "$BUILD/CMakeCache.txt"; then
    echo ">> compiler changed / stale cache — removing $BUILD"; rm -rf "$BUILD"
fi

GRACE_FLAGS=()
if [ "$GRACE_TF" = "off" ]; then
    GRACE_FLAGS+=( -D NO_GRACE_TF=ON )                 # FS + ACE only; no GRACE on the GPU
    echo ">> GRACE_TF=off: building pace/kk + grace/fs (CPU). GRACE will NOT run on the APU GPU."
else
    [ -n "${PYTHON:-}" ] || { echo "ERROR: GRACE_TF=on needs PYTHON=<tensorflow-rocm python>" >&2; exit 1; }
    GRACE_FLAGS+=( -D PACE_PYTHON_EXEC="$PYTHON" -D Python_EXECUTABLE="$PYTHON" )
    [ -n "${TF_LIB_FILE:-}" ] && GRACE_FLAGS+=( -D TF_LIB_FILE="$TF_LIB_FILE" )
    echo ">> GRACE_TF=on: using TensorFlow-ROCm from $PYTHON (experimental on gfx942)"
fi

cmake -S "$SRC/cmake" -B "$BUILD" \
    -C "$SCRIPT_DIR/cmake/lammps-packages-mpcdf.cmake" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=17 \
    -D CMAKE_CXX_STANDARD_REQUIRED=ON \
    -D BUILD_MPI=on -D BUILD_OMP=on \
    -D LAMMPS_MACHINE=viper_fork \
    -D MPI_CXX_SKIP_MPICXX=on \
    -D MPI_CXX_COMPILER=mpic++ \
    -D CMAKE_CXX_COMPILER="$HIPCC_WRAP" \
    -D CMAKE_CXX_FLAGS="-munsafe-fp-atomics" \
    -D PKG_PLUMED=off \
    -D PKG_ML-UF3=off \
    -D PKG_MC=on -D PKG_ML-PACE=on \
    -D PKG_VORONOI=on -D DOWNLOAD_VORO=off \
    -D VORO_LIBRARY="$VORO_LIB" -D VORO_INCLUDE_DIR="$VORO_INC" \
    -D PKG_KOKKOS=on \
    -D Kokkos_ENABLE_SERIAL=on \
    -D Kokkos_ENABLE_OPENMP=off \
    -D Kokkos_ENABLE_HIP=on \
    -D Kokkos_ARCH_AMD_GFX942_APU=on \
    -D Kokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS=on \
    -D FFT=KISS -D FFT_KOKKOS=HIPFFT \
    "${GRACE_FLAGS[@]}" \
    -D CMAKE_EXE_LINKER_FLAGS="-Wl,-rpath=\$ORIGIN/../lib64 -Wl,-rpath=$ROCM/lib/llvm/lib"

cmake --build "$BUILD" -j "$JOBS"
echo
echo "DONE: $BUILD/lmp_viper_fork"
echo ">> styles (grace/pace/atom_swap):"
( "$BUILD/lmp_viper_fork" -h 2>/dev/null | grep -iE 'grace|pace|atom/swap' | head -20 ) \
  || echo "   (could not run on the login node — verify in a job)"
