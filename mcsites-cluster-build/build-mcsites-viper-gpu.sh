#!/bin/bash
# ===========================================================================
# Build LAMMPS + MC-SITES on VIPER-GPU (AMD MI300A APU).
# Base: thermoatoms/lammps @ 24da74cd + feature/mc-sites patches.
# KOKKOS + HIP, arch gfx942 (APU variant).
#
# Full compile-n-bench package set + MC-SITES (PKG_MC + PKG_ML-PACE + PKG_VORONOI).
# ML-UF3 stays OFF on HIP (illegal cross-memory-space ScatterView copy). fix mc/sites
# has no Kokkos variant (v1) — its bookkeeping runs host-side while pair/MD use the GPU.
#
# Binary:  lmp_mcsites_fork24da74_viper_gpu
#
# GRACE/TensorFlow OFF by default. GRACE on the MI300A needs tensorflow-rocm for
# gfx942 (experimental): GRACE_TF=on PYTHON=~/tf-rocm/bin/python. Without it, keep off.
#
# RUN THIS ON A VIPER LOGIN NODE (viper11i/12i/13i). cd into /viper/ptmp/$USER/mcsites first.
#   ./build-mcsites-viper-gpu.sh
# ===========================================================================

FORK_URL="${FORK_URL:-https://github.com/thermoatoms/lammps.git}"
FORK_BRANCH="${FORK_BRANCH:-develop}"
FORK_COMMIT="${FORK_COMMIT:-24da74cd73323f5e7415fdd9a9670b88535464d3}"
MCSITES_BRANCH="${MCSITES_BRANCH:-feature/mc-sites}"
GRACE_TF="${GRACE_TF:-off}"       # TF-ROCm must be set up deliberately; default off
PYMODULE="${PYMODULE:-}"

if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh \
              "${MODULESHOME:+$MODULESHOME/init/bash}" \
              /mpcdf/soft/SLE_15/packages/x86_64/Modules/5.4.0/init/bash; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
module load gcc/14 rocm/6.3 openmpi_gpu/5.0 cmake
if [ "$GRACE_TF" != "off" ] && [ -n "${PYMODULE:-}" ]; then
    module load "$PYMODULE" && echo ">> loaded Python module for TF discovery: $PYMODULE"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(pwd)"
PATCHES_DIR="${PATCHES_DIR:-$SCRIPT_DIR/patches}"

LOG="${LOG:-$RUN_DIR/build-mcsites-viper-gpu-$(date +%Y%m%d-%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
echo ">> logging this run to: $LOG"

SRC="${SRC:-$RUN_DIR/lammps-mcsites}"
BUILD="${BUILD:-$SRC/build-mcsites-viper-gpu}"
JOBS="${JOBS:-8}"                 # hipcc is memory-hungry; don't overdo -j
PYTHON="${PYTHON:-}"              # only set to a tensorflow-rocm python if GRACE_TF=on

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
          VPATCH="$SRC/cmake/patches/voro-make.patch"
          [ -f "$VPATCH" ] || VPATCH="$SRC/lib/voronoi/voro-make.patch"
          ( cd "$VORO_DIR" && patch -b -p0 < "$VPATCH" )
      fi
      make -C "$VORO_DIR" CXX=g++ CFLAGS="-O3 -fPIC" )
fi
[ -f "$VORO_LIB" ] || { echo "ERROR: voro++ build failed" >&2; exit 1; }

# --- hipcc wrapper: drop CMake's c++98 probe std + pin gfx942 ----------------
# Keep any explicit -std (c++17/c++20 that the fork's newer Kokkos uses); only add
# c++17 for the bare project() probe where CMake injected -std=c++98.
REAL_HIPCC="$(command -v hipcc)"
HIPCC_WRAP="$RUN_DIR/hipcc-cxx17"
cat > "$HIPCC_WRAP" <<EOF
#!/bin/bash
args=(); has_std=0
for a in "\$@"; do
  case "\$a" in
    -std=c++98|-std=gnu++98|-std=c++03|-std=gnu++03) ;;
    -std=*) has_std=1; args+=("\$a") ;;
    *) args+=("\$a") ;;
  esac
done
[ "\$has_std" -eq 0 ] && args+=(-std=c++17)
exec "$REAL_HIPCC" --offload-arch=gfx942 "\${args[@]}"
EOF
chmod +x "$HIPCC_WRAP"
echo ">> hipcc wrapper: $HIPCC_WRAP  (real: $REAL_HIPCC)"

if [ -f "$BUILD/CMakeCache.txt" ] && ! grep -q "CMAKE_CXX_COMPILER:.*=${HIPCC_WRAP}\$" "$BUILD/CMakeCache.txt"; then
    echo ">> compiler changed / stale cache — removing $BUILD"; rm -rf "$BUILD"
fi

GRACE_FLAGS=()
if [ "$GRACE_TF" = "off" ]; then
    GRACE_FLAGS+=( -D NO_GRACE_TF=ON )
    echo ">> GRACE_TF=off: pace/kk (GPU ACE) + grace/fs (CPU). GRACE will NOT run on the APU GPU."
else
    [ -n "${PYTHON:-}" ] || { echo "ERROR: GRACE_TF=on needs PYTHON=<tensorflow-rocm python>" >&2; exit 1; }
    GRACE_FLAGS+=( -D PACE_PYTHON_EXEC="$PYTHON" -D Python_EXECUTABLE="$PYTHON" )
    [ -n "${TF_LIB_FILE:-}" ] && GRACE_FLAGS+=( -D TF_LIB_FILE="$TF_LIB_FILE" )
    echo ">> GRACE_TF=on: TensorFlow-ROCm from $PYTHON (experimental on gfx942)"
fi

cmake -S "$SRC/cmake" -B "$BUILD" \
    -C "$SCRIPT_DIR/cmake/lammps-packages-mpcdf.cmake" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=17 \
    -D CMAKE_CXX_STANDARD_REQUIRED=ON \
    -D BUILD_MPI=on -D BUILD_OMP=on \
    -D LAMMPS_MACHINE=mcsites_fork24da74_viper_gpu \
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
echo "DONE: $BUILD/lmp_mcsites_fork24da74_viper_gpu"
echo ">> mc-sites styles present?"
( "$BUILD/lmp_mcsites_fork24da74_viper_gpu" -h 2>/dev/null | grep -iE 'sites/voronoi|mc/sites' ) \
  || echo "   (could not run on the login node — verify in a job: srun ... -h | grep -iE 'sites/voronoi|mc/sites')"
