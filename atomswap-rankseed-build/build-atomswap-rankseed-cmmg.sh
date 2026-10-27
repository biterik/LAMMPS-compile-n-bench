#!/bin/bash
# ===========================================================================
# Build TWO LAMMPS binaries on cmmg from the SAME source tree and the SAME
# package set as the central module lammps/250722 (22 Jul 2025, Update 3):
#
#   bin/lmp_cmmg_22Jul2025u3_stock              unpatched (A/B reference)
#   bin/lmp_cmmg_22Jul2025u3_atomswap-rankseed  + patches/0001 (fix atom/swap:
#                                                 per-rank random_unequal, per-rank
#                                                 restart state) + patches/0002
#                                                 (fix gcmc: same constructor fix)
#
# Why: src/MC/fix_atom_swap.cpp seeds the "unequal" RNG identically on every
# MPI rank, so in semi-grand mode all ranks walk the same sequence of Metropolis
# numbers and accept swaps in bursts of ~nprocs (Cu-AT-DISLOCS thread 08,
# 2026-10-19). See patches/README.md.
#
# Same toolchain, cmake preset and flags as build-lammps-cmmg.sh (gcc/13 +
# impi/2021.16, full compile-n-bench package set, KOKKOS/OpenMP ZEN4). The
# source is pinned to the git TAG that matches the module, not to `stable`
# (which moved to 30 Sep 2026 on 2026-09-30).
#
# RUN THIS ON A cmmg LOGIN NODE (cmti001/002): the git clone and the package
# downloads (PLUMED, KIM, voro++, Eigen3, PACE) need internet, which compute
# nodes do not have. Clones + builds into the directory you launch it from,
# which must be under /cmmc/ptmp/$USER (never $HOME: quota).
#
#   cd /cmmc/ptmp/$USER/LAMMPS-ATOMSWAP-PATCH
#   /cmmc/ptmp/$USER/LAMMPS-compile-n-bench/atomswap-rankseed-build/build-atomswap-rankseed-cmmg.sh
#
# Idempotent: re-running resets src/MC to stock, rebuilds incrementally, re-applies.
# ===========================================================================

GCC_VER="${GCC_VER:-13}"
IMPI_VER="${IMPI_VER:-2021.16}"
CMAKE_VER="${CMAKE_VER:-3.30}"
MKL_VER="${MKL_VER:-2025.2}"      # BLAS/LAPACK for PLUMED
GSL_VER="${GSL_VER:-2.7}"         # GSL for PLUMED
LAMMPS_TAG="${LAMMPS_TAG:-stable_22Jul2025_update3}"   # == module lammps/250722 ("22 Jul 2025 - Update 3")
VERTAG="${VERTAG:-22Jul2025u3}"

# module function without a login shell (see build-lammps-cmmg.sh for why)
if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh \
              "${MODULESHOME:+$MODULESHOME/init/bash}" \
              /mpcdf/soft/SLE_15/packages/x86_64/Modules/5.4.0/init/bash; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
module load gcc/${GCC_VER}
module load impi/${IMPI_VER}
module load cmake/${CMAKE_VER}
module load mkl/${MKL_VER}
module load gsl/${GSL_VER}

set -euo pipefail

export I_MPI_CXX=g++
export I_MPI_CC=gcc

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(pwd)"
PATCHES_DIR="${PATCHES_DIR:-$SCRIPT_DIR/patches}"
PRESET="${PRESET:-$SCRIPT_DIR/../cmake/lammps-packages-mpcdf.cmake}"
case "$RUN_DIR" in
  "$HOME"|"$HOME"/*) echo "ERROR: launch from a /cmmc/ptmp/\$USER work dir, not \$HOME ($RUN_DIR)" >&2; exit 1;;
esac
[ -r "$PRESET" ] || { echo "ERROR: cmake preset not found: $PRESET" >&2; exit 1; }
[ -d "$PATCHES_DIR" ] || { echo "ERROR: patches dir not found: $PATCHES_DIR" >&2; exit 1; }

LOG="${LOG:-$RUN_DIR/build-atomswap-rankseed-cmmg-$(date +%Y%m%d-%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
echo ">> logging this run to: $LOG"

SRC="${SRC:-$RUN_DIR/lammps}"
BUILD="${BUILD:-$SRC/build-cmmg}"
BIN="${BIN:-$RUN_DIR/bin}"
JOBS="${JOBS:-16}"

# --- conda guard (a pyiron env shadowed mpicxx before) -----------------------
if command -v python >/dev/null && python -c 'import sys; sys.exit(0 if "conda" in sys.prefix or "mambaforge" in sys.prefix or "envs" in sys.prefix else 1)' 2>/dev/null; then
    echo "WARNING: a conda/mambaforge env is active ($(command -v python)). Run 'conda deactivate' and re-run." >&2
fi

# --- Intel MPI GNU wrappers (mpigcc / mpig++ on this oneAPI install) ---------
: "${I_MPI_ROOT:?I_MPI_ROOT not set -- is the impi/${IMPI_VER} module loaded?}"
MPI_BIN=""
for d in "$I_MPI_ROOT/bin" "$I_MPI_ROOT/intel64/bin"; do
    [ -d "$d" ] && MPI_BIN="$d" && break
done
pick(){ local n; for n in "$@"; do [ -n "${MPI_BIN:-}" ] && [ -x "$MPI_BIN/$n" ] && { echo "$MPI_BIN/$n"; return 0; }; done; return 0; }
MPICC="${MPICC:-$(pick mpigcc mpicc mpiicx)}"
MPICXX="${MPICXX:-$(pick mpig++ mpicxx mpigxx mpiicpx)}"
echo ">> MPICC=${MPICC:-NONE}  MPICXX=${MPICXX:-NONE}  g++ $(g++ -dumpfullversion 2>/dev/null || g++ -dumpversion)"
[ -n "${MPICXX:-}" ] && [ -n "${MPICC:-}" ] || { echo "ERROR: MPI wrappers not found in $MPI_BIN" >&2; exit 1; }

# --- source: the exact tag of the module ---------------------------------------
if [ ! -d "$SRC/.git" ]; then
    git clone --branch "$LAMMPS_TAG" --depth 1 https://github.com/lammps/lammps.git "$SRC"
fi
cd "$SRC"
git checkout -q -- src/MC/fix_atom_swap.cpp src/MC/fix_gcmc.cpp     # always start from stock
grep -q 'LAMMPS_VERSION "22 Jul 2025"' src/version.h && grep -q 'Update 3' src/version.h \
  || { echo "ERROR: $SRC is not 22 Jul 2025 Update 3 -- $(cat src/version.h | tr '\n' ' ')" >&2; exit 1; }
echo ">> source: $(git describe --tags --always) -- $(tr '\n' ' ' < src/version.h)"
for p in "$PATCHES_DIR"/000[12]-*.patch; do git apply --check "$p" && echo ">> patch applies cleanly: $(basename "$p")"; done

if [ -f "$BUILD/CMakeCache.txt" ] && ! grep -q "CMAKE_CXX_COMPILER:.*=${MPICXX}$" "$BUILD/CMakeCache.txt"; then
    echo ">> removing stale build dir $BUILD"; rm -rf "$BUILD"
fi

configure() {
cmake -S "$SRC/cmake" -B "$BUILD" \
    -C "$PRESET" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=17 \
    -D BUILD_MPI=on -D BUILD_OMP=on \
    -D LAMMPS_MACHINE=cmmg \
    -D CMAKE_C_COMPILER="$MPICC" \
    -D CMAKE_CXX_COMPILER="$MPICXX" \
    -D CMAKE_CXX_FLAGS="-O3 -march=znver4" \
    -D BLA_VENDOR=Intel10_64lp_seq \
    -D PKG_PLUMED=on -D DOWNLOAD_PLUMED=on -D PLUMED_MODE=static \
    -D PKG_KOKKOS=on \
    -D Kokkos_ENABLE_SERIAL=on \
    -D Kokkos_ENABLE_OPENMP=on \
    -D Kokkos_ARCH_ZEN4=on \
    -D FFT=KISS \
    -D CMAKE_EXE_LINKER_FLAGS="-Wl,-rpath=\$ORIGIN/../lib64"
}

mkdir -p "$BIN"

# --- pass 1: stock ----------------------------------------------------------------
echo ">> ===== pass 1: STOCK build ====="
configure
cmake --build "$BUILD" -j "$JOBS"
cp -f "$BUILD/lmp_cmmg" "$BIN/lmp_cmmg_${VERTAG}_stock"
echo ">> stock binary: $BIN/lmp_cmmg_${VERTAG}_stock"

# --- pass 2: patched ----------------------------------------------------------------
echo ">> ===== pass 2: PATCHED build (fix atom/swap + fix gcmc per-rank RNG) ====="
for p in "$PATCHES_DIR"/000[12]-*.patch; do git apply "$p"; echo ">> applied $(basename "$p")"; done
git diff --stat
cmake --build "$BUILD" -j "$JOBS"          # incremental: the two MC sources + link
cp -f "$BUILD/lmp_cmmg" "$BIN/lmp_cmmg_${VERTAG}_atomswap-rankseed"
echo ">> patched binary: $BIN/lmp_cmmg_${VERTAG}_atomswap-rankseed"

# --- record what was built ----------------------------------------------------------------
{
  echo "built: $(date -Is) on a cmmg login node"     # no user/host: this record gets copied into the public repo
  echo "source: $(git describe --tags --always)  $(tr '\n' ' ' < src/version.h)"
  echo "patches: $(ls "$PATCHES_DIR"/000[12]-*.patch | xargs -n1 basename | tr '\n' ' ')"
  echo "modules: gcc/$GCC_VER impi/$IMPI_VER cmake/$CMAKE_VER mkl/$MKL_VER gsl/$GSL_VER"
  md5sum "$BIN/lmp_cmmg_${VERTAG}_stock" "$BIN/lmp_cmmg_${VERTAG}_atomswap-rankseed"
  git diff > "$BIN/applied-patches.diff"; echo "diff: $BIN/applied-patches.diff ($(wc -l < "$BIN/applied-patches.diff") lines)"
} | tee "$BIN/BUILD-RECORD.txt"
git checkout -q -- src/MC/fix_atom_swap.cpp src/MC/fix_gcmc.cpp      # leave the tree stock again

echo
echo "DONE. Two binaries in $BIN:"
ls -l "$BIN"/lmp_cmmg_*
echo ">> installed packages (login-node singleton; MPI forced to shm):"
( unset I_MPI_PMI_LIBRARY; I_MPI_FABRICS=shm "$BIN/lmp_cmmg_${VERTAG}_atomswap-rankseed" -h 2>/dev/null \
    | sed -n '/Installed packages/,/^$/p' | head -12 ) \
  || echo "   (could not run on the login node -- verify in a job: srun $BIN/lmp_cmmg_${VERTAG}_atomswap-rankseed -h)"
