#!/bin/bash
# ===========================================================================
# Build LAMMPS (stable branch) on the cmti partition (Sustainable Materials).
# Hardware: 2x Intel Xeon Gold 6230 "Cascade Lake" (2x 20 = 40 cores/node,
# AVX-512). Same MPCDF cluster as cmmg, but the Intel nodes -> partition p.cmfe.
#
# Toolchain: Intel oneAPI (icpx/icx) + Intel MPI + MKL, with the LAMMPS **INTEL**
# package enabled (INTEL_ARCH=cpu) — the "proper" build for an Intel node, mirroring
# build-lammps-raven-cpu.sh. NOTE: `pair_style pace` (the benchmark) has no intel
# variant, so the ACE kernel runs as the standard CPU style; the INTEL package
# accelerates any other styles that support it. This is a fair Xeon CPU data point.
#
# RUN THIS ON A LOGIN NODE (cmti001/002): internet only on login nodes.
# Builds and git-clones into the directory you launch it from (not $HOME).
#
#   ./build-lammps-cmti.sh
# ===========================================================================

# ---------------------------------------------------------------------------
# Module setup FIRST, before `set -euo pipefail` (Lmod `module` can return
# non-zero / touch unset vars and would abort silently under strict mode).
# NO default module versions on MPCDF (pin them). intel/mkl confirmed on cmti
# (Jun 2026); impi is *hierarchical* — it only appears AFTER the intel compiler,
# so confirm with:  module load intel/2025.2 && module avail impi
# ---------------------------------------------------------------------------
INTEL_VER="${INTEL_VER:-2025.2}"   # oneAPI compilers (icpx/icx)
IMPI_VER="${IMPI_VER:-2021.16}"    # Intel MPI under intel/2025.2 — CONFIRM (see above)
MKL_VER="${MKL_VER:-2025.2}"       # BLAS/LAPACK + FFT (sets MKLROOT)
CMAKE_VER="${CMAKE_VER:-3.30}"

# Make the 'module' function available without a LOGIN shell.
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

set -euo pipefail

# Intel MPI wrappers should use the LLVM oneAPI compilers (icpx/icx), not classic.
export I_MPI_CXX=icpx
export I_MPI_CC=icx

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(pwd)"

LOG="${LOG:-$RUN_DIR/build-cmti-$(date +%Y%m%d-%H%M%S).log}"
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
BUILD="${BUILD:-$SRC/build-cmti}"
JOBS="${JOBS:-16}"

# --- locate the REAL Intel MPI wrappers (they live under I_MPI_ROOT, not PATH) ---
: "${I_MPI_ROOT:?I_MPI_ROOT not set — is the impi/${IMPI_VER} module loaded? (impi is hierarchical under intel/${INTEL_VER})}"
MPI_BIN=""
for d in "$I_MPI_ROOT/bin" "$I_MPI_ROOT/intel64/bin"; do
    [ -d "$d" ] && MPI_BIN="$d" && break
done
pick(){
    local n
    for n in "$@"; do
        [ -n "${MPI_BIN:-}" ] && [ -x "$MPI_BIN/$n" ] && { echo "$MPI_BIN/$n"; return 0; }
    done
    return 0
}
# Prefer the LLVM Intel wrappers (mpiicpx/mpiicx); mpiicpc/mpiicc work too via I_MPI_CXX/CC.
MPICXX="${MPICXX:-$(pick mpiicpx mpiicpc mpicxx)}"
MPICC="${MPICC:-$(pick mpiicx mpiicc mpicc)}"

echo ">> module list:"; module list 2>&1 | sed 's/^/   /'
echo ">> I_MPI_ROOT = $I_MPI_ROOT"
echo ">> MPICXX = ${MPICXX:-NOT FOUND}  (I_MPI_CXX=$I_MPI_CXX)"
echo ">> MPICC  = ${MPICC:-NOT FOUND}  (I_MPI_CC=$I_MPI_CC)"
if [ -z "${MPICXX:-}" ] || [ -z "${MPICC:-}" ]; then
    echo "ERROR: no usable Intel MPI wrapper found in $MPI_BIN" >&2
    echo "       Set by hand, e.g.  MPICC=/path/mpiicx MPICXX=/path/mpiicpx $0" >&2
    exit 1
fi
echo ">> icpx version: $(icpx --version 2>/dev/null | head -1 || echo '?')"
echo ">> build dir   : $BUILD"

# --- source: 'stable' tracks the newest stable release incl. its updates -----
if [ ! -d "$SRC/.git" ]; then
    git clone -b stable https://github.com/lammps/lammps.git "$SRC"
fi
cd "$SRC"

if [ -f "$BUILD/CMakeCache.txt" ] && ! grep -q "CMAKE_CXX_COMPILER:.*=${MPICXX}$" "$BUILD/CMakeCache.txt"; then
    echo ">> removing stale build dir $BUILD"
    rm -rf "$BUILD"
fi

cmake -S "$SRC/cmake" -B "$BUILD" \
    -C "$SCRIPT_DIR/cmake/lammps-packages-mpcdf.cmake" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=17 \
    -D BUILD_MPI=on -D BUILD_OMP=on \
    -D LAMMPS_MACHINE=cmti \
    -D CMAKE_C_COMPILER="$MPICC" \
    -D CMAKE_CXX_COMPILER="$MPICXX" \
    -D CMAKE_CXX_FLAGS="-O3 -xCORE-AVX512 -qopt-zmm-usage=high" \
    -D PKG_INTEL=on -D INTEL_ARCH=cpu \
    -D PKG_OPENMP=on -D PKG_OPT=on \
    -D USE_INTERNAL_LINALG=off -D BLA_VENDOR=Intel10_64lp_seq \
    -D FFT=MKL -D FFT_KOKKOS=KISS \
    -D PKG_PLUMED=off \
    -D PKG_KOKKOS=off \
    -D CMAKE_EXE_LINKER_FLAGS="-Wl,-rpath=\$ORIGIN/../lib64"

# Heavy external packages (KIM, VORONOI, MACHDYN, ML-PACE) build under icpx;
# ML-PACE (benchmark) and voro++ are fine. If KIM's Fortran fights the toolchain,
# add  -D PKG_KIM=off  (KIM is unused by the benchmark). PLUMED off here (needs GSL).

cmake --build "$BUILD" -j "$JOBS"
echo
echo "DONE: $BUILD/lmp_cmti"

echo ">> installed packages:"
( unset I_MPI_PMI_LIBRARY; I_MPI_FABRICS=shm "$BUILD/lmp_cmti" -h 2>/dev/null \
    | sed -n '/Installed packages/,/^$/p' | head -25 ) \
  || echo "   (could not run lmp_cmti on the login node — verify in a job: srun ... $BUILD/lmp_cmti -h)"
