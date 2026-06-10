#!/bin/bash
# ===========================================================================
# Build LAMMPS (stable branch) on RAVEN, CPU build (no GPU).
# Hardware: 2x Intel Xeon IceLake-SP Platinum 8360Y (72 physical cores/node,
#           AVX-512). This is the *CPU* counterpart to build-lammps-raven.sh
#           (which targets the A100 GPUs).
#
# Toolchain: Intel oneAPI (icpx/icx) + Intel MPI + MKL, with the LAMMPS **INTEL**
# package enabled (INTEL_ARCH=cpu). INTEL gives AVX-512-optimized variants of many
# styles; note that `pair_style pace` (our benchmark) has NO intel variant, so the
# ACE kernel runs as the standard CPU style — this build is a fair Intel-Xeon CPU
# data point, and the INTEL package accelerates anything that *does* support it.
#
# RUN THIS ON A RAVEN LOGIN NODE (raven01i..04i): internet only on login nodes.
# Builds and git-clones into the directory you launch it from (not $HOME).
#
#   ./build-lammps-raven-cpu.sh
# ===========================================================================

# ---------------------------------------------------------------------------
# Module setup FIRST, before `set -euo pipefail` (the Lmod `module` function can
# return non-zero / touch unset vars and would abort silently under strict mode).
# NO default module versions on MPCDF (gotcha 2): pin them. The trio below is
# confirmed available on Raven (Jun 2026). NOTE: impi is *hierarchical* — it only
# appears under the Intel compiler, so `module load intel/<ver>` must precede
# `module avail impi`. impi/2021.17 lives under intel/2025.3.
# ---------------------------------------------------------------------------
INTEL_VER="${INTEL_VER:-2025.3}"   # oneAPI compilers (icpx/icx)
IMPI_VER="${IMPI_VER:-2021.17}"    # Intel MPI (under intel/2025.3)
MKL_VER="${MKL_VER:-2025.3}"       # BLAS/LAPACK + FFT (sets MKLROOT)
CMAKE_VER="${CMAKE_VER:-3.30}"

# Make the 'module' function available without a LOGIN shell. (We deliberately do
# NOT use '#!/bin/bash -l': the login startup files exit for non-interactive
# shells, which made '-l' scripts terminate with no output.)
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

# Make the Intel MPI wrappers use the LLVM-based oneAPI compilers (icpx/icx), not
# the deprecated classic icpc/icc.
export I_MPI_CXX=icpx
export I_MPI_CC=icx

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(pwd)"

# --- Capture EVERYTHING (stdout + stderr) to a timestamped log file ----------
#     Override the path with:  LOG=/ptmp/$USER/raven-cpu.log ./build-lammps-raven-cpu.sh
LOG="${LOG:-$RUN_DIR/build-raven-cpu-$(date +%Y%m%d-%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
echo ">> logging this run to: $LOG"

# --- Detach from any active conda environment (gotcha 13) -------------------
# A child shell inherits conda's PATH/LD_LIBRARY_PATH (and an old libgfortran)
# even though `conda deactivate` isn't available non-interactively. Strip it so
# the build uses ONLY the Intel/GNU module toolchain.
_drop_conda() { printf '%s' "${1:-}" | tr ':' '\n' | grep -viE 'conda' | paste -sd: ; true; }
if [ -n "${CONDA_PREFIX:-}" ] || printf '%s' "${PATH}" | grep -qiE 'conda'; then
    echo ">> conda detected — removing it from PATH/LD_LIBRARY_PATH for a clean build"
    export PATH="$(_drop_conda "$PATH")"
    export LD_LIBRARY_PATH="$(_drop_conda "${LD_LIBRARY_PATH:-}")"
    export LIBRARY_PATH="$(_drop_conda "${LIBRARY_PATH:-}")"
    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE CONDA_SHLVL || true
fi

SRC="${SRC:-$RUN_DIR/lammps}"
BUILD="${BUILD:-$SRC/build-raven-cpu}"
JOBS="${JOBS:-16}"

# --- locate the REAL Intel MPI wrappers (they live under I_MPI_ROOT, not PATH) ---
: "${I_MPI_ROOT:?I_MPI_ROOT not set — is the impi/${IMPI_VER} module loaded?}"
MPI_BIN=""
for d in "$I_MPI_ROOT/bin" "$I_MPI_ROOT/intel64/bin"; do
    [ -d "$d" ] && MPI_BIN="$d" && break
done
# 'pick' must NEVER return non-zero (a non-zero command substitution in an
# assignment aborts the whole script silently under `set -e`).
pick(){
    local n
    for n in "$@"; do
        [ -n "${MPI_BIN:-}" ] && [ -x "$MPI_BIN/$n" ] && { echo "$MPI_BIN/$n"; return 0; }
    done
    return 0
}
# Prefer the LLVM Intel wrappers (mpiicpx/mpiicx). mpiicpc/mpiicc still work
# because I_MPI_CXX/I_MPI_CC force icpx/icx underneath.
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

# A stale cache from an earlier configure pins the wrong compiler and makes cmake
# no-op — start clean if the compiler doesn't match.
if [ -f "$BUILD/CMakeCache.txt" ] && ! grep -q "CMAKE_CXX_COMPILER:.*=${MPICXX}$" "$BUILD/CMakeCache.txt"; then
    echo ">> removing stale build dir $BUILD"
    rm -rf "$BUILD"
fi

cmake -S "$SRC/cmake" -B "$BUILD" \
    -C "$SCRIPT_DIR/cmake/lammps-packages-mpcdf.cmake" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_STANDARD=17 \
    -D BUILD_MPI=on -D BUILD_OMP=on \
    -D LAMMPS_MACHINE=raven_cpu \
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

# NOTE on heavy external packages (KIM, VORONOI, MACHDYN, ML-PACE): they download
# + build at configure time and are compiled by icpx. ML-PACE (the benchmark) and
# voro++ build fine; if KIM's Fortran step fights the oneAPI toolchain, just add
#   -D PKG_KIM=off
# above (KIM is unused by the PACE benchmark). PLUMED is off here (needs GSL).

cmake --build "$BUILD" -j "$JOBS"
echo
echo "DONE: $BUILD/lmp_raven_cpu"

# Smoke test: list installed packages. The binary is Intel-MPI-linked, so a bare
# run on the login node hits SLURM's PMI and aborts in MPI_Init. Force a true
# singleton init, and never let this optional check abort the successful build.
echo ">> installed packages:"
( unset I_MPI_PMI_LIBRARY; I_MPI_FABRICS=shm "$BUILD/lmp_raven_cpu" -h 2>/dev/null \
    | sed -n '/Installed packages/,/^$/p' | head -25 ) \
  || echo "   (could not run lmp_raven_cpu on the login node — verify in a job: srun ... $BUILD/lmp_raven_cpu -h)"
