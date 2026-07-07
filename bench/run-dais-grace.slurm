#!/bin/bash
# Self-contained GRACE single-GPU benchmark on DAIS (H200 / B200 / RTX PRO 6000).
# All paths verified 2026-07-07. Runs a PRE-FLIGHT that aborts in seconds if
# anything is missing, so a queued job never silently skips and wastes the wait.
#
# SUBMIT from your gracework-dais dir (outputs land there):
#   H200:  sbatch --export=GPU=h200       --gres=gpu:h200:1         --job-name=grace_dais_h200 bench/run-dais-grace.slurm
#   B200:  sbatch --export=GPU=b200       --gres=gpu:b200:1         --job-name=grace_dais_b200 bench/run-dais-grace.slurm
#   RTX :  sbatch --export=GPU=rtxpro6000 --gres=gpu:rtx_pro_6000:1 --job-name=grace_dais_rtx  bench/run-dais-grace.slurm
#
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000            # gpu1 is shared -> a memory limit is REQUIRED
#SBATCH --time=00:30:00
#SBATCH --output=grace_dais.%j.out

set -u
GPU="${GPU:-h200}"             # only used to tag log files

# ---- verified, hardcoded paths (confirmed 2026-07-07) --------------------
SUB=/viper/ptmp2/biterik/gracework-dais
LMP="$SUB/lammps-fork/build-dais-fork/lmp_dais_fork"
GC=/viper/ptmp2/biterik/gracework/grace-cache
MODEL_1L="$GC/GRACE-1L-SMAX-OMAT-large"
MODEL_2L="$GC/GRACE-2L-SMAX-OMAT-medium"
NX="${NX:-16}"                # 16^3 = 16,384 atoms, matches the other machines
# Force the correct cache; a stale GRACE_CACHE (/dais/fs/scratch/.cache/grace)
# in the login env otherwise shadows the models and the run skips silently.
export GRACE_CACHE="$GC"

# Resolve the input deck by checking real locations (no blind assumption).
IN=""
for c in "$SUB/in.grace_bench" \
         "$SUB/bench/in.grace_bench" \
         /viper/ptmp2/biterik/gracework/LAMMPS-compile-n-bench/bench/in.grace_bench \
         "${SLURM_SUBMIT_DIR:-$PWD}/in.grace_bench"; do
    if [ -f "$c" ]; then IN="$c"; break; fi
done

# ---- module env ----------------------------------------------------------
if ! command -v module >/dev/null 2>&1; then
    for f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${f:-}" ] && [ -r "$f" ] && . "$f" && break
    done
fi
unset LOADEDMODULES _LMFILES_ 2>/dev/null || true
module purge
module load gcc/14
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"

# ---- library path: gcc libstdc++ (CXXABI) : driver : CUDA wheels : TF ----
GCC_LIB="$(dirname "$(g++ -print-file-name=libstdc++.so.6 2>/dev/null)")"
NVLIBS="$(ls -d "$SUB"/tf-cuda/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)"
TF_PYLIB="$(ls -d "$SUB"/tf-cuda/lib/python*/site-packages/tensorflow 2>/dev/null | head -1)"
# RUNPATH beats LD_LIBRARY_PATH, so PRELOAD gcc/14's libstdc++ to get CXXABI_1.3.15.
export LD_PRELOAD="$GCC_LIB/libstdc++.so.6"
# /usr/lib64 supplies the driver libcuda.so.1 (NOT shipped by the nvidia wheels).
export LD_LIBRARY_PATH="$GCC_LIB:/usr/lib64:${NVLIBS}:${TF_PYLIB}:${LD_LIBRARY_PATH:-}"

# ---- PRE-FLIGHT: verify everything, abort loudly before wasting the run ---
echo "=========================================================="
echo " pre-flight $(date)  host=$(hostname)  GPU=$GPU"
echo "=========================================================="
fail=0
check() { if [ -e "$2" ]; then echo "  OK       $1: $2"; else echo "  MISSING  $1: $2"; fail=1; fi; }
check "binary"    "$LMP"
check "input"     "${IN:-<none found>}"
check "model 1L"  "$MODEL_1L"
check "model 2L"  "$MODEL_2L"
if [ -n "$NVLIBS" ]; then echo "  OK       nvidia libs: $NVLIBS"; else echo "  MISSING  nvidia CUDA wheel libs"; fail=1; fi
if [ -e /usr/lib64/libcuda.so.1 ]; then echo "  OK       driver: /usr/lib64/libcuda.so.1"; else echo "  MISSING  /usr/lib64/libcuda.so.1"; fail=1; fi
echo "  --- nvidia-smi ---"; nvidia-smi -L || fail=1
if [ "$fail" -ne 0 ]; then
    echo ">> PRE-FLIGHT FAILED — aborting before the 30 min allocation is wasted."
    exit 1
fi
echo ">> pre-flight OK — starting benchmark"
echo "=========================================================="

# ---- run DIRECTLY (not srun: srun resets env and drops LD_PRELOAD) --------
"$LMP" -in "$IN" -var pstyle grace -var model "$MODEL_1L" \
       -var nx "$NX" -var ny "$NX" -var nz "$NX" -var nsteps 100 \
       -log "log.grace_1l_dais_${GPU}"

"$LMP" -in "$IN" -var pstyle grace/2layer/chunk -var model "$MODEL_2L" \
       -var nx "$NX" -var ny "$NX" -var nz "$NX" -var nsteps 50 \
       -log "log.grace_2l_dais_${GPU}"
