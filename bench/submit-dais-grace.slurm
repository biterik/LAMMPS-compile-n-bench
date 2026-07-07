#!/bin/bash
# GRACE single-GPU benchmark on DAIS (NVIDIA H200 / B200 / RTX PRO 6000) via
# TensorFlow-CUDA. ONE binary (lmp_dais_fork, serial) runs on all three GPU
# types; select the GPU with sbatch --gres and tag the logs with GPU=<name>.
#
# SUBMIT (from your DAIS gracework-dais dir; --export=<vars> keeps the module
# env clean — see the viper notes on why --export=ALL breaks module loads):
#
#   H200:  sbatch --export=GPU=h200        --gres=gpu:h200:1        --job-name=grace_dais_h200 bench/submit-dais-grace.slurm
#   B200:  sbatch --export=GPU=b200        --gres=gpu:b200:1        --job-name=grace_dais_b200 bench/submit-dais-grace.slurm
#   RTX :  sbatch --export=GPU=rtxpro6000  --gres=gpu:rtx_pro_6000:1 --job-name=grace_dais_rtx bench/submit-dais-grace.slurm
#
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000            # DAIS gpu1 is shared -> a memory limit is required (<=375000 MB/GPU)
#SBATCH --time=00:30:00
#SBATCH --output=grace_dais.%j.out

GPU="${GPU:-h200}"               # only used to name the log files

if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi
unset LOADEDMODULES _LMFILES_ 2>/dev/null || true
module purge
module load gcc/14                # libstdc++ for the binary; TF brings its own CUDA
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
# Default to 1 so a failed GPU dlopen is visible in the log; override with
# TF_CPP_MIN_LOG_LEVEL=3 once the run is confirmed working, to quiet the flood.
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"

SUB="${SLURM_SUBMIT_DIR:-$PWD}"
LMP="${LMP:-$SUB/lammps-fork/build-dais-fork/lmp_dais_fork}"
# Models are shared with Viper — default to the existing gracework/grace-cache.
GRACE_CACHE="${GRACE_CACHE:-/viper/ptmp2/$USER/gracework/grace-cache}"
# TF-CUDA runtime: libtensorflow_cc.so.2 + the bundled nvidia-cu12 CUDA/cuDNN libs.
TF_PYLIB="${TF_PYLIB:-$(ls -d "$SUB"/tf-cuda/lib/python*/site-packages/tensorflow 2>/dev/null | head -1)}"
NVLIBS="$(ls -d "$SUB"/tf-cuda/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)"
# gcc/14's libstdc++ MUST win: the binary needs its newer CXXABI_1.3.15, but
# TensorFlow (built in a python-waterboa venv) pulls waterboa's older libstdc++
# via its RUNPATH. RUNPATH beats LD_LIBRARY_PATH, so prepending the dir is NOT
# enough -- we must LD_PRELOAD gcc/14's libstdc++ to override it.
GCC_LIB="$(dirname "$(g++ -print-file-name=libstdc++.so.6 2>/dev/null)")"
export LD_PRELOAD="$GCC_LIB/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
# Path order: gcc lib (CXXABI) : /usr/lib64 (driver libcuda.so.1 -- REQUIRED, the
# nvidia-cu12 wheels do NOT ship the driver) : nvidia-cu12 CUDA/cuDNN wheel libs :
# venv TF framework. Without /usr/lib64 TF prints "Could not find cuda drivers"
# and silently falls back to CPU (platform Host).
export LD_LIBRARY_PATH="$GCC_LIB:/usr/lib64:${NVLIBS:-}:${TF_PYLIB:-}:${LD_LIBRARY_PATH:-}"

# System size: default 16^3 = 16,384 atoms to compare directly with the other
# machines. These GPUs have 96-180 GB HBM, so bump it for a showcase run, e.g.
#   sbatch ... --export=GPU=h200,NX=40   (40^3 = 256,000 atoms)
NX="${NX:-16}"

MODEL_1L="${MODEL_1L:-$GRACE_CACHE/GRACE-1L-SMAX-OMAT-large}"
MODEL_2L="${MODEL_2L:-$GRACE_CACHE/GRACE-2L-SMAX-OMAT-medium}"

# Run the binary DIRECTLY, not via srun: srun re-execs through Slurm and resets
# the environment, dropping LD_PRELOAD (so the CXXABI override is lost). The
# batch script already runs on the allocated GPU node, and CUDA_VISIBLE_DEVICES
# from the gres request is inherited, so a serial single-rank exec is correct.
if [ -d "$MODEL_1L" ]; then
    "$LMP" -in in.grace_bench -var pstyle grace -var model "$MODEL_1L" \
         -var nx "$NX" -var ny "$NX" -var nz "$NX" -var nsteps 100 \
         -log "log.grace_1l_dais_${GPU}"
else
    echo ">> skip 1L: $MODEL_1L not found"
fi

if [ -d "$MODEL_2L" ]; then
    "$LMP" -in in.grace_bench -var pstyle grace/2layer/chunk -var model "$MODEL_2L" \
         -var nx "$NX" -var ny "$NX" -var nz "$NX" -var nsteps 50 \
         -log "log.grace_2l_dais_${GPU}"
else
    echo ">> skip 2L: $MODEL_2L not found"
fi
