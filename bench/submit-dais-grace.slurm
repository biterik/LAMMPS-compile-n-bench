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
export TF_CPP_MIN_LOG_LEVEL=3     # quiet TensorFlow's C++ log flood

SUB="${SLURM_SUBMIT_DIR:-$PWD}"
LMP="${LMP:-$SUB/lammps-fork/build-dais-fork/lmp_dais_fork}"
# Models are shared with Viper — default to the existing gracework/grace-cache.
GRACE_CACHE="${GRACE_CACHE:-/viper/ptmp2/$USER/gracework/grace-cache}"
# TF-CUDA runtime: libtensorflow_cc.so.2 + the bundled nvidia-cu12 CUDA/cuDNN libs.
TF_PYLIB="${TF_PYLIB:-$(ls -d "$SUB"/tf-cuda/lib/python*/site-packages/tensorflow 2>/dev/null | head -1)}"
NVLIBS="$(ls -d "$SUB"/tf-cuda/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)"
export LD_LIBRARY_PATH="${TF_PYLIB:-}:${NVLIBS:-}:${LD_LIBRARY_PATH:-}"

# System size: default 16^3 = 16,384 atoms to compare directly with the other
# machines. These GPUs have 96-180 GB HBM, so bump it for a showcase run, e.g.
#   sbatch ... --export=GPU=h200,NX=40   (40^3 = 256,000 atoms)
NX="${NX:-16}"

MODEL_1L="${MODEL_1L:-$GRACE_CACHE/GRACE-1L-SMAX-OMAT-large}"
MODEL_2L="${MODEL_2L:-$GRACE_CACHE/GRACE-2L-SMAX-OMAT-medium}"

if [ -d "$MODEL_1L" ]; then
    srun "$LMP" -in in.grace_bench -var pstyle grace -var model "$MODEL_1L" \
         -var nx "$NX" -var ny "$NX" -var nz "$NX" -var nsteps 100 \
         -log "log.grace_1l_dais_${GPU}"
else
    echo ">> skip 1L: $MODEL_1L not found"
fi

if [ -d "$MODEL_2L" ]; then
    srun "$LMP" -in in.grace_bench -var pstyle grace/2layer/chunk -var model "$MODEL_2L" \
         -var nx "$NX" -var ny "$NX" -var nz "$NX" -var nsteps 50 \
         -log "log.grace_2l_dais_${GPU}"
else
    echo ">> skip 2L: $MODEL_2L not found"
fi
