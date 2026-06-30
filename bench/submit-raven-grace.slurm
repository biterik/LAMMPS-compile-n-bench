#!/bin/bash
# GRACE single-GPU benchmark on Raven (1x NVIDIA A100 40GB).
# On the GPU, GRACE runs through TensorFlow-CUDA (this fork has no grace/fs/kk
# Kokkos path). Single process / single GPU:
#   * 1-layer SMAX-OMAT-large  via pair_style grace            (saved_model)
#   * 2-layer SMAX-OMAT-medium via pair_style grace/2layer/chunk
# TensorFlow uses the GPU automatically. ~16k atoms fits a 2L model in 40 GB.
#SBATCH --job-name=grace_raven
#SBATCH --output=grace_raven.%j.out
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:a100:1
#SBATCH --exclusive
#SBATCH --time=00:15:00

if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi
module purge
module load gcc/13 cuda/12.6 openmpi_gpu/5.0
export OMP_NUM_THREADS=1

# TensorFlow runtime must be findable at run time. Point this at the lib dir of
# the SAME python env whose TF the binary was built against (the CUDA wheel):
#   TF_PYLIB=~/tf-gpu/lib/python3.11/site-packages/tensorflow
[ -n "${TF_PYLIB:-}" ] && export LD_LIBRARY_PATH="$TF_PYLIB:${LD_LIBRARY_PATH:-}"

SUB="${SLURM_SUBMIT_DIR:-$PWD}"
LMP="${LMP:-$SUB/lammps-fork/build-raven-fork/lmp_raven_fork}"

MODEL_1L="${MODEL_1L:-$HOME/.cache/grace/GRACE-1L-SMAX-OMAT-large}"
MODEL_2L="${MODEL_2L:-$HOME/.cache/grace/GRACE-2L-SMAX-OMAT-medium}"

# 1) 1-layer model on the GPU (TensorFlow-CUDA).
if [ -d "$MODEL_1L" ]; then
    srun "$LMP" -in in.grace_bench -var pstyle grace -var model "$MODEL_1L" \
         -log log.grace_1l_raven
else
    echo ">> skip 1L: $MODEL_1L not found"
fi

# 2) 2-layer model on the GPU (heavier; keep nsteps modest for the dev cap).
if [ -d "$MODEL_2L" ]; then
    srun "$LMP" -in in.grace_bench -var pstyle grace/2layer/chunk -var model "$MODEL_2L" \
         -var nsteps 50 -log log.grace_2l_raven
else
    echo ">> skip 2L: $MODEL_2L not found"
fi
