#!/bin/bash
# GRACE single-GPU benchmark on Viper-GPU (1x AMD MI300A APU) — EXPERIMENTAL.
# This fork has no grace/fs/kk, so GRACE on this GPU is only possible through
# TensorFlow with a *ROCm* TensorFlow build (tensorflow-rocm) for gfx942, which
# is not guaranteed to work. Build with build-lammps-viper-fork.sh GRACE_TF=on
# and a tensorflow-rocm python first. If GRACE does not run on the APU, this is
# a known limitation — fall back to the CPU GRACE numbers (cmmg / viper-cpu).
#SBATCH --job-name=grace_viper
#SBATCH --output=grace_viper.%j.out
#SBATCH --partition=apudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=00:15:00

if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi
module purge
module load gcc/14 rocm/6.3 openmpi_gpu/5.0
export OMP_NUM_THREADS=1
export HSA_XNACK=1               # MI300A APU unified memory (as for the PACE run)

# tensorflow-rocm runtime must be findable at run time (same env used to build):
#   TF_PYLIB=~/tf-rocm/lib/python3.11/site-packages/tensorflow
[ -n "${TF_PYLIB:-}" ] && export LD_LIBRARY_PATH="$TF_PYLIB:${LD_LIBRARY_PATH:-}"

SUB="${SLURM_SUBMIT_DIR:-$PWD}"
LMP="${LMP:-$SUB/lammps-fork/build-viper-fork/lmp_viper_fork}"

MODEL_1L="${MODEL_1L:-$HOME/.cache/grace/GRACE-1L-SMAX-OMAT-large}"
MODEL_2L="${MODEL_2L:-$HOME/.cache/grace/GRACE-2L-SMAX-OMAT-medium}"

if [ -d "$MODEL_1L" ]; then
    srun "$LMP" -in in.grace_bench -var pstyle grace -var model "$MODEL_1L" \
         -log log.grace_1l_viper
else
    echo ">> skip 1L: $MODEL_1L not found"
fi

if [ -d "$MODEL_2L" ]; then
    srun "$LMP" -in in.grace_bench -var pstyle grace/2layer/chunk -var model "$MODEL_2L" \
         -var nsteps 50 -log log.grace_2l_viper
else
    echo ">> skip 2L: $MODEL_2L not found"
fi
