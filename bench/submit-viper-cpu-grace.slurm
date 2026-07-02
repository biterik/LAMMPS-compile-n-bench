#!/bin/bash
# GRACE CPU benchmark on Viper-CPU (full 2x AMD EPYC 9554 "Genoa" node = 128
# physical cores). Same two TensorFlow runs as submit-cmmg-grace.slurm:
# 1-layer via grace/1layer/chunk, 2-layer via grace/2layer/chunk (foundation
# models are not FS-exportable). Needs a TF-enabled binary + TF_PYLIB on the path.
#SBATCH --job-name=grace_viper_cpu
#SBATCH --output=grace_viper_cpu.%j.out
# #SBATCH --partition=cpu          # set your Viper-CPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:30:00

if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi
module purge
module load gcc/14 openmpi/5.0
export OMP_NUM_THREADS=1
# Quiet TensorFlow's per-rank C++ log flood (3 = hide INFO+WARNING+ERROR).
export TF_CPP_MIN_LOG_LEVEL=3

SUB="${SLURM_SUBMIT_DIR:-$PWD}"
LMP="${LMP:-$SUB/lammps-fork/build-viper-cpu-fork/lmp_viper_cpu_fork}"

MODEL_1L="${MODEL_1L:-${GRACE_CACHE:-$HOME/.cache/grace}/GRACE-1L-SMAX-OMAT-large}"
MODEL_2L="${MODEL_2L:-${GRACE_CACHE:-$HOME/.cache/grace}/GRACE-2L-SMAX-OMAT-medium}"

# 1) 1-layer model (TensorFlow) via grace/1layer/chunk, full node MPI.
if [ -d "$MODEL_1L" ]; then
    srun "$LMP" -in in.grace_bench -var pstyle grace/1layer/chunk -var model "$MODEL_1L" \
         -log log.grace_1l_viper_cpu
else
    echo ">> skip 1L: $MODEL_1L not found (grace_models download GRACE-1L-SMAX-OMAT-large)"
fi

# 2) 2-layer model (TensorFlow) via grace/2layer/chunk (heavier — small nsteps).
if [ -d "$MODEL_2L" ]; then
    echo ">> running 2-layer TF model on CPU — expect it to be slow"
    srun "$LMP" -in in.grace_bench -var pstyle grace/2layer/chunk -var model "$MODEL_2L" \
         -var nsteps 20 -log log.grace_2l_viper_cpu
else
    echo ">> skip 2L: $MODEL_2L not found (grace_models download GRACE-2L-SMAX-OMAT-medium)"
fi
