#!/bin/bash
# GRACE CPU benchmark on Viper-CPU (full 2x AMD EPYC 9554 "Genoa" node = 128
# physical cores). Same two runs as submit-cmmg-grace.slurm: headline grace/fs
# (1-layer, no TF) + optional 2-layer TensorFlow model.
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

SUB="${SLURM_SUBMIT_DIR:-$PWD}"
LMP="${LMP:-$SUB/lammps-fork/build-viper-cpu-fork/lmp_viper_cpu_fork}"

FS_MODEL="${FS_MODEL:-FS_model.yaml}"
MODEL_2L="${MODEL_2L:-${GRACE_CACHE:-$HOME/.cache/grace}/GRACE-2L-SMAX-OMAT-medium}"

# 1) Headline: FS 1-layer, full node, plain MPI, no TensorFlow.
srun "$LMP" -in in.grace_bench -var pstyle grace/fs -var model "$FS_MODEL" \
     -log log.grace_fs_viper_cpu

# 2) Optional: 2-layer TensorFlow model (CPU TF — slow; small nsteps).
if [ -d "$MODEL_2L" ]; then
    echo ">> running 2-layer TF model (grace/2layer/chunk) on CPU — expect it to be slow"
    srun "$LMP" -in in.grace_bench -var pstyle grace/2layer/chunk -var model "$MODEL_2L" \
         -var nsteps 20 -log log.grace_2l_viper_cpu
else
    echo ">> skip 2-layer TF run: $MODEL_2L not found (see GRACE.md to fetch it)"
fi
