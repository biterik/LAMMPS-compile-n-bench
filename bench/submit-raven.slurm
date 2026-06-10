#!/bin/bash
# PACE single-GPU benchmark on Raven (1x NVIDIA A100).
# 'gpudev' caps at 15 min — size nsteps to fit, or use the regular 'gpu' partition.
#SBATCH --job-name=pace_raven
#SBATCH --output=pace_raven.%j.out
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18      # 72 host cores / 4 GPUs per node
#SBATCH --gres=gpu:a100:1
#SBATCH --exclusive             # reserve the whole node: no co-scheduled job can contend (we still use 1 A100)
#SBATCH --time=00:15:00

# ensure 'module' exists without a login shell (avoids login-profile early exit)
if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
module load gcc/13 cuda/12.6 openmpi_gpu/5.0
export OMP_NUM_THREADS=1

# point LMP at the binary you built (build script puts it in <launch-dir>/lammps/build-raven/)
LMP="${LMP:-${SLURM_SUBMIT_DIR:-$PWD}/lammps/build-raven/lmp_raven}"

srun "$LMP" -k on g 1 -sf kk -pk kokkos newton on neigh half \
     -in in.pace_bench -log log.pace_raven
