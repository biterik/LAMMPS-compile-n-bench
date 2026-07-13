#!/bin/bash
# ===========================================================================
# Run a LAMMPS + MC-SITES job on RAVEN-GPU (1x NVIDIA A100).
# The MD/pair styles run on the GPU via Kokkos/CUDA (-sf kk). NOTE: fix mc/sites
# has no Kokkos variant (v1) — it runs its create/delete bookkeeping on the host
# and calls the (GPU-accelerated) pair style for each trial energy. That is
# expected; the pair evaluation is still on the A100.
#
# 'gpudev' caps at 15 min — size your run to fit, or use the regular 'gpu' partition.
#
# Usage:  cd into your run dir (in.* + potentials), then
#   sbatch submit-mcsites-raven-gpu.slurm
#   IN=in.my_run sbatch submit-mcsites-raven-gpu.slurm
# ===========================================================================
#SBATCH --job-name=mcsites_raven_gpu
#SBATCH --output=mcsites_raven_gpu.%j.out
#SBATCH --partition=gpudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18      # 72 host cores / 4 GPUs per node
#SBATCH --gres=gpu:a100:1
#SBATCH --exclusive
#SBATCH --time=00:15:00

if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
module load gcc/13 cuda/12.6 openmpi_gpu/5.0    # match build-mcsites-raven-gpu.sh
export OMP_NUM_THREADS=1

IN="${IN:-in.mc_sites.langmuir}"
LMP="${LMP:-${SLURM_SUBMIT_DIR:-$PWD}/lammps-mcsites/build-mcsites-raven-gpu/lmp_mcsites_fork24da74_raven_gpu}"

echo ">> binary : $LMP"
echo ">> input  : $IN"
srun "$LMP" -k on g 1 -sf kk -pk kokkos newton on neigh half \
     -in "$IN" -log "log.${SLURM_JOB_ID:-out}.mcsites_raven_gpu"
