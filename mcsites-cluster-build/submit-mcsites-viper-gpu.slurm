#!/bin/bash
# ===========================================================================
# Run a LAMMPS + MC-SITES job on VIPER-GPU (1x AMD MI300A APU).
# MD/pair styles run on the GPU via Kokkos/HIP (-sf kk). NOTE: fix mc/sites has
# no Kokkos variant (v1) — its create/delete bookkeeping runs host-side and calls
# the (GPU-accelerated) pair style for each trial energy. Expected behaviour.
#
# 'apudev' caps at 15 min — size your run to fit, or switch to -p gpu for longer.
#
# Usage:  cd into your run dir (in.* + potentials), then
#   sbatch submit-mcsites-viper-gpu.slurm
#   IN=in.my_run sbatch submit-mcsites-viper-gpu.slurm
# ===========================================================================
#SBATCH --job-name=mcsites_viper_gpu
#SBATCH --output=mcsites_viper_gpu.%j.out
#SBATCH --partition=apudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1            # one MI300A APU (verify gres name with sinfo)
#SBATCH --exclusive
#SBATCH --time=00:15:00

if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
module load gcc/14 rocm/6.3 openmpi_gpu/5.0     # match build-mcsites-viper-gpu.sh
export OMP_NUM_THREADS=1

# MI300A APU unified memory (HMM) requires XNACK enabled at runtime, or the first
# device access to a host allocation faults and the run dies right after Kokkos init.
export HSA_XNACK=1

IN="${IN:-in.mc_sites.langmuir}"
LMP="${LMP:-${SLURM_SUBMIT_DIR:-$PWD}/lammps-mcsites/build-mcsites-viper-gpu/lmp_mcsites_fork24da74_viper_gpu}"

echo ">> binary : $LMP"
echo ">> input  : $IN"
srun "$LMP" -k on g 1 -sf kk -pk kokkos newton on neigh half \
     -in "$IN" -log "log.${SLURM_JOB_ID:-out}.mcsites_viper_gpu"
