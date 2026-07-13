#!/bin/bash
# ===========================================================================
# Run a LAMMPS + MC-SITES job on VIPER-CPU (one full node = 2x AMD EPYC 9554,
# 128 physical cores). Pure MPI. fix mc/sites bookkeeping is serial (rank 0);
# energy evaluations are MPI-parallel.
#
# Usage:  cd into your run dir (in.* + potentials), then
#   sbatch submit-mcsites-viper-cpu.slurm
#   IN=in.my_run sbatch submit-mcsites-viper-cpu.slurm
# ===========================================================================
#SBATCH --job-name=mcsites_viper_cpu
#SBATCH --output=mcsites_viper_cpu.%j.out
# Viper-CPU partition; adjust if your project uses a named one.
# #SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=01:00:00

if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
module load gcc/14 openmpi/5.0            # match build-mcsites-viper-cpu.sh
export OMP_NUM_THREADS=1

IN="${IN:-in.mc_sites.langmuir}"
LMP="${LMP:-${SLURM_SUBMIT_DIR:-$PWD}/lammps-mcsites/build-mcsites-viper-cpu/lmp_mcsites_fork24da74_viper_cpu}"

echo ">> binary : $LMP"
echo ">> input  : $IN"
srun "$LMP" -in "$IN" -log "log.${SLURM_JOB_ID:-out}.mcsites_viper_cpu"
