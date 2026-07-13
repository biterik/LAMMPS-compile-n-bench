#!/bin/bash
# ===========================================================================
# Run a LAMMPS + MC-SITES job on RAVEN-CPU (one full node = 2x Xeon IceLake,
# 72 physical cores). Pure MPI, INTEL package engaged for styles that support it.
# fix mc/sites bookkeeping is serial (rank 0); energy evals are MPI-parallel.
#
# Usage:  cd into your run dir (in.* + potentials), then
#   sbatch submit-mcsites-raven-cpu.slurm
#   IN=in.my_run sbatch submit-mcsites-raven-cpu.slurm
# ===========================================================================
#SBATCH --job-name=mcsites_raven_cpu
#SBATCH --output=mcsites_raven_cpu.%j.out
# Raven routes CPU jobs to the default partition; uncomment to target one.
# #SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=01:00:00

if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
module load intel/2025.3 impi/2021.17 mkl/2025.3   # match build-mcsites-raven-cpu.sh
export OMP_NUM_THREADS=1

IN="${IN:-in.mc_sites.langmuir}"
LMP="${LMP:-${SLURM_SUBMIT_DIR:-$PWD}/lammps-mcsites/build-mcsites-raven-cpu/lmp_mcsites_fork24da74_raven_cpu}"

echo ">> binary : $LMP"
echo ">> input  : $IN"
# -pk intel 0 omp 1 / -sf intel : enable INTEL-accelerated styles where available.
srun "$LMP" -pk intel 0 omp 1 -sf intel -in "$IN" -log "log.${SLURM_JOB_ID:-out}.mcsites_raven_cpu"
