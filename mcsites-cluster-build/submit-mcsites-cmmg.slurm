#!/bin/bash
# ===========================================================================
# Run a LAMMPS + MC-SITES job on cmmg (full AMD EPYC 9754 node = 256 cores).
# Plain MPI, one rank per physical core. fix mc/sites does its MC bookkeeping
# serially on rank 0; the energy evaluations are MPI-parallel across all ranks.
#
# Usage:  cd into your PTMP run dir (holding your in.* input + potentials), then
#   sbatch submit-mcsites-cmmg.slurm                     # runs $IN below
#   IN=in.my_run sbatch submit-mcsites-cmmg.slurm        # override input
#   LMP=/path/to/lmp_mcsites_fork24da74_cmmg sbatch ...  # override binary
# ===========================================================================
#SBATCH --job-name=mcsites_cmmg
#SBATCH --output=mcsites_cmmg.%j.out
#SBATCH --partition=p.cmmg
#SBATCH --nodes=1
#SBATCH --ntasks=256           # full node (set 128 for one socket)
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
module load gcc/13 impi/2021.16          # match build-mcsites-cmmg.sh
export OMP_NUM_THREADS=1                   # HT on — 1 thread/rank, physical cores only

IN="${IN:-in.mc_sites.langmuir}"
LMP="${LMP:-${SLURM_SUBMIT_DIR:-$PWD}/lammps-mcsites/build-mcsites-cmmg/lmp_mcsites_fork24da74_cmmg}"

echo ">> binary : $LMP"
echo ">> input  : $IN"
srun "$LMP" -in "$IN" -log "log.${SLURM_JOB_ID:-out}.mcsites_cmmg"
