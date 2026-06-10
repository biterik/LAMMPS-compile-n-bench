#!/bin/bash
# PACE CPU benchmark on cmmg (full AMD EPYC 9754 node = 256 physical cores).
# Plain MPI, one rank per physical core, plain CPU pair_style pace.
#SBATCH --job-name=pace_cmmg
#SBATCH --output=pace_cmmg.%j.out
#SBATCH --partition=p.cmmg
#SBATCH --nodes=1
#SBATCH --ntasks=256           # full node, both sockets (set 128 for a one-socket run)
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

# ensure 'module' exists without a login shell (avoids login-profile early exit)
if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
# No default module versions on this cluster — pin them (match your build).
module load gcc/13
module load impi/2021.16
export OMP_NUM_THREADS=1        # HT is on — keep 1 thread/rank, physical cores only

# point LMP at the binary you built (build script puts it in <launch-dir>/lammps/build-cmmg/)
LMP="${LMP:-${SLURM_SUBMIT_DIR:-$PWD}/lammps/build-cmmg/lmp_cmmg}"

srun "$LMP" -in in.pace_bench -log log.pace_cmmg
