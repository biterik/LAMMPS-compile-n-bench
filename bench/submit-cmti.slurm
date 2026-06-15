#!/bin/bash
# PACE CPU benchmark on cmti — ONE FULL NODE (2x Intel Xeon Gold 6230
# "Cascade Lake" = 40 physical cores), with the LAMMPS INTEL package engaged.
#
# Note: `pair_style pace` has no INTEL-accelerated variant, so `-sf intel` falls
# back to the standard CPU ACE kernel for the pair style — a fair Xeon CPU data
# point. The INTEL package still accelerates any other styles that support it.
#SBATCH --job-name=pace_cmti
#SBATCH --output=pace_cmti.%j.out
#SBATCH --partition=p.cmfe     # cmti compute nodes; full-node (exclusive) partition
#SBATCH --nodes=1
#SBATCH --ntasks=40            # one full node: 2x 20-core Xeon Gold 6230 (physical cores)
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00        # 256k atoms x 500 steps on 40 cores is slow; lower -var nsteps if needed

# ensure 'module' exists without a login shell (avoids login-profile early exit)
if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
# Match the versions used by build-lammps-cmti.sh. (impi is hierarchical —
# it loads here because intel/2025.2 is loaded first on the same line.)
module load intel/2025.2 impi/2021.16 mkl/2025.2
export OMP_NUM_THREADS=1       # pure MPI, one rank per physical core

# point LMP at the binary you built (build script puts it in <launch-dir>/lammps/build-cmti/)
LMP="${LMP:-${SLURM_SUBMIT_DIR:-$PWD}/lammps/build-cmti/lmp_cmti}"

# -pk intel 0 omp 1 : enable INTEL package (0 coprocessors, 1 OpenMP thread/rank)
# -sf intel         : use intel-accelerated styles where available (pace falls back)
srun "$LMP" -pk intel 0 omp 1 -sf intel \
     -in in.pace_bench -log log.pace_cmti
