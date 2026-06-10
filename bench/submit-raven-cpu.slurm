#!/bin/bash
# PACE CPU benchmark on Raven — ONE FULL CPU NODE (2x Intel Xeon IceLake-SP
# Platinum 8360Y = 72 physical cores), with the LAMMPS INTEL package engaged.
#
# Note: `pair_style pace` has no INTEL-accelerated variant, so `-sf intel` falls
# back to the standard CPU ACE kernel for the pair style — this is a fair
# Intel-Xeon CPU data point (comparable to the EPYC cmmg run). The INTEL package
# still accelerates any other styles that support it.
#SBATCH --job-name=pace_raven_cpu
#SBATCH --output=pace_raven_cpu.%j.out
# Raven routes CPU jobs to the default partition; uncomment to target a specific one.
# #SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=72            # one full node: 2x 36-core IceLake (physical cores)
#SBATCH --cpus-per-task=1
#SBATCH --exclusive            # reserve the whole node — no co-scheduled job can contend
#SBATCH --time=00:45:00        # 256k atoms x 500 steps on 72 cores ~ 20-25 min; lower -var nsteps to fit a short partition

# ensure 'module' exists without a login shell (avoids login-profile early exit)
if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
# Match the versions used by build-lammps-raven-cpu.sh. (impi is hierarchical —
# it loads here because intel/2025.3 is loaded first on the same line.)
module load intel/2025.3 impi/2021.17 mkl/2025.3
export OMP_NUM_THREADS=1       # pure MPI, one rank per physical core

# point LMP at the CPU binary (build script puts it in <launch-dir>/lammps/build-raven-cpu/)
LMP="${LMP:-${SLURM_SUBMIT_DIR:-$PWD}/lammps/build-raven-cpu/lmp_raven_cpu}"

# -pk intel 0 omp 1  : enable the INTEL package (0 coprocessors, 1 OpenMP thread/rank)
# -sf intel          : use intel-accelerated styles where available (pace falls back)
srun "$LMP" -pk intel 0 omp 1 -sf intel \
     -in in.pace_bench -log log.pace_raven_cpu
