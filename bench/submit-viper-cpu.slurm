#!/bin/bash
# PACE CPU benchmark on Viper-CPU — ONE FULL NODE (2x AMD EPYC 9554 "Genoa" =
# 128 physical cores). Plain MPI, one rank per physical core, plain CPU
# pair_style pace (same input as every other machine).
#SBATCH --job-name=pace_viper_cpu
#SBATCH --output=pace_viper_cpu.%j.out
# Viper-CPU partition; adjust if your project uses a named one.
# #SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=128           # one full node: 2x 64-core EPYC 9554 (physical cores)
#SBATCH --cpus-per-task=1
#SBATCH --exclusive            # reserve the whole node — no co-scheduled job can contend
#SBATCH --time=00:30:00

# ensure 'module' exists without a login shell (avoids login-profile early exit)
if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
# Match the versions used by build-lammps-viper-cpu.sh.
module load gcc/14 openmpi/5.0
export OMP_NUM_THREADS=1       # pure MPI, one rank per physical core

# point LMP at the distinctly-named CPU binary (build-viper-cpu/lmp_viper_cpu)
LMP="${LMP:-${SLURM_SUBMIT_DIR:-$PWD}/lammps/build-viper-cpu/lmp_viper_cpu}"

srun "$LMP" -in in.pace_bench -log log.pace_viper_cpu
