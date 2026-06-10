#!/bin/bash
# PACE single-GPU benchmark on Viper-GPU (1x MI300A APU).
# 'apudev' caps at 15 min — size nsteps to fit, or switch to -p gpu for longer.
#SBATCH --job-name=pace_viper
#SBATCH --output=pace_viper.%j.out
#SBATCH --partition=apudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1            # one MI300A APU (verify gres name with `sinfo`/example scripts)
#SBATCH --exclusive             # reserve the whole node: no co-scheduled job can contend (we still use 1 APU)
#SBATCH --time=00:15:00

# ensure 'module' exists without a login shell (avoids login-profile early exit)
if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
module load gcc/14 rocm/6.3 openmpi_gpu/5.0
export OMP_NUM_THREADS=1

# The binary is built for the MI300A APU (Kokkos_ARCH_AMD_GFX942_APU), which uses
# unified memory: the GPU reads host allocations directly via HMM. That REQUIRES
# XNACK to be enabled at runtime — without it the first device access to a host
# allocation faults and the run dies silently right after Kokkos init. Enable it:
export HSA_XNACK=1

# point LMP at the binary you built (build script puts it in <launch-dir>/lammps/build-viper/)
LMP="${LMP:-${SLURM_SUBMIT_DIR:-$PWD}/lammps/build-viper/lmp_viper}"

srun "$LMP" -k on g 1 -sf kk -pk kokkos newton on neigh half \
     -in in.pace_bench -log log.pace_viper
