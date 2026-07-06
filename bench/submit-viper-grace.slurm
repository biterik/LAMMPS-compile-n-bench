#!/bin/bash
# GRACE single-GPU benchmark on Viper-GPU (1x AMD MI300A APU) via TF-ROCm.
# CONFIRMED WORKING (2026-07-06): TF-ROCm 2.17 does use the MI300A (gfx942), but
# is ~13x slower than the A100 for GRACE (1L 4.33, 2L 0.94 katom-step/s). The
# fork has no grace/fs/kk, so TF-ROCm is the only GPU path here.
#
# SUBMIT WITH A CLEAN ENVIRONMENT:
#   cd <gracework>            # must contain grace-cache/ , tf-rocm/ , lammps-fork/
#   sbatch --export=NONE /path/to/bench/submit-viper-grace.slurm
# --export=NONE avoids leaking your interactive module state into the job. That
# leak breaks 'module load rocm' inside the job, and TF then silently falls back
# to the CPU (platform Host) and blows the wall-clock limit. GRACE_CACHE and
# TF_PYLIB default to the submit dir, so nothing else needs to be exported.
#
# Build first:  GRACE_TF=on PYMODULE=python-waterboa/2024.06 \
#               PYTHON=<gracework>/tf-rocm/bin/python  build-lammps-viper-fork.sh
#SBATCH --job-name=grace_viper
#SBATCH --output=grace_viper.%j.out
#SBATCH --partition=apudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=00:15:00

if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi
# Belt-and-suspenders if submitted with --export=ALL: drop any inherited module
# state so 'module purge' starts clean (the real fix is --export=NONE, see header).
unset LOADEDMODULES _LMFILES_ 2>/dev/null || true
module purge
module load gcc/14 rocm/6.3 openmpi_gpu/5.0
export OMP_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3     # quiet TensorFlow's per-process C++ log flood
export HSA_XNACK=1                # MI300A APU unified memory

SUB="${SLURM_SUBMIT_DIR:-$PWD}"
LMP="${LMP:-$SUB/lammps-fork/build-viper-fork/lmp_viper_fork}"
# Self-contained defaults (work with --export=NONE): models + TF runtime live in
# the submit dir. Override GRACE_CACHE / TF_PYLIB via --export=... if your layout differs.
GRACE_CACHE="${GRACE_CACHE:-$SUB/grace-cache}"
TF_PYLIB="${TF_PYLIB:-$(ls -d "$SUB"/tf-rocm/lib/python*/site-packages/tensorflow 2>/dev/null | head -1)}"
[ -n "${TF_PYLIB:-}" ] && export LD_LIBRARY_PATH="$TF_PYLIB:${LD_LIBRARY_PATH:-}"

MODEL_1L="${MODEL_1L:-$GRACE_CACHE/GRACE-1L-SMAX-OMAT-large}"
MODEL_2L="${MODEL_2L:-$GRACE_CACHE/GRACE-2L-SMAX-OMAT-medium}"

# nsteps kept small so BOTH models finish inside the 15-min apudev window on the
# APU (TF-ROCm is slow). Throughput (katom-step/s) is step-count-independent.
if [ -d "$MODEL_1L" ]; then
    srun "$LMP" -in in.grace_bench -var pstyle grace -var model "$MODEL_1L" \
         -var nsteps 50 -log log.grace_1l_viper
else
    echo ">> skip 1L: $MODEL_1L not found (grace-cache under $SUB?)"
fi

if [ -d "$MODEL_2L" ]; then
    srun "$LMP" -in in.grace_bench -var pstyle grace/2layer/chunk -var model "$MODEL_2L" \
         -var nsteps 20 -log log.grace_2l_viper
else
    echo ">> skip 2L: $MODEL_2L not found (grace-cache under $SUB?)"
fi
