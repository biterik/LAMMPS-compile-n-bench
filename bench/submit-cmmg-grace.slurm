#!/bin/bash
# GRACE CPU benchmark on cmmg (full AMD EPYC 9754 node = 256 physical cores).
# Both runs use the downloaded TensorFlow saved_models (SMAX-OMAT foundation
# models are NOT FS-exportable, so grace/fs is not used):
#   * 1-layer via grace/1layer/chunk   * 2-layer via grace/2layer/chunk
# Requires a TF-enabled binary (build with GRACE_TF=on) and the TF runtime on
# LD_LIBRARY_PATH (TF_PYLIB). CPU TensorFlow is slow — keep nsteps modest.
#SBATCH --job-name=grace_cmmg
#SBATCH --output=grace_cmmg.%j.out
#SBATCH --partition=p.cmmg
#SBATCH --nodes=1
#SBATCH --ntasks=256
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi
module purge
module load gcc/13 impi/2021.16
export OMP_NUM_THREADS=1

SUB="${SLURM_SUBMIT_DIR:-$PWD}"
LMP="${LMP:-$SUB/lammps-fork/build-cmmg-fork/lmp_cmmg_fork}"

# --- model saved_model dirs (override with env vars) ------------------------
# Multi-GB — keep them in PTMP via GRACE_CACHE, NOT $HOME. Fetch with:
#   grace_models download GRACE-1L-SMAX-OMAT-large   GRACE-2L-SMAX-OMAT-medium
MODEL_1L="${MODEL_1L:-${GRACE_CACHE:-$HOME/.cache/grace}/GRACE-1L-SMAX-OMAT-large}"
MODEL_2L="${MODEL_2L:-${GRACE_CACHE:-$HOME/.cache/grace}/GRACE-2L-SMAX-OMAT-medium}"

# 1) 1-layer model (TensorFlow) via grace/1layer/chunk, full node MPI.
if [ -d "$MODEL_1L" ]; then
    srun "$LMP" -in in.grace_bench -var pstyle grace/1layer/chunk -var model "$MODEL_1L" \
         -log log.grace_1l_cmmg
else
    echo ">> skip 1L: $MODEL_1L not found (grace_models download GRACE-1L-SMAX-OMAT-large)"
fi

# 2) 2-layer model (TensorFlow) via grace/2layer/chunk (heavier — small nsteps).
if [ -d "$MODEL_2L" ]; then
    echo ">> running 2-layer TF model on CPU — expect it to be slow"
    srun "$LMP" -in in.grace_bench -var pstyle grace/2layer/chunk -var model "$MODEL_2L" \
         -var nsteps 20 -log log.grace_2l_cmmg
else
    echo ">> skip 2L: $MODEL_2L not found (grace_models download GRACE-2L-SMAX-OMAT-medium)"
fi
