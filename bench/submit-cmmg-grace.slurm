#!/bin/bash
# GRACE CPU benchmark on cmmg (full AMD EPYC 9754 node = 256 physical cores).
# Headline run: grace/fs (1-layer SMAX-OMAT, FS export, NO TensorFlow) — the
# cross-machine-comparable GRACE point. Optional second run: the 2-layer TF
# model via grace/2layer/chunk (CPU TensorFlow — much slower; runs only if the
# saved_model dir is present and TF runtime is on LD_LIBRARY_PATH).
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

# --- model files (override with env vars) -----------------------------------
# 1-layer FS export (see GRACE.md: gracemaker -r -s -sf -> FS_model.yaml):
FS_MODEL="${FS_MODEL:-FS_model.yaml}"
# 2-layer TensorFlow saved_model dir (grace_models download GRACE-2L-SMAX-OMAT-medium).
# Models are multi-GB — keep them in PTMP via GRACE_CACHE, NOT $HOME:
MODEL_2L="${MODEL_2L:-${GRACE_CACHE:-$HOME/.cache/grace}/GRACE-2L-SMAX-OMAT-medium}"

# 1) Headline: FS 1-layer, full node, plain MPI, no TensorFlow.
srun "$LMP" -in in.grace_bench -var pstyle grace/fs -var model "$FS_MODEL" \
     -log log.grace_fs_cmmg

# 2) Optional: 2-layer TensorFlow model (CPU TF is slow — keep nsteps small).
#    Needs the TF runtime libs on LD_LIBRARY_PATH (from your tensorflow venv).
if [ -d "$MODEL_2L" ]; then
    echo ">> running 2-layer TF model (grace/2layer/chunk) on CPU — expect it to be slow"
    srun "$LMP" -in in.grace_bench -var pstyle grace/2layer/chunk -var model "$MODEL_2L" \
         -var nsteps 20 -log log.grace_2l_cmmg
else
    echo ">> skip 2-layer TF run: $MODEL_2L not found (see GRACE.md to fetch it)"
fi
