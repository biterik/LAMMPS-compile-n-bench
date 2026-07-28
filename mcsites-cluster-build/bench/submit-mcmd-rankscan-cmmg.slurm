#!/bin/bash
# ===========================================================================
# RANK SCAN on cmmg: the viper-8B criterion-7 decider on an independent
# machine + compiler (2x EPYC 9754, 256 cores, gcc/impi build).
#
# One full-node job runs the SAME Ni/H MD+MC benchmark (in.mcmd_rankscan_nih,
# same seed 12345) sequentially at np = 1..256 and prints one RANKSCAN line
# per np.  The input runs fix mc/sites with `check yes` (patch 0010): every
# MC block verifies atom count, species accounting, and the Metropolis
# reference energy — a parallel bookkeeping bug aborts the run AT THE FIRST
# INCONSISTENT BLOCK instead of surfacing as silently wrong statistics.
#
# Requires the 10-patch binary (build-mcsites-cmmg.sh rebuilds the branch
# automatically since its guard probes for patch 0010; a 9-patch binary
# fails immediately on the unknown `check` keyword — that is the freshness
# check).
#
# Interpretation (2026-07-28 container scan: gcc/x86 showed NO np-dependence
# at np<=64, incl. sub-cutoff subdomains — see the session notes):
#   * acc flat over np, checks silent  -> fix is rank-consistent here too;
#     viper 0.28-vs-0.004 was a binary/build or machine effect, not MC logic.
#   * acc rises with np, or a check fires -> reproducible bug in hand; the
#     failing np + block number pinpoint it.
#
# Usage (cd into your PTMP run dir):
#   sbatch submit-mcmd-rankscan-cmmg.slurm
#   MU=-2.4 NPLIST="1 16 256" sbatch submit-mcmd-rankscan-cmmg.slurm
# Harvest:  grep -h RANKSCAN mcmd_rankscan_cmmg.*.out
# ===========================================================================
#SBATCH --job-name=mcmd_rankscan_cmmg
#SBATCH --output=mcmd_rankscan_cmmg.%j.out
#SBATCH --partition=p.cmmg
#SBATCH --nodes=1
#SBATCH --ntasks=256
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=02:00:00

if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
module load gcc/13 impi/2021.16          # match build-mcsites-cmmg.sh
export OMP_NUM_THREADS=1

# ================= USER CONFIG — baked-in defaults, EDIT IF PATHS MOVE =========
# cmmg policy: work under /u/$USER/PTMP (see cluster-lammps-info.txt).
POTDIR="${POTDIR:-/u/$USER/PTMP/POTENTIALS}"
POT="${POT:-ni_h_rcut4.90_rcut2.eam.alloy}"
LMP="${LMP:-/u/$USER/PTMP/mcsites/lammps-mcsites/build-mcsites-cmmg/lmp_mcsites_fork24da74_cmmg}"
IN="${IN:-${SLURM_SUBMIT_DIR:-$PWD}/in.mcmd_rankscan_nih}"
# ==============================================================================
HERE="${SLURM_SUBMIT_DIR:-$PWD}"
NCELL="${NCELL:-10}"; NMD="${NMD:-500}"; NEVERY="${NEVERY:-20}"
NTRIALS="${NTRIALS:-100}"; MU="${MU:--2.0}"; TEMP="${TEMP:-300.0}"; SEED="${SEED:-12345}"
NPLIST="${NPLIST:-1 2 4 8 16 32 64 128 256}"

[ -x "$LMP" ] || { echo "ERROR: binary not found: $LMP  (set LMP=/path/to/lmp_mcsites_fork24da74_cmmg)" >&2; exit 1; }
[ -f "$POTDIR/$POT" ] || { echo "ERROR: potential not found: $POTDIR/$POT" >&2; exit 1; }
[ -f "$IN" ] || { echo "ERROR: input not found: $IN" >&2; exit 1; }

echo ">> binary : $LMP"
echo ">> size   : ${NCELL}^3 fcc cells ($((4*NCELL*NCELL*NCELL)) Ni)  nmd=$NMD nevery=$NEVERY ntrials=$NTRIALS mu=$MU seed=$SEED"
echo ">> np list: $NPLIST   (check yes: any consistency failure aborts that np's run)"

for NP in $NPLIST; do
    LOG="$HERE/log.${SLURM_JOB_ID:-out}.rankscan_np${NP}"
    echo ">> ---- np=$NP ----"
    if srun --exact -n "$NP" "$LMP" -in "$IN" -log "$LOG" \
         -var potdir "$POTDIR" -var pot "$POT" -var N "$NCELL" -var nmd "$NMD" \
         -var nevery "$NEVERY" -var ntrials "$NTRIALS" -var mu "$MU" -var T "$TEMP" \
         -var seed "$SEED" -var check yes; then
        RES=$(grep '^BENCH_RESULT' "$LOG" || echo "BENCH_RESULT missing")
        echo "RANKSCAN np=$NP ${RES#BENCH_RESULT }"
    else
        echo "RANKSCAN np=$NP FAILED (consistency check or crash — see $LOG and any ERROR above)"
    fi
done

echo
echo "==================== rank-scan summary ===================="
grep -h "^RANKSCAN" "$HERE/mcmd_rankscan_cmmg.${SLURM_JOB_ID}.out" 2>/dev/null || true
echo "acc flat over np + no FAILED lines => rank-consistent on cmmg."
