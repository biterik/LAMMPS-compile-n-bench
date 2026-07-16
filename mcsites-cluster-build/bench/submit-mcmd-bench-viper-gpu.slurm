#!/bin/bash
# ===========================================================================
# MD + MC (fix mc/sites) benchmark on VIPER-GPU (1x AMD MI300A APU).
# Runs in.mcmd_bench_nih (EAM Ni/H) and prints a timing breakdown so you can
# estimate the wall time of a production MD+MC run on one GPU.
#
# The MD (and each MC trial's EAM energy eval) run on the GPU via -sf kk.
# fix mc/sites itself has no Kokkos variant (v1) — its create/delete bookkeeping
# runs on the host; on the MI300A that host access is cheap (unified memory).
#
# Requires a Ni/H EAM potential. Point POTDIR at the dir holding
#   ni_h_rcut4.90_rcut2.eam.alloy
#
# Usage (from your PTMP work dir, after building viper-gpu):
#   POTDIR=/viper/ptmp/$USER/potentials sbatch submit-mcmd-bench-viper-gpu.slurm
# Quick/tiny run (fits apudev's 15 min): also pass smaller sizes, e.g.
#   POTDIR=... NMD=500 NTRIALS=100 NCELL=10 sbatch -p apudev -t 00:15:00 submit-mcmd-bench-viper-gpu.slurm
# ===========================================================================
#SBATCH --job-name=mcmd_bench_viper_gpu
#SBATCH --output=mcmd_bench_viper_gpu.%j.out
#SBATCH --partition=gpu        # apudev caps at 15 min; use it only for a tiny run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1           # one MI300A APU (verify gres name with sinfo)
#SBATCH --exclusive
#SBATCH --time=00:30:00

if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
module load gcc/14 rocm/6.3 openmpi_gpu/5.0     # match build-mcsites-viper-gpu.sh
export OMP_NUM_THREADS=1
export HSA_XNACK=1              # MI300A unified memory (HMM) needs XNACK on

# --- inputs / knobs ---------------------------------------------------------
POTDIR="${POTDIR:?set POTDIR=/path with ni_h_rcut4.90_rcut2.eam.alloy}"
HERE="${SLURM_SUBMIT_DIR:-$PWD}"
IN="${IN:-$HERE/in.mcmd_bench_nih}"
LMP="${LMP:-$HERE/lammps-mcsites/build-mcsites-viper-gpu/lmp_mcsites_fork24da74_viper_gpu}"
LOG="${LOG:-$HERE/log.${SLURM_JOB_ID:-out}.mcmd_bench_viper_gpu}"

# optional size/step overrides (forwarded to the input's -var)
NCELL="${NCELL:-16}"; NMD="${NMD:-2000}"; NEVERY="${NEVERY:-20}"
NTRIALS="${NTRIALS:-400}"; MU="${MU:--2.4}"; TEMP="${TEMP:-300.0}"; SEED="${SEED:-12345}"
POT="${POT:-ni_h_rcut4.90_rcut2.eam.alloy}"   # EAM file in POTDIR; override e.g. POT=NiAlH_jea.eam.alloy

[ -x "$LMP" ] || { echo "ERROR: binary not found/executable: $LMP  (set LMP=/path/to/lmp_mcsites_fork24da74_viper_gpu)" >&2; exit 1; }
[ -f "$POTDIR/$POT" ] || { echo "ERROR: potential not found: $POTDIR/$POT  (set POTDIR/POT correctly)" >&2; exit 1; }

echo ">> binary : $LMP"
echo ">> potdir : $POTDIR"
echo ">> size   : ${NCELL}^3 fcc cells ($((4*NCELL*NCELL*NCELL)) Ni)  nmd=$NMD nevery=$NEVERY ntrials=$NTRIALS"

srun "$LMP" -k on g 1 -sf kk -pk kokkos newton on neigh half \
     -in "$IN" -log "$LOG" \
     -var potdir "$POTDIR" -var pot "$POT" -var N "$NCELL" -var nmd "$NMD" -var nevery "$NEVERY" \
     -var ntrials "$NTRIALS" -var mu "$MU" -var T "$TEMP" -var seed "$SEED"

# ===========================================================================
# Timing breakdown. Loop-time lines appear in order: [0] equil, [1] MD-only,
# [2] MD+MC. We use #1 and #2.
# ===========================================================================
echo
echo "==================== MD + MC timing estimate ===================="
awk '
  /Loop time of/ { n++; t[n]=$4 }                       # $4 = seconds
  /^BENCH_RESULT/ {
    for (i=1;i<=NF;i++){ split($i,a,"="); v[a[1]]=a[2] }
  }
  END {
    tMD = t[2]; tMDMC = t[3];                            # after the equil loop
    if (tMD=="" || tMDMC=="") { print "!! could not find both loop times — check the log"; exit }
    nmd=v["nmd"]; nev=v["nevery"]; ntr=v["ntrials"];
    natt=v["natt"]; nat=v["natoms"]; conc=v["conc"]; acc=v["acc"];
    dMC   = tMDMC - tMD;                                  # extra wall from MC over nmd steps
    tstep = tMD / nmd;                                   # s per MD step
    nblk  = nmd / nev;                                   # MC blocks in phase B
    tblk  = (nblk>0)? dMC/nblk : 0;                      # s per MC block
    ttri  = (natt>0)? dMC/natt : 0;                      # s per MC trial (full EAM eval)
    printf "system                : %d atoms (Ni+H), conc N/M = %.3f, MC acceptance = %.3f\n", nat, conc, acc;
    printf "MD-only phase         : %.3f s for %d steps\n", tMD, nmd;
    printf "MD+MC phase           : %.3f s for %d steps + %d blocks x %d trials (%d attempts)\n", tMDMC, nmd, nblk, ntr, natt;
    printf "-----------------------------------------------------------------\n";
    printf "per MD step           : %.4f ms\n", tstep*1e3;
    printf "per MC block          : %.4f ms   (%d trials/block)\n", tblk*1e3, ntr;
    printf "per MC trial (EAM eval): %.4f ms\n", ttri*1e3;
    printf "MC/MD cost ratio       : %.1fx  (one trial vs one MD step)\n", (tstep>0)? ttri/tstep : 0;
    printf "=================================================================\n";
    printf "ESTIMATOR for a production run of  Nmd MD steps + Nblk blocks x Ntr trials:\n";
    printf "   wall ~= Nmd * %.4g s  +  (Nblk*Ntr) * %.4g s\n", tstep, ttri;
    printf "   e.g. 100000 MD steps + 5000 blocks x 400 trials ~= %.1f min\n",
           (100000*tstep + 5000*400*ttri)/60.0;
  }
' "$LOG"
