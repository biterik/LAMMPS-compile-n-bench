#!/bin/bash
# ===========================================================================
# CROSS-CHECK: same MD+MC Ni/H benchmark on VIPER-CPU (2x EPYC 9554, 128 cores).
# Purpose: (1) confirm the GPU run's physics — compare the final "conc N/M" and
# MC acceptance printed below against the viper-gpu run (they should agree within
# statistics); (2) a full-node CPU timing reference for the same recipe.
# Pure MPI, no Kokkos — the fully supported path for fix mc/sites.
#
#   POTDIR=/viper/ptmp/$USER/potentials sbatch submit-mcmd-bench-viper-cpu.slurm
# ===========================================================================
#SBATCH --job-name=mcmd_bench_viper_cpu
#SBATCH --output=mcmd_bench_viper_cpu.%j.out
# #SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:30:00

if ! command -v module >/dev/null 2>&1; then
    for _f in /etc/profile.d/modules.sh "${MODULESHOME:+$MODULESHOME/init/bash}"; do
        [ -n "${_f:-}" ] && [ -r "$_f" ] && . "$_f" && break
    done
fi

module purge
module load gcc/14 openmpi/5.0
export OMP_NUM_THREADS=1

# ================= USER CONFIG — baked-in defaults, EDIT IF PATHS MOVE =========
# Bare `sbatch submit-mcmd-bench-viper-cpu.slurm` works with NO env vars.
# Override for one run with --export, e.g. sbatch --export=ALL,NCELL=24 <script>.
POTDIR="${POTDIR:-/viper/ptmp/biterik/POTENTIALS}"
POT="${POT:-ni_h_rcut4.90_rcut2.eam.alloy}"
LMP="${LMP:-/viper/ptmp2/biterik/LAMMPS-compile-n-bench/lammps-mcsites/build-mcsites-viper-cpu/lmp_mcsites_fork24da74_viper_cpu}"
IN="${IN:-/viper/ptmp2/biterik/LAMMPS-compile-n-bench/mcsites-cluster-build/bench/in.mcmd_bench_nih}"
# ==============================================================================
HERE="${SLURM_SUBMIT_DIR:-$PWD}"
LOG="${LOG:-$HERE/log.${SLURM_JOB_ID:-out}.mcmd_bench_viper_cpu}"
NCELL="${NCELL:-16}"; NMD="${NMD:-2000}"; NEVERY="${NEVERY:-20}"
NTRIALS="${NTRIALS:-400}"; MU="${MU:--2.4}"; TEMP="${TEMP:-300.0}"; SEED="${SEED:-12345}"

[ -x "$LMP" ] || { echo "ERROR: binary not found: $LMP  (set LMP=/path/to/lmp_mcsites_fork24da74_viper_cpu)" >&2; exit 1; }
[ -f "$POTDIR/$POT" ] || { echo "ERROR: potential not found: $POTDIR/$POT" >&2; exit 1; }

echo ">> binary : $LMP"
echo ">> size   : ${NCELL}^3 fcc cells ($((4*NCELL*NCELL*NCELL)) Ni)  nmd=$NMD nevery=$NEVERY ntrials=$NTRIALS"

srun "$LMP" -in "$IN" -log "$LOG" \
     -var potdir "$POTDIR" -var pot "$POT" -var N "$NCELL" -var nmd "$NMD" -var nevery "$NEVERY" \
     -var ntrials "$NTRIALS" -var mu "$MU" -var T "$TEMP" -var seed "$SEED"

echo
echo "==================== MD + MC timing estimate (CPU) ===================="
awk '
  /Loop time of/ { n++; t[n]=$4 }
  /^BENCH_RESULT/ { for (i=1;i<=NF;i++){ split($i,a,"="); v[a[1]]=a[2] } }
  END {
    tMD=t[2]; tMDMC=t[3];
    if (tMD=="" || tMDMC=="") { print "!! could not find both loop times"; exit }
    nmd=v["nmd"]; nev=v["nevery"]; ntr=v["ntrials"]; natt=v["natt"]; nat=v["natoms"]; conc=v["conc"]; acc=v["acc"];
    dMC=tMDMC-tMD; tstep=tMD/nmd; nblk=nmd/nev; ttri=(natt>0)?dMC/natt:0;
    printf "system                : %d atoms, conc N/M = %.3f, MC acceptance = %.3f\n", nat, conc, acc;
    printf "per MD step           : %.4f ms\n", tstep*1e3;
    printf "per MC trial (EAM eval): %.4f ms\n", ttri*1e3;
    printf "  --> compare conc N/M and acceptance to the viper-gpu run to validate the GPU path.\n";
  }
' "$LOG"
