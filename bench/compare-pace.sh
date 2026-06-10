#!/bin/bash
# ===========================================================================
# Compare PACE benchmark results across the three machines.
#
# Parses the LAMMPS log files, pulls the run size and timing, and prints a
# throughput table (timesteps/s, katom-step/s, ns/day) plus speed-up relative
# to the slowest run. Throughput is the fair cross-machine metric as long as
# atoms x steps is identical in every run.
#
# Usage:
#   ./compare-pace.sh                       # uses log.pace_{viper,raven,cmmg} in CWD
#   ./compare-pace.sh log1 log2 ...         # explicit log files
#
# Because the three clusters have separate filesystems, copy (scp) the logs
# into one place first, e.g.:
#   scp viper:.../log.pace_viper raven:.../log.pace_raven cmmg:.../log.pace_cmmg .
# ===========================================================================
set -uo pipefail

logs=("$@")
if [ ${#logs[@]} -eq 0 ]; then
    logs=(log.pace_viper log.pace_raven log.pace_cmmg)
fi

# split into existing / missing so a missing file can't abort awk
present=(); missing=()
for f in "${logs[@]}"; do
    if [ -f "$f" ]; then present+=("$f"); else missing+=("$f"); fi
done

if [ ${#present[@]} -eq 0 ]; then
    echo "No log files found (looked for: ${logs[*]})." >&2
    exit 1
fi

awk '
function bn(p,   s){ s=p; sub(/.*\//,"",s); sub(/^log\.pace[._-]?/,"",s); sub(/\.log$/,"",s); return (s==""?p:s) }
function trim(x){ gsub(/^[ \t]+|[ \t]+$/,"",x); return x }
BEGIN{
    fmt="%-12s %6s %10s %8s %12s %12s %12s %7s %7s %9s\n"
    printf fmt,"machine","procs","atoms","steps","loop_s","tstep/s","katom-st/s","pair%","comm%","speedup"
    printf "%s\n","-----------------------------------------------------------------------------------------------------------"
}
FNR==1{ if (file!=""){ flush() }; file=FILENAME; procs=atoms=steps=loop=tps=kas=0; pair_pct=comm_pct=neigh_pct=0; inb=0; tcol=0 }
# "Loop time of 123.4 on 128 procs for 2000 steps with 256000 atoms"
/Loop time of/{
    loop=$4; procs=$6; steps=$9; atoms=$12
}
# "Performance: 1.23 ns/day, ... , 45.6 timesteps/s, 78.9 katom-step/s"
/Performance:/{
    for(i=1;i<=NF;i++){
        if($i=="ns/day")    nsday=$(i-1)
        if($i=="timesteps/s")tps=$(i-1)
        if($i ~ /katom-step\/s/) kas=$(i-1)
    }
}
# --- MPI task timing breakdown: locate the %total column, then read Pair/Comm ---
/MPI task timing breakdown/{ inb=1; next }
inb && /%total/{                                  # header row: find the %total column
    m=split($0,a,"|"); for(j=1;j<=m;j++){ if(trim(a[j])=="%total") tcol=j }; next
}
inb{
    if($0 ~ /^[A-Za-z]/ && tcol>0){
        m=split($0,a,"|"); lbl=trim(a[1]); val=trim(a[tcol])+0
        if(lbl=="Pair")  pair_pct=val
        else if(lbl=="Comm")  comm_pct=val
        else if(lbl=="Neigh") neigh_pct=val
    }
    if($0 ~ /Nlocal/ || $0 ~ /^[[:space:]]*$/) inb=0
}
function flush(   k){
    if(loop>0){
        if(tps==0 && steps>0) tps=steps/loop
        if(kas==0 && atoms>0 && steps>0) kas=atoms*steps/loop/1000.0
        n++; M[n]=bn(file); P[n]=procs; A[n]=atoms; S[n]=steps; L[n]=loop; T[n]=tps; K[n]=kas; ND[n]=nsday
        PP[n]=pair_pct; CP[n]=comm_pct; NP[n]=neigh_pct
        if(kas>kmax) kmax=kas
        if(kmin==0 || kas<kmin) kmin=kas
    }
}
END{
    flush()
    for(i=1;i<=n;i++){
        sp=(kmin>0)? K[i]/kmin : 0
        printf "%-12s %6d %10d %8d %12.2f %12.2f %12.2f %6.1f%% %6.1f%% %8.2fx\n", M[i],P[i],A[i],S[i],L[i],T[i],K[i],PP[i],CP[i],sp
    }
    if(n>1){
        printf "\n(speedup = katom-step/s relative to the slowest run; ns/day also in logs)\n"
        printf "(pair%% / comm%% = share of wall time in force eval vs MPI communication;\n"
        printf " a high comm%% means the run is communication-bound — try fewer ranks / more atoms-per-core)\n"
        # warn if work differs
        for(i=2;i<=n;i++) if(A[i]!=A[1] || S[i]!=S[1]){
            printf "WARNING: atoms x steps differ between runs - throughputs are still comparable,\n"
            printf "         but wall times are NOT. Re-run with identical nx/ny/nz and nsteps.\n"; break
        }
    }
}
' "${present[@]}"

# note any missing logs (guard the expansion: under `set -u` an empty array
# would otherwise error as "missing[@]: unbound variable" on older bash)
if [ ${#missing[@]} -gt 0 ]; then
    for f in "${missing[@]}"; do
        echo "missing: $f (skipped)"
    done
fi
