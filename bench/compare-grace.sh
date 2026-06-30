#!/bin/bash
# ===========================================================================
# Compare GRACE benchmark results across machines / model flavours.
#
# Same parser as compare-pace.sh, but defaults to the log.grace_* files and
# keeps the flavour tag (fs / 1l / 2l + machine) in the first column. Throughput
# (katom-step/s) is the fair metric as long as atoms x steps is identical in
# every run you compare. NOTE: the GRACE table is at the smaller common size
# (~16k atoms), separate from the 256k PACE table — see in.grace_bench.
#
# Usage:
#   ./compare-grace.sh                 # all log.grace_* in CWD
#   ./compare-grace.sh log1 log2 ...   # explicit log files
#
# Gather logs onto one host first (separate cluster filesystems), e.g.:
#   scp cmmg:.../log.grace_fs_cmmg raven:.../log.grace_*_raven .
# ===========================================================================
set -uo pipefail

logs=("$@")
if [ ${#logs[@]} -eq 0 ]; then
    shopt -s nullglob
    logs=( log.grace_* )
    shopt -u nullglob
fi

present=(); missing=()
for f in "${logs[@]}"; do
    if [ -f "$f" ]; then present+=("$f"); else missing+=("$f"); fi
done
if [ ${#present[@]} -eq 0 ]; then
    echo "No GRACE log files found (looked for: ${logs[*]:-log.grace_*})." >&2
    exit 1
fi

awk '
function bn(p,   s){ s=p; sub(/.*\//,"",s); sub(/^log\.grace[._-]?/,"",s); sub(/\.log$/,"",s); return (s==""?p:s) }
function trim(x){ gsub(/^[ \t]+|[ \t]+$/,"",x); return x }
BEGIN{
    fmt="%-16s %6s %10s %8s %12s %12s %12s %7s %7s %9s\n"
    printf fmt,"run","procs","atoms","steps","loop_s","tstep/s","katom-st/s","pair%","comm%","speedup"
    printf "%s\n","---------------------------------------------------------------------------------------------------------------"
}
FNR==1{ if (file!=""){ flush() }; file=FILENAME; procs=atoms=steps=loop=tps=kas=0; pair_pct=comm_pct=neigh_pct=0; inb=0; tcol=0 }
/Loop time of/{ loop=$4; procs=$6; steps=$9; atoms=$12 }
/Performance:/{
    for(i=1;i<=NF;i++){
        if($i=="ns/day")    nsday=$(i-1)
        if($i=="timesteps/s")tps=$(i-1)
        if($i ~ /katom-step\/s/) kas=$(i-1)
    }
}
/MPI task timing breakdown/{ inb=1; next }
inb && /%total/{ m=split($0,a,"|"); for(j=1;j<=m;j++){ if(trim(a[j])=="%total") tcol=j }; next }
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
        n++; M[n]=bn(file); P[n]=procs; A[n]=atoms; S[n]=steps; L[n]=loop; T[n]=tps; K[n]=kas
        PP[n]=pair_pct; CP[n]=comm_pct
        if(kas>kmax) kmax=kas
        if(kmin==0 || kas<kmin) kmin=kas
    }
}
END{
    flush()
    for(i=1;i<=n;i++){
        sp=(kmin>0)? K[i]/kmin : 0
        printf "%-16s %6d %10d %8d %12.2f %12.2f %12.2f %6.1f%% %6.1f%% %8.2fx\n", M[i],P[i],A[i],S[i],L[i],T[i],K[i],PP[i],CP[i],sp
    }
    if(n>1){
        printf "\n(speedup = katom-step/s relative to the slowest run)\n"
        for(i=2;i<=n;i++) if(A[i]!=A[1] || S[i]!=S[1]){
            printf "NOTE: atoms x steps differ between runs — katom-step/s is still comparable,\n"
            printf "      but wall times are not. (1L vs 2L often use different nsteps on purpose.)\n"; break
        }
    }
}
' "${present[@]}"

if [ ${#missing[@]} -gt 0 ]; then
    for f in "${missing[@]}"; do echo "missing: $f (skipped)"; done
fi
