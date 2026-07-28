# MD + MC benchmark — EAM Ni/H on one GPU

Estimates how long a **combination of MC moves and MD steps** takes on a single GPU,
for the hydrogen-charging-of-nickel workflow (`fix mc/sites` + EAM Ni/H).

## Files

| File | Purpose |
|---|---|
| `in.mcmd_bench_nih` | the LAMMPS input: fcc Ni + EAM Ni/H, NPT MD, `fix mc/sites` GCMC on Voronoi octahedral sites. Runs a pure-MD phase then an MD+MC phase, both timed. |
| `submit-mcmd-bench-viper-gpu.slurm` | run on one MI300A (`-sf kk`), then auto-print a timing breakdown + estimator |
| `submit-mcmd-bench-viper-cpu.slurm` | same input on a full CPU node — physics cross-check + CPU reference |
| `in.mcmd_rankscan_nih` | rank-scan variant of the bench input: identical recipe + `fix mc/sites ... check yes` (per-block consistency verification, patch 0010) |
| `submit-mcmd-rankscan-cmmg.slurm` | one full cmmg node, runs the scan input at np = 1…256 sequentially (same seed), one `RANKSCAN` summary line per np |

## What it measures and why it's split in two phases

`fix mc/sites` does a **full energy evaluation per trial** (required for EAM/ACE/GRACE).
So a production run costs roughly

```
wall  ≈  Nmd · t_step   +   (Nmd/Nevery) · Ntrials · t_trial
         └── MD ────┘        └──────── MC ────────────┘
```

To get `t_step` and `t_trial` cleanly, the input runs **Phase A** (pure MD, `nmd`
steps) and **Phase B** (MD + MC, `nmd` steps). The submit script reads the two
"Loop time" values: `t_step = A/nmd`, and the MC cost is `B − A` spread over the
recorded number of trials → `t_trial`. It then prints a ready-to-use estimator.

## Status (updated 2026-07-27)

The Kokkos port (patches 0006-0009) works on the MI300A: under `-sf kk` the
suffix selects `fix mc/sites/kk` + `compute sites/voronoi/kk`, the catalogue is
populated (`Msites>0`, `natt>0`) and accept/reject behaves sensibly.

**Historical note:** the original "v1: Msites=0 under -sf kk" measurement was
confounded by an input-script bug — `reset_timestep 0` placed AFTER the
`fix mc/sites` definition disarmed the fix entirely (it arms its first MC block
at creation via next_reneighbor, which reset_timestep does not remap), giving
Msites=0/natt=0 in ANY mode. `in.mcmd_bench_nih` now resets the timestep BEFORE
defining the fix. Never `reset_timestep` after defining gcmc-like fixes.

**Path rule (MPCDF docs):** Viper-CPU and Viper-GPU are separate clusters;
compute nodes cannot access the other system's filesystems. Both submit scripts
therefore default to side-local paths (CPU: /viper/ptmp1/..., GPU:
/viper/ptmp2/...). Keep binary, input, and potential on the submitting side.

**Open item (2026-07-27, updated 2026-07-28):** at mu=-2.0 the 128-rank CPU run
accepts ~0.28 while 1-rank runs of the GPU binary (kk, kk+comm host, plain host)
agree at ~0.004 — note these are two DIFFERENT binaries. The viper rank scan
(CPU binary, np=1…128) is queued. Independently, a 2026-07-28 container scan
(gcc/x86, fork+9 patches, same bench input, NiAlH_jea potential) found NO
np-dependence at np = 1…64 — including sub-cutoff subdomains and ~14 atoms/rank —
in both the plateau (acc≈0.3) and sensitive (acc≈0.02) regimes, which shifts
suspicion toward a binary/build difference rather than MC logic. Patch 0010 adds
`check yes` to fix mc/sites (per-block atom-count / species / energy-bookkeeping
verification): use `in.mcmd_rankscan_nih` + `submit-mcmd-rankscan-cmmg.slurm`
(cmmg, independent machine+compiler) to decide. Do not treat high-rank mc/sites
results as production-grade until this is closed.

## Run it

```bash
# from your viper PTMP work dir, after building viper-gpu (and viper-cpu for the check)
cp <repo>/mcsites-cluster-build/bench/in.mcmd_bench_nih .
cp <repo>/mcsites-cluster-build/bench/submit-mcmd-bench-viper-gpu.slurm .
cp <repo>/mcsites-cluster-build/bench/submit-mcmd-bench-viper-cpu.slurm .

sbatch submit-mcmd-bench-viper-gpu.slurm     # defaults baked in (ptmp2 side-local)
sbatch submit-mcmd-bench-viper-cpu.slurm     # defaults baked in (ptmp1 side-local)
# NOTE: viper does not forward the submit environment - override only via
#   sbatch --export=ALL,VAR=value ...
```

`POTDIR` must contain `ni_h_rcut4.90_rcut2.eam.alloy` (the Korbmacher potential).
The timing table prints at the end of the `.out` file.

## Knobs (env vars on the submit line, or `-var` on the input)

| Var | Default | Meaning |
|---|---|---|
| `NCELL` | 16 | fcc cells/side → `4·N³` Ni atoms (10≈4k, 16≈16k, 24≈55k, 32≈131k) |
| `NMD` | 2000 | timed MD steps per phase |
| `NEVERY` | 20 | MC block every this many MD steps |
| `NTRIALS` | 400 | trial flips per MC block |
| `MU` | −2.4 | lattice-gas chemical potential (eV), ~plateau |
| `TEMP` | 300 | temperature (K) |

The default MD+MC phase does `NMD/NEVERY = 100` blocks × 400 = **40 000 full EAM
evals** — expect the MC phase to dominate. For a quick first number use e.g.
`NMD=500 NTRIALS=100 NCELL=10` (fits the 15-min `apudev` partition:
`sbatch -p apudev -t 00:15:00 …`). Scale `NCELL` up to see how the GPU per-eval cost
amortizes with system size — small systems are launch-latency bound.

## Reading the output

```
per MD step            : 0.4 ms
per MC trial (EAM eval): 0.6 ms
MC/MD cost ratio       : 1.5x
ESTIMATOR: wall ~= Nmd*4.0e-04 s + (Nblk*Ntr)*6.0e-04 s
```

The two numbers (`t_step`, `t_trial`) are what you plug into your own production
recipe. The `MC/MD cost ratio` tells you at a glance how much the MC blocks add:
with `Nevery=20, Ntrials=400` you do 400 evals per 20 MD steps, so MC will dominate
wall time unless you lower `Ntrials` or raise `Nevery`.
