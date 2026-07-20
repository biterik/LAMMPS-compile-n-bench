# MD + MC benchmark — EAM Ni/H on one GPU

Estimates how long a **combination of MC moves and MD steps** takes on a single GPU,
for the hydrogen-charging-of-nickel workflow (`fix mc/sites` + EAM Ni/H).

## Files

| File | Purpose |
|---|---|
| `in.mcmd_bench_nih` | the LAMMPS input: fcc Ni + EAM Ni/H, NPT MD, `fix mc/sites` GCMC on Voronoi octahedral sites. Runs a pure-MD phase then an MD+MC phase, both timed. |
| `submit-mcmd-bench-viper-gpu.slurm` | run on one MI300A (`-sf kk`), then auto-print a timing breakdown + estimator |
| `submit-mcmd-bench-viper-cpu.slurm` | same input on a full CPU node — physics cross-check + CPU reference |

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

## IMPORTANT — the MC half does NOT run under `-sf kk` (measured, v1)

Confirmed on Viper (MI300A): with `-sf kk`, `compute sites/voronoi` returns an
**empty catalogue** (`Msites=0`, `natt=0`) — the MC half is a silent no-op and the
"MD+MC" phase just measures MD again. Root cause: the compute reads coordinates via
`atom->x` on the host with no Kokkos device sync, so under a Kokkos run it never sees
the live (device) positions. This is the documented v1 limitation ("Not supported:
Kokkos/GPU/INTEL suffix versions").

Consequence: **the GPU accelerates only the MD.** Since each MC trial is a full
energy evaluation and those dominate a hydrogen-charging run, the MC bottleneck runs
on the host regardless — so the meaningful **MD+MC benchmark is the CPU one**
(`submit-mcmd-bench-viper-cpu.slurm`, full 128-core node, fully supported). Use the
GPU number only for the MD portion.

Two ways to see real MC numbers:
- **CPU benchmark** (recommended): build viper-cpu, run `submit-mcmd-bench-viper-cpu.slurm`.
- **Host mode on the existing GPU binary** (quick proof it's the Kokkos path, not the
  input): `sbatch --export=ALL,ACC=' ',NCELL=8,NMD=200,NTRIALS=50 submit-mcmd-bench-viper-gpu.slurm`
  — with `ACC` empty the binary runs plain host EAM and you should see `Msites>0`, `natt>0`.

## Run it

```bash
# from your viper PTMP work dir, after building viper-gpu (and viper-cpu for the check)
cp <repo>/mcsites-cluster-build/bench/in.mcmd_bench_nih .
cp <repo>/mcsites-cluster-build/bench/submit-mcmd-bench-viper-gpu.slurm .
cp <repo>/mcsites-cluster-build/bench/submit-mcmd-bench-viper-cpu.slurm .

POTDIR=/viper/ptmp/$USER/potentials sbatch submit-mcmd-bench-viper-gpu.slurm
POTDIR=/viper/ptmp/$USER/potentials sbatch submit-mcmd-bench-viper-cpu.slurm   # cross-check
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
