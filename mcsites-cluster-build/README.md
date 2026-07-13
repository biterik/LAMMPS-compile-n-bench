# LAMMPS + MC-SITES — cluster build & run kit

Build LAMMPS with the **full compile-n-bench package set** *plus* the **MC-SITES**
contribution (`compute sites/voronoi` + `fix mc/sites`), on the five MPCDF targets:
**cmmg, raven-gpu, raven-cpu, viper-cpu, viper-gpu**. Produces one uniquely-named
binary per machine, with matching SLURM submission scripts.

This kit is the MC-SITES counterpart of `mpcdf-lammps/` (the compile-n-bench repo):
same toolchains, same per-machine gotchas, same shared cmake preset — with the
MC-SITES source layered on and the binaries renamed so they never collide with your
existing `lmp_<machine>` benchmark builds.

---

## What was decided (and how to change it)

Two choices are baked in as defaults because they are the only *validated* path.
Both are overridable with environment variables.

1. **Source base = the thermoatoms GRACE fork + patches**, not upstream `stable`.
   MC-SITES was authored and passed its 43/43 test suite only against
   `thermoatoms/lammps @ 24da74cd` (base `patch_11Feb2026`). Its patches even edit
   fork-specific doc files, and on that fork `PKG_MC` *requires* `PKG_ML-PACE`
   (the fork's `MC/fix_atom_swap.cpp` `#include`s `pair_pace.h`). The fork build
   already compiles the **entire compile-n-bench package set** because it uses the
   same `cmake/lammps-packages-mpcdf.cmake` preset — so "all packages as before" is
   fully satisfied here.
   *Upstream-stable note:* the patches will **not** `git am` cleanly onto
   `stable_22Jul2025_update4` (patch 0003 edits doc files that differ there), and the
   code targets Feb 2026 / `develop`, not the older stable. If you want an
   upstream-stable build, it needs a separate port — say the word.

2. **GRACE / TensorFlow = OFF by default** (`GRACE_TF=off` → `-D NO_GRACE_TF=ON`).
   `fix mc/sites` works fully with EAM / MEAM / ACE / PACE with no TensorFlow env to
   set up — every build is one command. The GRACE energy-only fast path is opt-in
   (see [GRACE / TensorFlow](#grace--tensorflow-opt-in)).

---

## Binaries produced

The version tag `fork24da74` = `thermoatoms/lammps @ 24da74cd` (base `patch_11Feb2026`)
+ branch `feature/mc-sites` (5 patches). Every name encodes **fork/version + cluster**.

| Your target | Build script | Binary | Backend |
|---|---|---|---|
| cmmg | `build-mcsites-cmmg.sh` | `lmp_mcsites_fork24da74_cmmg` | Kokkos/OpenMP, Zen4 (AMD EPYC 9754) |
| raven-gpu | `build-mcsites-raven-gpu.sh` | `lmp_mcsites_fork24da74_raven_gpu` | Kokkos/CUDA, Ampere80 (A100) |
| raven-cpu | `build-mcsites-raven-cpu.sh` | `lmp_mcsites_fork24da74_raven_cpu` | Intel oneAPI + INTEL pkg (Xeon IceLake) |
| viper-cpu | `build-mcsites-viper-cpu.sh` | `lmp_mcsites_fork24da74_viper_cpu` | Kokkos/OpenMP, Zen4 (AMD EPYC 9554) |
| viper-gpu | `build-mcsites-viper-gpu.sh` | `lmp_mcsites_fork24da74_viper_gpu` | Kokkos/HIP, gfx942 APU (MI300A) |

> `raven-cpu` is **new** — there was no fork build for it before. It takes the proven
> compile-n-bench Intel recipe and adds MC-SITES + the fork's `ML-PACE` coupling.

---

## Contents

```
mcsites-cluster-build/
├── README.md                       ← this file
├── cmake/
│   └── lammps-packages-mpcdf.cmake ← shared full package set (copy of the compile-n-bench preset)
├── patches/
│   └── 0001..0005-*.patch          ← the MC-SITES contribution (git am-able on 24da74cd)
├── patch-mc-sites.sh               ← OPTIONAL: prepare the source tree without building
├── build-mcsites-cmmg.sh
├── build-mcsites-raven-gpu.sh
├── build-mcsites-raven-cpu.sh
├── build-mcsites-viper-cpu.sh
├── build-mcsites-viper-gpu.sh
├── submit-mcsites-cmmg.slurm
├── submit-mcsites-raven-gpu.slurm
├── submit-mcsites-raven-cpu.slurm
├── submit-mcsites-viper-cpu.slurm
└── submit-mcsites-viper-gpu.slurm
```

The whole folder is self-contained and rsync-able. The build scripts apply the
patches from `./patches/` automatically, so you never run `patch-mc-sites.sh` by hand
unless you want to.

---

## Cluster policy (unchanged from your setup)

- **Build on a login node** (git clone + package downloads need internet; compute
  nodes have none). **Run only inside SLURM jobs**, never on a login node.
- **Everything under PTMP, nothing in `$HOME`** — builds, clones, runs. The scripts
  build in the directory you launch them from, so `cd` into your PTMP work dir first.

---

## Quick start (per cluster)

Each build is **one command** run from your PTMP work dir on the matching login node.
It clones the fork, applies the MC-SITES patches onto `feature/mc-sites`, and compiles.

```bash
# 1. copy this kit to the cluster
rsync -av mcsites-cluster-build/  cmmg:/u/$USER/PTMP/mcsites/     # cmmg
#   raven:  raven:/ptmp/$USER/mcsites/     viper:  viper:/viper/ptmp/$USER/mcsites/

# 2. build (login node, from the PTMP work dir)
ssh cmmg
cd /u/$USER/PTMP/mcsites
./build-mcsites-cmmg.sh                 # -> lammps-mcsites/build-mcsites-cmmg/lmp_mcsites_fork24da74_cmmg
```

Analogous on the others: `./build-mcsites-raven-gpu.sh`, `./build-mcsites-raven-cpu.sh`,
`./build-mcsites-viper-cpu.sh`, `./build-mcsites-viper-gpu.sh`.

Common overrides (all optional):

```bash
JOBS=32 ./build-mcsites-cmmg.sh                 # parallel build jobs
SRC=/ptmp/$USER/shared/lammps-mcsites ./build-mcsites-raven-gpu.sh   # reuse one source tree
GRACE_TF=on PYTHON=~/tf-cpu/bin/python ./build-mcsites-viper-cpu.sh  # enable GRACE (see below)
```

If several clusters share a filesystem you can point them all at **one** `SRC` tree;
each writes to its own `build-mcsites-<cluster>/` subdir, so the binaries never clash.

---

## Package set

Identical to compile-n-bench (see the shared `cmake/lammps-packages-mpcdf.cmake` and
your `PACKAGES.md`) with MC-SITES layered on. The three packages MC-SITES needs are
forced ON in every build script: **`MC`** (the fix), **`VORONOI`** (the compute), and
**`ML-PACE`** (required by the fork's MC package). Per-machine exceptions are exactly
as before — `INTEL` only on raven-cpu, `PLUMED` only on cmmg, `ML-UF3` off on
viper-gpu (HIP), Kokkos backend per architecture.

---

## Running jobs

Put your input (`in.*`) and any potential files in a PTMP run dir, then submit the
matching script. Each defaults to `IN=in.mc_sites.langmuir` (the shipped example) and
auto-locates the binary; override with `IN=` and `LMP=`.

```bash
cp lammps-mcsites/examples/PACKAGES/mc_sites/in.mc_sites.langmuir .   # e.g. the example
sbatch submit-mcsites-cmmg.slurm
IN=in.h_charging sbatch submit-mcsites-viper-cpu.slurm
```

- **CPU (cmmg, viper-cpu, raven-cpu):** pure MPI, one rank per physical core. The MC
  bookkeeping in `fix mc/sites` is serial on rank 0; the per-trial **energy
  evaluations are MPI-parallel** across all ranks.
- **GPU (raven-gpu, viper-gpu):** MD/pair styles run on the device via `-sf kk`.
  `fix mc/sites` has **no Kokkos variant (v1)**, so it runs its create/delete
  bookkeeping host-side and calls the GPU-accelerated pair style for each trial
  energy — expected, and still uses the GPU for the heavy pair work. (viper-gpu also
  exports `HSA_XNACK=1`, required for the MI300A's unified memory.)

---

## Verify a build

```bash
# 1. the two new styles are present (run in a short interactive job / srun)
srun $LMP -h 2>/dev/null | grep -iE 'sites/voronoi|mc/sites'
#   -> compute ... sites/voronoi ...   and   fix ... mc/sites ...

# 2. the physics smoke test: octahedral lattice gas at mu=0 must give theta -> 0.5
cd lammps-mcsites/examples/PACKAGES/mc_sites
srun $LMP -in in.mc_sites.langmuir -log log.check        # watch the f_MC[6] column -> ~0.5

# 3. (optional) the full 43-check suite, driven against your binary
cd <this kit's sibling>/MC-SITES-LAMMPS   # your review-package checkout
python3 -m venv .venv && ./.venv/bin/pip install numpy pytest
LMP=/ptmp/$USER/mcsites/lammps-mcsites/build-mcsites-viper-cpu/lmp_mcsites_fork24da74_viper_cpu \
    ./.venv/bin/python -m pytest tests/ -v
```

---

## GRACE / TensorFlow (opt-in)

Only needed if you want the GRACE ML potentials + the energy-only MC fast path. Build
with `GRACE_TF=on` and point the build at a TensorFlow-providing Python:

| Cluster | TensorFlow flavour | Example |
|---|---|---|
| cmmg, viper-cpu, raven-cpu | CPU TF | `python -m venv ~/tf-cpu && ~/tf-cpu/bin/pip install tensorflow-cpu` → `GRACE_TF=on PYTHON=~/tf-cpu/bin/python ./build-mcsites-cmmg.sh` |
| raven-gpu | CUDA TF (ABI-match cuda/12.6) | `~/tf-gpu/bin/pip install tensorflow` → `GRACE_TF=on PYTHON=~/tf-gpu/bin/python ./build-mcsites-raven-gpu.sh` |
| viper-gpu | **tensorflow-rocm for gfx942 (experimental)** | `~/tf-rocm/bin/pip install tensorflow-rocm` → `GRACE_TF=on PYTHON=~/tf-rocm/bin/python ./build-mcsites-viper-gpu.sh` |

Notes: on raven-gpu, CUDA-TF vs nvcc/Kokkos runtime skew can break the link — fall
back to `GRACE_TF=off` and run GRACE from a separate plain-g++ + TF-CUDA build (your
`GRACE.md` "raven GRACE-TF fallback"). On viper-gpu, without tensorflow-rocm GRACE
cannot run on the APU GPU; the `GRACE_TF=off` binary still gives `pace/kk` (GPU ACE)
+ `grace/fs` (CPU). You may also set `PYMODULE=` to keep a compatible Python 3.10–3.12
module (e.g. `python-waterboa/2024.06`) loaded for TF discovery during configure.

---

## Troubleshooting

- **`git am` fails / half-applied** — a previous interrupted run can leave the tree
  mid-`am`. In the source tree: `git am --abort; git checkout develop;
  git branch -D feature/mc-sites`, then re-run the build script (it recreates the
  branch). The branch-exists check makes re-runs otherwise idempotent.
- **raven-cpu: KIM's Fortran fights oneAPI** — KIM is unused by `fix mc/sites`. If its
  download/build step fails, append `-D PKG_KIM=off` to the cmake block in
  `build-mcsites-raven-cpu.sh` and rebuild.
- **conda shadows the toolchain** — the CPU/GPU scripts strip conda from `PATH`
  automatically and warn on cmmg; if in doubt, `conda deactivate` before building.
- **`-l` / login-shell weirdness** — the scripts deliberately avoid `#!/bin/bash -l`
  and re-init the `module` function themselves (your non-interactive profile exits
  early otherwise).

---

## Provenance

MC-SITES: `feature/mc-sites`, 5 patches on `thermoatoms/lammps @ 24da74cd`
(LAMMPS "11 Feb 2026"), validated 43/43 on macOS (Apple clang 21, Open MPI), serial
+ MPI, SMALLBIG + BIGBIG. Build recipes derived from `mpcdf-lammps/build-lammps-*-fork.sh`
and `build-lammps-raven-cpu.sh`, sharing `cmake/lammps-packages-mpcdf.cmake`.
