# Implementation brief — Kokkos/GPU support for `compute sites/voronoi` + `fix mc/sites`

**Audience:** an autonomous coding agent (Claude Code) running **on Erik's Mac**, in the
LAMMPS checkout at `~/DEVEL/MC-SITES-LAMMPS/lammps` (branch `feature/mc-sites`). The Mac
has **no GPU**, so you implement and validate against a **Kokkos CPU build** (§7); the
final GPU runs on A100/MI300A are a separate sign-off (§8B). **Goal:** make hybrid MD/MC
with these two commands run correctly under `-sf kk` on **both** NVIDIA CUDA (A100 /
raven) and AMD HIP (MI300A / viper), so the MD *and* the per-trial energy evaluations
execute on the GPU.

Work autonomously: read the source, implement, build, run the acceptance tests, and
iterate until they pass. Only stop when all acceptance criteria are green or you are
genuinely blocked (and then write down exactly what and why).

---

## 1. Background & confirmed root cause

- Tree: `thermoatoms/lammps` @ `24da74cd`, branch `feature/mc-sites`.
- New files already present:
  `src/VORONOI/compute_sites_voronoi.{h,cpp}`, `src/MC/fix_mc_sites.{h,cpp}`.
- **Measured bug:** under `-sf kk`, `compute sites/voronoi` returns an *empty*
  catalogue (`Msites=0`), so `fix mc/sites` does zero trials (`natt=0`) — the MC half
  is a silent no-op. Reason (verified in source): the compute reads coordinates as
  `double **x = atom->x;` with **no device→host sync**, so in a Kokkos run the live
  coordinates (on the device) are never seen. The fix has the same problem plus it
  mutates atom arrays on the host without telling the device.
- On the MI300A (unified memory) this is partly masked; on the **A100 (no unified
  memory) it is a hard correctness failure**. Treat the **A100 as the strict oracle.**

## 2. Strategy — correctness first, minimal kernels

Do **not** rewrite the MC bookkeeping as GPU kernels. It is inherently serial and
cheap. The expensive part is the per-trial **full energy evaluation**, which already
runs on the device through `pair eam/alloy/kk` (and `pace/kk`, GRACE) *once the atom
data is coherent*. `compute sites/voronoi` is inherently host-only (Voro++ is a serial
CPU library) — the site-finding stays on the CPU; that is fine and cheap.

So the win is: **GPU-accelerated MD + GPU energy evaluations during MC**, not GPU
tessellation. Deliver that by adding Kokkos-aware `/kk` subclasses that keep the host
logic but insert the required device⇄host `sync`/`modified` calls.

## 3. Files to add (KOKKOS package)

- `src/KOKKOS/compute_sites_voronoi_kokkos.{h,cpp}` — `class ComputeSitesVoronoiKokkos : public ComputeSitesVoronoi`
- `src/KOKKOS/fix_mc_sites_kokkos.{h,cpp}` — `class FixMCSitesKokkos : public FixMCSites`
- Register with `ComputeStyle(sites/voronoi/kk,ComputeSitesVoronoiKokkos)` and
  `FixStyle(mc/sites/kk,FixMCSitesKokkos)` so `-sf kk` auto-selects them. These files
  compile only in KOKKOS builds, so `#include "atom_kokkos.h"` etc. is safe here.

## 4. Base-class refactor (no behavior change)

In `src/VORONOI/compute_sites_voronoi.*` and `src/MC/fix_mc_sites.*`, make the methods
the `/kk` subclass must wrap **`virtual` and `protected`** (they are likely `private`
now): at minimum the compute's local-build entry point, and in the fix the per-block
driver, `attempt_insertion()`, `attempt_deletion()`, and `energy_full()` (or whatever
they are named). Expose the members the subclass needs. The non-kk path must remain
behaviorally identical — verify by re-running the CPU test suite after this step.

## 5. Exact Kokkos idioms

```cpp
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "kokkos.h"
// ...
AtomKokkos *atomKK = (AtomKokkos *) atom;

// before any HOST read of positions/type/mask/tag (compute + fix):
atomKK->sync(Host, X_MASK | TYPE_MASK | MASK_MASK | TAG_MASK);

// after the fix MODIFIES atoms on the host (insert / delete / move / velocity):
atomKK->modified(Host, X_MASK | V_MASK | TYPE_MASK | MASK_MASK | TAG_MASK | IMAGE_MASK);
```

For `energy_full()` under Kokkos, mirror the host logic of `fix gcmc`'s full-energy
path but ensure `comm->borders()`, `neighbor->build()` and `force->pair->compute()`
run through the Kokkos path with coherent data: after changing `nlocal`/`natoms`, call
`atomKK->modified(Host, ALL_MASK)` so the device rebuilds its atom set before the pair
compute, then `sync(Host, ...)` back anything you must read on the host. Study
`src/KOKKOS/verlet_kokkos.cpp` and an existing atom-mutating Kokkos fix for the exact
call ordering; do not guess — read how they interleave `sync`/`modified` with
`comm`/`neighbor`.

## 6. Watch-outs (read before coding)

- **A100 has no unified memory** → every missing `sync`/`modified` is a *correctness*
  bug, not just a perf one. Do all correctness validation on raven/A100. Do **not**
  use the MI300A as the oracle — its unified memory hides missing syncs.
- Per trial, `energy_full` does a neighbor rebuild and (on A100) a host⇄device atom
  transfer. The first correct version **will be slow** on A100 for large `Ntrials`.
  That is acceptable for this task; note it and leave perf for a later stage.
- Preserve the **GRACE energy-only** hook (`pair->extract("compute_energy_only")`).
- Keep **SMALLBIG and BIGBIG** working; keep the non-kk path unchanged.
- The compute's local-array output format (columns) must be identical to the base so
  the fix reads it unchanged.

## 7. Build & iterate on the Mac (Kokkos CPU)

You have no GPU locally, so validate against a **Kokkos Serial/OpenMP** build. This
compiles the exact `/kk` classes, registers the `/kk` styles, and lets `-sf kk` select
them — enough to prove the code builds, the styles are picked up, and the physics is
correct. From `~/DEVEL/MC-SITES-LAMMPS/lammps`:

```bash
cmake -S cmake -B build-kk \
  -D PKG_MC=on -D PKG_VORONOI=on -D DOWNLOAD_VORO=on \
  -D PKG_ML-PACE=on -D NO_GRACE_TF=1 \
  -D PKG_KOKKOS=on -D Kokkos_ENABLE_SERIAL=on \
  -D BUILD_MPI=on -D CMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-kk -j          # rebuild incrementally after each edit
```

**Do NOT reuse the existing `build/` directory** — it was configured with
`PKG_KOKKOS=OFF`, so it silently compiles *none* of the `src/KOKKOS/*.cpp` files and
`-sf kk` on the result errors with "KOKKOS package not enabled." Use a **fresh
`build-kk`**, and before trusting any `-sf kk` test, assert Kokkos is actually on:
`grep -E '^PKG_KOKKOS' build-kk/CMakeCache.txt` must print `PKG_KOKKOS:BOOL=ON`, and
`build-kk/CMakeFiles` must contain `*sites_voronoi_kokkos*` / `*mc_sites_kokkos*`
object files. If those are missing, the `/kk` code was not compiled and no test is
meaningful.

Run the Kokkos path with `-k on -sf kk`, e.g.
`build-kk/lmp -k on t 1 -sf kk -in examples/PACKAGES/mc_sites/in.mc_sites.langmuir`.

**Limitation you must design around:** a Kokkos *Serial/OpenMP* build shares host
memory, so it will **not** expose a missing `sync`/`modified` — the very bug class we
are fixing. So: add every sync/modify from §5 **by construction** (do not wait for a
local failure to reveal them), and treat the Mac build as validating *compile +
style-selection + physics* only. The **A100** (separate device memory) is the real
verifier of sync correctness, and it runs on the cluster (§8B) — not on the Mac. The
GPU cluster build scripts (`mcsites-cluster-build/build-mcsites-raven-gpu.sh` for the
A100, `build-mcsites-viper-gpu.sh` for the MI300A) are for that sign-off step.

## 8. Acceptance criteria

### 8A. On the Mac (Kokkos CPU) — you run these; all must pass before you finish

1. **Builds clean** with the §7 cmake; the `/kk` styles are registered.
2. **Styles selected under kk + physics right.**
   `build-kk/lmp -k on t 1 -sf kk -in examples/PACKAGES/mc_sites/in.mc_sites.langmuir`
   uses `compute sites/voronoi/kk` + `fix mc/sites/kk` (verify in the log), and
   `f_MC[6]` converges to ≈0.5 (matches the committed reference logs). `Msites` and the
   attempt counter are **> 0** (i.e. no empty-catalogue no-op).
3. **Full suite green on the kk binary.**
   `LMP_BIN=<abs>/build-kk/lmp ./.venv/bin/python -m pytest ../tests/ -v`
   (tests live in `~/DEVEL/MC-SITES-LAMMPS/tests`; run Claude Code from
   `~/DEVEL/MC-SITES-LAMMPS` so both `lammps/` and `tests/` are in reach).
4. **No CPU regression.** A plain non-kk build still passes the full suite unchanged.
5. **Every sync/modify from §5 is present** at each host read / mutation point (self
   review — the CPU build can't catch a missing one).

### 8B. GPU sign-off on the cluster (A100 raven + MI300A viper) — Erik runs these

The Mac cannot reproduce separate device memory, so these are the real verifier and
are done on the cluster. Build with the GPU scripts, then in a GPU job:

6. **Catalogue non-empty under real device memory.** `in.mcmd_bench_nih` with `-sf kk`
   → `BENCH_RESULT` shows `Msites>0`, `natt>0` (currently 0). *This is the criterion the
   Mac cannot check — a missing sync only shows up here.*
7. **GPU vs CPU consistency.** Same seed: final `conc N/M` and acceptance agree with a
   host run within stochastic tolerance, on both A100 and MI300A.

Finish when all of **8A** passes on the Mac; generate the patches + `PORT-NOTES.md`, and
print the exact build+run commands for Erik to execute **8B**. (Only run 8B yourself if
ssh access to the cluster has been explicitly enabled for you — by default it is not.)

## 9. Git workflow & deliverable

- Branch: `git checkout -b feature/mc-sites-kokkos` off `feature/mc-sites`.
- Small commits: (1) base refactor to virtual/protected — no behavior change; (2)
  compute `/kk` sync; (3) fix `/kk` sync/modify + energy_full; (4) tests/examples; (5)
  docs page updates noting the `/kk` styles and the A100/MI300A results.
- When green, `git format-patch` the new commits so they append to the existing
  `000x-*.patch` series, and write `PORT-NOTES.md`: what was synced where, and the
  measured A100 + MI300A numbers (Msites, θ, per-step, per-trial).

## 10. Definition of done

**Your (agent, on the Mac) done:** all of **8A** passes, the non-kk CPU suite is
unregressed, patches are generated off `feature/mc-sites-kokkos`, `PORT-NOTES.md` is
written, and the exact **8B** build+run commands are printed for Erik. If an 8A
criterion cannot be met, stop and document the exact failure, file/line, and what you
tried. **Final done (with Erik):** 8B passes on A100 and MI300A.
