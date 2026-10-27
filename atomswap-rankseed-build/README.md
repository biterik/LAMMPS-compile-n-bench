# atomswap-rankseed-build — LAMMPS `fix atom/swap` per-rank RNG patch + cmmg build kit

**What is wrong.** `src/MC/fix_atom_swap.cpp` (every version from stable_22Jul2025 to
stable_30Sep2026 and `develop`, checked 2026-10-19) constructs its two Park–Miller
generators with the *same* seed on *every* MPI rank:

```c++
  // random number generator, same for all procs
  random_equal = new RanPark(lmp, seed);
  // random number generator, not the same for all procs      <-- the comment says one thing,
  random_unequal = new RanPark(lmp, seed);                    //     the code does another
```

In semi-grand mode (`attempt_semi_grand()`) the atom is picked with `random_equal`
(identical on all ranks, correct), but the *new type* and the *Metropolis acceptance
number* are drawn from `random_unequal` on the rank that owns the picked atom only.
Every rank therefore walks through the **same** sequence of acceptance numbers, each at
its own pace (one draw pair per attempt it owns). Since all ranks own atoms at about the
same rate, they reach a small number in that sequence at about the same time and
**~nprocs swaps are accepted within a few MC calls** — a burst — then nothing for a long
while. Long-run averages are unbiased only in the limit; the variance is inflated
roughly nprocs-fold, and any restart that replays the stream (our chunked templates,
which `unfix` before `write_restart` and recreate the fix with a literal seed) turns the
burst structure into a seed-dependent bias. `restart()` has the same defect: the single
stored rank-0 state is handed to **all** ranks.

Observed (Cu-AT-DISLOCS thread 08, 2026-10-18/19; Onat 2013 EAM, 568 280 atoms, 256
ranks, dmu = 0.8462 eV, bulk equilibrium ~406 Cu): continuous run 406 → 827 Cu within
0.05 ns, per-chunk reseeding gives saw-tooth jumps of +250..+500, same-seed chunks give
seed-dependent plateaus (~1000 vs ~1300).

`fix gcmc` has the identical constructor (and in stable_22Jul2025 the identical
rank-0-only restart). In `develop`, `fix gcmc` already *stores* per-rank
`random_unequal` states in the restart file (`MPI_Gather`, `nlocalvals`) but still seeds
all ranks identically in the constructor, so the restart machinery there is the pattern
this patch follows, and the constructor bug is shared.

**What the patches do.**

| patch | file | change |
|---|---|---|
| `0001-fix-atom-swap-per-rank-random_unequal.patch` | `src/MC/fix_atom_swap.cpp` | constructor: give `random_unequal` a distinct, well-mixed state per rank; `write_restart`: gather and store one state per rank (same layout as `develop`'s fix gcmc: `nlocalvals` after the global values); `restart()`: restore per-rank state when nprocs is unchanged, otherwise derive distinct per-rank states from the stored rank-0 value — never the same value on every rank. |
| `0002-fix-gcmc-per-rank-random_unequal.patch` | `src/MC/fix_gcmc.cpp` | constructor + restart: same per-rank derivation (minimal; stable_22Jul2025 only — `develop` needs only the constructor line). Not exercised by our tests. |

Per-rank state is derived with `RanPark::reset(int seed, double *coord)` — the Jenkins
hash LAMMPS already uses for per-atom seeding — fed with `(seed, rank, 0, 0)`.
A plain `seed + comm->me` is **not** sufficient for Park–Miller: seeds `s` and `s+1`
give first draws that differ by only 16807/2^31 ≈ 8e-6 and second draws in an
arithmetic progression (0.23, 0.36, 0.50, …), so the ranks would still start in step.
The hashed seeds for seed 857474 are all distinct over 256 ranks with first draws
0.30, 0.18, 0.14, 0.56, … `random_equal` is untouched.

`patch -p1` applies cleanly to stable_22Jul2025_update{3,4,6}, stable_30Sep2026 and
`develop` (0001; checked 2026-10-19). 0002 applies to stable_22Jul2025 only. Both
patched files compile against the update3 headers, and a minimal patched build (MANYBODY + MC, gcc 11, mpich) ran the A/B
input on 4 ranks on 2026-10-19: write_restart with the fix defined, restart() on the same rank count (per-rank states) and on
a different rank count (hashed fallback) all clean.

**Restart-file compatibility.** A restart written by the *patched* fix carries
`7 + nprocs` doubles instead of 6. The patched `restart()` falls back to the hashed
rank-0 state whenever the stored `nlocalvals` does not equal the current nprocs — which is
also what happens when it reads a restart written by the *stock* fix (the 7th value is
then whatever follows the fix block; it will not equal nprocs except by accident). Mixing
binaries across a restart is not recommended either way; our thread-08 templates
`unfix` before `write_restart`, so no fix state is stored there.

**Build (cmmg).** `build-atomswap-rankseed-cmmg.sh` — run on a login node, from a
`/cmmc/ptmp/$USER/...` work dir. It clones the exact tag of the central module
(`stable_22Jul2025_update3` = `lammps/250722`, "22 Jul 2025 - Update 3"; NB the `stable`
branch itself moved to 30 Sep 2026 on 2026-09-30, so `build-lammps-cmmg.sh`'s default
clone no longer matches the module), builds the stock binary first, applies the patches,
rebuilds incrementally, and leaves two binaries side by side:

```
<work-dir>/bin/lmp_cmmg_22Jul2025u3_stock
<work-dir>/bin/lmp_cmmg_22Jul2025u3_atomswap-rankseed
<work-dir>/bin/BUILD-RECORD.txt        # tag, modules, md5sums, applied diff
```

Package set, toolchain and flags are those of `build-lammps-cmmg.sh` (gcc/13 +
impi/2021.16, full compile-n-bench package set incl. MANYBODY, MC, MEAM, ML-PACE, KIM,
PLUMED, KOKKOS/OpenMP ZEN4). Difference to the module: no INTEL package (the module is an
Intel oneAPI build; fix atom/swap refuses `/intel` pair styles since update4 anyway), plus
KOKKOS. Expect ~1 h at `JOBS=16` on a login node because of the external packages.

**Verification.** The A/B test (bulk 8788-atom fcc Ni–Cu, Onat EAM, dmu 0.8462 eV,
`fix atom/swap 20 100 … semi-grand`, 1 / 64 / 256 ranks, stock vs patched vs module) lives
with the simulation project: `Cu-AT-DISLOCS-Ni-EAM-MEAM-ACE/08_MC-SEED-RESTART-TEST/verify-rankseed/`.
Expected: the stock binary shows jumps of O(nprocs) Cu within a few hundred MD steps on
64 and 256 ranks and none on 1 rank; the patched binary shows no rank dependence.

**Fork line.** The thermoatoms fork (`mcsites-cluster-build`, base 11 Feb 2026 "MCnoforce-localE") has the
same bug; the same fix, ported to its `fix_atom_swap.cpp` (8 global restart values incl. adaptive mu), is
`../mcsites-cluster-build/patches/0013-…patch` → mc-sites v1.3 (`lmp_mcsites_v13_cmmg`).

**Upstream.** Filed as https://github.com/lammps/lammps/issues/5269 (2026-10-27). Issue text drafted in the same thread folder
(`ISSUE-DRAFT_lammps_fix-atom-swap_per-rank-rng.md`). Same bug class as our own
`mcsites-cluster-build/patches/0005-fix-mc-sites-seed-velocity-RNG-per-rank…` (2026-07).
