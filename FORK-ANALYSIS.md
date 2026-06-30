# thermoatoms/lammps — base version & rebase assessment

Analysis date: 2026-06-30. Question (goal a): build a LAMMPS with the full MPCDF
package set **+ ACE (fast MC/MD) + GRACE + efficient MC**, ideally on the most
recent *stable* LAMMPS. (goal b — generic interstitial MC for EAM/MEAM/ACE/GRACE
— is to be developed, see end.)

## What the fork is based on

| | value |
|---|---|
| Repo / branch | `thermoatoms/lammps` @ `develop` |
| `version.h` | `"11 Feb 2026"` / `"MCnoforce-localE"` |
| Merge-base with upstream | `b75dfcc93` = **`patch_11Feb2026`** + 99 (commit "apip_local_average") |
| Commits on top of base | **25** (≈half are merge commits) |
| Companion repo | PACE evaluator pulled from `thermoatoms/lammps-user-pace` (`main`) via patched `ML-PACE.cmake` |

**Key point:** `patch_11Feb2026` is on LAMMPS' **monthly patch line and is ~7 months
NEWER than the most recent *stable*** (`stable_22Jul2025_update4`). Current upstream
develop is `patch_30Mar2026`. So the fork is *ahead* of stable, not behind it —
"rebase onto current stable" means moving the code **backwards**.

## What the 25 commits add (the value)

Efficient MC / MD on `fix atom/swap` (MC package):
- `noforce` keyword — skip force accumulation during MC energy evals.
- `localE` keyword — local coordination-shell energy (PACE/ACE only).
- multi-swap, hybrid/scaled `pace+pace`, time-variable `mu` and `T`, `adapt`,
  and a `split_cache` type-invariant trial optimization for alchemical TI.

GRACE (entirely added by the fork — absent from both stable and the Feb2026 base):
- full `pair_style grace` family: `pair_grace`, `pair_grace_fs`,
  `pair_grace_1layer_chunk`, `pair_grace_2layer_chunk`,
  `pair_grace_2layer_parallel`, `utils_grace` (~5,000 lines, all **new files**).

Other:
- `pair_eam.cpp` (+27) — a local single-atom-energy hook in EAM (directly
  relevant to goal b).
- small touches to `pair.cpp/.h`, `pair_hybrid_scaled.h`, `pair_pace.cpp/.h`.

## GRACE build dependency (you asked for ACE + GRACE fully)

`pair_grace.cpp` `#include <tensorflow/c/c_api.h>` → **GRACE needs the TensorFlow
C library**. The patched `ML-PACE.cmake` auto-handles it:
- discovers TF from a Python `tensorflow` install, else **downloads libtensorflow
  2.18** + `cppflow` at configure time (needs login-node internet, like your other
  external packages).
- the Linux URL it ships is the **GPU** build (`libtensorflow-gpu-linux-x86_64`);
  on a CPU node (cmmg) point `-D TF_LIB_FILE=` at a CPU `libtensorflow_cc.so.2`.
- `-D NO_GRACE_TF=ON` disables TF GRACE, leaving only the TF-free `pair_grace_fs`
  ("FS" explicit) variant — useful fallback if TF linking is painful.
- The fast ACE evaluator + GRACE kernels come from the `thermoatoms/lammps-user-pace`
  download, so that fork is part of the toolchain (pin it too for reproducibility).

## Rebase effort

The fork touches **10 existing files** (small) and adds **13 new files** (the GRACE
family — zero conflict risk, they just drop in). The only meaningful merge point is
`src/MC/fix_atom_swap.cpp` (fork rewrites it, +654 lines).

**Option 1 — forward-port onto newest upstream (`patch_30Mar2026`/develop): trivial.**
Since the fork base (Feb2026) is only ~6 weeks behind, upstream drift in fork-touched
files since the base is tiny: `fix_atom_swap.cpp` 3 lines, `pair.cpp` (fork adds just
1 line), everything else unchanged. ≈1 hour. Keeps newest upstream fixes. **Downside:
it's a patch release, not a "stable" tag.**

**Option 2 — backport onto `stable_22Jul2025_update4` (the most-recent *stable*):
more work, less benefit.** Conflict surface is essentially **one file**: upstream
changed `fix_atom_swap.cpp` by +106/−15 between stable and the Feb2026 base, to be
reconciled against the fork's rewrite; all other modified files are unchanged
stable→Feb2026, and GRACE is new files. *But*: (i) you'd lose ~7 months of upstream
fixes; (ii) the fork base sits on top of APIP (D. Immel) and other post-stable work —
the new GRACE/pace/utils code must be verified not to use post-Jul2025 APIs. Realistic
estimate: **half a day to a day** of merge + build-test, best done by someone who knows
the code (Sarath), for a result that is *older* than what you'd get from Option 1.

## Recommendation

For goal a), **don't backport to the Jul2025 stable** — it's the most effort for the
least benefit. Either:
- **(simplest)** build from the fork pinned at its current base (Feb2026) — recent,
  GRACE works, no merge work; or
- **forward-port to `patch_30Mar2026`** for the newest code (~1 h); or
- if you specifically need the maintained **stable line**, ask Sarath to rebase onto
  the **next** stable when it's cut (the `fix_atom_swap.cpp` reconciliation is the only
  real work) rather than back onto Jul2025.

Then adapt your existing `build-lammps-cmmg.sh`: clone `thermoatoms/lammps` (pinned
commit) instead of `lammps/lammps stable`; keep `PKG_MC` + `PKG_ML-PACE` on; add the
TensorFlow handling for GRACE (or `NO_GRACE_TF` + FS-only to start). Everything else
in the cmmg recipe (gcc/13 + impi, Kokkos/OpenMP ZEN4) is unchanged.

## For goal b (interstitial MC, to be developed)

The fork is the scaffolding: the `localE` machinery in `fix_atom_swap.cpp` plus the
per-pair-style single-atom-energy hook added to `pair_eam.cpp` are exactly the two
pieces a generic, pair-style-agnostic interstitial-insertion MC (EAM/MEAM/ACE/GRACE)
would generalize. Note the efficient `localE` path is currently PACE/ACE-only; the EAM
hook shows the pattern to extend to MEAM and GRACE.
