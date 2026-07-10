# Hardware selection guide — GRACE (and PACE/ACE) on GPUs

Which accelerator to run — and to **buy** — for GRACE and PACE machine-learning
interatomic potentials in LAMMPS, based on the measured benchmarks in this repo
(`GRACE.md`, `README.md`) plus device specs and rough mid-2026 prices. It answers
three questions: **how fast**, **how big a system fits**, and **what it costs** —
and flags the software (TensorFlow) caveats that move those numbers.

> **Scope & honesty note.** Throughput numbers are *measured* (fcc-Cu, TF-CUDA /
> TF-ROCm, this repo's scripts). **Atom-capacity numbers are estimates** — only one
> memory point is actually measured (16,384-atom 2L fits a 40 GB A100); §3 gives a
> recipe to measure the real ceiling. **Prices are rough** (mid-2026, single-unit,
> USD, vendor/volume-dependent) and shown only to size relative value, not to quote.

---

## TL;DR — what to buy

- **Running GRACE (graph-ACE, TensorFlow) and want the fastest single device:**
  **NVIDIA B200** — fastest here on both 1L and 2L, and the most memory (192 GB) for
  the largest systems. Highest cost.
- **Best price/performance for GRACE:** **NVIDIA RTX PRO 6000 Blackwell (96 GB).**
  A workstation card at roughly a third of a datacenter GPU's price that **matches
  the A100 on 1L and beats it (~1.3×) on 2L**. The value pick for workstations,
  small clusters, and cost-sensitive fleets.
- **Mature datacenter middle ground:** **NVIDIA H200** — ~0.8× a B200, 141 GB, and
  proven Hopper software. Sensible if B200 supply/price is prohibitive.
- **Running PACE/ACE (Kokkos-HIP/CUDA), not GRACE:** **AMD MI300A** is the *fastest
  single device* for PACE in this project (509 vs 360 katom-step/s for an A100) and
  has 128 GB unified memory. **But it is a poor GRACE device today** (TF-ROCm is
  ~13× slower than an A100) — see §4. Buy MI300A for ACE, not for GRACE.
- **A100:** legacy. Fine if you already own it; for a *new* purchase the RTX PRO
  6000 Blackwell beats it on performance, memory, and (often) price.
- **CPU-only for GRACE:** avoid except for validation/small jobs — it's 5–15× slower
  than a single GPU and carries the CPU-TensorFlow tax (§4).

---

## 1. Measured throughput (GRACE, fcc-Cu, 16,384 atoms)

Single device (GPU) or one full node (CPU). Higher `katom-step/s` = faster. "×A100"
is relative to the raven A100 40 GB; "×cmmg" is relative to a full 256-core EPYC node
(the CPU reference). 1L = `GRACE-1L-SMAX-OMAT-large`, 2L = `GRACE-2L-SMAX-OMAT-medium`.

| Device | Arch | 1L katom-step/s | ×A100 | ×cmmg | 2L katom-step/s | ×A100 | ×cmmg |
|---|---|--:|--:|--:|--:|--:|--:|
| **B200** | Blackwell (sm_100) | **147.00** | 2.6× | 14.7× | **25.28** | 2.0× | 7.0× |
| **H200** | Hopper (sm_90) | 114.26 | 2.0× | 11.4× | 19.80 | 1.5× | 5.5× |
| **A100 40 GB** | Ampere (sm_80) | 57.53 | 1.00× | 5.7× | 12.82 | 1.00× | 3.6× |
| **RTX PRO 6000** | Blackwell (sm_100) | 54.59 | 0.95× | 5.4× | 16.86 | 1.3× | 4.7× |
| **MI300A** | CDNA3 (gfx942) | 4.33 | 0.08× | 0.43× | 0.94 | 0.07× | 0.26× |
| cmmg (CPU) | 256-core EPYC 9754 | 10.03 | 0.17× | 1.00× | 3.61 | 0.28× | 1.00× |
| viper-cpu (CPU) | 128-core EPYC Genoa | _pending_ | | | _pending_ | | |

Notes:
- **All GPU runs are >97 % GPU-bound** (Pair% 96–100 %), so throughput reflects the
  GPU + TF kernels, not the host. The DAIS runs used different host thread counts
  (B200 12, H200 16, RTX 8) purely from `--cpus-per-task`; that does **not** explain
  the ranking.
- **The 1L vs 2L crossover for the RTX PRO 6000 is the interesting result.** On the
  light 1L model it ties the A100; on the heavier, message-passing 2L model it pulls
  ~1.3× *ahead* of the A100 — newer Blackwell tensor cores + slightly higher memory
  bandwidth (GDDR7) matter more as the model gets heavier.
- **MI300A is not slow silicon** — it's the TF-ROCm software stack (§4). The *same*
  chip is the project's fastest PACE/ACE device.

---

## 2. Device specs & rough prices

Vendor peak specs for orientation; real GRACE performance is in §1. Prices are
**approximate mid-2026 single-unit street/list** (USD) and move with supply — the
RTX PRO 6000, for one, jumped ~55 % on a GDDR7 shortage. Datacenter parts (H200,
B200, MI300A) are usually bought as 4–8-GPU nodes, not singly.

| Device | Memory | Mem type · bandwidth (peak) | ~Price/unit (mid-2026) | Buy form |
|---|--:|---|--:|---|
| **B200** | 192 GB | HBM3e · ~8 TB/s | **~$35–45k** | 8-GPU HGX/DGX (~$0.4–0.5M) |
| **H200** | 141 GB | HBM3e · ~4.8 TB/s | **~$30–40k** | 8-GPU HGX (~$0.3M); NVL also |
| **A100 40 GB** | 40 GB | HBM2e · ~1.56 TB/s | **~$8–12k new** (less used) | PCIe/SXM4; legacy |
| **RTX PRO 6000** | 96 GB | GDDR7 (ECC) · ~1.8 TB/s | **~$11–13k** (launch $8.5k) | single workstation/server card |
| **MI300A** | 128 GB | HBM3 *unified* CPU+GPU · ~5.3 TB/s | **~$10–15k (est.)** | 4-APU nodes; no public single-unit list |

Reading it: the **RTX PRO 6000 is the only "buy one card" option** — the datacenter
parts assume a node. Its GDDR7 bandwidth (~1.8 TB/s) is far below the HBM cards, yet
it still beats the A100 on GRACE, because GRACE-TF inference here isn't purely
bandwidth-bound. The **B200's 192 GB** is the memory headroom leader; the **MI300A's
128 GB is *unified*** (CPU and GPU share it — no host/device copies), which is a real
advantage for huge systems *if* the software path were competitive (it isn't yet for
GRACE; it is for PACE/ACE).

### Rough price/performance (GRACE, per $1,000, using mid-range price)

Indicative only — divides §1 throughput by the price above. Prices are volatile;
treat as ±50 %.

| Device | 1L katom-step/s per $1k | 2L katom-step/s per $1k |
|---|--:|--:|
| **RTX PRO 6000** | ~4.5 | **~1.4** |
| A100 40 GB (new) | ~5.8 | ~1.3 |
| B200 | ~3.7 | ~0.63 |
| H200 | ~3.3 | ~0.57 |
| MI300A | ~0.35 | ~0.08 |

On a *new-purchase* basis the **RTX PRO 6000 is the value leader for 2L** and near
the top for 1L; a still-available A100 competes on 1L $/perf but is EOL and loses on
2L, memory, and longevity. The B200/H200 buy you **absolute speed and memory**, not
$/perf. (Used-A100 pricing would shift its column up, but you inherit Ampere's age.)

---

## 3. How big a system fits — capacity per device (ESTIMATE + how to measure)

**Only one capacity point is measured:** 16,384 atoms with the **2-layer** model fits
on a **40 GB A100** (this repo's common benchmark size, chosen for exactly that
reason). Everything in the table below is a **planning estimate** from roughly linear
memory scaling (graph-ACE working set grows ~linearly in atom count, on top of a
fixed TF/CUDA + model-weights overhead of a few GB). The **2-layer model needs
markedly more memory per atom** than 1-layer (it builds a larger, semi-local
message-passing graph). **Measure before you rely on it** — recipe below.

Rough max atoms per device (single GPU, order-of-magnitude, **verify!**):

| Device | Usable HBM/VRAM | ~Max atoms, 1L | ~Max atoms, 2L |
|---|--:|--:|--:|
| A100 40 GB | 40 GB | ~150–250k | ~40–70k *(16k measured-OK)* |
| RTX PRO 6000 | 96 GB | ~400–600k | ~100–170k |
| MI300A | 128 GB (unified) | ~500–800k | ~140–220k |
| H200 | 141 GB | ~600–900k | ~150–250k |
| B200 | 192 GB | ~800k–1.2M | ~200–350k |

These assume the GPU is dedicated to one LAMMPS rank (the serial TF-CUDA build).
The unified-memory **MI300A** can in principle spill into the full 128 GB shared
pool without explicit host copies, which may let it exceed the linear estimate for
very large systems — again, measure. The H200/B200 columns are consistent with the
submit scripts' own suggestion to run `NX=40` (256,000 atoms) as a showcase.

### Measurement recipe — find the real ceiling

Sweep the box size until the run OOMs; the last size that completes is the ceiling.
fcc atom count = `4·nx·ny·nz`. Use the same binary/env as `run-dais-grace.slurm`
(the `LD_PRELOAD`/`LD_LIBRARY_PATH` block matters), just vary `nx=ny=nz` and keep
`nsteps` tiny:

```bash
# in your gracework dir, on the GPU node (env exactly as run-dais-grace.slurm)
GPU=${GPU:-h200}
for NX in 16 24 32 40 48 56 64 80; do
  atoms=$((4*NX*NX*NX))
  echo "=== NX=$NX  atoms=$atoms ==="
  "$LMP" -in in.grace_bench -var pstyle grace -var model "$MODEL_1L" \
      -var nx $NX -var ny $NX -var nz $NX -var nsteps 5 \
      -log log.cap_1l_${GPU}_nx${NX} 2>&1 | tail -5
  # OOM shows as TF "RESOURCE_EXHAUSTED / OOM when allocating tensor" or a CUDA
  # alloc failure. The largest NX that FINISHES is the 1L ceiling for this GPU.
done
# repeat with: -var pstyle grace/2layer/chunk -var model "$MODEL_2L"  (2L ceiling)
# watch memory live in another shell:  watch -n1 nvidia-smi
```

Record `(GPU, model, max_atoms, peak MiB from nvidia-smi)` and replace the estimate
row. A dozen 5-step runs cost minutes and turn the table above into fact.

---

## 4. The software caveat that governs all of this: TensorFlow

GRACE in this fork runs through **TensorFlow**, not Kokkos (there is **no
`grace/fs/kk`**). That single fact drives most of the surprises above and most of the
upside from future updates.

**What has bitten us (all now fixed in the scripts):**

- **Pinned TF 2.18 C-API tarball → no GPU platform.** The fork's cmake will, if it
  can't find a full TF in your venv, **download the `libtensorflow-gpu 2.18` C-API
  tarball** and link that. It lacks the XLA **GPU** platform the GRACE saved-models
  need → `NOT_FOUND: could not find registered platform`, or a silent CPU fallback.
  **Fix ("strategy #2"):** build against the **venv's full `libtensorflow_cc.so.2`**
  (`PYTHON=.../tf-cuda/bin/python`, no `TF_LIB_FILE`) — the same library raven uses.
- **Blackwell (B200 / RTX PRO 6000) needs a newer TF/CUDA than 2.18.** Blackwell is
  `sm_100` and wants CUDA 12.8+. We dodged this entirely by using an **unpinned
  `pip install tensorflow`** (current CUDA build, ≥2.19, shipping CUDA-12.8 wheels) in
  the venv — so the *same serial binary* ran on Ampere, Hopper, **and** Blackwell with
  only a `--gres` change. (Because GRACE is TF, there's no per-arch GPU compile to get
  wrong — unlike a Kokkos `pace/kk` build, which *would* need separate sm_90/sm_100
  builds.) **Record the exact working version** — an unpinned venv drifts:
  `pip show tensorflow` + `python -c "import tensorflow as tf;
  print(tf.__version__, tf.sysconfig.get_build_info()['cuda_version'])"`.
- **TF-ROCm on MI300A is immature.** TF-ROCm 2.17 (from AMD's ROCm-6.3 repo) *does*
  run GRACE on gfx942 (confirmed on-GPU) but ~13× slower than an A100 — a
  **software** gap in TF-ROCm's kernels for these graph-ACE ops, not a hardware one.
  Also submit with `--export=NONE` or a leaked module env silently drops it to CPU.
- **CPU-TensorFlow tax.** GRACE on CPU is ~40× (1L) to ~110× (2L) slower *per atom*
  than PACE/ACE on the same CPU — fine for validation, not production.
- **Runtime plumbing** (documented in the submit scripts): `LD_PRELOAD` gcc/14's
  `libstdc++` (TF's venv drags an older one in via RUNPATH → missing `CXXABI_1.3.15`);
  put `/usr/lib64` on `LD_LIBRARY_PATH` for the driver `libcuda.so.1` (the nvidia
  wheels don't ship it); run the binary **directly, not via `srun`** (srun resets the
  env and drops the preload).
- **FS export doesn't work for the SMAX-OMAT foundation models** (`KeyError:
  RadialBasis`) — so there is **no TensorFlow-free CPU path** for these models; you're
  on TF everywhere.

**What would change for the better (and would move the buying math):**

- **`grace/fs/kk` (a native Kokkos FS GPU path) is the big one.** It would drop the
  TensorFlow dependency entirely, almost certainly speed up the NVIDIA GPUs, and —
  most importantly — give the **MI300A a competitive path** (Kokkos-HIP, where the
  same APU is already the fastest PACE device). That single addition could turn the
  MI300A from "avoid for GRACE" into a top contender, and de-risk the whole TF stack.
  Ask **Sarath** (who pulls GRACE from **Yury**).
- **Maturing TF-ROCm** (newer releases with better gfx942 kernels) would shrink the
  MI300A gap even without `grace/fs/kk` — it's a software curve, so it improves for
  free over time.
- **Newer TF/CUDA on Blackwell** could start exploiting FP8/FP4 datapaths; if
  GRACE/TF ever emit reduced-precision inference, the B200/RTX PRO 6000 gain again.
- **Practical:** because the venv is unpinned, *pin the exact TF once you're happy*
  for reproducible rebuilds — just keep a version that still ships
  `libtensorflow_cc.so.2` (what strategy #2 links).

---

## 5. Decision guide

**Q: Is your workload GRACE (graph-ACE, TF) or PACE/ACE (Kokkos)?** This is the
fork in the road — the two disagree on the best hardware.

- **GRACE-first → buy NVIDIA.**
  - *Max speed & biggest systems, budget available:* **B200** (fastest, 192 GB).
  - *Best value / workstations / smaller or cost-sensitive clusters:* **RTX PRO 6000
    Blackwell** — ≈A100 on 1L, >A100 on 2L, 96 GB, ~1/3 the price, and it's a single
    card you can drop into a workstation. Caveats: GDDR7 not HBM (lower bandwidth for
    future bandwidth-bound kernels), no NVLink multi-GPU scale-out, workstation-class
    support/cooling.
  - *Proven datacenter middle:* **H200** if B200 is unavailable/too dear.
  - *Skip:* new A100 purchases (RTX PRO 6000 dominates); MI300A for GRACE (until
    `grace/fs/kk`).
- **PACE/ACE-first → the MI300A is excellent** (fastest single device here, 128 GB
  unified) and A100/H200/B200 also strong via Kokkos-CUDA. GRACE would come "for
  free-ish" on the NVIDIA parts; on the MI300A it stays weak until the Kokkos FS path
  lands.
- **Mixed GRACE+PACE shop:** NVIDIA Blackwell (B200 for the top end, RTX PRO 6000 for
  volume) covers *both* today. Betting on MI300A for both is a bet on `grace/fs/kk`
  and TF-ROCm maturing — reasonable on a 3–5-year horizon given the silicon, but not
  today's safe choice for GRACE.

**Rules of thumb**
- Size the memory to your *largest* intended system at your *heaviest* model layer
  (2L costs the most/atom). Measure with §3 before committing — don't trust the
  estimate for procurement.
- One modern GPU replaces 5–15 full CPU nodes for GRACE; CPU is for validation.
- Budget the **software** as well as the silicon: a GRACE fleet is a **TensorFlow**
  fleet. Pin versions, keep the venv-`libtensorflow_cc.so.2` build path, and track
  `grace/fs/kk` — it would reshuffle this entire guide, especially for AMD.

---

## Sources & provenance

- **Throughput:** measured in this repo — `GRACE.md` §6, `DAIS-STATUS.md`,
  `bench/compare-grace.sh` output (fcc-Cu, 16,384 atoms; DAIS B200/H200/RTX PRO 6000
  via `bench/run-dais-grace.slurm`, TF-CUDA).
- **Capacity:** one measured anchor (16k 2L on a 40 GB A100); the rest are linear
  estimates — measure with §3.
- **Prices (approximate, mid-2026, single-unit, USD; vendor/volume-dependent):**
  B200 ~$35–45k, H200 ~$30–40k/unit, A100 40 GB ~$8–12k new, RTX PRO 6000 Blackwell
  ~$11–13k (launched ~$8.5k), MI300A ~$10–15k est. (sold in 4-APU nodes; no public
  single-unit list). See the linked market references in the chat/commit for the
  underlying quotes; these are order-of-magnitude only.
