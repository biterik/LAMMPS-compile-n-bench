# ===========================================================================
# Common LAMMPS package set for the MPCDF builds (Viper-GPU, Raven, cmmg).
#
# This mirrors the "Installed packages" block of Erik's lammps_250722_config.txt
# (the central lammps/250722 module), MINUS the Intel-only package, because
# Viper and cmmg are AMD and Raven uses CUDA — INTEL would be unused/unbuildable.
#
# Used via:  cmake -C <this file> ...   (see the per-machine build-*.sh scripts)
# ===========================================================================

# --- build-wide settings to match the reference build ----------------------
set(LAMMPS_SIZES         smallbig CACHE STRING "" FORCE)   # -DLAMMPS_SMALLBIG (as in the dump)
set(BUILD_MPI            ON       CACHE BOOL   "" FORCE)
set(USE_INTERNAL_LINALG  ON       CACHE BOOL   "" FORCE)   # avoid an external LAPACK dependency

# --- packages with NO external library (safe everywhere) -------------------
foreach(pkg
    AMOEBA ASPHERE BOCS BODY BPM BROWNIAN CG-DNA CG-SPICA CLASS2 COLLOID
    CORESHELL DIELECTRIC DIFFRACTION DIPOLE DPD-BASIC DPD-MESO DPD-REACT
    DPD-SMOOTH DRUDE EFF ELECTRODE EXTRA-COMMAND EXTRA-COMPUTE EXTRA-DUMP
    EXTRA-FIX EXTRA-MOLECULE EXTRA-PAIR FEP GRANULAR INTERLAYER KSPACE
    LEPTON MANYBODY MC MEAM MESONT MISC ML-IAP ML-POD ML-RANN ML-SNAP
    ML-UF3 MOFFF MOLECULE OPENMP OPT ORIENT PERI PHONON PLUGIN POEMS QEQ
    REACTION REAXFF REPLICA RHEO RIGID SHOCK SPH SPIN SRD TALLY UEF YAFF
    ML-PACE)                                # ML-PACE is the benchmark potential (pace/kk on GPU)
  set(PKG_${pkg} ON CACHE BOOL "" FORCE)
endforeach()

# COLVARS ships a bundled lib (lib/colvars) — no download needed.
set(PKG_COLVARS ON CACHE BOOL "" FORCE)
# COMPRESS gives gzip/zstd dump compression (needs system zlib, present everywhere).
set(PKG_COMPRESS ON CACHE BOOL "" FORCE)

# ===========================================================================
# HEAVY / EXTERNAL-LIBRARY packages.
# These auto-download and build extra sources at CONFIGURE time, which only
# works on a login node (compute nodes have no internet). They add build time
# and are the most likely source of build friction on the GPU toolchains.
# They are NOT needed for the PACE benchmark — comment any out if a build fails.
# ===========================================================================
set(PKG_ML-PACE ON  CACHE BOOL "" FORCE)   # (already on above) downloads pacelib
set(DOWNLOAD_PACELIB ON CACHE BOOL "" FORCE)

set(PKG_KIM     ON  CACHE BOOL "" FORCE)   # downloads + builds kim-api
set(DOWNLOAD_KIM ON CACHE BOOL "" FORCE)

set(PKG_VORONOI ON  CACHE BOOL "" FORCE)   # downloads + builds voro++
set(DOWNLOAD_VORO ON CACHE BOOL "" FORCE)

set(PKG_MACHDYN ON  CACHE BOOL "" FORCE)   # needs the Eigen3 headers
set(DOWNLOAD_EIGEN3 ON CACHE BOOL "" FORCE)

# PLUMED is intentionally NOT set here. It is CPU-only, unused by the PACE
# benchmark, and pulls in BLAS/LAPACK+GSL. It is enabled ONLY in the cmmg build
# script (which loads MKL + GSL). The Raven/Viper GPU builds omit it, so it
# defaults OFF there — no BLAS/GSL needed on the GPU machines.

# ---------------------------------------------------------------------------
# NOTE on offline builds: if your login node blocks these downloads, point the
# *_URL variables at pre-fetched tarballs, e.g.
#   -D PACELIB_URL=file:///path/pacelib-v.2023.11.25.fix2.tar.gz
#   -D KIM_URL=file:///path/kim-api-2.2.1.txz
#   -D EIGEN3_URL=file:///path/eigen-3.4.0.tar.gz
#   -D VORO_URL=file:///path/voro++-0.4.6.tar.gz
#   -D PLUMED_URL=file:///path/plumed-src-2.9.4.tgz
# (exactly as in your Raven snippet).
# ---------------------------------------------------------------------------
