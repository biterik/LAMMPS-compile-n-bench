#!/bin/bash
# ===========================================================================
# OPTIONAL standalone helper: obtain the fork + apply the MC-SITES patches onto
# branch feature/mc-sites, WITHOUT building. The build-mcsites-*.sh scripts do
# this automatically, so you only need this if you want to prepare/inspect the
# source tree separately (e.g. run the pytest validation suite against it).
#
# LOGIN-NODE / laptop script (git + network only, no compute).
#
#   SRC=/ptmp/$USER/mcsites/lammps-mcsites ./patch-mc-sites.sh
#   # then, e.g.:  SRC=$SRC ./build-mcsites-cmmg.sh   (skips re-cloning)
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORK_URL="${FORK_URL:-https://github.com/thermoatoms/lammps.git}"
FORK_BRANCH="${FORK_BRANCH:-develop}"
FORK_COMMIT="${FORK_COMMIT:-24da74cd73323f5e7415fdd9a9670b88535464d3}"
MCSITES_BRANCH="${MCSITES_BRANCH:-feature/mc-sites}"
PATCHES_DIR="${PATCHES_DIR:-$SCRIPT_DIR/patches}"
SRC="${SRC:-$(pwd)/lammps-mcsites}"

case "$SRC" in
  "$HOME") echo "ERROR: refusing to use \$HOME as SRC" >&2; exit 1;;
esac
[ -d "$PATCHES_DIR" ] || { echo "ERROR: PATCHES_DIR not found: $PATCHES_DIR" >&2; exit 1; }

if [ ! -d "$SRC/.git" ]; then
    git clone -b "$FORK_BRANCH" "$FORK_URL" "$SRC"
fi
cd "$SRC"
git fetch --all -q || true

if git rev-parse -q --verify "$MCSITES_BRANCH" >/dev/null; then
    echo ">> branch $MCSITES_BRANCH already exists at: $(git log -1 --format='%h %s' "$MCSITES_BRANCH")"
    echo ">> delete it first (git branch -D $MCSITES_BRANCH) to re-apply."
    exit 0
fi

git checkout -q "$FORK_COMMIT"
git checkout -q -b "$MCSITES_BRANCH"
git am "$PATCHES_DIR"/00*.patch

echo ">> $MCSITES_BRANCH ready at: $(git log -1 --format='%H %s')"
echo ">> patches applied: $(git rev-list --count "$FORK_COMMIT"..HEAD)  (expected 5)"
echo ">> new files:"
git show --stat --oneline HEAD~4..HEAD 2>/dev/null | grep -E 'compute_sites_voronoi|fix_mc_sites' | sed 's/^/     /' || true
