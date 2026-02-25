#!/usr/bin/env bash
# Run PNAPixelDataset benchmarks on the current branch and on dev, then compare.
#
# Prerequisites:
#   - PNA062_unstim_PBMCs_1000cells_S02_S2.layout.pxl
#   - PNA062_PHA_PBMCs_1000cells_S04_S4.layout.pxl
#   in the repo root (otherwise benchmarks are skipped).
#
# Usage:
#   ./scripts/run_pixeldataset_benchmark_compare.sh
#
# Or via bash:
#   bash scripts/run_pixeldataset_benchmark_compare.sh
#
# Output: benchmark results for current branch, then for dev, then a comparison
# (dev run vs current-branch run). You are left on your original branch.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BENCHMARK_TEST="tests/pna/pixeldataset/test_benchmark_pixeldataset.py"
STORAGE="${REPO_ROOT}/.benchmarks"

# Check benchmark test exists
if [[ ! -f "$BENCHMARK_TEST" ]]; then
  echo "Error: Benchmark file not found: $BENCHMARK_TEST"
  exit 1
fi

ORIGINAL_BRANCH="$(git branch --show-current)"
echo "=== Original branch: $ORIGINAL_BRANCH ==="

# 1) Run on current branch and save
echo ""
echo "=== Running benchmarks on current branch ($ORIGINAL_BRANCH) ==="
uv run pytest \
  --benchmark-enable \
  --benchmark-only \
  --benchmark-save=current-branch \
  -v \
  "$BENCHMARK_TEST"

# 2) Checkout dev and run (compare against latest = current-branch run)
echo ""
echo "=== Switching to dev and running benchmarks ==="
git checkout dev

uv run pytest \
  --benchmark-enable \
  --benchmark-only \
  --benchmark-save=dev-branch \
  --benchmark-compare \
  -v \
  "$BENCHMARK_TEST"

# 3) Return to original branch
echo ""
echo "=== Switching back to $ORIGINAL_BRANCH ==="
git checkout "$ORIGINAL_BRANCH"

echo ""
echo "Done. You are on branch: $(git branch --show-current)"
echo "Saved runs are in: ${STORAGE}/"
echo "To compare again: uv run pytest --benchmark-only --benchmark-compare=0001 $BENCHMARK_TEST"
