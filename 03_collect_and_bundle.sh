#!/usr/bin/env bash
set -euo pipefail

EVAL_ROOT="${EVAL_ROOT:-eval}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"

mkdir -p "${RESULTS_ROOT}/registry" "${RESULTS_ROOT}/figures_bundle"

echo "==== COLLECT RESULTS from ${EVAL_ROOT} -> ${RESULTS_ROOT}/registry"
python scripts/collect_results.py \
  --root "${EVAL_ROOT}" \
  --outdir "${RESULTS_ROOT}/registry" \
  --with-ci-cols \
  --strict

echo
echo "==== BUNDLE FIGURES from ${EVAL_ROOT} -> ${RESULTS_ROOT}/figures_bundle"
python scripts/make_figures.py \
  --root "${EVAL_ROOT}" \
  --outdir "${RESULTS_ROOT}/figures_bundle" \
  --layout by_kind \
  --write-tex \
  --include-embeddings

echo
echo "[collect+bundle] done"
