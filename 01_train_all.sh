#!/usr/bin/env bash
set -euo pipefail

# ------------------- User knobs (override via env vars) -------------------
RUNS_ROOT="${RUNS_ROOT:-runs}"

# Training seeds (3–5 is typical; keep fixed for defendability)
TRAIN_SEEDS_STR="${TRAIN_SEEDS:-0 1 2}"

# If you want to override epochs/lr/etc, do it via YAML or train.py flags.
# -------------------------------------------------------------------------

EXPS=(
  "mixing-grid"
  "search-grid"
  "transfer-line"
  "transfer-regular"
  "robust-line-disorder"
)

echo "[train] runs_root=${RUNS_ROOT}"
echo "[train] seeds=${TRAIN_SEEDS_STR}"
echo

mkdir -p "${RUNS_ROOT}"

for EXP in "${EXPS[@]}"; do
  CFG="acpl/configs/experiments/${EXP}.yaml"
  for SEED in ${TRAIN_SEEDS_STR}; do
    OUTDIR="${RUNS_ROOT}/${EXP}/seed${SEED}"
    mkdir -p "${OUTDIR}"
    echo "==== TRAIN exp=${EXP} seed=${SEED} -> ${OUTDIR}"
    scripts/train.py \
      --config "${CFG}" \
      --seed "${SEED}" \
      --run_dir "${OUTDIR}"
  done
done

echo
echo "[train] done"
