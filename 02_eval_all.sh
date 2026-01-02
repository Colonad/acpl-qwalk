#!/usr/bin/env bash
set -euo pipefail

# ------------------- User knobs (override via env vars) -------------------
RUNS_ROOT="${RUNS_ROOT:-runs}"
EVAL_ROOT="${EVAL_ROOT:-eval}"

DEVICE="${DEVICE:-cuda}"        # or cpu; eval.py also has auto if you pass nothing
EVAL_SEEDS="${EVAL_SEEDS:-10}"  # CI seeds in eval.py (0..EVAL_SEEDS-1)
EPISODES="${EPISODES:-256}"

BASELINE_KIND="${BASELINE_KIND:-hadamard}"

# Ablations (space-separated). Set to "" to disable.
ABLATIONS_STR="${ABLATIONS:-NoPE GlobalCoin TimeFrozen NodePermute}"

# If STRICT_ABLATIONS=1, fail fast if any requested ablation can't be applied.
STRICT_ABLATIONS="${STRICT_ABLATIONS:-1}"

# Mask sensitivity modes (space-separated). Leave empty to disable.
# IMPORTANT: these must match what your eval.py expects as mode names.
MASK_SENS_STR="${MASK_SENSITIVITY:-}"

# Robust sweep knobs (only used for robust-line-disorder)
ROB_SIGMA="${ROB_SIGMA:-0,0.02,0.05,0.1}"
ROB_TRIALS="${ROB_TRIALS:-5}"
ROB_BOOT="${ROB_BOOTSTRAP:-256}"

# -------------------------------------------------------------------------

mkdir -p "${EVAL_ROOT}"

# Build common flag arrays (bash-safe)
COMMON_EVAL_FLAGS=(--device "${DEVICE}" --seeds "${EVAL_SEEDS}" --episodes "${EPISODES}" --plots --report)

ABL_FLAGS=()
if [[ -n "${ABLATIONS_STR}" ]]; then
  # shellcheck disable=SC2206
  ABL_LIST=(${ABLATIONS_STR})
  ABL_FLAGS+=(--ablations "${ABL_LIST[@]}")
  if [[ "${STRICT_ABLATIONS}" == "1" ]]; then
    ABL_FLAGS+=(--strict-ablations)
  fi
fi

MS_FLAGS=()
if [[ -n "${MASK_SENS_STR}" ]]; then
  # shellcheck disable=SC2206
  MS_LIST=(${MASK_SENS_STR})
  MS_FLAGS+=(--mask-sensitivity "${MS_LIST[@]}")
fi

# Provenance note (best-effort)
GIT_REV="unknown"
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_REV="$(git rev-parse --short HEAD || echo unknown)"
fi

TRAIN_SEEDS_STR="${TRAIN_SEEDS:-0 1 2}"

run_eval_ckpt () {
  local ckpt_dir="$1"
  local outdir="$2"
  local suite="$3"
  local note="$4"

  python scripts/eval.py \
    --ckpt "${ckpt_dir}" \
    --outdir "${outdir}" \
    --suite "${suite}" \
    "${COMMON_EVAL_FLAGS[@]}" \
    "${ABL_FLAGS[@]}" \
    "${MS_FLAGS[@]}" \
    --override "note=${note}" \
    --override "git_rev=${GIT_REV}"
}

run_eval_baseline () {
  local cfg_path="$1"
  local outdir="$2"
  local suite="$3"
  local note="$4"

  python scripts/eval.py \
    --policy baseline \
    --config "${cfg_path}" \
    --outdir "${outdir}" \
    --suite "${suite}" \
    "${COMMON_EVAL_FLAGS[@]}" \
    --baseline-kind "${BASELINE_KIND}" \
    --override "note=${note}" \
    --override "git_rev=${GIT_REV}"
}

echo "[eval] runs_root=${RUNS_ROOT}"
echo "[eval] eval_root=${EVAL_ROOT}"
echo "[eval] device=${DEVICE} eval_seeds=${EVAL_SEEDS} episodes=${EPISODES}"
echo "[eval] ablations='${ABLATIONS_STR}' strict=${STRICT_ABLATIONS}"
echo "[eval] mask_sensitivity='${MASK_SENS_STR}'"
echo "[eval] baseline_kind=${BASELINE_KIND}"
echo "[eval] git_rev=${GIT_REV}"
echo

# ------------------------------ MIXING ------------------------------
EXP="mixing-grid"
CFG="acpl/configs/experiments/${EXP}.yaml"
for SEED in ${TRAIN_SEEDS_STR}; do
  CKPT_DIR="${RUNS_ROOT}/${EXP}/seed${SEED}"
  OUTDIR="${EVAL_ROOT}/${EXP}/seed${SEED}/main"
  echo "==== EVAL exp=${EXP} train_seed=${SEED} -> ${OUTDIR}"
  run_eval_ckpt "${CKPT_DIR}" "${OUTDIR}" "${EXP}" "mixing_main_seed${SEED}"
done
echo "==== BASELINE exp=${EXP}"
run_eval_baseline "${CFG}" "${EVAL_ROOT}/${EXP}/baseline/${BASELINE_KIND}" "${EXP}" "mixing_baseline_${BASELINE_KIND}"

# ------------------------------ SEARCH ------------------------------
EXP="search-grid"
CFG="acpl/configs/experiments/${EXP}.yaml"
for SEED in ${TRAIN_SEEDS_STR}; do
  CKPT_DIR="${RUNS_ROOT}/${EXP}/seed${SEED}"
  OUTDIR="${EVAL_ROOT}/${EXP}/seed${SEED}/main"
  echo "==== EVAL exp=${EXP} train_seed=${SEED} -> ${OUTDIR}"
  run_eval_ckpt "${CKPT_DIR}" "${OUTDIR}" "${EXP}" "search_main_seed${SEED}"
done
echo "==== BASELINE exp=${EXP}"
run_eval_baseline "${CFG}" "${EVAL_ROOT}/${EXP}/baseline/${BASELINE_KIND}" "${EXP}" "search_baseline_${BASELINE_KIND}"

# ------------------------------ TRANSFER (line) ------------------------------
EXP="transfer-line"
CFG="acpl/configs/experiments/${EXP}.yaml"
for SEED in ${TRAIN_SEEDS_STR}; do
  CKPT_DIR="${RUNS_ROOT}/${EXP}/seed${SEED}"
  OUTDIR="${EVAL_ROOT}/${EXP}/seed${SEED}/main"
  echo "==== EVAL exp=${EXP} train_seed=${SEED} -> ${OUTDIR}"
  run_eval_ckpt "${CKPT_DIR}" "${OUTDIR}" "${EXP}" "transfer_line_main_seed${SEED}"
done
echo "==== BASELINE exp=${EXP}"
run_eval_baseline "${CFG}" "${EVAL_ROOT}/${EXP}/baseline/${BASELINE_KIND}" "${EXP}" "transfer_line_baseline_${BASELINE_KIND}"

# ------------------------------ TRANSFER (regular) ------------------------------
EXP="transfer-regular"
CFG="acpl/configs/experiments/${EXP}.yaml"
for SEED in ${TRAIN_SEEDS_STR}; do
  CKPT_DIR="${RUNS_ROOT}/${EXP}/seed${SEED}"
  OUTDIR="${EVAL_ROOT}/${EXP}/seed${SEED}/main"
  echo "==== EVAL exp=${EXP} train_seed=${SEED} -> ${OUTDIR}"
  run_eval_ckpt "${CKPT_DIR}" "${OUTDIR}" "${EXP}" "transfer_regular_main_seed${SEED}"
done
echo "==== BASELINE exp=${EXP}"
run_eval_baseline "${CFG}" "${EVAL_ROOT}/${EXP}/baseline/${BASELINE_KIND}" "${EXP}" "transfer_regular_baseline_${BASELINE_KIND}"

# Cross-generalization (novelty): evaluate each trained policy under the other config
echo "==== CROSS-GEN: line-trained evaluated on regular config"
for SEED in ${TRAIN_SEEDS_STR}; do
  python scripts/eval.py \
    --ckpt "${RUNS_ROOT}/transfer-line/seed${SEED}" \
    --config "acpl/configs/experiments/transfer-regular.yaml" \
    --outdir "${EVAL_ROOT}/xfer_cross/line_to_regular/seed${SEED}" \
    --suite "xfer_line_to_regular" \
    "${COMMON_EVAL_FLAGS[@]}" \
    "${ABL_FLAGS[@]}" \
    "${MS_FLAGS[@]}" \
    --override "note=cross_line_to_regular_seed${SEED}" \
    --override "git_rev=${GIT_REV}"
done

echo "==== CROSS-GEN: regular-trained evaluated on line config"
for SEED in ${TRAIN_SEEDS_STR}; do
  python scripts/eval.py \
    --ckpt "${RUNS_ROOT}/transfer-regular/seed${SEED}" \
    --config "acpl/configs/experiments/transfer-line.yaml" \
    --outdir "${EVAL_ROOT}/xfer_cross/regular_to_line/seed${SEED}" \
    --suite "xfer_regular_to_line" \
    "${COMMON_EVAL_FLAGS[@]}" \
    "${ABL_FLAGS[@]}" \
    "${MS_FLAGS[@]}" \
    --override "note=cross_regular_to_line_seed${SEED}" \
    --override "git_rev=${GIT_REV}"
done

# ------------------------------ ROBUST ------------------------------
EXP="robust-line-disorder"
CFG="acpl/configs/experiments/${EXP}.yaml"
for SEED in ${TRAIN_SEEDS_STR}; do
  CKPT_DIR="${RUNS_ROOT}/${EXP}/seed${SEED}"
  OUTDIR="${EVAL_ROOT}/${EXP}/seed${SEED}/main"
  echo "==== EVAL exp=${EXP} train_seed=${SEED} -> ${OUTDIR}"
  run_eval_ckpt "${CKPT_DIR}" "${OUTDIR}" "${EXP}" "robust_main_seed${SEED}"
done
echo "==== BASELINE exp=${EXP}"
run_eval_baseline "${CFG}" "${EVAL_ROOT}/${EXP}/baseline/${BASELINE_KIND}" "${EXP}" "robust_baseline_${BASELINE_KIND}"

# Robustness sweep (thesis-critical)
for SEED in ${TRAIN_SEEDS_STR}; do
  echo "==== ROBUST-SWEEP exp=${EXP} train_seed=${SEED}"
  python scripts/eval.py \
    --ckpt "${RUNS_ROOT}/${EXP}/seed${SEED}" \
    --outdir "${EVAL_ROOT}/${EXP}/seed${SEED}/robust_sweep" \
    --suite "${EXP}" \
    "${COMMON_EVAL_FLAGS[@]}" \
    "${ABL_FLAGS[@]}" \
    --robust-sweep \
    --robust-sweep-kinds edge_phase coin_dephase \
    --robust-sweep-sigma "${ROB_SIGMA}" \
    --robust-sweep-trials "${ROB_TRIALS}" \
    --robust-sweep-bootstrap "${ROB_BOOT}" \
    --robust-sweep-include-ablations \
    --override "note=robust_sweep_seed${SEED}" \
    --override "git_rev=${GIT_REV}"
done

echo
echo "[eval] done"
