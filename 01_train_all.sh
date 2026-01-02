#!/usr/bin/env bash
set -euo pipefail

# ------------------- Threading safety (WSL/BLAS) -------------------
# Some BLAS/OpenMP combinations on WSL can crash (segfault) under heavy
# linear algebra (e.g., complex matmuls/eigendecompositions). Limiting
# threads is a pragmatic stability win. Override via env vars if desired.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

# ------------------- User knobs (override via env vars) -------------------
RUNS_ROOT="${RUNS_ROOT:-runs}"
# Base config selection (CPU vs CUDA).
# - Set BASE_CFG explicitly to override.
# - Or set DEVICE_PREF=cpu|cuda|auto to auto-pick *_cpu.yaml or *_cuda.yaml when present.
DEVICE_PREF="${DEVICE_PREF:-auto}"  # auto|cpu|cuda
AUTO_DEVICE="$(python - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)"
if [[ "${DEVICE_PREF}" == "auto" ]]; then
  DEVICE_PREF="${AUTO_DEVICE}"
fi

if [[ -z "${BASE_CFG:-}" ]]; then
  if [[ "${DEVICE_PREF}" == "cuda" && -f acpl/configs/train_cuda.yaml ]]; then
    BASE_CFG="acpl/configs/train_cuda.yaml"
  elif [[ "${DEVICE_PREF}" == "cpu" && -f acpl/configs/train_cpu.yaml ]]; then
    BASE_CFG="acpl/configs/train_cpu.yaml"
  else
    BASE_CFG="acpl/configs/train.yaml"
  fi
fi

# Training seeds (space-separated)
TRAIN_SEEDS_STR="${TRAIN_SEEDS:-0 1 2 3 4}"
# allow commas in TRAIN_SEEDS, e.g. "0 1 2,3,4"
TRAIN_SEEDS_STR="${TRAIN_SEEDS_STR//,/ }"

# Behavior toggles
SKIP_FINISHED="${SKIP_FINISHED:-1}"           # skip only if ckpt_epoch >= desired_epochs
RESUME_INCOMPLETE="${RESUME_INCOMPLETE:-1}"   # resume if ckpt exists but not finished
FORCE_RESTART="${FORCE_RESTART:-0}"           # wipe run dir before training (DANGEROUS)

# Auto-fix invalid coin family (Option B + transfer-regular safety)
# If coin.family=su2 but graph family is not line/cycle, switch to exp.
AUTO_COIN_FIX="${AUTO_COIN_FIX:-1}"           # set 0 to disable
AUTO_COIN_FAMILY="${AUTO_COIN_FAMILY:-exp}"   # exp or cayley

PYTHON_BIN="${PYTHON_BIN:-python}"
# -------------------------------------------------------------------------

EXPS=(
  "mixing-grid"
  "search-grid"
  "transfer-line"
  "transfer-regular"
  "robust-line-disorder"
)

echo "[train] runs_root=${RUNS_ROOT}"
echo "[train] base_cfg=${BASE_CFG}"
echo "[train] seeds=${TRAIN_SEEDS_STR}"
echo "[train] device_pref=${DEVICE_PREF} (auto_device=${AUTO_DEVICE})"
echo "[train] skip_finished=${SKIP_FINISHED} resume_incomplete=${RESUME_INCOMPLETE} force_restart=${FORCE_RESTART}"
echo "[train] auto_coin_fix=${AUTO_COIN_FIX} auto_coin_family=${AUTO_COIN_FAMILY}"
echo

mkdir -p "${RUNS_ROOT}"

# Best-effort provenance
GIT_REV="unknown"
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_REV="$(git rev-parse --short HEAD || echo unknown)"
fi

for EXP in "${EXPS[@]}"; do
  EXP_CFG="acpl/configs/experiments/${EXP}.yaml"
  if [[ ! -f "${EXP_CFG}" ]]; then
    echo "[warn] missing config: ${EXP_CFG} (skipping exp=${EXP})"
    continue
  fi

  for SEED in ${TRAIN_SEEDS_STR}; do
    OUTDIR="${RUNS_ROOT}/${EXP}/seed${SEED}"
    mkdir -p "${OUTDIR}"

    MERGED_CFG="${OUTDIR}/config_merged.yaml"
    CKPT_LAST="${OUTDIR}/model_last.pt"

    if [[ "${FORCE_RESTART}" == "1" ]]; then
      echo "==== TRAIN exp=${EXP} seed=${SEED} -> ${OUTDIR} (FORCE_RESTART=1)"
      rm -f "${CKPT_LAST}" "${OUTDIR}/model_best.pt" "${OUTDIR}/pt_target.png" || true
      # keep merged config for provenance; wipe metrics so you don’t mix runs
      rm -f "${OUTDIR}/metrics.jsonl" "${OUTDIR}/eval_ci.txt" "${OUTDIR}/eval_ci.json" || true
    fi

    # ---- Merge base + exp cfg -> OUTDIR/config_merged.yaml, set run_dir, set seed ----
    # Also applies the Option B coin-family auto-fix if enabled.
    EPOCHS_WANT="$("${PYTHON_BIN}" - <<PY
import os
import yaml
from pathlib import Path

base = Path("${BASE_CFG}")
exp  = Path("${EXP_CFG}")
out  = Path("${MERGED_CFG}")
out.parent.mkdir(parents=True, exist_ok=True)

def deep_merge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        r = dict(a)
        for k, v in b.items():
            r[k] = deep_merge(r.get(k), v)
        return r
    return b if b is not None else a

cfg_base = yaml.safe_load(base.read_text()) if base.exists() else {}
cfg_exp  = yaml.safe_load(exp.read_text()) if exp.exists() else {}
cfg = deep_merge(cfg_base or {}, cfg_exp or {})

# Device handling: force cfg['device'] based on DEVICE_PREF (cpu|cuda) from bash.
cfg['device'] = os.environ.get('DEVICE_PREF', cfg.get('device','auto'))

cfg.setdefault("train", {})
cfg["train"]["run_dir"] = str(Path("${OUTDIR}"))
cfg["seed"] = int("${SEED}")

# Option B: auto-fix invalid SU2 on non-line/cycle graphs (transfer-regular bug)
auto_fix = int("${AUTO_COIN_FIX}") == 1
auto_family = str("${AUTO_COIN_FAMILY}").lower().strip()

data = cfg.get("data", {}) or {}
coin = cfg.get("coin", {}) or {}
if not isinstance(coin, dict):
    coin = {"family": str(coin)}
    cfg["coin"] = coin

graph_family = str(data.get("family", "")).lower().strip()
coin_family  = str(coin.get("family", "su2")).lower().strip()

# treat these as deg<=2 families
line_like = graph_family in ("line", "cycle")

if auto_fix and (coin_family == "su2") and (not line_like):
    # record the fix in-config for provenance
    coin["family"] = auto_family if auto_family in ("exp", "cayley") else "exp"
    coin.setdefault("auto_fix_notes", {})
    coin["auto_fix_notes"]["reason"] = "su2 invalid for max-degree>2 graphs; auto-switched"
    coin["auto_fix_notes"]["from"] = "su2"
    coin["auto_fix_notes"]["to"] = coin["family"]
    coin["auto_fix_notes"]["graph_family"] = graph_family

out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

epochs = int((cfg.get("train", {}) or {}).get("epochs", 1))
print(epochs)
PY
)"
    echo "[merge_cfg] wrote ${MERGED_CFG}"
    echo "[merge_cfg] epochs = ${EPOCHS_WANT}"

    # ---- Check existing checkpoint epoch ----
    CKPT_EPOCH=0
    CKPT_OK=0
    if [[ -f "${CKPT_LAST}" ]]; then
      CKPT_EPOCH="$("${PYTHON_BIN}" - <<PY
import torch
p="${CKPT_LAST}"
try:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    ep = int(ck.get("epoch", 0)) if isinstance(ck, dict) else 0
    print(ep)
except Exception:
    print(-1)
PY
)"
      if [[ "${CKPT_EPOCH}" -ge 0 ]]; then
        CKPT_OK=1
      fi
    fi

    RESUME_ARGS=()
    if [[ "${CKPT_OK}" == "1" ]]; then
      if [[ "${SKIP_FINISHED}" == "1" && "${CKPT_EPOCH}" -ge "${EPOCHS_WANT}" ]]; then
        echo "==== TRAIN exp=${EXP} seed=${SEED} -> ${OUTDIR} (SKIP_FINISHED: ckpt_epoch=${CKPT_EPOCH} >= epochs=${EPOCHS_WANT})"
        continue
      fi
      if [[ "${RESUME_INCOMPLETE}" == "1" && "${CKPT_EPOCH}" -lt "${EPOCHS_WANT}" ]]; then
        echo "==== TRAIN exp=${EXP} seed=${SEED} -> ${OUTDIR} (RESUME: ckpt_epoch=${CKPT_EPOCH} < epochs=${EPOCHS_WANT})"
        RESUME_ARGS=(--resume "${OUTDIR}")
      else
        echo "==== TRAIN exp=${EXP} seed=${SEED} -> ${OUTDIR} (RESTART_FROM_SCRATCH: resume disabled)"
      fi
    else
      if [[ -f "${CKPT_LAST}" ]]; then
        echo "==== TRAIN exp=${EXP} seed=${SEED} -> ${OUTDIR} (BAD CHECKPOINT: ${CKPT_LAST})"
        echo "     Suggest: FORCE_RESTART=1 to wipe this run dir, or delete ${CKPT_LAST}."
        exit 2
      fi
      echo "==== TRAIN exp=${EXP} seed=${SEED} -> ${OUTDIR}"
    fi

    # Run training.
    # Note: we pass --seed for clarity; run_dir is in the merged YAML for provenance.
    scripts/train.py \
      --config "${MERGED_CFG}" \
      --seed "${SEED}" \
      "${RESUME_ARGS[@]}" \
      --override "log.run_name=${EXP}_seed${SEED}" \
      --override "log.wandb.run_name=${EXP}_seed${SEED}" \
      --override "git_rev=${GIT_REV}"
  done
done

echo
echo "[train] done"