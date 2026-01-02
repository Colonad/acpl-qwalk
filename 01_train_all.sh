#!/usr/bin/env bash
set -euo pipefail

# ------------------- User knobs (override via env vars) -------------------
RUNS_ROOT="${RUNS_ROOT:-runs}"
BASE_CFG="${BASE_CFG:-acpl/configs/train.yaml}"

# Training seeds (space-separated)
TRAIN_SEEDS_STR="${TRAIN_SEEDS:-0 1 2}"

# If 1: skip a run_dir if a last checkpoint exists.
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# If we must replace su2 on high-degree graphs, choose this:
#   exp   (recommended default)  OR  cayley
ANYDEG_COIN_FAMILY="${ANYDEG_COIN_FAMILY:-exp}"
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
echo "[train] skip_existing=${SKIP_EXISTING}"
echo "[train] anydeg_coin_family=${ANYDEG_COIN_FAMILY}"
echo

mkdir -p "${RUNS_ROOT}"

for EXP in "${EXPS[@]}"; do
  EXP_CFG="acpl/configs/experiments/${EXP}.yaml"

  for SEED in ${TRAIN_SEEDS_STR}; do
    OUTDIR="${RUNS_ROOT}/${EXP}/seed${SEED}"
    mkdir -p "${OUTDIR}"

    LAST_CKPT="${OUTDIR}/model_last.pt"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${LAST_CKPT}" ]]; then
      # Still write merged config for provenance consistency
      :
    fi

    MERGED_CFG="${OUTDIR}/config_merged.yaml"

    # ---------------- Merge + normalize + safety coin override ----------------
    python3 - <<'PY' "${BASE_CFG}" "${EXP_CFG}" "${MERGED_CFG}" "${ANYDEG_COIN_FAMILY}"
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
exp_path  = Path(sys.argv[2])
out_path  = Path(sys.argv[3])
anydeg    = str(sys.argv[4])

def deep_merge(a, b):
    # b overrides a
    if not isinstance(a, dict) or not isinstance(b, dict):
        return b
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_yaml(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

base = load_yaml(base_path)
exp  = load_yaml(exp_path)

# Support a simple "defaults:" dict (your experiment yamls use this sometimes)
defaults = exp.get("defaults", {}) or {}
if "defaults" in exp:
    exp = dict(exp)
    exp.pop("defaults", None)

cfg = dict(base)

# Apply defaults in a deterministic order
if isinstance(defaults, dict):
    for _, rel in sorted(defaults.items(), key=lambda kv: str(kv[0])):
        try:
            p = Path(rel)
            dcfg = load_yaml(p)
            cfg = deep_merge(cfg, dcfg)
        except Exception:
            # Defaults are "best effort"; keep robust.
            pass

# Apply experiment overrides last
cfg = deep_merge(cfg, exp)

# ---- Schema fixups (so experiment variants can actually override base) ----
# Some configs place coin under model.coin; train.py reads top-level coin.
if isinstance(cfg.get("model"), dict) and isinstance(cfg["model"].get("coin"), dict):
    cfg["coin"] = deep_merge(cfg.get("coin", {}) or {}, cfg["model"]["coin"])

# Some configs use training.*; train.py reads train.*
if isinstance(cfg.get("training"), dict) and not isinstance(cfg.get("train"), dict):
    cfg["train"] = cfg["training"]

# ---- Safety rule: su2 only valid if max degree <= 2 ----
data = cfg.get("data", {}) or {}
fam  = str(data.get("family", "") or "").lower()

def needs_any_degree_coin(data: dict) -> bool:
    fam = str(data.get("family", "") or "").lower()
    if fam in ("line", "path", "cycle", "ring"):
        return False
    if fam == "regular":
        d = (data.get("regular", {}) or {}).get("d", None)
        try:
            return int(d) > 2
        except Exception:
            return True
    # grid / er / ws and anything unknown are treated as "possibly >2"
    if fam in ("grid", "er", "ws"):
        return True
    return True

coin = cfg.get("coin", {}) or {}
coin_fam = str(coin.get("family", "") or "").lower()

if needs_any_degree_coin(data) and coin_fam == "su2":
    cfg.setdefault("coin", {})
    cfg["coin"]["family"] = anydeg
    cfg.setdefault("_notes", {})
    cfg["_notes"]["coin_family_auto_override"] = (
        f"auto: su2-> {anydeg} because data.family='{fam}' may exceed degree 2"
    )

# Write merged config
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

# Print key provenance info (matches your prior logging style)
epochs = None
if isinstance(cfg.get("train"), dict):
    epochs = cfg["train"].get("epochs", None)
print(f"[merge_cfg] wrote {out_path}")
print(f"[merge_cfg] epochs = {epochs}")
print(f"[merge_cfg] coin.family = {cfg.get('coin',{}).get('family',None)}")
print(f"[merge_cfg] data.family = {cfg.get('data',{}).get('family',None)}")
PY
    # ----------------------------------------------------------------------

    if [[ "${SKIP_EXISTING}" == "1" && -f "${LAST_CKPT}" ]]; then
      echo "==== TRAIN exp=${EXP} seed=${SEED} -> ${OUTDIR} (SKIP_EXISTING=1: found last checkpoint)"
      continue
    fi

    echo "==== TRAIN exp=${EXP} seed=${SEED} -> ${OUTDIR}"
    scripts/train.py \
      --config "${MERGED_CFG}" \
      --seed "${SEED}" \
      --run_dir "${OUTDIR}"
  done
done

echo
echo "[train] done"
