#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

CFG="acpl/configs/train.yaml"   # base config; we override fields via CLI
SUITE="basic_valid"             # eval suite name used by scripts/eval.py
EPISODES=256
EVAL_SEEDS=(0 1 2 3 4)
MODEL_SEEDS=(0 1 2 3 4)

# 1) tests
pytest -q

# 2) train (S random inits)
for SEED in "${MODEL_SEEDS[@]}"; do
  RUN="runs/exp1_transfer-line_L64_T64_exp_lr1e-3_bidirectional_seed${SEED}"

  python scripts/train.py \
    --config "$CFG" \
    --seed "$SEED" \
    --run_dir "$RUN" \
    data.family=line \
    data.num_nodes=64 \
    sim.steps=64 \
    task.target_index=63 \
    coin.family=exp \
    model.controller.bidirectional=true \
    optim.lr=1e-3 \
    train.epochs=120
done

# 3) eval each seed-run
for SEED in "${MODEL_SEEDS[@]}"; do
  RUN="runs/exp1_transfer-line_L64_T64_exp_lr1e-3_bidirectional_seed${SEED}"
  CKPT="${RUN}/model_best.pt"

  python scripts/eval.py \
    --ckpt "$CKPT" \
    --outdir "${RUN}/eval/${SUITE}" \
    --suite "$SUITE" \
    --episodes "$EPISODES" \
    --seeds "${EVAL_SEEDS[@]}" \
    --ablations NoPE GlobalCoin TimeFrozen NodePermute \
    --plots
done

# 4) aggregate across model seeds (reads each run's eval summary.json)
python - <<'PY'
import glob, json, math, statistics
paths = sorted(glob.glob("runs/exp1_transfer-line_*_seed*/eval/basic_valid/summary.json"))
if not paths:
    raise SystemExit("No summary.json files found. Did eval run?")
summaries = []
for p in paths:
    j = json.load(open(p, "r"))
    base = (j.get("base") or {}).get("summary") or {}
    summaries.append(base)

keys = sorted({k for s in summaries for k in s.keys() if isinstance(s.get(k), (int, float))})
print(f"Found {len(paths)} runs.")
for k in keys:
    vals = [s[k] for s in summaries if k in s and isinstance(s[k], (int,float))]
    if len(vals) >= 2:
        m = statistics.mean(vals)
        se = statistics.stdev(vals) / math.sqrt(len(vals))
        print(f"{k}: {m:.6g} ± {se:.3g} (n={len(vals)})")
    elif len(vals) == 1:
        print(f"{k}: {vals[0]:.6g} (n=1)")
PY
