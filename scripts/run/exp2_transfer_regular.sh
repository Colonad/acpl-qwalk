#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

CFG="acpl/configs/train.yaml"
SUITE="basic_valid"
EPISODES=256
EVAL_SEEDS=(0 1 2 3 4)
MODEL_SEEDS=(0 1 2 3 4)

pytest -q

for SEED in "${MODEL_SEEDS[@]}"; do
  RUN="runs/exp2_transfer-regular_N64_d3_T64_exp_lr2.5e-4_seed${SEED}"

  python scripts/train.py \
    --config "$CFG" \
    --seed "$SEED" \
    --run_dir "$RUN" \
    data.family=regular \
    data.num_nodes=64 \
    data.d_reg=3 \
    data.seed=1234 \
    sim.steps=64 \
    task.target_index=-1 \
    coin.family=exp \
    optim.lr=2.5e-4 \
    train.epochs=160
done

for SEED in "${MODEL_SEEDS[@]}"; do
  RUN="runs/exp2_transfer-regular_N64_d3_T64_exp_lr2.5e-4_seed${SEED}"
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
