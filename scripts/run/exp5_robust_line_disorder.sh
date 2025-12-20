#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
pytest -q

RUN="runs/exp5_robust-line_L64_T64_sigma0.1_seed0"
python scripts/train.py \
  --config acpl/configs/train.yaml \
  --seed 0 \
  --run_dir "$RUN" \
  data.family=line data.num_nodes=64 \
  task.name=robust \
  sim.steps=64
