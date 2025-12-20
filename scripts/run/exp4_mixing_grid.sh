#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
pytest -q

RUN="runs/exp4_mixing-grid_L32_T128_seed0"
python scripts/train.py \
  --config acpl/configs/train.yaml \
  --seed 0 \
  --run_dir "$RUN" \
  data.family=grid data.grid.Lx=32 data.grid.Ly=32 \
  task.name=mixing \
  sim.steps=128
