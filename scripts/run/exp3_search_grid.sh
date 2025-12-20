#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

pytest -q

# Placeholder run_dir naming (adjust once the search trainer is active)
RUN="runs/exp3_search-grid_L32_T64_seed0"

# Expected pattern once task dispatch exists:
python scripts/train.py \
  --config acpl/configs/train.yaml \
  --seed 0 \
  --run_dir "$RUN" \
  data.family=grid data.grid.Lx=32 data.grid.Ly=32 \
  task.name=search \
  sim.steps=64
