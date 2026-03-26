#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/junn1us/Library/Mobile Documents/com~apple~CloudDocs/ISYE URP"
PY="$ROOT/.venv-1/bin/python"

cd "$ROOT"

SEEDS=(18 19 20)
ITERS=200
EPISODES_PER_ITER=50

for SEED in "${SEEDS[@]}"; do
  OUT="ppo_fixture_policy_seed${SEED}_it${ITERS}.pt"
  echo "[RUN] seed=${SEED} iters=${ITERS} episodes_per_iter=${EPISODES_PER_ITER} out=${OUT}"
  "$PY" ppo_finetune.py \
    --pretrain-teacher \
    --force-scratch \
    --iters "$ITERS" \
    --episodes-per-iter "$EPISODES_PER_ITER" \
    --seed "$SEED" \
    --out "$OUT"
done

echo "[DONE] Multiseed training complete."
