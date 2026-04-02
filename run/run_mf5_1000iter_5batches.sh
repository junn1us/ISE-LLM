#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv-1/bin/python"

HF_MODEL_ID="${HF_MODEL_ID:-}"
HF_MAX_NEW_TOKENS="${HF_MAX_NEW_TOKENS:-256}"
HF_TEMPERATURE="${HF_TEMPERATURE:-0.2}"
HF_TOP_P="${HF_TOP_P:-0.9}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-model-id)
      HF_MODEL_ID="$2"
      shift 2
      ;;
    --hf-max-new-tokens)
      HF_MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --hf-temperature)
      HF_TEMPERATURE="$2"
      shift 2
      ;;
    --hf-top-p)
      HF_TOP_P="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: bash scripts/run_mf5_1000iter_5batches.sh [--hf-model-id <id>] [--hf-max-new-tokens <n>] [--hf-temperature <t>] [--hf-top-p <p>]"
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ ! -x "$PY" ]]; then
  echo "[ERROR] Python not found at $PY"
  exit 1
fi

cd "$ROOT"
mkdir -p "$ROOT/logs"

HF_ARGS=()
if [[ -n "$HF_MODEL_ID" ]]; then
  HF_ARGS=(
    --hf-model-id "$HF_MODEL_ID"
    --hf-max-new-tokens "$HF_MAX_NEW_TOKENS"
    --hf-temperature "$HF_TEMPERATURE"
    --hf-top-p "$HF_TOP_P"
  )
  echo "[INFO] GPT teacher enabled: $HF_MODEL_ID"
else
  echo "[INFO] Using default local teacher (no HF model id provided)."
fi

declare -a BATCHES=(5 6 7 8 9)
for B in "${BATCHES[@]}"; do
  case "$B" in
    5) SEEDS=(61 62 63 64) ;;
    6) SEEDS=(65 66 67 68) ;;
    7) SEEDS=(69 70 71 72) ;;
    8) SEEDS=(73 74 75 76) ;;
    9) SEEDS=(77 78 79 80) ;;
    *) echo "[ERROR] Unknown batch $B"; exit 1 ;;
  esac

  echo "[BATCH $B] seeds=${SEEDS[*]}"
  CKPTS=""
  SEEDCSV=""

  for S in "${SEEDS[@]}"; do
    OUT="$ROOT/ppo_fixture_policy_local_1000iter_mf5_b${B}_s${S}.pt"
    echo "[TRAIN] batch=$B seed=$S out=$(basename "$OUT")"
    "$PY" "$ROOT/ppo_finetune.py" \
      --pretrain-teacher \
      --force-scratch \
      --iters 1000 \
      --episodes-per-iter 5 \
      --teacher-samples 64 \
      --anchor-mode grid100 \
      --grid-size 10 \
      --max-fixtures 5 \
      --seed "$S" \
      --out "$OUT" \
      "${HF_ARGS[@]}"

    if [[ -z "$CKPTS" ]]; then
      CKPTS="$OUT"
      SEEDCSV="$S"
    else
      CKPTS="$CKPTS,$OUT"
      SEEDCSV="$SEEDCSV,$S"
    fi
  done

  echo "[COMPARE] batch=$B"
  "$PY" "$ROOT/scripts/compare_deformation_runs.py" \
    --checkpoints "$CKPTS" \
    --seeds "$SEEDCSV" \
    --anchor-mode grid100 \
    --grid-size 10 \
    --max-fixtures 5 \
    --show-fixtures \
    --enforce-fixture-count 5 \
    --out-figure "$ROOT/outputs/deformation_comparison_1000iter_mf5_batch${B}_5fixtures.png" \
    --out-metrics "$ROOT/outputs/deformation_metrics_1000iter_mf5_batch${B}_5fixtures.csv" \
    --out-pairwise "$ROOT/outputs/deformation_pairwise_1000iter_mf5_batch${B}_5fixtures.csv"
done

echo "[SUMMARY] aggregating batches 5-9"
"$PY" "$ROOT/scripts/summarize_mf5_1000iter_batches.py"

echo "[DONE] Completed 1000-iter MF5 process for 5 additional batches."
