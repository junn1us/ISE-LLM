#!/usr/bin/env bash
set -euo pipefail

# Run this ON the remote Ubuntu server after SSH login.
# Usage:
#   bash scripts/remote_gpu_run_gpt.sh [options]
# Example:
#   bash scripts/remote_gpu_run_gpt.sh --gpus 0,1,2,3
#   bash scripts/remote_gpu_run_gpt.sh --mode long --background --log-file logs/gpt_long.log

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="0,1,2,3,4,5,6,7"
MODE="short"
BACKGROUND=0
LOG_FILE=""
SKIP_SMOKE=0

ITERS=""
EPISODES_PER_ITER=""
TEACHER_SAMPLES=""
OUT_FILE=""

usage() {
  cat <<'EOF'
Usage: bash scripts/remote_gpu_run_gpt.sh [options]

Options:
  --gpus <ids>                 Comma-separated GPU ids (default: 0,1,2,3,4,5,6,7)
  --mode <short|long>          short: quick validation, long: extended run
  --background                 Launch in background with nohup
  --log-file <path>            Append logs to this file
  --skip-smoke                 Skip GPT smoke test step
  --iters <n>                  Override PPO iterations
  --episodes-per-iter <n>      Override PPO episodes per iteration
  --teacher-samples <n>        Override distillation sample count
  --out <path>                 Override output checkpoint path
  -h, --help                   Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPU_IDS="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --background)
      BACKGROUND=1
      shift
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --skip-smoke)
      SKIP_SMOKE=1
      shift
      ;;
    --iters)
      ITERS="$2"
      shift 2
      ;;
    --episodes-per-iter)
      EPISODES_PER_ITER="$2"
      shift 2
      ;;
    --teacher-samples)
      TEACHER_SAMPLES="$2"
      shift 2
      ;;
    --out)
      OUT_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "short" && "$MODE" != "long" ]]; then
  echo "[ERROR] --mode must be one of: short, long"
  exit 1
fi

if [[ "$MODE" == "long" ]]; then
  : "${ITERS:=100}"
  : "${EPISODES_PER_ITER:=20}"
  : "${TEACHER_SAMPLES:=512}"
  : "${OUT_FILE:=ppo_fixture_policy_gpt_grid100_long.pt}"
else
  : "${ITERS:=5}"
  : "${EPISODES_PER_ITER:=5}"
  : "${TEACHER_SAMPLES:=32}"
  : "${OUT_FILE:=ppo_fixture_policy_gpt_grid100_smoketest.pt}"
fi

if [[ -z "$LOG_FILE" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="logs/gpt_${MODE}_${TS}.log"
fi

cd "$ROOT"
mkdir -p "$(dirname "$LOG_FILE")"

if [[ "$BACKGROUND" -eq 1 ]]; then
  CMD=(
    bash "$0"
    --gpus "$GPU_IDS"
    --mode "$MODE"
    --log-file "$LOG_FILE"
    --iters "$ITERS"
    --episodes-per-iter "$EPISODES_PER_ITER"
    --teacher-samples "$TEACHER_SAMPLES"
    --out "$OUT_FILE"
  )
  if [[ "$SKIP_SMOKE" -eq 1 ]]; then
    CMD+=(--skip-smoke)
  fi
  nohup "${CMD[@]}" >> "$LOG_FILE" 2>&1 &
  PID=$!
  echo "[LAUNCHED] PID=$PID"
  echo "[LAUNCHED] Log: $LOG_FILE"
  echo "[LAUNCHED] Monitor: tail -f $LOG_FILE"
  exit 0
fi

exec > >(tee -a "$LOG_FILE") 2>&1

if [[ -x "$ROOT/.venv-1/bin/python" ]]; then
  PY="$ROOT/.venv-1/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "[ERROR] python3 not found"
  exit 1
fi

echo "[INFO] Host: $(hostname)"
echo "[INFO] Working dir: $ROOT"
echo "[INFO] Python: $PY"
echo "[INFO] CUDA_VISIBLE_DEVICES=$GPU_IDS"
echo "[INFO] Mode: $MODE"
echo "[INFO] Log file: $LOG_FILE"
echo "[INFO] Config: iters=$ITERS episodes_per_iter=$EPISODES_PER_ITER teacher_samples=$TEACHER_SAMPLES out=$OUT_FILE"

nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false

if [[ "$SKIP_SMOKE" -eq 0 ]]; then
  echo "[STEP 1/2] GPT teacher smoke test"
  "$PY" scripts/run_gpt_teacher_smoke.py \
    --model-id openai/gpt-oss-120b \
    --anchor-mode grid100 \
    --grid-size 10 \
    --max-fixtures -1
else
  echo "[STEP 1/2] GPT teacher smoke test (skipped)"
fi

echo "[STEP 2/2] ${MODE} GPT E2E distill+PPO run"
"$PY" ppo_finetune.py \
  --pretrain-teacher \
  --force-scratch \
  --iters "$ITERS" \
  --episodes-per-iter "$EPISODES_PER_ITER" \
  --teacher-samples "$TEACHER_SAMPLES" \
  --anchor-mode grid100 \
  --grid-size 10 \
  --max-fixtures -1 \
  --hf-model-id openai/gpt-oss-120b \
  --hf-max-new-tokens 256 \
  --hf-temperature 0.2 \
  --hf-top-p 0.9 \
  --out "$OUT_FILE"

echo "[INFO] Final GPU status"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

echo "[DONE] Remote GPT workflow finished."
