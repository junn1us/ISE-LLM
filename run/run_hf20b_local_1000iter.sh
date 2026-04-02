#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv-1/bin/python"

MODEL_ID="openai/gpt-oss-20b"
CACHE_DIR="${ROOT}/hf_cache"
HF_MAX_NEW_TOKENS="256"
HF_TEMPERATURE="0.2"
HF_TOP_P="0.9"
SEED="91"
OUT_CKPT="${ROOT}/ppo_fixture_policy_local_1000iter_hf20b_s91.pt"
LOCAL_ONLY="0"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_hf20b_local_1000iter.sh [options]

Options:
  --seed <n>                 Seed (default: 91)
  --cache-dir <path>         HF cache directory (default: <repo>/hf_cache)
  --out <path>               Output checkpoint path
  --local-files-only         Do not download; load only from local cache
  --hf-max-new-tokens <n>    Generation max tokens (default: 256)
  --hf-temperature <v>       Sampling temperature (default: 0.2)
  --hf-top-p <v>             Sampling top-p (default: 0.9)
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      SEED="$2"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --out)
      OUT_CKPT="$2"
      shift 2
      ;;
    --local-files-only)
      LOCAL_ONLY="1"
      shift
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

if [[ ! -x "$PY" ]]; then
  echo "[ERROR] Python not found: $PY"
  exit 1
fi

mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR"

cd "$ROOT"

if [[ "$LOCAL_ONLY" != "1" ]]; then
  echo "[STEP] Prefetching ${MODEL_ID} to ${CACHE_DIR}"
  "$PY" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${MODEL_ID}",
    cache_dir="${CACHE_DIR}",
)
print("prefetch_done")
PY
else
  echo "[STEP] Local-files-only mode enabled; skipping download"
fi

echo "[STEP] Running 1000-iter training with HF teacher (${MODEL_ID})"
CMD=(
  "$PY" "$ROOT/ppo_finetune.py"
  --pretrain-teacher
  --force-scratch
  --iters 1000
  --episodes-per-iter 5
  --teacher-samples 64
  --anchor-mode grid100
  --grid-size 10
  --max-fixtures 5
  --seed "$SEED"
  --hf-model-id "$MODEL_ID"
  --hf-max-new-tokens "$HF_MAX_NEW_TOKENS"
  --hf-temperature "$HF_TEMPERATURE"
  --hf-top-p "$HF_TOP_P"
  --hf-cache-dir "$CACHE_DIR"
  --out "$OUT_CKPT"
)

if [[ "$LOCAL_ONLY" == "1" ]]; then
  CMD+=(--hf-local-files-only)
fi

"${CMD[@]}"

echo "[DONE] HF20B 1000-iter run completed: $OUT_CKPT"
