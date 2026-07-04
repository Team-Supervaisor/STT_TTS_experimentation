#!/usr/bin/env bash
# Full Qwen3-14B baseline-vs-DSpark benchmark. Run on a single GPU with >=40 GB
# VRAM (Qwen3-14B bf16 ~28 GB + ~3B draft ~6 GB + KV). H100-80GB is comfortable.
#
# Companion to run_cloud_gemma.sh (Gemma-4-12B) and run_local_qwen.sh (6 GB PoC).
# For other Qwen sizes, override TARGET/DRAFT, e.g.:
#   TARGET=Qwen/Qwen3-8B DRAFT=deepseek-ai/dspark_qwen3_8b_block7 bash run_cloud_qwen.sh
set -euo pipefail

# Locate the DeepSpec clone (must contain deepspec/ + benchmark/bench.py). Honors $REPO_DIR.
if [ -z "${REPO_DIR:-}" ]; then
  _sd="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if   [ -f "$_sd/../deepspec/__init__.py" ]; then REPO_DIR="$(cd "$_sd/.." && pwd)"
  elif [ -f "$HOME/DeepSpec/deepspec/__init__.py" ]; then REPO_DIR="$HOME/DeepSpec"
  else echo "ERROR: set REPO_DIR=/path/to/DeepSpec (the dir containing deepspec/ and benchmark/)"; exit 1; fi
fi
cd "$REPO_DIR"
source .venv/bin/activate

TARGET="${TARGET:-Qwen/Qwen3-14B}"
DRAFT="${DRAFT:-deepseek-ai/dspark_qwen3_14b_block7}"
LOAD_MODE="${LOAD_MODE:-bf16}"       # bf16 (>=40GB) | 8bit (24GB, degraded) | 4bit
TASK="${TASK:-gsm8k}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"    # 0.0 = greedy => DSpark is lossless => clean speedup
OUT_DIR="${OUT_DIR:-$REPO_DIR/benchmark/results}"
mkdir -p "$OUT_DIR"
# stamp from the target basename so alternate sizes get distinct files
STAMP="$(basename "$TARGET" | tr '[:upper:]' '[:lower:]' | tr -d '.')_${LOAD_MODE}_${TASK}"

echo "[download] target + draft (this pulls ~34 GB the first time)"
hf download "$TARGET"
hf download "$DRAFT"

echo "[run] baseline vs DSpark | target=$TARGET load=$LOAD_MODE task=$TASK"
python3 benchmark/bench.py \
  --target-name-or-path "$TARGET" \
  --draft-name-or-path "$DRAFT" \
  --mode both \
  --load-mode "$LOAD_MODE" \
  --task "$TASK" \
  --num-samples "$NUM_SAMPLES" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --output "$OUT_DIR/${STAMP}.json"

echo
echo "[done] results -> $OUT_DIR/${STAMP}.json"
