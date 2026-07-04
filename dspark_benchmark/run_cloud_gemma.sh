#!/usr/bin/env bash
# Full Gemma-4-12B baseline-vs-DSpark benchmark. Run on a single GPU with >=40 GB
# VRAM (A100-40GB / L40S-48GB / H100-80GB) for clean BF16 numbers.
#
# A 24 GB card (4090 / L4 / A10) can only run this with LOAD_MODE=8bit, which
# perturbs the target's hidden states and lowers DSpark acceptance -> not a
# faithful "real" number. Use >=40 GB and BF16 for the headline comparison.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/DeepSpec}"
cd "$REPO_DIR"
source .venv/bin/activate

TARGET="${TARGET:-google/gemma-4-12B-it}"
DRAFT="${DRAFT:-deepseek-ai/dspark_gemma4_12b_block7}"
LOAD_MODE="${LOAD_MODE:-bf16}"       # bf16 (>=40GB) | 8bit (24GB, degraded) | 4bit
TASK="${TASK:-gsm8k}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"    # 0.0 = greedy => DSpark is lossless => clean speedup
OUT_DIR="${OUT_DIR:-$REPO_DIR/benchmark/results}"
mkdir -p "$OUT_DIR"
STAMP="gemma4_12b_${LOAD_MODE}_${TASK}"

echo "[download] target + draft (this pulls ~30 GB the first time)"
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
echo "Tip: sweep tasks with:  for t in gsm8k humaneval mt-bench; do TASK=\$t bash benchmark/run_cloud_gemma.sh; done"
