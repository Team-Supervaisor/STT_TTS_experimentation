#!/usr/bin/env bash
# LOCAL proof-of-concept on the 6 GB RTX 4050. This is a PIPELINE validation, not
# a faithful benchmark: the 12 GB (bf16) Qwen3-4B + 1.4B draft do not fit in 6 GB,
# so the target is loaded 4-bit (NF4). Quantizing the target perturbs the hidden
# states the DSpark draft was trained on, so acceptance/speedup here UNDER-reports
# the real numbers -- those come from the >=40 GB cloud run in BF16.
#
# Purpose: prove install -> load -> speculative-decode loop -> instrumentation all
# work end-to-end on this exact software stack before spending on cloud GPU time.
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

TARGET="${TARGET:-Qwen/Qwen3-4B}"
DRAFT="${DRAFT:-deepseek-ai/dspark_qwen3_4b_block7}"
LOAD_MODE="${LOAD_MODE:-4bit}"        # target precision
DRAFT_LOAD_MODE="${DRAFT_LOAD_MODE:-4bit}"  # draft too: bf16 draft alone OOMs 6 GB
NUM_SAMPLES="${NUM_SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/benchmark/results}"

# expandable_segments avoids fragmentation OOM on the tiny card
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

python3 benchmark/bench.py \
  --target-name-or-path "$TARGET" \
  --draft-name-or-path "$DRAFT" \
  --mode both \
  --load-mode "$LOAD_MODE" \
  --draft-load-mode "$DRAFT_LOAD_MODE" \
  --task gsm8k \
  --num-samples "$NUM_SAMPLES" \
  --warmup 1 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature 0.0 \
  --output "$OUT_DIR/local_qwen3_4b_${LOAD_MODE}.json"
