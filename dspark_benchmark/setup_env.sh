#!/usr/bin/env bash
# One-time environment setup for the DSpark benchmark. Works on a fresh cloud box
# or locally. Clones DeepSpec, wires this benchmark/ folder in, builds a venv, and
# installs the exact pinned deps validated on the RTX 4050 (torch 2.9.1+cu128).
#
#   scp -r benchmark/  user@cloud:~/           # copy this folder to the cloud box
#   ssh user@cloud 'bash ~/benchmark/setup_env.sh'
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$HOME/DeepSpec}"

if [ ! -d "$REPO_DIR/.git" ]; then
  echo "[setup] cloning DeepSpec -> $REPO_DIR"
  git clone --depth 1 https://github.com/deepseek-ai/DeepSpec.git "$REPO_DIR"
fi

# wire this benchmark folder into the repo (unless it already lives there)
if [ "$SCRIPT_DIR" != "$REPO_DIR/benchmark" ]; then
  echo "[setup] copying benchmark/ into $REPO_DIR"
  cp -r "$SCRIPT_DIR" "$REPO_DIR/benchmark"
fi

cd "$REPO_DIR"
if command -v uv >/dev/null 2>&1; then
  uv venv .venv --python 3.11
  source .venv/bin/activate
  uv pip install -r requirements.txt accelerate huggingface_hub nvidia-ml-py bitsandbytes
else
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -U pip
  pip install -r requirements.txt accelerate huggingface_hub nvidia-ml-py bitsandbytes
fi

echo
echo "[setup] done. Verifying CUDA..."
python3 - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "| gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"vram: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
PY
echo
echo "Next:  export HF_TOKEN=hf_...   (if models are gated for your account)"
echo "       bash $REPO_DIR/benchmark/run_cloud_gemma.sh"
