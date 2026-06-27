#!/usr/bin/env bash
# Launch the realtime avatar demo on a single 6 GB GPU:
#   0) pre-build the avatar cache while the GPU is free (face detect + latents)
#   1) Chatterbox TTS service  (main venv, T3 in fp16 ~2.3 GB)
#   2) MuseTalk + UI server     (MuseTalk venv, loads from cache ~2.4 GB)
# Both fit together with ~0.5 GB headroom. Ctrl-C stops both. Open the printed URL.
set -uo pipefail

PROJECT=/home/pragay/WWAI/STT_TTS_experimentation
APP="$PROJECT/realtime_avatar"
MUSE_ROOT="$PROJECT/MuseTalk"
MAIN_PY=/home/pragay/WWAI/.venv/bin/python
MUSE_PY="$MUSE_ROOT/.venv/bin/python"

TTS_PORT="${TTS_PORT:-8765}"
UI_PORT="${UI_PORT:-8011}"          # 8000 is often taken; override with UI_PORT=...
export TTS_URL="http://127.0.0.1:${TTS_PORT}"
export AVATAR_IMAGE="${AVATAR_IMAGE:-data/video/sun.mp4}"
export AVATAR_ID="${AVATAR_ID:-sun}"
export MUSETALK_BATCH="${MUSETALK_BATCH:-2}"
export MUSETALK_FPS="${MUSETALK_FPS:-25}"   # smooth mouth; on a 6 GB GPU set =10 for local realtime
export DISPLAY_WIDTH="${DISPLAY_WIDTH:-512}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cleanup() { echo; echo "stopping..."; [ -n "${TTS_PID:-}" ] && kill $TTS_PID 2>/dev/null; [ -n "${UI_PID:-}" ] && kill $UI_PID 2>/dev/null; wait 2>/dev/null; }
trap cleanup INT TERM EXIT

echo "[0] pre-building avatar cache (GPU free; loads dwpose/face models once)"
( cd "$APP" && MUSETALK_ROOT="$MUSE_ROOT" PREP_FORCE="${PREP_FORCE:-0}" "$MUSE_PY" prep_avatar.py ) \
  || { echo "    avatar prep failed"; exit 1; }

echo "[1] starting Chatterbox TTS service on :$TTS_PORT"
"$MAIN_PY" -m uvicorn tts_service:app --app-dir "$APP" --host 127.0.0.1 --port "$TTS_PORT" --log-level warning &
TTS_PID=$!
echo "    waiting for TTS (loads + warms up Chatterbox)..."
for i in $(seq 1 120); do
  curl -fsS "http://127.0.0.1:${TTS_PORT}/health" 2>/dev/null | grep -q '"ready": *true' && { echo "    TTS ready."; break; }
  kill -0 $TTS_PID 2>/dev/null || { echo "    TTS failed to start"; exit 1; }
  sleep 2
done

echo "[2] starting MuseTalk + UI server on :$UI_PORT"
echo "    >>> open  http://localhost:${UI_PORT}  in your browser <<<"
MUSETALK_ROOT="$MUSE_ROOT" "$MUSE_PY" -m uvicorn server:app --app-dir "$APP" --host 0.0.0.0 --port "$UI_PORT" --log-level info &
UI_PID=$!
wait $UI_PID
