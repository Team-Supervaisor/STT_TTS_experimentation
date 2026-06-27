"""Standalone end-to-end check of the MuseTalk streaming engine (no UI, no TTS).

Loads the engine, prepares the sample avatar, streams frames for a bundled
example wav, measures latency/throughput/VRAM, and writes an mp4 to eyeball.

Run (inside MuseTalk venv, from anywhere):
    MUSETALK_ROOT=/.../MuseTalk \
    /.../MuseTalk/.venv/bin/python /.../realtime_avatar/verify_engine.py
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from musetalk_engine import MuseTalkEngine  # chdir -> MuseTalk root on import

import torch

AVATAR = os.environ.get("VERIFY_AVATAR", "data/avatars/sample_face.png")
AUDIO = os.environ.get("VERIFY_AUDIO", "data/audio/eng.wav")
FPS = int(os.environ.get("MUSETALK_FPS", "25"))
BBOX_SHIFT = int(os.environ.get("MUSETALK_BBOX_SHIFT", "0"))
EXTRA_MARGIN = int(os.environ.get("MUSETALK_EXTRA_MARGIN", "10"))
# Derive a cache id from the avatar filename so different avatars don't collide.
AVATAR_ID = os.environ.get("VERIFY_AVATAR_ID", os.path.splitext(os.path.basename(AVATAR))[0])
OUT_DIR = "results/verify"
os.makedirs(OUT_DIR, exist_ok=True)
DISP_W = 512


def main():
    t0 = time.time()
    eng = MuseTalkEngine(version="v15", batch_size=4, fps=FPS, extra_margin=EXTRA_MARGIN)
    print(f"[verify] model load: {time.time()-t0:.1f}s  (fps={FPS} bbox_shift={BBOX_SHIFT} extra_margin={EXTRA_MARGIN})")

    t1 = time.time()
    eng.prepare_avatar(AVATAR, avatar_id=AVATAR_ID, bbox_shift=BBOX_SHIFT, force=True)
    print(f"[verify] avatar prep: {time.time()-t1:.1f}s")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    frames = []
    t_start = time.time()
    t_first = None
    for f in eng.stream_frames(AUDIO):
        if t_first is None:
            t_first = time.time() - t_start
            print(f"[verify] time-to-first-frame: {t_first*1000:.0f} ms")
        h, w = f.shape[:2]
        if w > DISP_W:
            f = cv2.resize(f, (DISP_W, int(h * DISP_W / w)), interpolation=cv2.INTER_AREA)
        frames.append(f)
    gen_s = time.time() - t_start

    n = len(frames)
    fps_eff = n / gen_s if gen_s > 0 else 0
    audio_s = n / eng.fps
    print(f"[verify] frames={n}  gen={gen_s:.2f}s  throughput={fps_eff:.1f} fps  "
          f"(target {eng.fps})  audio_len={audio_s:.2f}s  realtime_factor={gen_s/max(audio_s,1e-6):.2f}x")
    if torch.cuda.is_available():
        print(f"[verify] peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # write mp4 (silent) then mux audio
    if n:
        hh, ww = frames[0].shape[:2]
        silent = os.path.join(OUT_DIR, f"_silent_{AVATAR_ID}.mp4")
        vw = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), eng.fps, (ww, hh))
        for f in frames:
            vw.write(f)
        vw.release()
        out = os.path.join(OUT_DIR, f"sample_{AVATAR_ID}_{FPS}fps.mp4")
        os.system(f'ffmpeg -y -v error -i "{silent}" -i "{AUDIO}" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "{out}"')
        print(f"[verify] wrote {os.path.abspath(out)}")
    print(f"[verify] TOTAL {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
