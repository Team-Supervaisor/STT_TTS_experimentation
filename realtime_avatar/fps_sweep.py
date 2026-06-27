"""Sweep target video fps at batch=4 to find the realtime crossover on this GPU."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.backends.cudnn.benchmark = True
from musetalk_engine import MuseTalkEngine

AUDIO = os.environ.get("VERIFY_AUDIO", "data/audio/eng_6s.wav")
eng = MuseTalkEngine(version="v15", batch_size=4, fps=25)
eng.prepare_avatar("data/avatars/sample_face.png", avatar_id="sample_v15", force=False)

for _ in eng.stream_frames(AUDIO):  # warmup (cudnn autotune)
    pass

for fps in [25, 15, 12, 10]:
    eng.fps = fps
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); tf = None; n = 0
    for _f in eng.stream_frames(AUDIO):
        if tf is None:
            tf = time.time() - t0
        n += 1
    dt = time.time() - t0
    audio_s = n / fps
    tag = "REALTIME-OK" if dt <= audio_s else "too slow"
    print(f"[fps] target={fps:2d}  frames={n}  gen={dt:5.2f}s  genfps={n/dt:5.1f}  "
          f"ttff={tf*1000:4.0f}ms  rt_factor={dt/audio_s:.2f}x  {tag}", flush=True)
