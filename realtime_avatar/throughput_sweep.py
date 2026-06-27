"""Sweep batch size (and optionally model version) to find the fastest config
on this GPU. Engine is loaded once; avatar is cached."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.backends.cudnn.benchmark = True
from musetalk_engine import MuseTalkEngine

VERSION = os.environ.get("SWEEP_VERSION", "v15")
AUDIO = os.environ.get("VERIFY_AUDIO", "data/audio/eng_6s.wav")
AVATAR = "data/avatars/sample_face.png"
BATCHES = [int(x) for x in os.environ.get("SWEEP_BATCHES", "4,8,16,32").split(",")]

eng = MuseTalkEngine(version=VERSION, batch_size=BATCHES[0], fps=25)
eng.prepare_avatar(AVATAR, avatar_id=f"sample_{VERSION}", force=False)

# warmup (cudnn autotune + cache whisper path)
for _ in eng.stream_frames(AUDIO):
    pass

for bs in BATCHES:
    eng.batch_size = bs
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    t_first = None
    n = 0
    for _f in eng.stream_frames(AUDIO):
        if t_first is None:
            t_first = time.time() - t0
        n += 1
    dt = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    audio_s = n / eng.fps
    print(f"[sweep] {VERSION} batch={bs:3d}  frames={n}  gen={dt:5.2f}s  "
          f"fps={n/dt:5.1f}  ttff={t_first*1000:4.0f}ms  rt_factor={dt/audio_s:.2f}x  vram={vram:.2f}GB",
          flush=True)
