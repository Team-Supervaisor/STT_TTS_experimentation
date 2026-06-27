"""Pre-build the avatar cache (face detect + latents + masks) as a standalone
step while the GPU is free. After this, the server loads from cache and never
needs dwpose/face-align/face-parse, keeping its resident VRAM ~2 GB so it fits
alongside Chatterbox on a 6 GB GPU.

Run (MuseTalk venv):
    MUSETALK_ROOT=/.../MuseTalk AVATAR_IMAGE=... AVATAR_ID=... \
    /.../MuseTalk/.venv/bin/python realtime_avatar/prep_avatar.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from musetalk_engine import MuseTalkEngine

IMG = os.environ.get("AVATAR_IMAGE", "data/avatars/sample_face.png")
AID = os.environ.get("AVATAR_ID", "sample")
FPS = int(os.environ.get("MUSETALK_FPS", "25"))
BBOX_SHIFT = int(os.environ.get("MUSETALK_BBOX_SHIFT", "0"))
EXTRA_MARGIN = int(os.environ.get("MUSETALK_EXTRA_MARGIN", "10"))
FORCE = os.environ.get("PREP_FORCE", "1") == "1"

eng = MuseTalkEngine(version="v15", batch_size=4, fps=FPS, extra_margin=EXTRA_MARGIN)
eng.prepare_avatar(IMG, avatar_id=AID, bbox_shift=BBOX_SHIFT, force=FORCE)
print("AVATAR_CACHE_READY", flush=True)
