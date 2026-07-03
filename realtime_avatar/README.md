# Realtime MuseTalk Avatar (Chatterbox TTS → lip-synced photo)

Type text in a browser; a photo-realistic avatar speaks it back with MuseTalk
V1.5 lip-sync. Video frames are **streamed as they are generated** (not rendered
to a file first), driven off the audio clock for A/V sync.

## Architecture (two processes — forced by a dependency conflict)

Chatterbox pins `transformers==5.2.0` / `diffusers==0.29.0`; MuseTalk needs
`transformers==4.39.2` / `diffusers==0.30.2`. They cannot share a venv, so:

```
 browser ──ws──► server.py (MuseTalk venv)            tts_service.py (main venv)
   ▲  audio+frames │  split into sentences                 │ Chatterbox text→wav
   └───────────────┤  for each sentence:                   │
                   ├─ POST /tts ──────────────────────────►│  (24 kHz wav)
                   │◄──────────── wav ──────────────────────┘
                   └─ MuseTalkEngine.stream_frames(wav) → push JPEG frames over ws
```

- `tts_service.py` — FastAPI, **main venv** (`/home/pragay/WWAI/.venv`). `POST /tts`.
- `server.py` — FastAPI + WebSocket UI, **MuseTalk venv** (`MuseTalk/.venv`).
- `musetalk_engine.py` — loads MuseTalk once, prepares the avatar once (cached to
  `MuseTalk/results/realtime_avatars/`), streams frames per audio clip.
- `static/index.html` — browser UI; plays each sentence's audio gaplessly via Web
  Audio and renders the frame matching the current audio time.

Sentences **pipeline**: while the browser plays sentence N, the server is already
generating sentence N+1 — so playback is progressive, not "render everything then
play".

## Run

```bash
bash realtime_avatar/run_all.sh
# then open http://localhost:8000
```

Environment knobs (optional):
- `AVATAR_IMAGE` — avatar source: a **photo OR a short idle video** (mp4/mov/...).
  Path is absolute or relative to `MuseTalk/`. Default `data/avatars/sample_face.png`.
- `AVATAR_ID` — cache key. **Use a new id for a new avatar** (else it reuses the old
  cached prep). Or set `PREP_FORCE=1` to rebuild the same id.
- `MUSETALK_FPS` — render fps. Default 25 (smooth; H100 does this realtime). On a
  6 GB GPU set `=10` for local realtime, or judge quality via offline renders.
- `MUSETALK_BBOX_SHIFT` / `MUSETALK_EXTRA_MARGIN` — mouth-crop tuning (see below).
- Idle loop: when not speaking, the client loops the avatar's base frames (natural
  head motion) instead of freezing. The base clip is sampled at the **render fps**
  and capped at 6 s, so motion plays at **real-time speed**.
  - `IDLE_MAX` — idle-loop length in frames (default 120 ≈ 4.8 s at 25 fps).
  - `IDLE_FPS` — override idle playback fps (default = render fps). Lower it (e.g.
    `IDLE_FPS=18`) for an even calmer idle.
  - Use a calm idle source clip (person at rest) so the resting face looks natural.
- `MUSETALK_BATCH`, `DISPLAY_WIDTH`, `TTS_DEVICE=cpu`.

```bash
# new avatar from an idle video, smooth 25 fps:
AVATAR_IMAGE=/path/to/idle_clip.mp4 AVATAR_ID=myperson bash run_all.sh
```

## Improving lip-sync quality ("feels like it's actually speaking")

1. **Drive from a short idle video, not a still photo.** MuseTalk only repaints the
   mouth; on a frozen photo the head/eyes don't move, which reads as a puppet. A
   few-second clip with natural micro-motion + blinks makes it look alive. Pass the
   video as `AVATAR_IMAGE` (frames are auto-extracted, capped at 160, looped
   ping-pong). This is the single biggest improvement.
2. **Use 25 fps.** At 10 fps the mouth is choppy; 25 fps is far more natural (default
   now; realtime on an H100).
3. **Tune the mouth crop** if articulation looks off: `MUSETALK_BBOX_SHIFT` (try -7..+7;
   shifts the mouth region up/down) and `MUSETALK_EXTRA_MARGIN` (chin room, default 10).

### Evaluate quality offline (no realtime lag)

`verify_engine.py` renders a full MP4 at the target fps so you can judge quality
without the live-stream lag (useful on the 6 GB box):

```bash
cd MuseTalk
MUSETALK_ROOT=$PWD VERIFY_AVATAR=data/video/yongen.mp4 \
  VERIFY_AUDIO=data/audio/eng.wav MUSETALK_FPS=25 \
  .venv/bin/python ../realtime_avatar/verify_engine.py
# writes results/verify/sample_<id>_25fps.mp4
```

## Performance on this machine (RTX 4050 Laptop, 6 GB) — measured

- MuseTalk lip-sync throughput maxes at **~10 fps** here (small batches are fastest;
  the GPU saturates at 256×256, so batch 16/32 are *slower*). So the demo renders at
  **fps=10**, where generation keeps pace with realtime audio (rt_factor ≈ 1.0).
- After startup warmup: **time-to-first-audio ≈ 0.8–1.8 s**, **time-to-first-frame
  ≈ 2–4.5 s** (first sentence), then continuous.
- **Both models fit in 6 GB**: Chatterbox ~2.3 GB (T3 in fp16) + MuseTalk ~2.4 GB
  (loaded from avatar cache) = ~5.2 GB used, ~0.5 GB free at batch=2.

How the 6 GB fit is achieved (all automatic in the scripts):
1. **Chatterbox T3 token model runs in fp16** (vocoder stays fp32) → ~3.2 → ~2.3 GB.
2. TTS **warms up then `empty_cache()` at startup** so its steady-state (~2.3 GB) is
   reached *before* MuseTalk loads (avoids a transient-peak OOM).
3. MuseTalk loads the UNet **fp16 on CPU then moves to GPU** (low load peak) and
   loads the avatar **from cache**, so dwpose/face-align/face-parse (prep-only,
   ~0.4 GB) are never resident during serving.
4. Avatar cache is **pre-built while the GPU is free** (step [0] in run_all.sh).

If you hit OOM (e.g. other GPU users), set `MUSETALK_BATCH=1` or `TTS_DEVICE=cpu`.

The design keeps **audio realtime and continuous**; video is driven off the audio
clock and holds the last frame if generation briefly lags.

## Integrating into a realtime voice bot

See **[INTEGRATION.md](INTEGRATION.md)** for the full step-by-step guide (architecture,
options, the engine API, audio contract, A/V sync, idle states, and a production
checklist).

In short: `MuseTalkEngine.stream_frames(audio)` is the integration point. Run it in
the MuseTalk venv as a separate worker, feed it your bot's TTS audio (per sentence),
and forward the yielded frames to your client.
