# Integrating the Avatar into a Realtime Voice Bot

This guide shows how to add the MuseTalk lip-sync avatar to an existing realtime
voice bot (STT → LLM → TTS). For running/tuning the standalone demo, see
[README.md](README.md).

---

## TL;DR — the whole integration surface

The avatar is one class with one streaming method. Feed it the **audio your bot's
TTS already produces**; it yields **lip-synced video frames** as they are generated.

```python
from musetalk_engine import MuseTalkEngine

engine = MuseTalkEngine(version="v15", fps=25)     # load once at startup
engine.prepare_avatar("idle_clip.mp4", avatar_id="agent")   # once per avatar (cached)
engine.warmup()

# ...per bot reply, once you have the reply's audio as a wav file:
for frame_bgr in engine.stream_frames("reply.wav"):   # generator, yields as it decodes
    jpg = cv2.imencode(".jpg", frame_bgr)[1].tobytes()
    send_to_client(jpg)                                # + play reply.wav on the client
```

Everything else (the WebSocket server, the TTS service, the browser page) is a
**reference implementation** of that loop you can copy or replace.

---

## The one hard rule: MuseTalk runs in its own process

MuseTalk pins `transformers==4.39.2` / `diffusers==0.30.2`; most modern TTS
(including Chatterbox) pin `transformers>=5`. **They cannot share a Python venv.**

So in your bot, the avatar is a **separate process/service** with its own venv
(`MuseTalk/.venv`, Python 3.10 + torch 2.0.1). Your bot talks to it over a socket
(WebSocket/gRPC/ZeroMQ) or runs it as a dedicated worker. You do **not** import
`MuseTalkEngine` into your main bot process unless that process happens to have no
conflicting deps.

---

## Where it fits in the voice-bot loop

```
  mic ─► STT ─► LLM ─► TTS ──audio──►  MuseTalk engine  ──frames──►  client
                                   (stream_frames)          (renders video,
                                                             plays the audio)
```

The avatar is a **pure consumer of TTS audio**. It does not care how the audio was
produced. The only new data flowing out is a stream of video frames, time-aligned
to that audio at a fixed fps.

---

## Pick an integration option

### Option A — Reuse the bundled server, point it at your TTS (fastest)

`server.py` already implements text → TTS → lip-sync → WebSocket streaming, with
sentence pipelining and an idle loop. It calls a TTS HTTP service via `TTS_URL`.

- Your bot connects to `ws://host:8011/ws` and sends `{"type":"speak","text":"..."}`.
- To use **your** TTS instead of Chatterbox, make it expose `POST /tts {text} → WAV
  bytes` and set `TTS_URL=http://your-tts:port`. (Contract: 16-bit PCM WAV, any
  sample rate; the engine resamples.) See `tts_service.py` for the shape.
- The server streams back `sentence` (audio) and `frame` (JPEG) messages; the
  browser client in `static/index.html` shows how to play them in sync.

Good when your bot can hand off *text* and let this service own TTS+video.

### Option B — Write a thin worker around `MuseTalkEngine` (recommended for custom bots)

When your bot already produces audio (its own streaming TTS), skip the bundled TTS
and drive the engine directly. Minimal worker (runs in the MuseTalk venv):

```python
# avatar_worker.py  — run with MuseTalk/.venv/bin/python, MUSETALK_ROOT set
import os, io, cv2, numpy as np, soundfile as sf
from musetalk_engine import MuseTalkEngine     # chdir -> MUSETALK_ROOT on import

engine = MuseTalkEngine(version="v15", fps=25, batch_size=2)
engine.prepare_avatar(os.environ["AVATAR_SRC"], avatar_id="agent")
engine.warmup()

def synth_video(wav_path):
    """Yield JPEG-encoded lip-sync frames for one utterance's audio."""
    for frame_bgr in engine.stream_frames(wav_path):
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            yield buf.tobytes()

# Wrap synth_video() in your transport of choice: a WebSocket endpoint, a gRPC
# stream, a ZeroMQ PUB socket — whatever your bot already uses. Your bot sends the
# reply audio (a wav path or bytes); this worker streams frames back.
```

Your bot's turn handler then does, per reply:
1. get reply audio from your TTS (per sentence is ideal — lower latency),
2. send it to the avatar worker,
3. forward the returned frames to the client while the client plays the audio.

### Option C — In-process (only if your bot has no `transformers>=5` dependency)

If (and only if) your bot's runtime can live in the MuseTalk venv, you can import
`MuseTalkEngine` directly and call `stream_frames` inline. Rare in practice.

---

## The engine API (reference)

All in `musetalk_engine.py`. Runs in the MuseTalk venv; importing it `chdir`s to
`MUSETALK_ROOT` (set the env var or it defaults to the repo path).

| Call | What it does |
|---|---|
| `MuseTalkEngine(version="v15", fps=25, batch_size=2, extra_margin=10, avatar_max_seconds=6)` | Load UNet+VAE+Whisper. Call **once** at startup. |
| `prepare_avatar(src, avatar_id, bbox_shift=0, force=False)` | One-time per avatar. `src` = photo **or** idle video. Caches to `results/realtime_avatars/v15_<id>/`. Loads face-detect models only when building (not when serving from cache). |
| `warmup()` | One dummy inference so cuDNN autotune doesn't add seconds to the first reply. |
| `stream_frames(audio_path) -> Iterator[np.ndarray]` | **The integration point.** Yields blended **BGR uint8** frames (full avatar frame) one at a time, emitting each UNet batch as it decodes. |
| `idle_frame() -> np.ndarray` | The avatar's neutral base frame (BGR). |

**Frame format:** `np.ndarray`, shape `(H, W, 3)`, **BGR**, uint8 (OpenCV order). You
encode it (JPEG/PNG) or feed it to a video encoder. Size = the prepared avatar's
frame size (source downscaled to ≤768px by default).

---

## Audio contract & how to feed streaming TTS

- `stream_frames` takes a **path to a WAV** (mono; any sample rate — it resamples to
  16 kHz internally for the Whisper features).
- **Per-utterance / per-sentence is the practical unit.** Most streaming TTS emit
  audio sentence-by-sentence; call `stream_frames` once per sentence and pipeline
  (generate sentence N+1's audio while streaming sentence N's frames). The bundled
  `server.py` shows this pipelining.
- **In-memory audio (no temp file):** the engine currently reads a path. Either
  write the TTS output to a temp `.wav` (simplest), or add a small array variant:

  ```python
  # convenience wrapper if you have audio as a numpy array
  import tempfile, soundfile as sf
  def stream_frames_from_array(engine, audio, sr):
      with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
          sf.write(f.name, audio, sr); path = f.name
      try:
          yield from engine.stream_frames(path)
      finally:
          os.unlink(path)
  ```

  (If you want, this can be added as a first-class `engine.stream_frames_array()`.)

---

## A/V sync on the client

Drive video off the **audio clock**, not wall-clock:

1. Start playing the utterance's audio.
2. For the current audio time `t`, show frame index `floor(t * fps)`.
3. If that frame hasn't arrived yet (generation briefly lagging), **hold the last
   frame** rather than stalling audio. Audio stays continuous; video catches up.

`static/index.html` implements exactly this (Web Audio + a requestAnimationFrame
loop). On an H100 generation keeps up at 25 fps so there's no lag.

---

## Idle / listening / thinking states (don't freeze)

A talking head that freezes between turns looks dead. When the bot is **listening or
thinking** (not speaking), loop the avatar's **base frames** (its natural head
motion). The server exposes:

- `GET /idle_meta` → `{"n": <frames>, "fps": <render fps>}`
- `GET /idle_frame/{i}.jpg` → the i-th base frame

The client preloads these and ping-pong-loops them at the render fps whenever no
speaking frame is due. Use a **calm idle source clip** (person at rest, slight
movement/blinks) so the resting face looks natural. `_build_idle_loop()` in
`server.py` is the reference.

---

## Avatar preparation (one-time, do it before serving)

`prepare_avatar()` runs face detection + VAE encoding + mask building over the
source frames — **slow (~40 s) and needs the heavy face-detect models**. Do it once,
offline, **while the GPU is free** (e.g. `prep_avatar.py`), so your live service
starts from cache and never loads those models alongside your TTS. Changing
`bbox_shift`/`extra_margin` requires re-running this.

Use a **video** source, not a still photo — the natural head motion under the
repainted mouth is what makes it read as actually speaking.

---

## Performance & VRAM

- **H100 (your target):** MuseTalk runs ~25–30 fps (realtime) and both the avatar and
  a TTS model fit trivially. Use defaults; you don't need the small-GPU tricks.
- **Small GPU (≤8 GB):** see README — fp16 T3 for the TTS, warmup+`empty_cache`,
  load-avatar-from-cache, `MUSETALK_BATCH=2`. Lip-sync tops out ~10 fps there.
- Generation cost is per output frame (UNet + VAE decode). Throughput scales with
  GPU, not batch size (small batches are fastest on weak GPUs).

---

## Production transport

The demo ships **JPEG frames over WebSocket** — fine for LAN/testing. For production:

- **WebRTC** (H.264/VP8) is the right call: far less bandwidth than per-frame JPEG,
  built-in A/V sync and jitter buffering. Feed the BGR frames from `stream_frames`
  into your encoder (e.g. `aiortc` `VideoStreamTrack`) and the TTS audio into an
  audio track.
- Keep the avatar worker **stateful per session** (one prepared avatar, one engine)
  and **serialize GPU access** (the demo uses an async lock) if multiple sessions
  share a GPU.

---

## Integration checklist

- [ ] Avatar worker runs in the **MuseTalk venv** (isolated from your bot's deps).
- [ ] `MUSETALK_ROOT` env points at the MuseTalk repo.
- [ ] Avatar **cache pre-built** (calm idle video) before serving.
- [ ] `engine.warmup()` called at startup.
- [ ] Per reply: TTS audio (per sentence) → `stream_frames` → encode → client.
- [ ] Client drives video off the **audio clock**; holds last frame if behind.
- [ ] **Idle loop** plays while listening/thinking.
- [ ] GPU access serialized if sessions share a GPU; one engine per worker.
- [ ] (Production) frames go out over **WebRTC**, not per-frame JPEG.
