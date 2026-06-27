"""Realtime avatar server (runs in the isolated MuseTalk venv).

Serves the browser UI and a WebSocket that, for each submitted text:
  1. splits it into sentences,
  2. for each sentence: asks the Chatterbox TTS microservice for audio, then
     streams MuseTalk lip-sync frames over the socket as the VAE decodes them.

Sentences pipeline naturally: while the browser plays sentence N, the server is
already generating sentence N+1, so playback is progressive (not "render the
whole video, then play").

Run (inside MuseTalk venv):
    MUSETALK_ROOT=/.../MuseTalk \
    /.../MuseTalk/.venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import tempfile

import cv2
import httpx
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Resolve our app dir BEFORE importing the engine (engine chdir's to MuseTalk root).
APP_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | server | %(message)s")
logger = logging.getLogger("server")

from musetalk_engine import MuseTalkEngine  # noqa: E402  (chdir happens on import)

TTS_URL = os.environ.get("TTS_URL", "http://127.0.0.1:8765")
AVATAR_IMAGE = os.environ.get("AVATAR_IMAGE", "data/video/sun.mp4")
AVATAR_ID = os.environ.get("AVATAR_ID", "sun")
DISPLAY_WIDTH = int(os.environ.get("DISPLAY_WIDTH", "512"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))
# Small batches are both fastest on this GPU and leave VRAM headroom next to the
# TTS process (they reduce per-step activation memory).
BATCH_SIZE = int(os.environ.get("MUSETALK_BATCH", "2"))
# 25 fps for smooth, natural mouth motion (an H100 generates this in realtime; a
# 6 GB laptop will lag the live stream at 25 fps — lower MUSETALK_FPS for local
# realtime testing, or evaluate quality via offline renders).
TARGET_FPS = int(os.environ.get("MUSETALK_FPS", "25"))
# Mouth-articulation tuning. bbox_shift nudges the mouth crop up/down (affects how
# open the mouth reads); extra_margin (v15) adds chin room below the mouth.
BBOX_SHIFT = int(os.environ.get("MUSETALK_BBOX_SHIFT", "0"))
EXTRA_MARGIN = int(os.environ.get("MUSETALK_EXTRA_MARGIN", "10"))

# Current (live-tunable) prep params; changing them rebuilds the avatar.
cur_bbox_shift = BBOX_SHIFT
cur_extra_margin = EXTRA_MARGIN

engine: MuseTalkEngine | None = None
_engine_lock = asyncio.Lock()  # serialize GPU access across connections

# Idle loop: the avatar's natural base-video frames, pre-encoded as JPEG. The
# client plays these (ping-pong) whenever it isn't speaking, so the avatar keeps
# moving instead of freezing. The frames are at render-fps density, so playing them
# back at the render fps reproduces the source's real-time pace.
_IDLE_FPS_OVERRIDE = os.environ.get("IDLE_FPS")  # optional manual override
IDLE_MAX = int(os.environ.get("IDLE_MAX", "120"))  # consecutive frames (~loop length)
idle_jpegs: list[bytes] = []

app = FastAPI(title="Realtime MuseTalk Avatar")


def _idle_fps() -> float:
    if _IDLE_FPS_OVERRIDE:
        return float(_IDLE_FPS_OVERRIDE)
    return float(engine.fps) if engine else 25.0


def _build_idle_loop():
    """Pre-encode a consecutive run of the avatar's natural base frames. Consecutive
    (not subsampled) keeps them at render-fps density so playback speed is natural."""
    global idle_jpegs
    base = engine.avatar.frame_list_cycle
    fwd = base[: max(1, len(base) // 2)]   # forward frames (cycle is fwd + reversed)
    fwd = fwd[:IDLE_MAX]                    # consecutive segment from the start
    idle_jpegs = [_encode_jpg(f) for f in fwd]
    logger.info("idle loop: %d frames @ %.0f fps (%.1fs)", len(idle_jpegs), _idle_fps(),
                len(idle_jpegs) / max(_idle_fps(), 1e-6))


@app.on_event("startup")
def _startup():
    global engine
    logger.info("initializing MuseTalk engine ...")
    engine = MuseTalkEngine(version="v15", batch_size=BATCH_SIZE, fps=TARGET_FPS, extra_margin=EXTRA_MARGIN)
    engine.prepare_avatar(AVATAR_IMAGE, avatar_id=AVATAR_ID, bbox_shift=BBOX_SHIFT)
    engine.warmup()  # trigger cuDNN autotune now, not on the first sentence
    _build_idle_loop()
    logger.info("engine + avatar ready; TTS at %s", TTS_URL)


@app.get("/")
def index():
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"))


@app.get("/health")
def health():
    return {"engine_ready": engine is not None, "fps": engine.fps if engine else None}


@app.get("/idle.jpg")
def idle_jpg():
    from fastapi.responses import Response as _Resp

    frame = engine.idle_frame()
    return _Resp(content=_encode_jpg(frame), media_type="image/jpeg")


@app.get("/idle_meta")
def idle_meta():
    return {"n": len(idle_jpegs), "fps": _idle_fps()}


@app.get("/idle_frame/{i}.jpg")
def idle_frame_i(i: int):
    from fastapi.responses import Response as _Resp

    if not idle_jpegs:
        return _Resp(status_code=503)
    return _Resp(content=idle_jpegs[i % len(idle_jpegs)], media_type="image/jpeg")


app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")


_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


def split_sentences(text: str, max_chars: int = 220) -> list[str]:
    text = " ".join(text.strip().split())
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    # Further break over-long sentences on commas so latency-to-first-audio stays low.
    out: list[str] = []
    for p in parts:
        if len(p) <= max_chars:
            out.append(p)
            continue
        buf = ""
        for clause in re.split(r"(?<=,)\s+", p):
            if len(buf) + len(clause) + 1 > max_chars and buf:
                out.append(buf.strip())
                buf = clause
            else:
                buf = (buf + " " + clause).strip()
        if buf:
            out.append(buf.strip())
    return out


def _encode_jpg(frame_bgr) -> bytes:
    h, w = frame_bgr.shape[:2]
    if w > DISPLAY_WIDTH:
        scale = DISPLAY_WIDTH / w
        frame_bgr = cv2.resize(frame_bgr, (DISPLAY_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else b""


async def _call_tts(client: httpx.AsyncClient, sentence: str, params: dict) -> bytes:
    payload = {"text": sentence, **params}
    r = await client.post(f"{TTS_URL}/tts", json=payload, timeout=120.0)
    r.raise_for_status()
    return r.content


async def _handle_retune(websocket: WebSocket, loop, msg: dict) -> None:
    """Rebuild the avatar with new bbox_shift/extra_margin (prep-time params).
    This re-runs face detection + latents + masks (~40 s for a video)."""
    global cur_bbox_shift, cur_extra_margin
    bs = int(msg.get("bbox_shift", cur_bbox_shift))
    em = int(msg.get("extra_margin", cur_extra_margin))
    await websocket.send_json({"type": "retuning", "bbox_shift": bs, "extra_margin": em})

    def _rebuild():
        engine.extra_margin = em
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        engine.prepare_avatar(AVATAR_IMAGE, avatar_id=AVATAR_ID, bbox_shift=bs, force=True)
        engine.warmup()
        _build_idle_loop()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async with _engine_lock:
        try:
            await loop.run_in_executor(None, _rebuild)
            cur_bbox_shift, cur_extra_margin = bs, em
            await websocket.send_json({"type": "retuned", "bbox_shift": bs, "extra_margin": em})
        except Exception as e:  # noqa: BLE001
            logger.exception("retune failed")
            await websocket.send_json({"type": "error", "message": f"retune failed (often VRAM on a small GPU): {e}"})


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    await websocket.send_json({
        "type": "config",
        "avatar_id": AVATAR_ID,
        "fps": engine.fps if engine else None,
        "bbox_shift": cur_bbox_shift,
        "extra_margin": cur_extra_margin,
    })
    async with httpx.AsyncClient() as client:
        try:
            while True:
                msg = await websocket.receive_json()
                mtype = msg.get("type")
                if mtype == "retune":
                    await _handle_retune(websocket, loop, msg)
                    continue
                if mtype != "speak":
                    continue
                text = (msg.get("text") or "").strip()
                if not text:
                    continue
                params = {
                    "exaggeration": float(msg.get("exaggeration", 0.5)),
                    "cfg_weight": float(msg.get("cfg_weight", 0.5)),
                    "temperature": float(msg.get("temperature", 0.8)),
                }
                if msg.get("prompt_audio"):
                    params["prompt_audio"] = msg["prompt_audio"]

                sentences = split_sentences(text)
                await websocket.send_json({"type": "start", "n_sentences": len(sentences), "fps": engine.fps})

                # 1-ahead TTS prefetch: synthesize sentence N+1 (separate TTS
                # process) while we stream sentence N's frames, to shrink the
                # inter-sentence gap caused by non-streaming Chatterbox.
                next_task = (
                    asyncio.create_task(_call_tts(client, sentences[0], params))
                    if sentences else None
                )

                for s_idx, sentence in enumerate(sentences):
                    # 1) await this sentence's audio, then immediately kick off the next
                    try:
                        wav_bytes = await next_task
                    except Exception as e:
                        await websocket.send_json({"type": "error", "message": f"TTS failed: {e}"})
                        break
                    if s_idx + 1 < len(sentences):
                        next_task = asyncio.create_task(_call_tts(client, sentences[s_idx + 1], params))

                    await websocket.send_json({
                        "type": "sentence",
                        "idx": s_idx,
                        "text": sentence,
                        "fps": engine.fps,
                        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
                    })

                    # 2) Stream MuseTalk frames for this sentence as they decode.
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp.write(wav_bytes)
                    tmp.close()
                    try:
                        async with _engine_lock:
                            gen = engine.stream_frames(tmp.name)
                            i = 0

                            def _next():
                                try:
                                    return next(gen)
                                except StopIteration:
                                    return None

                            while True:
                                frame = await loop.run_in_executor(None, _next)
                                if frame is None:
                                    break
                                jpg = await loop.run_in_executor(None, _encode_jpg, frame)
                                await websocket.send_json({
                                    "type": "frame",
                                    "s": s_idx,
                                    "i": i,
                                    "jpg_b64": base64.b64encode(jpg).decode("ascii"),
                                })
                                i += 1
                    finally:
                        os.unlink(tmp.name)

                    await websocket.send_json({"type": "sentence_end", "idx": s_idx, "n_frames": i})

                await websocket.send_json({"type": "done"})
        except WebSocketDisconnect:
            logger.info("client disconnected")
        except Exception as e:  # noqa: BLE001
            logger.exception("ws error: %s", e)
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
