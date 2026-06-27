"""Chatterbox TTS microservice (runs in the MAIN venv).

Isolated from MuseTalk because Chatterbox pins transformers==5.2.0 / diffusers
==0.29.0, which conflict with MuseTalk's transformers==4.39.2. Exposes a tiny
HTTP API: POST /tts {text,...} -> 24 kHz WAV bytes.

Run:
    /home/pragay/WWAI/.venv/bin/python -m uvicorn tts_service:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import io
import logging
import os

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | tts | %(message)s")
logger = logging.getLogger("tts_service")

app = FastAPI(title="Chatterbox TTS service")
_model = None
_sr = 24000
_force_cpu = os.environ.get("TTS_DEVICE", "").lower() == "cpu"


def _load():
    global _model, _sr
    if _model is not None:
        return
    from chatterbox.tts import ChatterboxTTS

    device = "cpu" if _force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("loading ChatterboxTTS on %s ...", device)
    _model = ChatterboxTTS.from_pretrained(device=device)
    _sr = int(getattr(_model, "sr", 24000))

    # Run the big T3 token model in fp16 to roughly halve VRAM (~3.2 -> ~2.3 GB),
    # so Chatterbox fits next to MuseTalk on a 6 GB GPU. The s3gen vocoder + voice
    # encoder stay fp32 for audio quality. We coerce T3's conditioning tensors to
    # fp16 at inference entry so it survives generate()'s emotion_adv rebuilds.
    if device == "cuda" and os.environ.get("TTS_FP16_T3", "1") == "1":
        _model.t3 = _model.t3.half()
        _orig_t3_inference = _model.t3.inference

        def _t3_inference_fp16(*args, t3_cond=None, **kwargs):
            if t3_cond is not None:
                for _n, _v in vars(t3_cond).items():
                    if torch.is_tensor(_v) and _v.is_floating_point() and _v.dtype != torch.float16:
                        setattr(t3_cond, _n, _v.half())
            return _orig_t3_inference(*args, t3_cond=t3_cond, **kwargs)

        _model.t3.inference = _t3_inference_fp16
        logger.info("T3 token model running in fp16 (vocoder fp32)")
    # Warm up once, then drop the allocator to its steady state (~2.3 GB). Doing this
    # *before* we report ready means MuseTalk (other process) loads with ~3.4 GB free
    # on the shared 6 GB GPU instead of ~2.5 GB right after load (which OOMs it).
    if torch.cuda.is_available():
        try:
            with torch.no_grad():
                _model.generate("Warming up.", exaggeration=0.5, cfg_weight=0.5, temperature=0.8)
        except Exception as e:  # noqa: BLE001
            logger.warning("warmup gen failed (non-fatal): %s", e)
        torch.cuda.empty_cache()
    logger.info("ChatterboxTTS ready (sr=%d)", _sr)


@app.on_event("startup")
def _startup():
    _load()


class TTSRequest(BaseModel):
    text: str
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    prompt_audio: str | None = None


@app.get("/health")
def health():
    return {"ready": _model is not None, "sr": _sr, "cuda": torch.cuda.is_available()}


@app.post("/tts")
def tts(req: TTSRequest):
    _load()
    kwargs = dict(
        exaggeration=float(req.exaggeration),
        cfg_weight=float(req.cfg_weight),
        temperature=float(req.temperature),
    )
    if req.prompt_audio:
        kwargs["audio_prompt_path"] = req.prompt_audio

    with torch.no_grad():
        wav = _model.generate(req.text, **kwargs)

    audio = wav.detach().to("cpu").float().numpy().squeeze()
    audio = np.clip(audio, -1.0, 1.0)
    # Release cached GPU blocks so MuseTalk (separate process, shared 6 GB GPU)
    # has headroom while it streams frames for this sentence.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    buf = io.BytesIO()
    sf.write(buf, audio, _sr, format="WAV", subtype="PCM_16")
    data = buf.getvalue()
    logger.info("synth: %d chars -> %.2fs audio", len(req.text), len(audio) / _sr)
    return Response(content=data, media_type="audio/wav", headers={"X-Sample-Rate": str(_sr)})
