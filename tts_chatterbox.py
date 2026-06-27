"""Chatterbox (Resemble AI) runner (aligned with the other local runners).

Chatterbox is an open-source zero-shot TTS with optional voice cloning from a
short reference clip. The basic API is offline (no token streaming), so this
runner mirrors the non-streaming path of the VoXtream/MeloTTS runners.

Install:
    pip install chatterbox-tts

Key knobs (Chatterbox-specific):
    --exaggeration   emotion/intensity (0.5 = neutral; raise for more expressive)
    --cfg-weight     classifier-free guidance weight (lower = slower/steadier pace)
    --temperature    sampling temperature
    --prompt-audio   reference WAV for voice cloning (optional; omit for default voice)
"""

from __future__ import annotations

import argparse
import logging
import os
from time import perf_counter
from typing import Optional

import numpy as np

import tts_common as common


logger = logging.getLogger(__name__)


def _wav_to_mono_float32(wav) -> np.ndarray:
    """Convert a Chatterbox torch tensor (1, N) / (N,) to mono float32 numpy."""

    try:
        import torch  # type: ignore

        if isinstance(wav, torch.Tensor):
            wav = wav.detach().to("cpu").float().numpy()
    except Exception:
        pass
    return common.to_mono_float32(np.asarray(wav, dtype=np.float32))


def synthesize_chatterbox(
    text: str,
    *,
    use_gpu: bool,
    prompt_audio: Optional[str] = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
) -> tuple[np.ndarray, int, float, float]:
    """Synthesize with Chatterbox.

    Returns: (audio, samplerate, gen_s, init_s)
    """

    from chatterbox.tts import ChatterboxTTS  # type: ignore

    device = "cuda" if use_gpu else "cpu"

    init_t0 = perf_counter()
    model = ChatterboxTTS.from_pretrained(device=device)
    init_s = perf_counter() - init_t0

    gen_kwargs: dict = {
        "exaggeration": float(exaggeration),
        "cfg_weight": float(cfg_weight),
        "temperature": float(temperature),
    }
    if prompt_audio:
        gen_kwargs["audio_prompt_path"] = prompt_audio

    gen_t0 = perf_counter()
    wav = model.generate(text, **gen_kwargs)
    gen_s = perf_counter() - gen_t0

    audio = _wav_to_mono_float32(wav)
    sr = int(getattr(model, "sr", 24000))
    return audio, sr, float(gen_s), float(init_s)


def main(argv: Optional[list[str]] = None) -> int:
    common.setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, required=True)
    p.add_argument(
        "--prompt-audio",
        type=str,
        default="",
        help="Reference WAV for voice cloning (optional; omit for Chatterbox default voice).",
    )
    p.add_argument("--gpu", action="store_true", help="Force GPU")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--no-play", action="store_true")
    p.add_argument("--print-marks", action="store_true")
    p.add_argument("--wav-out", type=str, default="")
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--jitter", type=float, default=0.0)
    p.add_argument("--exaggeration", type=float, default=0.5, help="Emotion/intensity (0.5 = neutral)")
    p.add_argument("--cfg-weight", type=float, default=0.5, help="CFG weight (lower = steadier pace)")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    args = p.parse_args(argv)

    # Device selection identical to the other local runners.
    if args.cpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True
    else:
        use_gpu = common.gpu_is_available()

    common.log_json(logger, {"device": "cuda" if use_gpu else "cpu"})

    prompt_audio = args.prompt_audio.strip()
    if prompt_audio and not os.path.exists(prompt_audio):
        logger.error("Prompt audio not found: %s", prompt_audio)
        return 1

    audio, sr, gen_s, init_s = synthesize_chatterbox(
        args.text,
        use_gpu=use_gpu,
        prompt_audio=prompt_audio or None,
        exaggeration=float(args.exaggeration),
        cfg_weight=float(args.cfg_weight),
        temperature=float(args.temperature),
    )

    metrics = common.compute_metrics(gen_s, audio, sr)

    common.log_json(
        logger,
        {
            "model": "chatterbox",
            "streaming": False,
            "voice_cloned": bool(prompt_audio),
            "samplerate": sr,
            "init_s": round(float(init_s), 3),
            "gen_s": round(metrics.gen_s, 3),
            "ttfc_s": round(metrics.gen_s, 3),
            "ttft_s": round(metrics.gen_s, 3),
            "audio_s": round(metrics.audio_s, 3),
            "rtf": round(metrics.rtf, 3),
        },
    )

    # Only compute fallback marks when we're going to use them.
    marks = []
    if bool(args.print_marks) or (not args.no_play):
        phonemes = common.text_to_phonemes_g2p_en(args.text)
        marks = common.build_fallback_marks(phonemes, metrics.audio_s, jitter_s=float(args.jitter))

    if args.wav_out:
        common.save_wav_pcm16(args.wav_out, audio, sr)

    if args.print_marks:
        for m in marks:
            common.log_json(logger, {"t": round(m.t, 3), "phoneme": m.phoneme, "viseme": m.viseme})

    if not args.no_play:
        common.play_audio_with_marks(audio, sr, marks, chunk_size=int(args.chunk), logger=logger)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
