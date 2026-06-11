# pyright: reportMissingImports=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false

"""Evaluate NeMo/NVIDIA ASR checkpoints on a JSONL dataset.

Expected JSONL format:
  {"audio": "path/to.wav", "text": "reference transcript", ...}

Example:
  /home/pragay/WWAI/.venv/bin/python stt_nemotron_eval.py \
    --data-jsonl datasets/sales_synth_vibevoice/train.jsonl \
    --model nvidia/nemotron-3.5-asr-streaming-0.6b \
    --batch-size 1 \
    --save-preds outputs/nemotron_vibevoice_preds.jsonl

Notes:
- This script measures end-to-end utterance latency around `model.transcribe(...)`.
- For a streaming checkpoint, this is still useful on a laptop, but it is not the
  same as token-by-token / first-partial latency from a real streaming loop.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import statistics
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _log_json(logger_: logging.Logger, obj: object, *, level: int = logging.INFO) -> None:
    try:
        msg = json.dumps(obj, ensure_ascii=False)
    except TypeError:
        msg = json.dumps(str(obj), ensure_ascii=False)
    logger_.log(level, msg)


def _require(pkg: str, hint: str) -> None:
    raise RuntimeError(f"Missing dependency '{pkg}'. Install: {hint}")


def _resolve_audio_path(p: str, *, audio_root: Path) -> str:
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((audio_root / pp).resolve())


def _load_rows(data_jsonl: Path, *, audio_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with data_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "audio" not in row or "text" not in row:
                raise ValueError(f"{data_jsonl}:{line_no} is missing required 'audio' or 'text' fields")
            row["audio"] = _resolve_audio_path(str(row["audio"]), audio_root=audio_root)
            rows.append(row)
    return rows


def _wav_duration_s(path: str) -> float:
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        sample_rate = wf.getframerate()
    if sample_rate <= 0:
        return 0.0
    return float(frames) / float(sample_rate)


def _text_norm(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"_", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_pred_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    for attr in ("text", "pred_text", "transcript"):
        value = getattr(item, attr, None)
        if isinstance(value, str):
            return value
    if isinstance(item, dict):
        for key in ("text", "pred_text", "transcript"):
            value = item.get(key)
            if isinstance(value, str):
                return value
    return str(item)


# Prompt-conditioned Nemotron models emit a trailing language token, e.g.
# "do you have headphones <en-US>". Strip any <...> tags so they don't pollute WER/CER.
_TAG_RE = re.compile(r"\s*<[^<>]*>\s*")


def _strip_tags(text: str) -> str:
    return _TAG_RE.sub(" ", str(text)).strip()


def _is_prompt_model(model: Any) -> dict | None:
    """Return the model's prompt_dictionary if it is a prompt-conditioned model, else None.

    Prompt-conditioned Nemotron checkpoints (EncDecRNNTBPEModelWithPrompt) require a
    per-utterance language prompt index. The offline `transcribe([paths...])` path leaves
    the cut language unset, so the dataset crashes with "Unknown prompt key: 'None'".
    We instead feed a manifest carrying `lang`/`prompt_mode` per entry (see _transcribe).
    """
    try:
        pd = model.cfg.model_defaults.get("prompt_dictionary")
    except Exception:
        return None
    return dict(pd) if pd else None


def _transcribe(
    model: Any,
    audio_paths: list[str],
    durations_s: list[float],
    *,
    batch_size: int,
    prompt_dict: dict | None,
    target_lang: str,
    prompt_mode: str,
    tmp_dir: str,
    tag: str,
) -> list[Any]:
    """Run model.transcribe, routing prompt-conditioned models through a manifest.

    For prompt models we write a temp JSONL manifest with `lang`/`prompt_mode` per entry;
    NeMo reads those into the lhotse cut (supervision.language + cut.custom['prompt_mode'])
    so the prompt index resolves deterministically. Non-prompt models use the plain path.
    """
    if prompt_dict is not None:
        entries = [
            {
                "audio_filepath": str(p),
                "duration": float(d) if d and d > 0 else 100000.0,
                "text": "",
                "lang": target_lang,
                "prompt_mode": prompt_mode,
            }
            for p, d in zip(audio_paths, durations_s)
        ]
        manifest_path = os.path.join(tmp_dir, f"manifest_{tag}.jsonl")
        with open(manifest_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        return model.transcribe(manifest_path, batch_size=batch_size)

    try:
        return model.transcribe(audio_paths, batch_size=batch_size)
    except TypeError:
        return model.transcribe(audio_paths)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    idx = (len(values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(values[lo])
    frac = idx - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def _load_model(nemo_asr: Any, model_name: str, device: str) -> tuple[Any, float]:
    try:
        import nemo
        nemo_version = getattr(nemo, "__version__", "unknown")
    except Exception:
        nemo_version = "unknown"

    load_t0 = time.perf_counter()
    try:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    except ModuleNotFoundError as e:
        if "rnnt_bpe_models_prompt" in str(e):
            raise RuntimeError(
                "This checkpoint requires a NeMo build newer than any PyPI release. "
                f"Detected NeMo version: {nemo_version} (the latest on PyPI does NOT ship "
                "`nemo.collections.asr.models.rnnt_bpe_models_prompt`). "
                "`pip install -U nemo-toolkit[asr]` will NOT help. Install NeMo from git main instead:\n"
                '  uv pip install -U "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"\n'
                "Then rerun the script."
            ) from e
        raise
    except TypeError as e:
        msg = str(e)
        if "Can't instantiate abstract class ASRModel" in msg:
            raise RuntimeError(
                "NeMo could not restore this pretrained checkpoint because the installed NeMo package "
                f"(detected version: {nemo_version}) is older than the model definition. "
                "The latest PyPI release is not new enough; install NeMo from git main instead:\n"
                '  uv pip install -U "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"'
            ) from e
        raise

    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model, time.perf_counter() - load_t0


def main(argv: list[str] | None = None) -> int:
    _setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--data-jsonl", type=str, required=True)
    p.add_argument("--audio-root", type=str, default=".")

    p.add_argument("--model", type=str, default="nvidia/nemotron-3.5-asr-streaming-0.6b")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--warmup-samples", type=int, default=1)
 
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--normalize-text", action="store_true")
    p.add_argument("--save-preds", type=str, default="", help="Optional JSONL output with predictions")

    # Prompt-conditioned models (e.g. nvidia/nemotron-3.5-asr-streaming-*) need a language
    # prompt per utterance. These are ignored for non-prompt checkpoints.
    p.add_argument(
        "--target-lang",
        type=str,
        default="en-US",
        help="Language prompt key for prompt-conditioned models (e.g. en-US, en, es-ES, zh-CN).",
    )
    p.add_argument(
        "--prompt-mode",
        type=str,
        default="langID",
        choices=["langID", "auto"],
        help="Prompt mode for prompt-conditioned models: 'langID' forces --target-lang; 'auto' is language-agnostic.",
    )

    p.add_argument(
        "--itn",
        action="store_true",
        help="Also report WER/CER after inverse text normalization of predictions "
        "(spoken->written, e.g. 'two point one'->'2.1', 'PS five'->'PS5') via text_itn.py.",
    )

    args = p.parse_args(argv)

    try:
        import torch
    except Exception:
        _require("torch", "Install torch for your CUDA/CPU setup")

    try:
        from jiwer import cer, wer
    except Exception:
        _require("jiwer", "uv pip install -U jiwer --python /home/pragay/WWAI/.venv/bin/python")

    try:
        import nemo.collections.asr as nemo_asr
    except OSError as e:
        msg = str(e)
        if "libtorchaudio" in msg or "torchaudio" in msg:
            raise RuntimeError(
                "NeMo import failed because `torchaudio` is binary-incompatible with your installed `torch`. "
                "Make sure `torch` and `torchaudio` use the same version and CUDA build. "
                "In zsh, remember to quote extras like this: `pip install -U \"nemo-toolkit[asr]\"`."
            ) from e
    except Exception:
        _require(
            "nemo_toolkit[asr]",
            "uv pip install -U nemo_toolkit[asr] jiwer --python /home/pragay/WWAI/.venv/bin/python",
        )

    device: str
    if str(args.device) == "cpu":
        device = "cpu"
    elif str(args.device) == "cuda":
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data_jsonl = Path(args.data_jsonl).resolve()
    audio_root = Path(args.audio_root).resolve()
    rows = _load_rows(data_jsonl, audio_root=audio_root)

    if not rows:
        raise ValueError(f"No samples found in {data_jsonl}")

    if int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]

    model, model_load_s = _load_model(nemo_asr, args.model, device)

    bs = max(1, int(args.batch_size))

    prompt_dict = _is_prompt_model(model)
    if prompt_dict is not None:
        if args.prompt_mode == "langID" and args.target_lang not in prompt_dict:
            raise SystemExit(
                f"--target-lang '{args.target_lang}' is not a valid prompt key for this model. "
                f"Available keys (first 20): {list(prompt_dict)[:20]}"
            )
        logger.info(
            "Prompt-conditioned model detected: prompt_mode=%s, target_lang=%s",
            args.prompt_mode,
            args.target_lang,
        )

    tmp_dir = tempfile.mkdtemp(prefix="nemotron_eval_")

    def _run_transcribe(audio_paths: list[str], durations_s: list[float], tag: str) -> list[Any]:
        return _transcribe(
            model,
            audio_paths,
            durations_s,
            batch_size=bs,
            prompt_dict=prompt_dict,
            target_lang=args.target_lang,
            prompt_mode=args.prompt_mode,
            tmp_dir=tmp_dir,
            tag=tag,
        )

    warmup_n = max(0, min(int(args.warmup_samples), len(rows)))
    if warmup_n:
        warmup_rows = rows[:warmup_n]
        warmup_paths = [str(row["audio"]) for row in warmup_rows]
        warmup_durs = [
            float(row.get("duration_s")) if row.get("duration_s") is not None else _wav_duration_s(str(row["audio"]))
            for row in warmup_rows
        ]
        logger.info("Running %d warmup sample(s)", warmup_n)
        _run_transcribe(warmup_paths, warmup_durs, tag="warmup")

    itn_fn = None
    if args.itn:
        try:
            from text_itn import inverse_normalize as itn_fn
        except Exception as e:
            raise SystemExit(
                "--itn requested but could not import `text_itn.inverse_normalize`. "
                "Ensure text_itn.py is alongside this script."
            ) from e

    preds: list[str] = []
    preds_itn: list[str] = []
    refs: list[str] = []
    latency_s_list: list[float] = []
    rtf_list: list[float] = []
    audio_duration_s_list: list[float] = []

    save_path = Path(args.save_preds).resolve() if str(args.save_preds).strip() else None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    f_out = open(save_path, "w", encoding="utf-8") if save_path else None

    try:
        eval_rows = rows[warmup_n:]

        for start in range(0, len(eval_rows), bs):
            chunk = eval_rows[start : start + bs]
            audio_paths = [str(row["audio"]) for row in chunk]
            refs_chunk = [str(row["text"]).strip() for row in chunk]
            durations_chunk = [
                float(row.get("duration_s")) if row.get("duration_s") is not None else _wav_duration_s(str(row["audio"]))
                for row in chunk
            ]

            infer_t0 = time.perf_counter()
            batch_preds_raw = _run_transcribe(audio_paths, durations_chunk, tag=f"chunk_{start}")
            batch_latency_s = time.perf_counter() - infer_t0

            batch_preds = [_strip_tags(_extract_pred_text(item)) for item in batch_preds_raw]
            if len(batch_preds) != len(chunk):
                raise RuntimeError(
                    f"Expected {len(chunk)} predictions but got {len(batch_preds)} from model.transcribe()"
                )

            # We only have batch wall time from transcribe(), so assign the same per-item
            # latency estimate within the batch for comparable summary stats.
            per_item_latency_s = batch_latency_s / max(1, len(chunk))

            for row, ref_t, pred_t, audio_dur_s in zip(chunk, refs_chunk, batch_preds, durations_chunk):
                # Inverse text normalization (spoken->written) is applied to the raw
                # prediction before any WER normalization, so digits/brands line up.
                pred_itn_t = itn_fn(pred_t) if itn_fn is not None else pred_t

                if args.normalize_text:
                    ref_eval = _text_norm(ref_t)
                    pred_eval = _text_norm(pred_t)
                    pred_itn_eval = _text_norm(pred_itn_t)
                else:
                    ref_eval = ref_t
                    pred_eval = pred_t
                    pred_itn_eval = pred_itn_t

                preds.append(pred_eval)
                preds_itn.append(pred_itn_eval)
                refs.append(ref_eval)
                latency_s_list.append(per_item_latency_s)
                audio_duration_s_list.append(audio_dur_s)
                rtf_list.append(per_item_latency_s / audio_dur_s if audio_dur_s > 0 else 0.0)

                if f_out is not None:
                    out_row = {
                        "id": row.get("id"),
                        "audio": row["audio"],
                        "ref": ref_t,
                        "pred": pred_t,
                        "ref_eval": ref_eval,
                        "pred_eval": pred_eval,
                        "audio_duration_s": audio_dur_s,
                        "latency_s_est": per_item_latency_s,
                        "rtf_est": per_item_latency_s / audio_dur_s if audio_dur_s > 0 else None,
                    }
                    if itn_fn is not None:
                        out_row["pred_itn"] = pred_itn_t
                        out_row["pred_itn_eval"] = pred_itn_eval
                    f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    finally:
        if f_out is not None:
            f_out.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    total_audio_s = float(sum(audio_duration_s_list))
    total_latency_s = float(sum(latency_s_list))

    metrics = {
        "n_total_rows": len(rows),
        "n_warmup": warmup_n,
        "n_eval": len(refs),
        "wer": float(wer(refs, preds)) if refs else None,
        "cer": float(cer(refs, preds)) if refs else None,
        "wer_itn": float(wer(refs, preds_itn)) if (refs and itn_fn is not None) else None,
        "cer_itn": float(cer(refs, preds_itn)) if (refs and itn_fn is not None) else None,
        "model": str(args.model),
        "device": device,
        "batch_size": max(1, int(args.batch_size)),
        "normalize_text": bool(args.normalize_text),
        "itn": bool(itn_fn is not None),
        "model_load_s": model_load_s,
        "latency_s_mean": statistics.fmean(latency_s_list) if latency_s_list else None,
        "latency_s_p50": _percentile(sorted(latency_s_list), 0.50),
        "latency_s_p95": _percentile(sorted(latency_s_list), 0.95),
        "latency_s_p99": _percentile(sorted(latency_s_list), 0.99),
        "audio_s_total": total_audio_s,
        "audio_s_mean": statistics.fmean(audio_duration_s_list) if audio_duration_s_list else None,
        "rtf_mean": statistics.fmean(rtf_list) if rtf_list else None,
        "rtf_p50": _percentile(sorted(rtf_list), 0.50),
        "rtf_p95": _percentile(sorted(rtf_list), 0.95),
        "throughput_audio_s_per_wall_s": (total_audio_s / total_latency_s) if total_latency_s > 0 else None,
    }

    _log_json(logger, {"event": "eval_done", **metrics})
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
