# pyright: reportMissingImports=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false

"""LoRA fine-tune OpenAI Whisper on JSONL manifests produced by this repo.

Expected JSONL format (one per line):
  {"audio": "path/to.wav", "text": "transcript", ...}

This script is intentionally optimized for quick iteration:
- supports limiting samples (e.g. --max-train-samples 200)
- LoRA adapters only (no full fine-tune)

Example quick check:
  /home/pragay/WWAI/.venv/bin/python stt_whisper_lora_finetune.py \
    --train-jsonl datasets/sales_synth_melotts/train.jsonl \
    --eval-jsonl datasets/sales_synth_melotts/val.jsonl \
    --model openai/whisper-large-v3 \
    --language en --task transcribe \
    --output-dir outputs/whisper-large-v3-lora-quick \
    --max-train-samples 200 --max-eval-samples 50 \
    --num-train-epochs 1 \
    --per-device-train-batch-size 8 --gradient-accumulation-steps 2 \
    --learning-rate 1e-4 \
    --bf16

Dependencies (install into your venv):
  uv pip install -U transformers datasets accelerate peft evaluate jiwer \
    --python /home/pragay/WWAI/.venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import tts_common as common

logger = logging.getLogger(__name__)


def _require(pkg: str, hint: str) -> None:
    raise RuntimeError(f"Missing dependency '{pkg}'. Install: {hint}")


def _as_list_csv(s: str) -> list[str]:
    parts = [p.strip() for p in str(s).split(",")]
    return [p for p in parts if p]


def _resolve_audio_path(p: str, *, audio_root: Path) -> str:
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((audio_root / pp).resolve())


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    input_dtype: Any = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # input_features: log-mels, shape (80, n_frames)
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.input_dtype is not None:
            batch["input_features"] = batch["input_features"].to(dtype=self.input_dtype)

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

        # Whisper uses a leading BOS token that we can safely ignore for loss.
        if labels.shape[1] > 0 and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def main(argv: list[str] | None = None) -> int:
    common.setup_logging()

    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", type=str, required=True)
    p.add_argument("--eval-jsonl", type=str, default="")
    p.add_argument("--audio-root", type=str, default=".", help="Prefix for relative audio paths")

    p.add_argument("--model", type=str, default="openai/whisper-large-v3")
    p.add_argument("--language", type=str, default="en")
    p.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"])

    p.add_argument("--output-dir", type=str, required=True)

    # Quick-iteration knobs
    p.add_argument("--max-train-samples", type=int, default=200, help="0 means no limit")
    p.add_argument("--max-eval-samples", type=int, default=50, help="0 means no limit")
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="If eval-jsonl is not provided or empty, create an eval split from train with this ratio (0 disables).",
    )

    # Training hyperparams
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=0.0)

    # H100-friendly defaults (you can still override).
    p.add_argument("--per-device-train-batch-size", type=int, default=8)
    p.add_argument("--per-device-eval-batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=2)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")

    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,v_proj",
        help="Comma-separated module name suffixes (common for Whisper: q_proj,v_proj)",
    )

    # Generation/eval
    p.add_argument("--generation-max-length", type=int, default=225)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)

    args = p.parse_args(argv)

    # Imports (fail with a clear error)
    has_cuda = False
    try:
        import torch

        has_cuda = bool(torch.cuda.is_available())
        if has_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        _require("torch", "Install torch for your CUDA/CPU setup")

    # Precision: default to BF16 on CUDA (ideal for H100) unless you explicitly ask for FP16.
    use_bf16 = bool(args.bf16) or (has_cuda and not bool(args.fp16))
    use_fp16 = bool(args.fp16) and not use_bf16

    try:
        from datasets import Audio, load_dataset
    except Exception:
        _require(
            "datasets",
            "uv pip install -U datasets --python /home/pragay/WWAI/.venv/bin/python",
        )

    # transformers' Whisper stack imports audio utilities that require `soxr`.
    try:
        import soxr  # noqa: F401
    except Exception:
        _require("soxr", "uv pip install -U soxr")

    try:
        from transformers.models.whisper import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.trainer_seq2seq import Seq2SeqTrainer
        from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
    except Exception as e:
        # Avoid misreporting optional sub-deps (e.g., soxr) as missing transformers.
        if isinstance(e, ModuleNotFoundError) and getattr(e, "name", None) == "soxr":
            _require("soxr", "uv pip install -U soxr")
        _require(
            "transformers",
            "uv pip install -U transformers accelerate",
        )

    try:
        from peft import LoraConfig, get_peft_model
    except Exception:
        _require("peft", "uv pip install -U peft --python /home/pragay/WWAI/.venv/bin/python")

    try:
        import evaluate
    except Exception:
        _require(
            "evaluate",
            "uv pip install -U evaluate jiwer --python /home/pragay/WWAI/.venv/bin/python",
        )

    audio_root = Path(args.audio_root).resolve()

    data_files: dict[str, str] = {"train": str(Path(args.train_jsonl).resolve())}
    if str(args.eval_jsonl).strip():
        eval_path = Path(args.eval_jsonl).resolve()
        if eval_path.exists() and eval_path.stat().st_size > 0:
            data_files["eval"] = str(eval_path)
        else:
            common.log_json(logger, {"event": "eval_skipped", "reason": "eval-jsonl missing or empty"})

    ds = load_dataset("json", data_files=data_files)

    def _fix_paths(batch: dict[str, Any]) -> dict[str, Any]:
        batch["audio"] = _resolve_audio_path(batch["audio"], audio_root=audio_root)
        return batch

    ds = ds.map(_fix_paths)

    # Load processor/model
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.config.use_cache = False

    # Prompt the decoder for the target language/task
    if hasattr(processor.tokenizer, "set_prefix_tokens"):
        try:
            processor.tokenizer.set_prefix_tokens(language=str(args.language), task=str(args.task))
        except Exception:
            pass
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=str(args.language), task=str(args.task))
    model.config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []

    # LoRA wrap
    lora = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        target_modules=_as_list_csv(str(args.lora_target_modules)),
    )
    model = get_peft_model(model, lora)

    # Ensure model weights dtype matches the precision mode, so generation (which
    # may run outside autocast) doesn't hit dtype mismatches in conv layers.
    if has_cuda:
        if use_bf16:
            model = model.to(dtype=torch.bfloat16)
        elif use_fp16:
            model = model.to(dtype=torch.float16)

    # Dataset audio decoding + feature extraction
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    def _prepare(batch: dict[str, Any]) -> dict[str, Any]:
        # batch["audio"] is {"array": np.ndarray, "sampling_rate": int, "path": str}
        audio = batch["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)

        # Whisper expects 16k mono
        if arr.ndim > 1:
            arr = np.mean(arr, axis=1)

        feats = processor.feature_extractor(arr, sampling_rate=int(audio["sampling_rate"])).input_features[0]
        batch["input_features"] = feats

        # Tokenize transcript
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    remove_cols = [c for c in ds["train"].column_names if c not in ("audio", "text")]
    ds = ds.map(_prepare, remove_columns=remove_cols)

    # If eval wasn't provided (or was empty), create a held-out eval split from the synthetic train set.
    if "eval" not in ds and float(args.val_ratio) > 0.0:
        split = ds["train"].train_test_split(test_size=float(args.val_ratio), seed=int(args.seed))
        ds = {"train": split["train"], "eval": split["test"]}

    # Optional subsampling
    if int(args.max_train_samples) > 0 and len(ds["train"]) > int(args.max_train_samples):
        ds["train"] = ds["train"].shuffle(seed=int(args.seed)).select(range(int(args.max_train_samples)))

    if "eval" in ds and int(args.max_eval_samples) > 0 and len(ds["eval"]) > int(args.max_eval_samples):
        ds["eval"] = ds["eval"].shuffle(seed=int(args.seed)).select(range(int(args.max_eval_samples)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep input dtype consistent with model weights under fp16/bf16 to avoid
    # float32 inputs hitting half-precision conv layers during generation.
    input_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else None)
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, input_dtype=input_dtype)

    wer = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        label_ids = np.array(pred.label_ids, copy=True)

        # Replace -100 in the labels as we can not decode them.
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        return {"wer": float(wer.compute(predictions=pred_str, references=label_str))}

    ta_kwargs: dict[str, Any] = {
        "output_dir": str(out_dir),
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "per_device_eval_batch_size": int(args.per_device_eval_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "learning_rate": float(args.learning_rate),
        "warmup_steps": int(args.warmup_steps),
        "weight_decay": float(args.weight_decay),
        "num_train_epochs": float(args.num_train_epochs),
        "fp16": bool(use_fp16),
        "bf16": bool(use_bf16),
        "tf32": bool(has_cuda),
        "logging_steps": int(args.logging_steps),
        "save_steps": int(args.save_steps),
        "save_total_limit": 2,
        "generation_max_length": int(args.generation_max_length),
        "report_to": [],
        "remove_unused_columns": False,
        "seed": int(args.seed),
    }

    # transformers renamed `evaluation_strategy` -> `eval_strategy` in newer versions.
    import inspect

    _ta_params = set(inspect.signature(Seq2SeqTrainingArguments.__init__).parameters)
    if "evaluation_strategy" in _ta_params:
        _eval_key = "evaluation_strategy"
    elif "eval_strategy" in _ta_params:
        _eval_key = "eval_strategy"
    else:
        _eval_key = "evaluation_strategy"

    if "eval" in ds:
        ta_kwargs.update(
            {
                _eval_key: "steps",
                "eval_steps": int(args.save_steps),
                "predict_with_generate": True,
            }
        )
    else:
        ta_kwargs.update({_eval_key: "no", "predict_with_generate": False})

    train_args = Seq2SeqTrainingArguments(**ta_kwargs)

    trainer = Seq2SeqTrainer(
        args=train_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"] if "eval" in ds else None,
        data_collator=collator,
        compute_metrics=compute_metrics if "eval" in ds else None,
    )

    common.log_json(
        logger,
        {
            "event": "train_start",
            "model": args.model,
            "train_n": len(ds["train"]),
            "eval_n": len(ds["eval"]) if "eval" in ds else 0,
            "output_dir": str(out_dir),
            "cuda_available": bool(has_cuda),
        },
    )

    trainer.train()

    # Save adapters + processor
    trainer.model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))

    # Save a small run manifest
    (out_dir / "run_config.json").write_text(
        json.dumps({"args": vars(args)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    common.log_json(logger, {"event": "train_done", "output_dir": str(out_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
