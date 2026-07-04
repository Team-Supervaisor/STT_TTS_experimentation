#!/usr/bin/env python3
"""
DSpark speculative-decoding benchmark.

Compares, on the SAME prompts and SAME sampling settings:
  * baseline : plain autoregressive decoding with the target model only
  * dspark   : DSpark speculative decoding (draft proposes a block, target verifies)

and reports, for each:
  * wall-clock latency per sample, end-to-end throughput (tok/s), decode tok/s, TTFT
  * number of target forward passes (the thing speculative decoding actually cuts)
  * DSpark acceptance length (mean tokens committed per target verify pass)
  * peak GPU memory (torch allocated/reserved) and live GPU util / power (via NVML)

At temperature 0 (default) speculative decoding is provably lossless, so baseline and
dspark emit identical tokens and the tok/s ratio is a clean apples-to-apples speedup.

Single GPU. Same script drives the local Qwen3-4B proof-of-concept and the cloud
Gemma-4-12B run -- only --target-name-or-path / --draft-name-or-path change.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from types import SimpleNamespace

# Make `deepspec` importable regardless of CWD (repo root is this file's parent's parent).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# init_dist() reads these; a single-process nccl group needs them set before import.
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29577")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from deepspec.eval.base_evaluator import (
    assert_no_final_target_layer,
    generate_decoding_sample,
    has_stop_token,
    load_and_process_dataset,
    resolve_stop_token_ids,
)
from deepspec.eval.dspark import Gemma4DSparkEvaluator, Qwen3DSparkEvaluator
from deepspec.data.parser import encode_chat_messages
from deepspec.utils.sampling import logits_to_probs, sample_from_probs
from deepspec.utils import seed_all
from transformers import AutoConfig

EVALUATORS = {
    "Qwen3DSparkModel": Qwen3DSparkEvaluator,
    "Gemma4DSparkModel": Gemma4DSparkEvaluator,
}


# --------------------------------------------------------------------------- #
# GPU sampling (NVML) -- runs in a background thread while a phase executes.
# --------------------------------------------------------------------------- #
class GPUSampler(threading.Thread):
    def __init__(self, index: int = 0, interval: float = 0.05):
        super().__init__(daemon=True)
        self.index = index
        self.interval = interval
        self._stop_evt = threading.Event()
        self.util: list[float] = []
        self.mem_used_mb: list[float] = []
        self.power_w: list[float] = []
        self._ok = False

    def run(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
        except Exception:
            return
        self._ok = True
        while not self._stop_evt.is_set():
            try:
                self.util.append(float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu))
                self.mem_used_mb.append(float(pynvml.nvmlDeviceGetMemoryInfo(handle).used) / 1e6)
                try:
                    self.power_w.append(float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0)
                except Exception:
                    pass
            except Exception:
                pass
            self._stop_evt.wait(self.interval)

    def stop(self) -> dict:
        self._stop_evt.set()
        self.join(timeout=2.0)
        if not self._ok or not self.util:
            return {"available": False}
        return {
            "available": True,
            "gpu_util_mean_pct": round(statistics.fmean(self.util), 1),
            "gpu_util_peak_pct": round(max(self.util), 1),
            "gpu_mem_used_peak_mb": round(max(self.mem_used_mb), 1) if self.mem_used_mb else None,
            "power_mean_w": round(statistics.fmean(self.power_w), 1) if self.power_w else None,
            "power_peak_w": round(max(self.power_w), 1) if self.power_w else None,
            "samples": len(self.util),
        }


# --------------------------------------------------------------------------- #
# Baseline: plain autoregressive decoding, target model only.
# Mirrors the harness sampling (logits_to_probs + sample_from_probs) and stop
# handling so it is directly comparable to the DSpark path.
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def generate_baseline(
    *,
    target_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    stop_token_ids: list[int] | None,
) -> SimpleNamespace:
    assert input_ids.size(0) == 1, "bsz=1 only"
    device = input_ids.device
    num_input = input_ids.shape[1]
    past = DynamicCache()

    # prefill -> first token (this is the TTFT segment)
    torch.cuda.synchronize()
    t_prefill0 = time.perf_counter()
    pos = torch.arange(num_input, device=device).unsqueeze(0)
    out = target_model(
        input_ids=input_ids,
        position_ids=pos,
        past_key_values=past,
        use_cache=True,
        logits_to_keep=1,
    )
    cur = sample_from_probs(logits_to_probs(out.logits, temperature))
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t_prefill0

    produced = 1
    cache_len = num_input
    stop = has_stop_token(cur, stop_token_ids)

    t_decode0 = time.perf_counter()
    while produced < max_new_tokens and not stop:
        pos_t = torch.tensor([[cache_len]], device=device)
        out = target_model(
            input_ids=cur,
            position_ids=pos_t,
            past_key_values=past,
            use_cache=True,
            logits_to_keep=1,
        )
        cache_len += 1
        cur = sample_from_probs(logits_to_probs(out.logits, temperature))
        produced += 1
        stop = has_stop_token(cur, stop_token_ids)
    torch.cuda.synchronize()
    decode_time = time.perf_counter() - t_decode0

    return SimpleNamespace(
        num_output_tokens=produced,
        target_forward_passes=produced,  # 1 prefill + (produced-1) decode steps == produced calls
        ttft=ttft,
        decode_time=decode_time,
        total_time=ttft + decode_time,
    )


# --------------------------------------------------------------------------- #
# Precision helpers. Quantization (8/4-bit) is a memory-fit crutch for the 6 GB
# local box only; the target's perturbed hidden states lower DSpark acceptance.
# The faithful cloud run uses bf16 for both.
# --------------------------------------------------------------------------- #
def build_quant_config(mode: str):
    if mode == "bf16":
        return None
    from transformers import BitsAndBytesConfig

    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if mode == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(mode)


def _load_causal_lm(name, mode, device, attn):
    qc = build_quant_config(mode)
    if qc is None:
        return (
            AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation=attn)
            .to(device)
            .eval()
        )
    # bnb models load directly on-device via device_map; cannot .to() after.
    return AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation=attn,
        quantization_config=qc, device_map={"": device},
    ).eval()


# --------------------------------------------------------------------------- #
# Evaluator subclass: pick load precision per model and pin to a single device
# without touching the repo, reusing its DSpark callbacks.
# --------------------------------------------------------------------------- #
def make_evaluator(evaluator_cls, load_mode: str, draft_load_mode: str):
    class BenchEvaluator(evaluator_cls):
        def build_models(self):
            attn = self.EVAL_ATTN_IMPLEMENTATION
            target = _load_causal_lm(
                self.args.target_name_or_path, load_mode, self.device, attn
            )
            draft_qc = build_quant_config(draft_load_mode)
            if draft_qc is None:
                draft = (
                    self.draft_model_cls.from_pretrained(
                        self.args.draft_name_or_path, dtype=torch.bfloat16,
                        attn_implementation=attn,
                    )
                    .to(self.device)
                    .eval()
                )
            else:
                draft = self.draft_model_cls.from_pretrained(
                    self.args.draft_name_or_path, dtype=torch.bfloat16,
                    attn_implementation=attn,
                    quantization_config=draft_qc, device_map={"": self.device},
                ).eval()
            assert_no_final_target_layer(target, draft.target_layer_ids)
            tok = AutoTokenizer.from_pretrained(self.args.target_name_or_path)
            return target, draft, tok

    return BenchEvaluator


@torch.inference_mode()
def run_dspark_sample(evaluator, input_ids, max_new_tokens, temperature, stop_token_ids):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = generate_decoding_sample(
        target_model=evaluator.target_model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        max_proposal_tokens=evaluator.max_proposal_tokens,
        temperature=temperature,
        stop_token_ids=stop_token_ids,
        init_context=evaluator._init_context,
        propose=evaluator._propose,
        update=evaluator._update,
        post_verify=None,
    )
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    accepts = list(res.acceptance_lengths) if res.acceptance_lengths else []
    return SimpleNamespace(
        num_output_tokens=int(res.num_output_tokens),
        target_forward_passes=int(res.verify_count) + 1,  # +1 prefill
        acceptance_length=(statistics.fmean(accepts) if accepts else 0.0),
        verify_count=int(res.verify_count),
        total_time=total,
    )


# --------------------------------------------------------------------------- #
def load_prompts(tokenizer, task: str, n: int, seed: int, device):
    rows = load_and_process_dataset(task)
    rows = rows[:n]
    prompts = []
    for r in rows:
        messages = [{"role": "user", "content": r["turns"][0]}]
        ids = encode_chat_messages(
            tokenizer, messages, add_generation_prompt=True, enable_thinking=False
        ).to(device)
        prompts.append(ids)
    return prompts


def summarize(tag, samples):
    tot_tok = sum(s.num_output_tokens for s in samples)
    tot_time = sum(s.total_time for s in samples)
    tot_fwd = sum(s.target_forward_passes for s in samples)
    per_lat = [s.total_time for s in samples]
    d = {
        "mode": tag,
        "num_samples": len(samples),
        "total_output_tokens": tot_tok,
        "total_wall_time_s": round(tot_time, 3),
        "throughput_tok_s": round(tot_tok / tot_time, 2) if tot_time else 0.0,
        "mean_latency_s": round(statistics.fmean(per_lat), 3),
        "target_forward_passes": tot_fwd,
    }
    if hasattr(samples[0], "ttft"):
        d["mean_ttft_s"] = round(statistics.fmean(s.ttft for s in samples), 4)
    if hasattr(samples[0], "acceptance_length"):
        d["mean_acceptance_length"] = round(
            statistics.fmean(s.acceptance_length for s in samples), 3
        )
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-name-or-path", required=True)
    ap.add_argument("--draft-name-or-path", default=None,
                    help="DSpark draft repo/path; required unless --mode baseline")
    ap.add_argument("--mode", choices=["baseline", "dspark", "both"], default="both")
    ap.add_argument("--load-mode", choices=["bf16", "8bit", "4bit"], default="bf16",
                    help="target model precision")
    ap.add_argument("--draft-load-mode", choices=["bf16", "8bit", "4bit"], default="bf16",
                    help="draft model precision (use 4bit only to fit tiny GPUs)")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=980406)
    ap.add_argument("--output", default=None, help="write JSON results here")
    args = ap.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    seed_all(args.seed)

    result = {
        "config": {
            "target": args.target_name_or_path,
            "draft": args.draft_name_or_path,
            "mode": args.mode,
            "load_mode": args.load_mode,
            "task": args.task,
            "num_samples": args.num_samples,
            "warmup": args.warmup,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "phases": {},
    }

    # ---- build models --------------------------------------------------- #
    evaluator = None
    if args.mode in ("dspark", "both"):
        assert args.draft_name_or_path, "--draft-name-or-path required for dspark"
        draft_cfg = AutoConfig.from_pretrained(args.draft_name_or_path)
        arch = draft_cfg.architectures[0]
        evaluator_cls = EVALUATORS[arch]
        BenchCls = make_evaluator(evaluator_cls, args.load_mode, args.draft_load_mode)
        eval_args = SimpleNamespace(
            target_name_or_path=args.target_name_or_path,
            draft_name_or_path=args.draft_name_or_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            confidence_threshold=0.0,
            tensorboard_dir=None,
            step=None,
            seed=args.seed,
            tasks=[],
        )
        evaluator = BenchCls(local_rank=0, args=eval_args)
        evaluator.confidence_head_recorder = None  # disable diagnostic overhead
        target_model = evaluator.target_model
        tokenizer = evaluator.tokenizer
    else:
        # baseline-only: load target directly (no draft needed)
        target_model = _load_causal_lm(args.target_name_or_path, args.load_mode, device, "sdpa")
        tokenizer = AutoTokenizer.from_pretrained(args.target_name_or_path)

    stop_token_ids = resolve_stop_token_ids(target_model, tokenizer)
    prompts = load_prompts(
        tokenizer, args.task, args.num_samples + args.warmup, args.seed, device
    )
    warm, bench_prompts = prompts[: args.warmup], prompts[args.warmup :]
    result["config"]["model_load_mem_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
    print(f"[load] target+draft resident: {torch.cuda.memory_allocated()/1e6:.0f} MB "
          f"allocated / {torch.cuda.memory_reserved()/1e6:.0f} MB reserved", flush=True)

    def run_phase(tag):
        # warmup (untimed)
        for p in warm:
            if tag == "baseline":
                generate_baseline(
                    target_model=target_model, input_ids=p,
                    max_new_tokens=min(args.max_new_tokens, 16),
                    temperature=args.temperature, stop_token_ids=stop_token_ids,
                )
            else:
                run_dspark_sample(evaluator, p, min(args.max_new_tokens, 16),
                                  args.temperature, stop_token_ids)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        sampler = GPUSampler()
        sampler.start()
        samples = []
        for i, p in enumerate(bench_prompts):
            if tag == "baseline":
                s = generate_baseline(
                    target_model=target_model, input_ids=p,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, stop_token_ids=stop_token_ids,
                )
            else:
                s = run_dspark_sample(evaluator, p, args.max_new_tokens,
                                      args.temperature, stop_token_ids)
            samples.append(s)
            print(f"  [{tag}] sample {i+1}/{len(bench_prompts)}: "
                  f"{s.num_output_tokens} tok, {s.total_time:.2f}s, "
                  f"{s.num_output_tokens/s.total_time:.1f} tok/s"
                  + (f", accept_len={s.acceptance_length:.2f}" if hasattr(s, 'acceptance_length') else ""),
                  flush=True)
        gpu = sampler.stop()
        summ = summarize(tag, samples)
        summ["peak_torch_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        summ["peak_torch_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1e6, 1)
        summ["gpu"] = gpu
        return summ

    if args.mode in ("baseline", "both"):
        print("\n=== BASELINE (target-only autoregressive) ===", flush=True)
        result["phases"]["baseline"] = run_phase("baseline")
    if args.mode in ("dspark", "both"):
        print("\n=== DSPARK (speculative decoding) ===", flush=True)
        result["phases"]["dspark"] = run_phase("dspark")

    # ---- comparison ----------------------------------------------------- #
    if args.mode == "both":
        b, d = result["phases"]["baseline"], result["phases"]["dspark"]
        result["comparison"] = {
            "throughput_speedup_x": round(d["throughput_tok_s"] / b["throughput_tok_s"], 3)
            if b["throughput_tok_s"] else None,
            "latency_speedup_x": round(b["mean_latency_s"] / d["mean_latency_s"], 3)
            if d["mean_latency_s"] else None,
            "target_forward_pass_reduction_x": round(
                b["target_forward_passes"] / d["target_forward_passes"], 3
            ) if d["target_forward_passes"] else None,
            "mean_acceptance_length": d.get("mean_acceptance_length"),
        }

    print("\n" + "=" * 70)
    print(json.dumps(result, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[saved] {args.output}", flush=True)


if __name__ == "__main__":
    main()
