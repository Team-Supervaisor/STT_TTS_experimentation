# DSpark benchmark: baseline vs speculative decoding

A thin, instrumented driver around DeepSpec's DSpark evaluator that answers the
practical question **"how much faster is the target model with DSpark, and at what
GPU cost?"** — which the stock `eval.py` does *not* measure (it reports acceptance
length only, no wall-clock latency / throughput / GPU usage).

## What it measures

For `--mode both` it runs the **same prompts** through two paths and compares them:

| Path | What runs |
|------|-----------|
| `baseline` | plain autoregressive decoding, target model only |
| `dspark`   | DSpark speculative decoding (1.4–3B draft proposes a block of 7, target verifies) |

Per path it reports:

- **throughput_tok_s** — end-to-end output tokens / wall time
- **mean_latency_s** — wall time per sample
- **mean_ttft_s** — time-to-first-token (baseline only; prefill segment)
- **target_forward_passes** — the quantity speculative decoding actually cuts
- **mean_acceptance_length** — avg tokens committed per target verify pass (DSpark)
- **peak_torch_alloc_mb / reserved_mb** — GPU memory footprint
- **gpu.gpu_util_mean/peak_pct, power_mean/peak_w** — live NVML utilization + power

And a `comparison` block: `throughput_speedup_x`, `latency_speedup_x`,
`target_forward_pass_reduction_x`.

> **Why temperature 0 (default).** At greedy decoding, speculative decoding is
> provably lossless — baseline and DSpark emit identical tokens — so the tok/s
> ratio is a clean apples-to-apples speedup, not a quality/speed tradeoff.

## Usage

```bash
python benchmark/bench.py \
  --target-name-or-path google/gemma-4-12B-it \
  --draft-name-or-path deepseek-ai/dspark_gemma4_12b_block7 \
  --mode both --load-mode bf16 \
  --task gsm8k --num-samples 16 --max-new-tokens 512 \
  --output benchmark/results/gemma.json
```

`--load-mode {bf16,8bit,4bit}` controls **target** precision (draft stays bf16).
`--task` is any file in `eval_datasets/` (gsm8k, math500, humaneval, mbpp,
mt-bench, alpaca, ...).

## GPU sizing (target = Gemma-4 12B, draft ~3B)

| VRAM | Mode | Notes |
|------|------|-------|
| ≥40 GB (A100-40 / L40S-48 / H100-80) | `bf16` | **recommended** — faithful headline numbers |
| 24 GB (4090 / L4 / A10) | `8bit` | fits, but quantizing the target lowers acceptance → under-reports |
| 6 GB (RTX 4050) | — | cannot run 12B at all; use for the Qwen3-4B pipeline check only |

## Files

- `bench.py` — the instrumented driver (single GPU, bsz=1, reuses DeepSpec's DSpark loop)
- `setup_env.sh` — clone DeepSpec + venv + install pinned deps (local or cloud)
- `run_local_qwen.sh` — 6 GB proof-of-concept: Qwen3-4B (4-bit) + 1.4B draft
- `run_cloud_gemma.sh` — full Gemma-4-12B baseline-vs-DSpark run on ≥40 GB
- `results/` — JSON outputs
