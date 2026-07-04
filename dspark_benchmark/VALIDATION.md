# Local validation report — DSpark benchmark pipeline

**Goal of this step:** prove the entire DSpark speculative-decoding benchmark works
end-to-end on the exact software stack *before* spending on cloud GPU time. The real
Gemma-4-12B numbers come from the cloud run — see "Next" below.

**Machine:** RTX 4050 Laptop, **6 GB VRAM**, 15 GB RAM · torch 2.9.1+cu128 · transformers 5.10.2 · CUDA driver 13.2

## What was validated ✅

| Step | Result |
|------|--------|
| venv + pinned deps install | OK (torch 2.9.1+cu128 sees the RTX 4050, sm_89) |
| HF access (token) | Qwen3-4B, Gemma-4-12B-it, both DSpark drafts — **all ungated, reachable** |
| Custom DSpark modeling loads on transformers 5.10.2 | OK (`Qwen3DSparkModel`, 1.39 B params, block_size 7, confidence head) |
| Target + draft load on GPU | OK (both 4-bit NF4 to fit 6 GB → 4.6 GB resident) |
| **DSpark speculative-decoding loop executes** | OK |
| Baseline (target-only) path | OK |
| Instrumentation: latency, tok/s, TTFT, forward-pass count, acceptance length, GPU mem/util/power, JSON | OK |

## Local numbers (Qwen3-4B, gsm8k, greedy, 4 samples × 128 tok)

| Metric | Baseline | DSpark | Δ |
|---|---|---|---|
| Throughput (tok/s) | 24.2 | **44.4** | **1.83×** |
| Mean latency / sample (s) | 5.29 | 2.89 | **1.83×** |
| Target forward passes | 512 | **96** | **5.33× fewer** |
| Mean acceptance length | — | 5.73 / 7 | — |
| Peak GPU mem (MB) | 5754 | 5867 | — |
| GPU util (mean) | 57% | 97% | — |
| Power (mean W) | 56.5 | 74.0 | — |

## ⚠️ Why these are NOT the real numbers

Both target **and** draft are loaded **4-bit NF4** — the only way to fit a 12 GB
(bf16) model pair into 6 GB. Quantization:
- perturbs the target hidden states the draft consumes → acceptance is not faithful,
- changes the compute path → tok/s and GPU-util are not representative of a datacenter GPU,
- degrades output quality (irrelevant here — we only measure the pipeline).

The **1.83× / 5.33×** here is a *lower-bound sanity signal* that DSpark is wired up and
working, not a number to report. Report only the bf16 cloud results.

## Reproduce locally

```bash
bash benchmark/run_local_qwen.sh
```

## Next — real Gemma-4-12B run on cloud (bf16, ≥40 GB GPU)

1. Provision one A100-40GB / L40S-48GB / H100-80GB (RunPod / Lambda / Vast).
2. `scp -r benchmark/ user@box:~/`
3. `ssh user@box 'export HF_TOKEN=hf_xxx; bash ~/benchmark/setup_env.sh'`
4. `ssh user@box 'bash ~/DeepSpec/benchmark/run_cloud_gemma.sh'`
   → writes `benchmark/results/gemma4_12b_bf16_gsm8k.json`

Then the same script does Qwen next: `TARGET=Qwen/Qwen3-14B DRAFT=deepseek-ai/dspark_qwen3_14b_block7 bash benchmark/run_cloud_gemma.sh`.
