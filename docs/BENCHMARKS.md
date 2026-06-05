# Benchmarks — NVFP4 KV (B2) vs fp8 on SM120

**Hardware/model:** StepFun **Step-3.7-Flash** (198B MoE, NVFP4 weights, head_dim 128, H_kv 4, NHD),
2× **RTX PRO 6000 Blackwell** (SM120), TP=2, `--enable-expert-parallel`,
`--max-model-len 131072`. vLLM 0.1.dev16944 + FlashInfer 0.6.11.post2.
**Production config: MTP speculative decode, K=1** (`--speculative-config method=mtp K=1`).

KV pool size and max concurrency are read straight from the vLLM startup log
(`GPU KV cache size: N tokens` / `Maximum concurrency for 131072 tokens per request`).
Decode throughput is `bench_client.py` (256-token completions, `ignore_eos`, 2 reps;
aggregate tok/s across the concurrent requests).

## Capacity + decode throughput (MTP K=1)

| KV dtype | util | KV pool (tokens) | vs fp8 | concurrency @131k | decode tok/s — 1 / 2 / 4 conc |
|---|---|---:|---:|---:|---|
| fp8 | 0.92 | 1,663,988 | 1.00× | 12.70× | 114.6 / 170.5 / 347.4 |
| **nvfp4 (B2)** | **0.92** | **2,960,263** | **1.78×** | **22.59×** | 111.8 / 171.2 / 316.9 |
| nvfp4 (B2) | 0.93 | 3,083,173 | 1.85× | 23.52× | 121.2 / 175.0 / 372.3 |

- **At matched util 0.92, NVFP4-B2 holds 1.78× the fp8 KV pool** — the full byte ceiling (128/72),
  because B2 carries **zero** SF scratch (no +5.5% V-SF cache, no hidden lazy allocation). Under the
  earlier B1 (+5.5% cache) you had to drop util below fp8's, which ate the advantage back to ~1.5×.
- **Decode speed is at parity**: single-stream 111.8 vs 114.6 tok/s = **97.6%** of fp8 (within
  run-to-run noise at 2 reps); aggregate throughput equal-to-higher. B2's in-kernel de-swizzle adds a
  few register ops per V-SF element but no extra global traffic, so there is no throughput regression.
- **B2 has no util penalty vs fp8**, so you can also push util to 0.93 → 3.08M tokens (1.85×) with the
  same stability; sweep up from 0.92 under your peak concurrency.

## MTP value (why K=1 is the production default)

Single-stream decode, fp8 KV @ util 0.92:

| | MTP off | MTP K=1 | gain |
|---|---:|---:|---:|
| decode tok/s (conc 1) | 96.0 | 114.6 | **+19%** |
| KV pool (tokens) | 1,667,816 | 1,663,988 | −0.2% (draft VRAM is negligible at this util) |

MTP K=1 buys ~+19% single-stream throughput for a negligible KV-pool cost, and **works unchanged on
the B2 NVFP4 path** (`supports_spec_as_decode=False`; the K-step verify goes through the FA2 prefill
path, the draft 1-token decode is CUDA-graph captured). This was the reason to validate capacity and
speed with MTP on rather than off.

## Quality (KV precision, separate study)

6-config sensitivity study (bf16 / fp8 / nvfp4 KV × MTP{0,1}, weights fixed NVFP4), noise-floor
controlled. Headline: **fp8 KV is statistically lossless; NVFP4 KV costs +0.01–0.02 nats/token PPL**
(4–10× the noise floor, monotonic bf16≤fp8≤nvfp4), with **retrieval (RULER-hard) intact**.
Full numbers and methodology: [`kv_quality_compare.md`](kv_quality_compare.md),
[`KV_QUALITY_STUDY.md`](KV_QUALITY_STUDY.md).

## Reproduce

```bash
# capacity + speed (this table), production MTP K=1:
MTP=1 bash cap_speed_bench.sh        # fp8@0.92, nvfp4@0.92, nvfp4@0.93
# quality (6-config):
bash run_kv_quality.sh
```
