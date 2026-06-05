# NVFP4 KV cache for vLLM on SM120 (RTX PRO 6000 / consumer Blackwell)

Enable **`--kv-cache-dtype nvfp4`** on **SM120** GPUs (RTX PRO 6000 Blackwell, GB202 — compute
capability 12.0/12.1) in vLLM, using FlashInfer's **FA2** attention backend. The patch feeds the
NVFP4 scale-factors to FA2 with **explicit SF strides** and **de-swizzles the V scale-factor inside
the kernel**, so the SF cache costs **zero** extra bytes and the full ~1.78× capacity is realized.

SM120 ships no trtllm-gen NVFP4-KV cubins, so vLLM's default NVFP4-KV path doesn't work there. This
repo routes NVFP4 KV through the arch-generic FA2 kernel (dequantizes `e2m1`→bf16 in registers,
computes with standard `mma.sync`) and patches it so **both** the K and V scale-factors are read
directly from the interleaved cache — no parallel SF scratch at all.

## What you get

Measured on Step-3.7-Flash (198B MoE, NVFP4 weights), 2× RTX PRO 6000 (SM120), TP=2, **MTP K=1**,
`--max-model-len 131072` — full table in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md):

| KV dtype | util | KV pool (tokens) | vs fp8 | concurrency @131k | decode tok/s (1 conc) |
|---|---|---:|---:|---:|---:|
| fp8 | 0.92 | 1,663,988 | 1.00× | 12.70× | 114.6 |
| **nvfp4** | 0.92 | **2,960,263** | **1.78×** | **22.59×** | 111.8 |
| nvfp4 | 0.93 | 3,083,173 | 1.85× | 23.52× | 121.2 |

- **1.78× the fp8 KV pool at matched util** (the physical byte ceiling, 128/72) — because the SF cache
  is **+0%**. The earlier interim approach ("B1") kept a contiguous de-swizzled V-SF cache (+5.5%) that
  was unaccounted by the profiler and forced util *below* fp8's, eating the gain back to ~1.5×. This
  version (B2) de-swizzles V-SF **in the kernel** → no scratch → util matches fp8 → full ceiling.
- **Decode speed at parity** (~91–100% of fp8; single-stream 111.8 vs 114.6 tok/s = 97.6%, within
  run-to-run noise; min observed 91% at 4-way concurrency).
- **MTP (speculative decode) compatible and validated** — +19% single-stream throughput, no backend
  change. CUDA-graph (`FULL_AND_PIECEWISE`) compatible.
- **Quality:** fp8 KV is statistically lossless; NVFP4 KV costs **+0.01–0.04 nats/token PPL** (≈4–15×
  the noise floor, monotonic; worst at 8k context under MTP), retrieval intact — see
  [`docs/KV_QUALITY_STUDY.md`](docs/KV_QUALITY_STUDY.md).

## How it works (1 paragraph)

The FA2 NVFP4 kernel originally derived SF gmem strides as `data_stride / 8`, with no independent SF
stride input. vLLM stores the cache interleaved (`[K_data|K_scale|V_data|V_scale]` per page), so that
derivation mis-addresses the page term by 8×. This patch makes the kernel take **explicit
`sf_stride_page/h/n`** parameters and reads both scale-factors straight from the interleaved cache:
- **K-SF** is read **directly** from the cache view (K is stored linear) — zero scratch.
- **V-SF** is stored 4-token-swizzled by the writer; the kernel reads it **in place** and applies the
  inverse swizzle to the gmem offset **per element, in registers** — zero scratch (this is "B2";
  the older B1 de-swizzled it into a +5.5% parallel cache instead).

The SF strides are derived automatically in the FlashInfer binding from the SF tensors vLLM already
passes (`maybe_{k,v}_cache_sf`), so no jinja/binding/wrapper changes are needed for the symmetric
path — only the kernel and the codegen helper. See [`docs/DESIGN.md`](docs/DESIGN.md).

### Independent K/V precision (K=fp8 / V=nvfp4)

The kernel additionally splits `DTypeKV` into independent `DTypeK`/`DTypeV` (bit-identical when
equal), so K can be fp8 (more precise) while V is nvfp4 (compressed). This path is **numerically
validated** (`harness/h_layout_mixed.py`, cos 0.995) but its **vLLM serving wiring is a follow-up**
(see [`docs/KV_QUALITY_STUDY.md`](docs/KV_QUALITY_STUDY.md)); the study shows symmetric NVFP4 is
already near-lossless, so the expected upside is small.

## Requirements (version-pinned)

| component | version |
|---|---|
| GPU | SM120 (compute capability **12.0 / 12.1**) — RTX PRO 6000 Blackwell etc. |
| vLLM | **0.1.dev16944** (`vllm.__version__`) |
| FlashInfer | **0.6.11.post2** |
| model | **standard attention** + `--kv-cache-dtype nvfp4` (NVFP4 KV via vLLM's writer). head_dim ∈ {64,128,256,512} |

The patches are unified diffs against those exact versions. On a different vLLM/FlashInfer release
the source will have drifted — re-base the patches (the changes are small and well-commented).

**Not supported:** MLA models (DeepSeek-V*), Mamba/SSM, attention sinks on this path; non-SM120 GPUs
(the patch self-disables — `_use_fa2_for_nvfp4_kv_on_sm120()` gates on cc 12.0/12.1, so it's inert
elsewhere and safe to leave applied). Independent K/V (mixed) precision is kernel-validated but not
yet selectable at serve time.

> **Validation status:** numerically verified (cos 0.995, NHD **and** HND) via
> [`harness/h_layout_b2.py`](harness/h_layout_b2.py) (symmetric) and
> [`harness/h_layout_mixed.py`](harness/h_layout_mixed.py) (K=fp8/V=nvfp4), and end-to-end on a 198B
> MoE (StepFun Step-3.7-Flash, head_dim 128, H_kv 4, NHD, TP=2) **with MTP K=1** — capacity, decode
> speed, and a 6-config quality study. Other head_dims / HND-in-production are handled by design (the
> kernel is fully stride- and layout-driven) but not yet exercised live — run the harness for your
> shape before relying on it.

## Install

### Option A — Docker (recommended)

```bash
# Build a patched image FROM your vLLM image (must be the pinned versions above).
docker build -t vllm-nvfp4-kv-sm120 --build-arg BASE_IMAGE=<your-vllm-image> .
```

### Option B — patch an existing install in place

```bash
# Applies to the active Python env's site-packages (vllm + flashinfer must be the pinned versions).
./apply_patches.sh
```

`apply_patches.sh` copies the four modified files (`src/`) over their site-packages locations (and
refuses to run if the detected vLLM/FlashInfer versions don't match, to avoid silent API mismatches).
The equivalent unified diffs are in [`patches/`](patches/) for review (regenerated against the pinned
versions; they apply cleanly and reproduce `src/` exactly).

## Usage

```bash
# Minimal: add --kv-cache-dtype nvfp4. SM120 FA2 path activates automatically.
vllm serve <model> --kv-cache-dtype nvfp4 --gpu-memory-utilization 0.92 [...]
```

Or use the provided launcher (Docker, persists the JIT cache so restarts are fast):

```bash
MODEL_DIR=/path/to/model GPU_UTIL=0.92 IMAGE=vllm-nvfp4-kv-sm120 ./run.sh
```

MTP (speculative decode) is compatible and recommended in production — add
`--speculative-config '{"method":"mtp","model":"/draft","num_speculative_tokens":1}'` if your model
ships a draft.

### `--gpu-memory-utilization` tuning

B2 carries **no** SF over-allocation (unlike B1's +5.5% cache), so util can sit **right where fp8
runs**:
- **0.92** → 1.78× fp8 pool, robust (recommended starting point).
- **0.93** → a bit more pool (1.85×), still stable here.
- Sweep up from fp8's util under your peak concurrency; there is no longer a hidden lazy allocation to
  leave headroom for.

## Verify your shape

```bash
# inside the patched image / env:
python harness/h_layout_b2.py        # symmetric NVFP4 K+V
python harness/h_layout_mixed.py     # independent K=fp8 / V=nvfp4
# expect: cos ≈ 0.995 (NVFP4 quant error), "-> PASS" for NHD and HND.
```

Edit `H_q/H_kv/D/page` at the top of the harness to match your model before trusting a new shape.

## License

Apache-2.0 (see [LICENSE](LICENSE)). Contains modifications to FlashInfer and vLLM, both Apache-2.0;
see [NOTICE](NOTICE).
