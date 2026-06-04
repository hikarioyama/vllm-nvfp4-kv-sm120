# NVFP4 KV cache for vLLM on SM120 (RTX PRO 6000 / consumer Blackwell)

Enable **`--kv-cache-dtype nvfp4`** on **SM120** GPUs (RTX PRO 6000 Blackwell, GB202 — compute
capability 12.0/12.1) in vLLM, using FlashInfer's **FA2** attention backend with an **explicit
scale-factor (SF) stride** kernel patch.

SM120 ships no trtllm-gen NVFP4-KV cubins, so vLLM's default NVFP4-KV path doesn't work there.
This repo routes NVFP4 KV through the arch-generic FA2 kernel (which dequantizes `e2m1`→bf16 in
registers and computes with standard `mma.sync`), and patches that kernel to take **explicit SF
strides** so the per-block scale-factor cache is tiny instead of 2.25× over-allocated.

## What you get

- **~1.78× KV bytes/token vs fp8** (NVFP4 = 144 B/tok/head K+V vs fp8 256 B) — the physical ceiling.
- **Real pool ≈ 1.5× fp8** in practice (validated: 2.47M vs ~1.6M tokens on one model), because the
  persistent V-SF cache adds only **+5.5%** (down from +25% with the naïve approach), so
  `--gpu-memory-utilization` can stay high.
- **~95–104% of fp8 decode speed** (no regression; faster than the naïve NVFP4-KV at low concurrency
  because K is read zero-copy directly from the cache).
- CUDA-graph + MTP (speculative decode) compatible.

## How it works (1 paragraph)

The FA2 NVFP4 kernel originally derived SF gmem strides as `data_stride / 8`, with no independent SF
stride input. vLLM stores the cache interleaved (`[K_data|K_scale|V_data|V_scale]` per page), so to
feed that kernel you had to repack **both** K and V scales into a **sparse** buffer (block stride =
`data_page/8`) — a 2.25× over-allocation (~+25% of the KV pool). This patch makes the kernel take
**explicit `sf_stride_page/h/n`** parameters. Then:
- **K-SF** is read **directly from the interleaved cache view** (K is stored linear) — zero copy.
- **V-SF** is de-swizzled into a **contiguous** parallel cache (real size, no over-alloc) — **+5.5%**.

The SF strides are derived automatically in the FlashInfer binding from the SF tensors that vLLM
already passes (`maybe_{k,v}_cache_sf`), so no jinja/binding/wrapper changes are needed — only the
kernel and the codegen helper. See [`docs/DESIGN.md`](docs/DESIGN.md).

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
elsewhere and safe to leave applied).

> **Validation status:** numerically verified (cos 0.995, byte-exact V de-swizzle, NHD **and** HND)
> via [`harness/h_layout_explicit.py`](harness/h_layout_explicit.py), and end-to-end on a 198B MoE
> (StepFun Step-3.7-Flash, head_dim 128, H_kv 4, NHD, TP=2). Other head_dims / HND-in-production are
> handled by design (the kernel is fully stride- and layout-driven) but not yet exercised live —
> run the harness for your shape before relying on it.

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

`apply_patches.sh` copies the three modified files over their site-packages locations (and refuses
to run if the detected vLLM/FlashInfer versions don't match, to avoid silent API mismatches). The
equivalent unified diffs are in [`patches/`](patches/) for review.

## Usage

```bash
# Minimal: add --kv-cache-dtype nvfp4. SM120 FA2 path activates automatically.
vllm serve <model> --kv-cache-dtype nvfp4 --gpu-memory-utilization 0.88 [...]
```

Or use the provided launcher (Docker, persists the JIT cache so restarts are fast):

```bash
MODEL_DIR=/path/to/model GPU_UTIL=0.88 IMAGE=vllm-nvfp4-kv-sm120 ./run.sh
```

### `--gpu-memory-utilization` tuning (important)

The +5.5% V-SF cache is allocated lazily and is **not counted by vLLM's memory profiler**. So set
util a bit below where you'd run fp8, or the first real request can OOM:
- too high (e.g. 0.90 here) → biggest pool but OOM at inference (runtime headroom too thin).
- **0.88** → robust, ~1.5× fp8 pool (recommended starting point).
- **0.87** → extra safety margin.

Sweep down from your fp8 util until the first request is stable under your peak concurrency.

## Verify your shape

```bash
# inside the patched image / env:
python harness/h_layout_explicit.py
# expect: cos ≈ 0.995 (NVFP4 quant error), "V byte-exact ... True", "-> PASS" for NHD and HND.
```

Edit `H_q/H_kv/D/page` at the top of the harness to match your model before trusting a new shape.

## License

Apache-2.0 (see [LICENSE](LICENSE)). Contains modifications to FlashInfer and vLLM, both Apache-2.0;
see [NOTICE](NOTICE).
