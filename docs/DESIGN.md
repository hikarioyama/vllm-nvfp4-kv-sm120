# Design: NVFP4 KV cache on SM120 via FlashInfer FA2 + explicit SF strides

## Background

NVFP4 KV cache stores each element as 4-bit `e2m1` data plus an `fp8_e4m3` **block scale** per 16
elements, plus a per-tensor global scale. Per token, per KV head, per side (K or V), for head_dim
128:

| | data | block scale | total |
|---|---|---|---|
| fp8 | 128 B | — | **128 B** |
| nvfp4 | 64 B (4-bit × 128) | 8 B (head/16) | **72 B** |

So NVFP4 K+V = 144 B/token/head vs fp8 256 B → **1.78× capacity ceiling** (128/72). (2× is
impossible without dropping the block scales, which is what makes NVFP4 accurate.)

On **SM100** (datacenter Blackwell) vLLM uses trtllm-gen NVFP4-KV cubins. **SM120** (RTX PRO 6000,
GB202) has no such cubins. But FlashInfer's **FA2** attention kernel handles NVFP4 KV in an
arch-generic way: it dequantizes `e2m1`→bf16 in registers (`cvt.e2m1x2`, SM120 has the `sm_120f` HW
path) and computes with standard Ampere `mma.sync` — no tcgen05/tmem/trtllm-gen. So the FA2 path
works on SM120; the job is to feed it the scale factors correctly.

## The three wiring problems (and how the base SM120 path solved them)

vLLM stores the cache interleaved per page as `[K_data | K_scale | V_data | V_scale]`. Feeding FA2:

1. **Per-tensor global scale** must be passed (`k_scale`/`v_scale` = `layer._{k,v}_scale_float`),
   else cos collapses.
2. **V block-scale is 4-token swizzled** by the writer (trtllm-compatible) but FA2 reads it linearly
   → V must be **de-swizzled**. K is stored linear.
3. **SF page-stride mismatch** — the FA2 kernel derived the SF gmem stride as
   `data_stride / SF_CONTAINERS` (SF_CONTAINERS = NVFP4_SF_VEC_SIZE/2 = 8) with **no independent SF
   stride input**. In the interleaved layout the SF view shares the *data* page stride, so
   `data_stride/8` mis-addresses the page term by 8× → multi-page garbage.

The base SM120 path (before this patch) worked around #3 by repacking **both** K and V scales into a
**sparse scratch** whose strides are exactly `(data_page//8, data_head, data_token, 1)`. That sparse
block stride (`data_page/8`) is ~2.25× the real per-block SF size → **+25% of the KV pool**,
unaccounted by vLLM's profiler, forcing a low `--gpu-memory-utilization` (~0.80). At that util the
real token pool was roughly a *tie* with fp8 — the 1.78×/byte advantage was eaten by the util hit.

## This patch: explicit SF strides

Make the kernel take **explicit** `sf_stride_page/h/n` instead of deriving `data_stride/8`. Then:

- **K-SF**: pass the interleaved cache view's *own* strides → the kernel reads K **directly from the
  cache** (K is linear). **Zero scratch.**
- **V-SF**: keep a **contiguous** de-swizzled parallel cache (real size, no 2.25× over-alloc) and
  pass its contiguous strides. **+5.5% of pool** (V only).

SF-cache overhead **+25% → +5.5%**, so util can rise back toward fp8's, and the real token pool
exceeds fp8 (measured ~1.5× at a robust util).

### Why it's a small change

`maybe_k_cache_sf` / `maybe_v_cache_sf` are FlashInfer **additional tensors** — they arrive at the
binding as `TensorView`s, which carry `.stride()`. So we don't need new wrapper args, jinja edits, or
binding edits: the codegen helper `generate_additional_params` auto-emits `<name>_stride_page/h/n`
Params fields and a **layout-aware setter** (`page=stride(0)`, and `n`/`h` per `QKVLayout`, read from
the int64 `layout` arg already in scope). The kernel reads `params.maybe_{k,v}_cache_sf_stride_*` and
passes K-strides to K-producer calls, V-strides to V-producer calls.

### Files

| file | change |
|---|---|
| `flashinfer/data/include/flashinfer/attention/prefill.cuh` | `page_produce_kv_sf` / `produce_kv_sf` take explicit `sf_stride_page/h/n`; 3 kernels (single/ragged/paged) read `params.maybe_{k,v}_cache_sf_stride_*` and split K/V strides across all 12 call sites |
| `flashinfer/jit/attention/utils.py` | `generate_additional_params` auto-emits stride fields + layout-aware setter for the `maybe_{k,v}_cache_sf` tensors |
| `vllm/v1/attention/backends/flashinfer.py` | SM120 FA2 NVFP4-KV backend: K direct (interleaved view), V de-swizzled into a persistent **contiguous** per-layer cache via a graph-safe Triton fill (`_nvfp4_v_sf_fill_kernel`, separate src/dst strides), incremental (O(new tokens)); fallback = K direct + dynamic contiguous V scratch |

## CUDA-graph + MTP

The V-SF cache is filled incrementally at `do_kv_cache_update` time (only the new tokens, fixed-grid
Triton kernel keyed on `slot_mapping`), so `forward()` just reads fixed tensors → no dynamic-shape
repack → FULL_AND_PIECEWISE capture works. MTP (spec decode) needs no backend change
(`supports_spec_as_decode` stays False; the K-step verify goes through the FA2 prefill path; the
draft 1-token decode is captured).

## The util caveat (operational)

The +5.5% V-SF cache is allocated lazily during the first real forward and is **not** counted by
vLLM's memory profiler. So util slightly below fp8's is required, or the first request OOMs. Sweep
down from your fp8 util:

| util (this setup) | pool | note |
|---|---|---|
| 0.90 | ~2.71M | OOM at inference (runtime headroom too thin) |
| 0.88 | ~2.47M | robust, ~1.5× fp8 (recommended) |
| 0.87 | ~2.34M | extra margin |

A future improvement ("B2") removes the V cache entirely by applying the 4-token swizzle to the V-SF
gmem offset **inside** the kernel (read swizzled V directly) → +0% overhead → util can match fp8 →
full 1.78× ceiling. Not implemented here (the vectorized SF load + per-element swizzle is fiddly).

## Verification

`harness/h_layout_explicit.py` writes a real cache via `reshape_and_cache_flash`, passes the K view
+ contiguous V cache through the FA2 wrapper with explicit strides, and checks:
- attention output **cos vs an fp32 reference** (≈0.995 = NVFP4 quant error),
- **byte-exact** V de-swizzle vs a full-tensor reference,
- both **NHD and HND** layouts.

Because K is read from a view whose strides would be mis-read under the old `data/8` derivation, a
PASS proves the explicit-stride kernel actually compiled and ran.
