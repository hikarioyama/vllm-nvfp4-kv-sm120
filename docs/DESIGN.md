# Design: NVFP4 KV cache on SM120 via FlashInfer FA2 (in-kernel V-SF de-swizzle)

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

## The three wiring problems

vLLM stores the cache interleaved per page as `[K_data | K_scale | V_data | V_scale]`. Feeding FA2:

1. **Per-tensor global scale** must be passed (`k_scale`/`v_scale` = `layer._{k,v}_scale_float`),
   else cos collapses.
2. **V block-scale is 4-token swizzled** by the writer (trtllm-compatible); FA2's smem fill reads it
   linearly → the swizzle has to be undone somewhere.
3. **SF page-stride mismatch** — the FA2 kernel derived the SF gmem stride as
   `data_stride / SF_CONTAINERS` (SF_CONTAINERS = NVFP4_SF_VEC_SIZE/2 = 8) with **no independent SF
   stride input**. In the interleaved layout the SF view shares the *data* page stride, so
   `data_stride/8` mis-addresses the page term by 8× → multi-page garbage.

### How the naïve SM100-style path failed on SM120

The pre-patch path worked around #3 by repacking **both** K and V scales into a **sparse scratch**
whose strides are exactly `(data_page//8, data_head, data_token, 1)`. That sparse block stride
(`data_page/8`) is ~2.25× the real per-block SF size → **+25% of the KV pool**, unaccounted by
vLLM's memory profiler, forcing a low `--gpu-memory-utilization` (~0.80). At that util the real
token pool was roughly a *tie* with fp8 — the 1.78×/byte advantage was eaten by the util hit.

## The patch — explicit SF strides + in-kernel V de-swizzle

Two layers, both shipped here:

### Layer A — explicit SF strides (fixes #3, zero K scratch)

Make the kernel take **explicit** `sf_stride_page/h/n` instead of deriving `data_stride/8`. Then
**K-SF** is passed the interleaved cache view's *own* strides → the kernel reads K **directly from
the cache** (K is stored linear). **Zero scratch for K.**

### Layer B — in-kernel V-SF de-swizzle (fixes #2, zero V scratch)

The V block-scale is stored **swizzled** in the interleaved cache (trtllm 4-token layout). Instead
of de-swizzling it into a separate contiguous parallel cache (an earlier interim approach that cost
**+5.5%** of the KV pool, "B1"), the kernel reads the swizzled V-SF **in place** from the cache view
and applies the inverse 4-token swizzle to the gmem offset **per element, in registers**. So V-SF
needs **zero scratch** too.

**Net: +0% SF over-allocation for both K and V.** No persistent SF cache, nothing hidden from the
memory profiler, so `--gpu-memory-utilization` can sit right where fp8 runs and the **full ~1.78×
byte ceiling is realized as real tokens** (1.78× measured vs fp8 at matched util 0.92: 2.96M vs
1.66M tokens, MTP K=1). This is the "B2" design; it supersedes the +5.5% V-SF cache.

### Why the binding stays small

`maybe_k_cache_sf` / `maybe_v_cache_sf` are FlashInfer **additional tensors** — they arrive at the
binding as `TensorView`s carrying `.stride()`. The codegen helper `generate_additional_params`
auto-emits `<name>_stride_page/h/n` Params fields and a **layout-aware setter** (`page=stride(0)`,
`n`/`h` per `QKVLayout`, read from the int64 `layout` arg already in scope). The kernel reads
`params.maybe_{k,v}_cache_sf_stride_*`, passes K-strides to K-producer calls and V-strides to
V-producer calls, and (B2) computes the de-swizzled V-SF offset inline — so no new wrapper args,
jinja edits, or binding signature changes are needed for the symmetric path.

### Files (symmetric NVFP4 path)

| file | change |
|---|---|
| `flashinfer/data/include/flashinfer/attention/prefill.cuh` | `page_produce_kv_sf` / `produce_kv_sf` take explicit `sf_stride_page/h/n`; V-SF read in place + **in-kernel 4-token de-swizzle**; 3 kernels (single/ragged/paged) split K/V strides across all call sites |
| `flashinfer/data/include/flashinfer/page.cuh` | `paged_kv_t` independent V stride + `protective_get_{k,v}_offset` (split from `protective_get_kv_offset`) — **required**, `prefill.cuh` calls these so a cold JIT compile fails without it |
| `flashinfer/jit/attention/utils.py` | `generate_additional_params` auto-emits stride fields + layout-aware setter for the `maybe_{k,v}_cache_sf` tensors |
| `vllm/v1/attention/backends/flashinfer.py` | SM120 FA2 NVFP4-KV backend: K and V-SF both read directly from the interleaved cache view (no parallel scratch); SM120 gate `_use_fa2_for_nvfp4_kv_on_sm120()` |

The four (incl. `page.cuh`) are also provided as reference unified diffs in [`patches/`](../patches/)
(regenerated against the pinned versions; `apply_patches.sh` installs the full files from `src/`).
**`page.cuh` is required** — `prefill.cuh` calls `paged_kv_t::protective_get_{k,v}_offset` (the
B2 split), which stock `page.cuh` does not define, so a cold JIT compile fails without it.

## Independent K/V precision (K=fp8 / V=nvfp4) — validated, serving wiring is follow-up

The kernel's `DTypeKV` template parameter is split into independent `DTypeK` / `DTypeV` (trailing
defaults `DTypeK_=DTypeKV_, DTypeV_=DTypeKV_`, so `dtype_k==dtype_v` is **bit-identical** to before).
With `K=fp8` the SF path is `compile-time` skipped via `is_fp4_type_v<DTypeK>` (fp8 carries no block
scale — only the per-tensor scale), while `V=nvfp4` still loads + in-kernel de-swizzles its V-SF.
This lets you spend bytes where precision matters more (K) and compress V.

Integration is complete across four layers and **numerically validated** (`harness/h_layout_mixed.py`,
cos 0.99513 NHD & HND; symmetric regression `h_layout_b2.py` cos 0.99499 intact):
1. kernel split (`prefill.cuh` / `page.cuh` independent V stride),
2. wrapper API (`plan(dtype_k=, dtype_v=)`, `run(kv_cache_sf=(None, v_sf))`),
3. JIT module URI (`_mixk_*_mixv_*` suffix so mixed kernels cache separately),
4. FFI arg binding (both SF slots passed, `None` for the non-fp4 side).

What is **not** yet wired is the vLLM **serving** forward path (selecting the mixed dtype via a flag
and feeding the split cache views through `plan()/run()` at request time). See
[`KV_QUALITY_STUDY.md`](KV_QUALITY_STUDY.md) for the remaining steps and the measured headroom (the
6-config study shows symmetric NVFP4 KV is already near-lossless, so mixed's expected upside is the
+0.01–0.02 nats PPL gap — small, hence deferred).

## CUDA-graph + MTP

With B2 there is no incremental SF-cache fill at all — both K-SF and V-SF are read straight from the
fixed interleaved cache during `forward()`, so there is no dynamic-shape repack and
`FULL_AND_PIECEWISE` CUDA-graph capture works. MTP (spec decode) needs no backend change
(`supports_spec_as_decode` stays False; the K-step verify goes through the FA2 prefill path; the
draft 1-token decode is captured).

## Verification

`harness/h_layout_b2.py` writes a real cache via `reshape_and_cache_flash`, passes the K view + the
**in-place swizzled** V-SF view through the FA2 wrapper with explicit strides, and checks the
attention output **cos vs an fp32 reference** (≈0.995 = NVFP4 quant error) for both **NHD and HND**.
A PASS proves the explicit-stride kernel and the in-kernel de-swizzle compiled and ran correctly
(K is read from a view whose strides would be mis-read under the old `data/8` derivation).
`harness/h_layout_mixed.py` does the same for K=fp8 / V=nvfp4. The earlier `h_layout_explicit.py`
(B1, contiguous V-SF cache) is kept for historical comparison.
