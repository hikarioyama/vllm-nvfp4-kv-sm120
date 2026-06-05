# Mixed K/V Precision Design

## Scope

This change adds machinery for independent K/V cache precision, targeting K=fp8/e4m3 and V=nvfp4/e2m1. The mixed path is guarded by `dtype_k != dtype_v` in the Python cache plumbing. When `dtype_k == dtype_v`, the legacy unified `[num_blocks, 2, ...]` cache path and the existing `DTypeKV` defaults remain in use.

## `prefill.cuh` DType Sites

Residual `DTypeKV` sites are legacy/shared:

| Lines | Classification | Notes |
|---|---|---|
| 65-81 | shared | `params_dtype_k/v` fall back to `Params::DTypeKV` unless generated params expose `DTypeK`/`DTypeV`. |
| 144-183 | shared | `KernelTraits` keeps `DTypeKV_` for ABI/default dispatch, then adds `DTypeK_` and `DTypeV_` defaulting to it. |
| 333-406 | shared dispatcher | `DTypeKVFor<produce_v>` selects K or V dtype at compile time. |
| 509, 618 | shared dispatcher | K/V-specific SF loaders select dtype through `DTypeKVFor<produce_v>`. |
| 1652, 1677 | shared legacy param | Single prefill still receives legacy `Params::DTypeKV`; local K/V aliases come from `KTraits`. |
| 1973, 2032 | shared launch default | Single prefill dispatch instantiates `KernelTraits<..., DTypeKV, ..., DTypeK, DTypeV>`. |
| 2111 | shared legacy param | Ragged prefill uses legacy params but K/V local pointer types. |
| 2466, 2504 | shared paged wrapper | `paged_kv_t<DTypeKV>` remains the binding type; K/V data pointers are reinterpreted to side-specific types. |
| 2934, 2986 | shared launch default | Ragged dispatch uses side-specific shared-memory sizing and traits. |
| 3063, 3116 | shared launch default | Paged dispatch uses side-specific shared-memory sizing and traits. |

K-classified sites:

| Lines | Notes |
|---|---|
| 113-120, 133 | `SharedStorageQKVO` stores `k_smem` as `DTypeK` and sizes K-SF by `is_fp4_type_v<DTypeK>`. |
| 160, 167-168 | `UPCAST_STRIDE_K` and `SWIZZLE_MODE_K` derive from `DTypeK_`. |
| 194-199 | invalid-config checks and shared storage use K dtype for K-side restrictions. |
| 815-990 | rotary and `compute_qk` use `DTypeK`; K-SF multiply is compiled only for FP4 K. |
| 1705, 1814-1826 | single prefill K pointer uses K packing/upcast. |
| 2145, 2309-2317 | ragged K pointer uses K packing/upcast. |
| 2656-2668, 2760-2767, 2829 | paged K column offset and data pointer use K dtype. |

V-classified sites:

| Lines | Notes |
|---|---|
| 113-120, 136 | `SharedStorageQKVO` stores `v_smem` as `DTypeV` and sizes V-SF by `is_fp4_type_v<DTypeV>`. |
| 161, 169-170 | `UPCAST_STRIDE_V` and `SWIZZLE_MODE_V` derive from `DTypeV_`. |
| 1260-1362 | `compute_sfm_v` uses `DTypeV`; V-SF multiply remains active for FP4 V. |
| 1706, 1815-1826 | single prefill V pointer uses V packing/upcast. |
| 2146, 2310-2323 | ragged V pointer uses V packing/upcast. |
| 2657-2678, 2761-2769, 2850 | paged V column offset and data pointer use V dtype. |

## KernelTraits Signature

Adopted signature:

```cpp
template <MaskMode MASK_MODE_, uint32_t CTA_TILE_Q_, uint32_t NUM_MMA_Q_,
          uint32_t NUM_MMA_KV_, uint32_t NUM_MMA_D_QK_, uint32_t NUM_MMA_D_VO_,
          uint32_t NUM_WARPS_Q_, uint32_t NUM_WARPS_KV_,
          PosEncodingMode POS_ENCODING_MODE_, typename DTypeQ_, typename DTypeKV_,
          typename DTypeO_, typename DTypeQKAccum_, typename IdType_,
          typename AttentionVariant_, typename DTypeK_ = DTypeKV_,
          typename DTypeV_ = DTypeKV_>
struct KernelTraits;
```

`DTypeK_` and `DTypeV_` are trailing defaults, so existing symmetric instantiations bind exactly as before.

## K=fp8 vs V=nvfp4 Branching

The decisive branches are now side-specific:

- `page_produce_kv_sf<false>` and `produce_kv_sf<false>` return at compile time when `DTypeK` is not FP4.
- `compute_qk` applies K block scale factors only under `is_fp4_type_v<DTypeK>`.
- `page_produce_kv_sf<true>` and `compute_sfm_v` remain active under `is_fp4_type_v<DTypeV>`, preserving the existing B2 V-SF in-kernel de-swizzle.
- `produce_kv` / `page_produce_kv` choose 64-bit packed FP4 loads independently for K or V.

## Cache Shapes

Symmetric cache keeps the current shape:

```text
(num_blocks, 2, block_size, num_kv_heads, head_size)          # fp8
(num_blocks, 2, block_size, num_kv_heads, head//2 + head//16) # nvfp4
```

Mixed cache returns separate K/V shapes:

```text
K fp8:   (num_blocks, block_size, num_kv_heads, head_size)
V nvfp4: (num_blocks, block_size, num_kv_heads, head//2 + head//16)
```

For `head_size=128`, K last_dim is `128`, V last_dim is `64 + 8 = 72`.

## Writer

No new custom op is strictly required for the writer. The existing `reshape_and_cache_flash` op has one dtype argument, so the mixed path writes with two calls:

1. dtype=`dtype_k`, real K cache, scratch V cache.
2. dtype=`dtype_v`, scratch K cache, real V cache.

This is additive and isolated behind `self.is_kvcache_mixed`. A dedicated mixed writer op would remove scratch allocation and duplicated quantization work, but correctness does not require it.

## Review Markers And GPU Risks

`// REVIEW(mixed):` / `# REVIEW(mixed):` markers:

- `prefill.cuh:172` shared K/V swizzle/thread-layout assumption.
- `prefill.cuh:350` side-specific packed global load width.
- `prefill.cuh:425` side-specific paged load dtype and FP4 predicate.
- `prefill.cuh:510` K=fp8 SF skip vs V=nvfp4 SF load.
- `prefill.cuh:619` contiguous SF load side split.
- `prefill.cuh:932` K-SF dequant branch.
- `prefill.cuh:1316` V-SF dequant branch.
- `prefill.cuh:1816` single-prefill K/V pointer arithmetic.
- `prefill.cuh:2311` ragged K/V pointer arithmetic.
- `prefill.cuh:2658` paged K/V offset split and stride risk.
- `prefill.cuh:2762` refreshed paged prefetch offsets.
- `utils.py:51` K/V SF codegen filter.
- `flashinfer.py:150` mixed tuple cache split.
- `flashinfer.py:592` separate mixed cache shapes.
- `flashinfer.py:869` one `kv_data_type` planning limitation.
- `flashinfer.py:1840` K has no real SF, V keeps swizzled SF.
- `flashinfer.py:2273` two-call mixed writer.

GPU validation risks:

- The current generated paged binding still has one `paged_kv_t<DTypeKV>` and one stride set. True separate K/V last dimensions need generated params/bindings to expose V strides, or a paged descriptor with independent K/V strides.
- Existing wrapper planning has one `kv_data_type`; mixed JIT must emit `Params::DTypeK`/`Params::DTypeV` and instantiate the trailing `KernelTraits` types.
- The compatibility K-SF alias in Python exists only to satisfy older tuple-unpack helpers. The mixed JIT should filter K-SF out so fp8 K never reads it.
- The two-call writer should be validated against `harness/h_layout_mixed.py` for B2 V-SF layout and K fp8 scale behavior.

## Symmetric Path Identity

When `dtype_k == dtype_v`, Python does not enter `self.is_kvcache_mixed`, `get_kv_cache_shape` returns the existing unified shape, `do_kv_cache_update` calls `reshape_and_cache_flash` once exactly as before, and forward uses the existing symmetric nvfp4/fp8 paths. In C++, `DTypeK_` and `DTypeV_` default to `DTypeKV_`, so the new aliases collapse to the old type and the side-specific predicates reduce to the previous `DTypeKV` behavior.
