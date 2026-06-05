# Mixed K/V Precision Design, Pass 2

## Scope

This pass threads independent K/V dtypes through the FlashInfer paged decode/prefill wrapper path used by:

```python
flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    workspace, kv_layout, use_tensor_cores=True, backend="fa2"
)
```

The target mixed layout is K=fp8/e4m3 and V=nvfp4/e2m1. K uses per-tensor `k_scale` and no block-scale-factor tensor. V uses packed uint8 nvfp4 data plus swizzled fp8 scale factors, with the existing B2 in-kernel V-SF de-swizzle.

## Wrapper API

`BatchDecodeWithPagedKVCacheWrapper.plan()` now accepts:

```python
dtype_k: Optional[Union[str, torch.dtype]] = None
dtype_v: Optional[Union[str, torch.dtype]] = None
```

`BatchPrefillWithPagedKVCacheWrapper.plan()` accepts the same arguments.

Defaults are backward compatible:

- If `dtype_k is None`, it falls back to `kv_data_type`.
- If `dtype_v is None`, it falls back to `kv_data_type`.
- If `kv_data_type is None`, it falls back to `dtype_k` when provided, otherwise `q_data_type`.

For the mixed harness:

```python
w.plan(...,
       q_data_type=torch.bfloat16,
       kv_data_type=torch.float8_e4m3fn,
       dtype_k=torch.float8_e4m3fn,
       dtype_v=torch.uint8)
o = w.run(q, (k_fp8_cache, v_nvfp4_data),
          k_scale=k_scale,
          v_scale=v_scale,
          kv_cache_sf=(None, v_sf))
```

`run()` accepts `(None, v_sf)` through `_unpack_paged_kv_cache`, so fp8 K can omit K-SF while nvfp4 V still supplies V-SF.

## Module URI And JIT Selection

For symmetric K/V dtype, module factory arguments are unchanged. This preserves the existing URI and cache key.

For `dtype_k != dtype_v`, the module factory appends `(dtype_k, dtype_v)` to the module URI arguments and forwards the side dtypes to generators that expose `dtype_k`/`dtype_v`; if a generator does not expose keyword parameters, the dtypes are appended positionally. This prevents mixed modules from colliding with symmetric fp8 or symmetric nvfp4 JIT cache entries.

The generated params templates now expose:

```cpp
using DTypeK = ...;
using DTypeV = ...;
```

`prefill.cuh` already consumes those aliases through `params_dtype_k<Params>` and `params_dtype_v<Params>`.

## SF Skip

`jit/attention/utils.py` filters `maybe_k_cache_sf` and `maybe_v_cache_sf` independently. It treats `torch.uint8` as the Python carrier dtype for nvfp4 cache data. Therefore:

- K=fp8: no `maybe_k_cache_sf` parameter is emitted.
- V=nvfp4/uint8: `maybe_v_cache_sf` and explicit SF strides are emitted.

The Python `get_batch_prefill_module()` closure mirrors this for mixed modules by passing only the FP4 side scale tensors to the generated FA2 function. Symmetric modules still receive both scale arguments exactly as before.

## K/V Strides

`paged_kv_t` now has independent V strides (`v_stride_page`, `v_stride_n`, `v_stride_h`) plus `protective_get_k_offset()` and `protective_get_v_offset()`. The legacy constructors initialize V strides equal to K strides.

The paged prefill and decode bindings pass separate K/V tensor strides. The FA2 prefill kernel now computes K load offsets from K strides and V load offsets from V strides, so a K last dimension of 128 can coexist with a V data last dimension of 64 and V-SF side tensor of 8.

## Harness

`harness/h_layout_mixed.py`:

- Allocates separate K fp8 and V nvfp4 physical caches.
- Writes K and V through two existing `reshape_and_cache_flash` calls.
- Tests both NHD and HND by permuting physical views.
- Plans with `dtype_k=torch.float8_e4m3fn`, `dtype_v=torch.uint8`.
- Runs with `kv_cache_sf=(None, v_sf)`.
- Prints PASS when output is finite and cosine similarity against bf16 reference attention is `>= 0.99`.

## Symmetric Compatibility

When `dtype_k == dtype_v` or both are omitted, wrapper factories do not append dtype metadata, generated module URIs are unchanged, and the run path still passes the same K-SF/V-SF argument pair. The new `paged_kv_t` fields are initialized to the old stride values in legacy constructors, so existing symmetric nvfp4 and fp8 paths should remain bit-identical.

## GPU Validation Risks

- Generator support for mixed dtype arguments must be confirmed in the full FlashInfer JIT package. This repo slice contains the wrappers and templates, but not every generator implementation.
- The target path is FA2 paged prefill used by tensor-core decode. CUDA-core decode still has a single `DTypeKV` attention kernel and is not the validation target.
- `reshape_and_cache_flash(..., "fp8", ...)` must write K fp8 into a standalone 4D fp8 cache on the target vLLM build.
- The V nvfp4 data/SF split relies on `nvfp4_kv_cache_split_views()` preserving the B2 swizzled scale layout after HND permutation.
- Opus should run `harness/h_layout_mixed.py` on GPU and also rerun symmetric `h_layout_b2.py` to verify `cos >= 0.99` and symmetric regression safety.
