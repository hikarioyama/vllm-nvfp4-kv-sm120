# Mixed K/V Precision Design, Pass 3

## Scope

This pass threads `dtype_k` / `dtype_v` through the FlashInfer JIT module layer in:

- `fi-src/flashinfer/jit/attention/modules.py`
- `fi-src/flashinfer/prefill.py`
- `fi-src/flashinfer/decode.py`

The target path is mixed K=fp8/e4m3 and V=nvfp4 carried as `torch.uint8`.

## URI Suffix

`get_batch_prefill_uri()`, `get_batch_decode_uri()`, and the batch prefill attention-sink URI helper now accept:

```python
dtype_k: Optional[torch.dtype] = None
dtype_v: Optional[torch.dtype] = None
```

When `dtype_k` and `dtype_v` are both present and differ, the URI appends:

```text
_mixk_{filename_safe_dtype_map[dtype_k]}_mixv_{filename_safe_dtype_map[dtype_v]}
```

When either side dtype is missing, or when `dtype_k == dtype_v`, the suffix is empty. This keeps symmetric and legacy module URIs byte-identical.

## Jinja Context

`gen_batch_prefill_module()` and `gen_batch_decode_module()` now accept `dtype_k` / `dtype_v` and pass them through to the customization generators.

`gen_customize_batch_prefill_module()` and `gen_customize_batch_decode_module()` add `dtype_k` / `dtype_v` to the Jinja render context only for mixed K/V dtypes. Symmetric modules omit these keys, so the pass2 template defaults:

```cpp
using DTypeK = {{ dtype_k|default(dtype_kv, true) }};
using DTypeV = {{ dtype_v|default(dtype_kv, true) }};
```

continue to render through `dtype_kv` exactly as before.

## Wrapper Call Sites

`prefill.py` and `decode.py` no longer call `get_batch_*_uri()` with `_module_args_with_mixed_kv(...)`. The URI helpers receive the original positional args plus keyword `dtype_k` / `dtype_v`, avoiding the prior `13 were given` failure.

The `_call_jit_factory()` fallback remains for older factories, but `modules.py` now exposes keyword-aware generator signatures for the batch prefill/decode path.

## Upstream-File Dependency

This source slice does not include the upstream file that defines `dtype_map`, `dtype_map_kv`, and `filename_safe_dtype_map`; `modules.py` imports them from `flashinfer.jit.utils` via `from ..utils import ...`.

Remaining dependency by filename:

- `flashinfer/jit/utils.py`: upstream should keep `filename_safe_dtype_map` and `dtype_map_kv` entries for `torch.float8_e4m3fn`, `torch.float8_e5m2`, and `torch.uint8`. This pass uses `setdefault()` in `modules.py` for URI-safe fp8/uint8 names and the `torch.uint8 -> uint8_t` KV render fallback, preserving upstream entries when present.

No GPU compile or launch was run in this pass.
