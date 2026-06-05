"""Mixed K/V harness: K=fp8_e4m3, V=nvfp4 with B2 in-kernel V-SF de-swizzle.

Run inside the same container style as h_layout_b2.py, with modified flashinfer
sources mounted and a fresh FlashInfer JIT cache. Success is finite attention
output with cosine similarity >= 0.99 against bf16 reference attention for both
NHD and HND cache layouts.
"""

import math
import traceback

import torch
import flashinfer
from vllm import _custom_ops as vops
from vllm.utils.torch_utils import nvfp4_kv_cache_full_dim, nvfp4_kv_cache_split_views


torch.manual_seed(37)
dev = "cuda:0"
H_q, H_kv, D = 64, 4, 128
page = 16
full_dim = nvfp4_kv_cache_full_dim(D)
num_blocks = 12
T = 40
blocks = [5, 2, 7]
kgs, vgs = 0.05, 0.05

print(
    "flashinfer",
    flashinfer.__version__,
    "mixed K=fp8_e4m3 V=nvfp4 full_dim",
    full_dim,
)


key = torch.randn(T, H_kv, D, dtype=torch.bfloat16, device=dev) * 0.5
value = torch.randn(T, H_kv, D, dtype=torch.bfloat16, device=dev) * 0.5
slot_mapping = torch.tensor(
    [blocks[i // page] * page + (i % page) for i in range(T)],
    dtype=torch.long,
    device=dev,
)
k_scale_t = torch.tensor(kgs, dtype=torch.float32, device=dev)
v_scale_t = torch.tensor(vgs, dtype=torch.float32, device=dev)

# REVIEW(mixed): K and V are written through the existing single-dtype writer in
# two calls. K real + V scratch for fp8, then K scratch + V real for nvfp4.
k_phys = torch.empty(
    num_blocks, page, H_kv, D, dtype=torch.float8_e4m3fn, device=dev
)
v_scratch_fp8 = torch.empty_like(k_phys)
v_phys = torch.empty(num_blocks, page, H_kv, full_dim, dtype=torch.uint8, device=dev)
k_scratch_nvfp4 = torch.empty_like(v_phys)

vops.reshape_and_cache_flash(
    key, value, k_phys, v_scratch_fp8, slot_mapping, "fp8", k_scale_t, v_scale_t
)
vops.reshape_and_cache_flash(
    key, value, k_scratch_nvfp4, v_phys, slot_mapping, "nvfp4", k_scale_t, v_scale_t
)
print("WRITER OK (K fp8 separate, V nvfp4 separate with swizzled V-SF)")

q = torch.randn(1, H_q, D, dtype=torch.bfloat16, device=dev) * 0.5
sm_sc = 1 / math.sqrt(D)
grp = H_q // H_kv
ws = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=dev)


def ref_attn():
    out = torch.zeros(1, H_q, D, dtype=torch.float32, device=dev)
    K = key.float()
    V = value.float()
    for h in range(H_q):
        w = torch.softmax((q[0, h].float() @ K[:, h // grp, :].T) * sm_sc, dim=-1)
        out[0, h] = w @ V[:, h // grp, :]
    return out


R = ref_attn()


def cmp(o):
    a = o.reshape(-1).float()
    b = R.reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = (a - b).norm().item() / (b.norm().item() + 1e-9)
    return cos, rel


def test_layout(layout):
    print(f"\n========== LAYOUT {layout} mixed K=fp8 V=nvfp4 ==========")
    if layout == "NHD":
        k_data = k_phys
        v_side = v_phys
    else:
        k_data = k_phys.permute(0, 2, 1, 3)
        v_side = v_phys.permute(0, 2, 1, 3)

    (v_data,), (v_sf,) = nvfp4_kv_cache_split_views(v_side)
    print(f"k_data.shape={tuple(k_data.shape)} k_data.stride={k_data.stride()}")
    print(f"v_data.shape={tuple(v_data.shape)} v_data.stride={v_data.stride()}")
    print(f"v_sf.shape={tuple(v_sf.shape)} v_sf.stride={v_sf.stride()}")

    try:
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws, layout, use_tensor_cores=True, backend="fa2"
        )
        ip = torch.tensor([0, len(blocks)], dtype=torch.int32, device=dev)
        idx = torch.tensor(blocks, dtype=torch.int32, device=dev)
        lp = torch.tensor([((T - 1) % page) + 1], dtype=torch.int32, device=dev)
        w.plan(
            ip,
            idx,
            lp,
            H_q,
            H_kv,
            D,
            page,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.float8_e4m3fn,
            dtype_k=torch.float8_e4m3fn,
            dtype_v=torch.uint8,
        )
        o = w.run(
            q,
            (k_data, v_data),
            k_scale=kgs,
            v_scale=vgs,
            kv_cache_sf=(None, v_sf),
        ).float()
        cos, rel = cmp(o)
        finite = torch.isfinite(o).all().item()
        ok = "PASS" if cos >= 0.99 and finite else "FAIL"
        print(f"[{layout}] attention cos={cos:.5f} rel={rel:.4f} finite={finite} -> {ok}")
    except Exception as exc:
        print(f"[{layout}] attn ERR {exc}")
        traceback.print_exc()


test_layout("NHD")
test_layout("HND")
