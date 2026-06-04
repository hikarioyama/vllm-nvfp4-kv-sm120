"""B2 harness — verify the fa2 kernel reads V-SF DIRECTLY from the IN-PLACE swizzled cache
view and de-swizzles per element inside the kernel (no separate contiguous V-SF cache, no
Triton fill). Both K-SF and V-SF are passed as the raw interleaved cache views (zero-copy);
the kernel reads K linear and V de-swizzled. Success = attention cos >= 0.99 + finite, which
proves the in-kernel de-swizzle (prefill.cuh page_produce_kv_sf<true>) is correct.

Run inside a container with the modified flashinfer sources mounted (attn/prefill.cuh +
jit/attention/utils.py) and a FRESH FLASHINFER JIT cache so the B2 kernel recompiles."""
import torch, math, traceback
import flashinfer
from vllm import _custom_ops as vops
from vllm.utils.torch_utils import nvfp4_kv_cache_split_views, nvfp4_kv_cache_full_dim

torch.manual_seed(3)
dev = "cuda:0"
H_q, H_kv, D = 64, 4, 128          # H_kv=4 matches Step3.7 TP=2 production (NHD nb,page,4,sd)
page = 16
full_dim = nvfp4_kv_cache_full_dim(D)  # 72
num_blocks = 12
print("flashinfer", flashinfer.__version__, "full_dim", full_dim, "(B2 in-kernel V de-swizzle)")


def _trtllm_v_scale_to_linear_for_fa2(v_scale, kv_layout):
    scale_dim = v_scale.shape[-1]
    if kv_layout == "NHD":
        num_pages, page_size, num_kv_heads, _ = v_scale.shape
        swz = (num_pages, page_size // 4, 4, num_kv_heads, scale_dim // 4, 4)
        inv = (0, 1, 5, 3, 2, 4)
        out = (num_pages, page_size, num_kv_heads, scale_dim)
    else:
        num_pages, num_kv_heads, page_size, _ = v_scale.shape
        swz = (num_pages, num_kv_heads, page_size // 4, 4, scale_dim // 4, 4)
        inv = (0, 1, 2, 5, 3, 4)
        out = (num_pages, num_kv_heads, page_size, scale_dim)
    return v_scale.reshape(swz).permute(inv).reshape(out)


# one writer, NHD physical (nb,2,page,H,fd)
phys = torch.zeros(num_blocks, 2, page, H_kv, full_dim, dtype=torch.uint8, device=dev)
T = 40
key = torch.randn(T, H_kv, D, dtype=torch.bfloat16, device=dev) * 0.5
value = torch.randn(T, H_kv, D, dtype=torch.bfloat16, device=dev) * 0.5
blocks = [5, 2, 7]
slot_mapping = torch.tensor([blocks[i // page] * page + (i % page) for i in range(T)], dtype=torch.long, device=dev)
kgs, vgs = 0.05, 0.05
k_scale_t = torch.tensor(kgs, dtype=torch.float32, device=dev)
v_scale_t = torch.tensor(vgs, dtype=torch.float32, device=dev)
vops.reshape_and_cache_flash(key, value, phys[:, 0], phys[:, 1], slot_mapping, "nvfp4", k_scale_t, v_scale_t)
print("WRITER OK (NHD physical, V-SF stored SWIZZLED)")

q = torch.randn(1, H_q, D, dtype=torch.bfloat16, device=dev) * 0.5
sm_sc = 1 / math.sqrt(D); grp = H_q // H_kv


def ref_attn():
    out = torch.zeros(1, H_q, D, dtype=torch.float32, device=dev)
    K = key.float(); V = value.float()
    for h in range(H_q):
        w = torch.softmax((q[0, h].float() @ K[:, h // grp, :].T) * sm_sc, dim=-1)
        out[0, h] = w @ V[:, h // grp, :]
    return out


R = ref_attn()
ws = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=dev)


def cmp(o):
    a = o.reshape(-1).float(); b = R.reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item(), (a - b).norm().item() / (b.norm().item() + 1e-9)


def test_layout(layout):
    print(f"\n========== LAYOUT {layout} (B2: V-SF read SWIZZLED + in-kernel de-swizzle) ==========")
    kv_perm = phys if layout == "NHD" else phys.permute(0, 1, 3, 2, 4)
    (k_data, v_data), (k_sf, v_sf) = nvfp4_kv_cache_split_views(kv_perm)
    print(f"k_sf.shape={tuple(k_sf.shape)} k_sf.stride={k_sf.stride()} (K direct)")
    print(f"v_sf.shape={tuple(v_sf.shape)} v_sf.stride={v_sf.stride()} (V swizzled, in-kernel deswz)")

    # B2: pass BOTH K-SF and V-SF as the raw interleaved cache views. No v_dst, no Triton fill.
    try:
        k_sf_fp8 = k_sf.view(torch.float8_e4m3fn)
        v_sf_fp8 = v_sf.view(torch.float8_e4m3fn)
    except Exception as e:
        print(f"[{layout}] .view(fp8) FAILED ({e}); falling back to uint8 pass-through")
        k_sf_fp8 = k_sf; v_sf_fp8 = v_sf

    try:
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, layout, use_tensor_cores=True, backend="fa2")
        ip = torch.tensor([0, len(blocks)], dtype=torch.int32, device=dev)
        idx = torch.tensor(blocks, dtype=torch.int32, device=dev)
        lp = torch.tensor([((T - 1) % page) + 1], dtype=torch.int32, device=dev)
        w.plan(ip, idx, lp, H_q, H_kv, D, page, q_data_type=torch.bfloat16, kv_data_type=torch.uint8)
        o = w.run(q, (k_data, v_data), k_scale=kgs, v_scale=vgs,
                  kv_cache_sf=(k_sf_fp8, v_sf_fp8)).float()
        c, r = cmp(o)
        ok = "PASS" if (c >= 0.99 and torch.isfinite(o).all().item()) else "FAIL"
        print(f"[{layout}] attention cos={c:.5f} rel={r:.4f} finite={torch.isfinite(o).all().item()} -> {ok}")
    except Exception as e:
        print(f"[{layout}] attn ERR {e}"); traceback.print_exc()


test_layout("NHD")
test_layout("HND")
