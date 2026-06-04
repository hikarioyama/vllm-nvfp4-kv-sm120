"""Explicit-SF-stride harness (Task B): verify the patched fa2 kernel reads K directly
from the interleaved cache view and V from a CONTIGUOUS de-swizzled cache, both via
explicit SF strides auto-derived in the binding. Must run inside a container with the
modified flashinfer sources mounted (attn/prefill.cuh + jit/attention/utils.py) and a
fresh FLASHINFER_CACHE_DIR so the explicit-stride kernel actually recompiles."""
import torch, math, traceback
import flashinfer
import triton
import triton.language as tl
from vllm import _custom_ops as vops
from vllm.utils.torch_utils import nvfp4_kv_cache_split_views, nvfp4_kv_cache_full_dim

torch.manual_seed(3)
dev = "cuda:0"
H_q, H_kv, D = 64, 4, 128          # H_kv=4 matches Step3.7 TP=2 production (NHD nb,page,4,sd)
page = 16
full_dim = nvfp4_kv_cache_full_dim(D)  # 72
num_blocks = 12
print("flashinfer", flashinfer.__version__, "full_dim", full_dim)


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


def _build_v_deswz_src_offsets(layout, page_size, scale_dim, tok_stride, device):
    p, sd = page_size, scale_dim
    toks = torch.arange(p, device=device, dtype=torch.int32)
    ds = torch.arange(sd, device=device, dtype=torch.int32)
    if layout == "NHD":
        store = (toks.view(1, p, 1, 1) * tok_stride + ds.view(1, 1, 1, sd)).expand(1, p, 1, sd).contiguous()
        off = _trtllm_v_scale_to_linear_for_fa2(store, "NHD")[0, :, 0, :]
    else:
        store = (toks.view(1, 1, p, 1) * tok_stride + ds.view(1, 1, 1, sd)).expand(1, 1, p, sd).contiguous()
        off = _trtllm_v_scale_to_linear_for_fa2(store, "HND")[0, 0, :, :]
    return off.reshape(-1).to(torch.int32).contiguous()


@triton.jit
def _nvfp4_v_sf_fill_kernel(
    slot_mapping_ptr, v_src_ptr, v_dst_ptr, v_src_off_ptr,
    src_blk_stride, src_head_stride, dst_blk_stride, dst_tok_stride, dst_head_stride,
    PAGE_SIZE: tl.constexpr, SD: tl.constexpr, H_KV: tl.constexpr,
):
    i = tl.program_id(0)
    s = tl.load(slot_mapping_ptr + i)
    valid = s >= 0
    s_safe = tl.where(valid, s, 0)
    pg = s_safe // PAGE_SIZE
    tok = s_safe % PAGE_SIZE
    offs_d = tl.arange(0, SD)
    dst_region = tok * dst_tok_stride + offs_d
    v_src_region = tl.load(v_src_off_ptr + tok * SD + offs_d)
    for h in tl.static_range(H_KV):
        sb = pg * src_blk_stride + h * src_head_stride
        db = pg * dst_blk_stride + h * dst_head_stride
        vv = tl.load(v_src_ptr + sb + v_src_region, mask=valid, other=0)
        tl.store(v_dst_ptr + db + dst_region, vv, mask=valid)


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
print("WRITER OK (NHD physical)")

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
    print(f"\n========== LAYOUT {layout} (explicit SF strides) ==========")
    kv_perm = phys if layout == "NHD" else phys.permute(0, 1, 3, 2, 4)
    (k_data, v_data), (k_sf, v_sf) = nvfp4_kv_cache_split_views(kv_perm)
    if layout == "NHD":
        _, page_size, h_kv, sd = k_sf.shape; tok_dim, head_dim = 1, 2
    else:
        _, h_kv, page_size, sd = k_sf.shape; tok_dim, head_dim = 2, 1
    src_tok = k_sf.stride(tok_dim); src_head = k_sf.stride(head_dim)
    print(f"k_sf.shape={tuple(k_sf.shape)} k_sf.stride={k_sf.stride()} (K read DIRECT, no scratch)")

    # V: contiguous de-swizzled cache (real size), filled by the V-only kernel.
    v_dst = torch.empty(tuple(v_sf.shape), dtype=torch.uint8, device=dev).zero_()
    dst_tok = v_dst.stride(tok_dim); dst_head = v_dst.stride(head_dim)
    print(f"v_dst.shape={tuple(v_dst.shape)} v_dst.stride={v_dst.stride()} (V contiguous, +5.5%)")
    tbl = _build_v_deswz_src_offsets(layout, page_size, sd, src_tok, dev)
    _nvfp4_v_sf_fill_kernel[(T,)](
        slot_mapping, v_sf.view(torch.uint8), v_dst, tbl,
        v_sf.stride(0), src_head, v_dst.stride(0), dst_tok, dst_head,
        PAGE_SIZE=page_size, SD=sd, H_KV=h_kv)

    # V byte-exact vs full-tensor reference de-swizzle
    v_ref = _trtllm_v_scale_to_linear_for_fa2(v_sf, layout).contiguous()
    okv = True
    for i in range(T):
        s = slot_mapping[i].item(); p = s // page; t = s % page
        if layout == "NHD":
            vr = v_ref[p, t, :, :]; vd = v_dst[p, t, :, :]
        else:
            vr = v_ref[p, :, t, :]; vd = v_dst[p, :, t, :]
        if not torch.equal(vd.reshape(-1).view(torch.uint8), vr.reshape(-1).view(torch.uint8)):
            okv = False
    print(f"[{layout}] V byte-exact vs reference deswz: {okv}")

    # K view dtype reinterpret check (must work for zero-copy in production)
    try:
        k_sf_fp8 = k_sf.view(torch.float8_e4m3fn)
        v_dst_fp8 = v_dst.view(torch.float8_e4m3fn)
    except Exception as e:
        print(f"[{layout}] .view(fp8) FAILED ({e}); falling back to uint8 pass-through")
        k_sf_fp8 = k_sf; v_dst_fp8 = v_dst

    try:
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, layout, use_tensor_cores=True, backend="fa2")
        ip = torch.tensor([0, len(blocks)], dtype=torch.int32, device=dev)
        idx = torch.tensor(blocks, dtype=torch.int32, device=dev)
        lp = torch.tensor([((T - 1) % page) + 1], dtype=torch.int32, device=dev)
        w.plan(ip, idx, lp, H_q, H_kv, D, page, q_data_type=torch.bfloat16, kv_data_type=torch.uint8)
        o = w.run(q, (k_data, v_data), k_scale=kgs, v_scale=vgs,
                  kv_cache_sf=(k_sf_fp8, v_dst_fp8)).float()
        c, r = cmp(o)
        ok = "PASS" if (c >= 0.99 and torch.isfinite(o).all().item()) else "FAIL"
        print(f"[{layout}] attention cos={c:.5f} rel={r:.4f} finite={torch.isfinite(o).all().item()} -> {ok}")
    except Exception as e:
        print(f"[{layout}] attn ERR {e}"); traceback.print_exc()


test_layout("NHD")
test_layout("HND")
