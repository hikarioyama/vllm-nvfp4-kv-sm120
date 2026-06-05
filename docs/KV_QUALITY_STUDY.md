# KV-cache 量子化 品質劣化スタディ + 独立 K/V 精度カーネル (Step3.7-Flash, SM120)

Step3.7-Flash の KV を NVFP4 化(B2)した代償の **品質劣化を定量化**し、ハイブリッド(K 保護)へ
投資する前に「混合 K=fp8/V=nvfp4 が対称 nvfp4 より良いか」を判断するためのスタディ。
weights は全構成 NVFP4 固定、**KV の持ち方だけ**を変えて感度の高い指標で測る。

計画: `~/.claude/plans/kv-plan-dreamy-hejlsberg.md`

---

## ステータス (2026-06-05)

| 項目 | 状態 |
|---|---|
| Part 2: 感度ハーネス3本 + orchestrator | ✅ 完成 (`harness/{ppl,divergence,ruler_hard,compare_kv_quality}.py`, `run_kv_quality.sh`) |
| Part 2: 6 構成ベンチ (bf16/fp8/nvfp4 × MTP) + ノイズ床 control | ✅ 完走 → `docs/kv_quality_compare.md` |
| Part 1: 独立精度カーネル (K=fp8 / V=nvfp4) | ✅ **数値検証 PASS** (cos=0.995, 対称回帰 intact) |
| Part 1: mixed を vLLM サーバ forward に配線 → mixed×2 採取 | ⏳ **follow-up**（下記） |
| 8 構成 compare | ⏳ mixed×2 採取後 |

---

## Part 2 — 6 構成の結論（確定）

ノイズ床 control(bf16-off 2 回ロード)で解釈を確定。**唯一帰属可能な信号は PPL**。

- **PPL ノイズ床**: 8k/32k/64k で ±0.001〜0.003 nats（2k は ±0.078 で計測不能 → 捨てる）。
- **fp8 KV = 統計的にロスレス**（Δnats 床と同等）。
- **nvfp4 KV = +0.01〜0.02 nats の実劣化**（床の 4〜10 倍, 単調性 bf16≤fp8≤nvfp4 成立）。
- **RULER-hard**: multikey 飽和(1.0)/multivalue≈0.9/vartrack≈0.5。KV 精度差は n=48 で noise 内 → retrieval は測定上無傷。
- **Greedy 分岐**: 床 first-div=15/agree=0.19。fp8・nvfp4・MTP on/off 全部この床と同等 →
  **greedy は本 reasoning モデルでカオス的、KV 精度に帰属不能**（方法論的知見, PPL が信頼レーン）。
- **含意**: 対称 nvfp4 は既に良い(劣化小)。混合の上積み余地は +0.01〜0.02 nats の範囲。

詳細表 → `docs/kv_quality_compare.md`。
教訓: `prompt_logprobs` は O(n_tokens×vocab) の fp32 log_softmax を一括 materialize するため
長文脈で OOM → eval は **util 0.70 / PPL 上限 64k**（128k PPL は原理的不可、長文脈信号は RULER+divergence が担う）。

---

## Part 1 — 独立精度カーネル（検証済み）

`DTypeKV` → `DTypeK`/`DTypeV` 分割で K と V を別精度に。**K=fp8(per-tensor scale, SF 無し) /
V=nvfp4(swizzled SF + B2 in-kernel de-swizzle)** を実現。検証は `harness/h_layout_mixed.py`:

```
[NHD] mixed K=fp8 V=nvfp4: cos=0.99513 PASS
[HND] mixed K=fp8 V=nvfp4: cos=0.99513 PASS
対称回帰 h_layout_b2.py: cos=0.99499 PASS  (壊していない)
```

統合は 4 層（全て additive, `dtype_k==dtype_v` でビット不変）:
1. `prefill.cuh` / `page.cuh`: KernelTraits の trailing `DTypeK_=DTypeKV_, DTypeV_=DTypeKV_`、
   `paged_kv_t` に独立 V stride、K=fp8 は `is_fp4_type_v<DTypeK>` で SF 経路を compile-time skip。
2. wrapper `decode.py`/`prefill.py`: `plan(..., dtype_k=, dtype_v=)`、`run(kv_cache_sf=(None, v_sf))`。
3. JIT module `jit/attention/modules.py`: mixed 時のみ URI に `_mixk_*_mixv_*` suffix、gen→jinja に dtype_k/v。
4. `prefill.py` の fa2 paged_run: mixed でも K-SF/V-SF 両スロットを渡す（非 fp4 側 None）。

検証コマンド: `cd mixed-dev && bash run_mixed_harness.sh`（リポジトリ版は `run_mixed_harness.sh`)。

---

## Follow-up — mixed×2 採取（残作業）

カーネルは検証済み。残るは vLLM **サーバ forward 経路**への配線（`src/vllm/v1/attention/backends/flashinfer.py`）:

1. **vLLM core 受理**: `vllm/config/cache.py` の `CacheDType = Literal[...]` に `"fp8_nvfp4"`(と `"mixed_fp8_nvfp4"`)を追加し、
   `_validate_cache_dtype` を通す（core ファイル patch + launcher で mount）。backend の `supported_kv_cache_dtypes` にも追加。
2. **forward 配線**: `self._context.plan()/run()` と `self._new_tokens.plan()/run()`（backend ~L475/493/519/534）の
   mixed 経路で wrapper に `dtype_k=cache_dtype_k, dtype_v=cache_dtype_v` と `kv_cache_sf=(None, v_sf)` を渡す
   （現状 `REVIEW(mixed): wrapper planning still exposes one kv_data_type` の TODO）。tuple cache は `_split_mixed_kv_cache_views` 利用。
3. **launcher**: `step37-up-eval.sh` の mixed 分岐を `--kv-cache-dtype fp8_nvfp4` に修正（現状の `VLLM_KV_DTYPE_K/V` env は backend が読まない）。util 0.70。
4. mixed サーバ起動 → smoke(France→Paris/coherence) → `PATTERNS="mixed:mixed:0 mixed:mixed:1" bash run_kv_quality.sh` → `compare_kv_quality.py` で 8 構成 Δ 行列。
   compare の verdict は mixed 行が入れば自動で「混合 vs 対称 nvfp4」を出す。

---

## 再現

- 6 構成ベンチ: `bash run_kv_quality.sh`（util 0.70, PPL 2k/8k/32k/64k, RULER 4k–128k）。
- mixed カーネル検証: `bash run_mixed_harness.sh`。
- 集計: `python harness/compare_kv_quality.py` → `docs/kv_quality_compare.md`。
- 結果 JSON は `~/bench/results/kvq_*.json`（.gitignore 対象、リポジトリ外）。
