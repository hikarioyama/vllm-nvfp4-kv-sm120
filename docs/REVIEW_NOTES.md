# Adversarial review — feat/hybrid-b2 → main

Before merging the B2 (in-kernel V-SF de-swizzle) + independent-K/V branch to `main`, the full
delta `124d457..HEAD` was reviewed adversarially by **Codex (GPT-5.5)** plus **3 independent Opus
agents** (kernel×MTP correctness / mixed-dormancy & regression / docs-vs-measured). Outcome: one
**blocker** (packaging), fixed and re-validated; the rest are minors — the doc minors are fixed, the
code minors are unreachable in the shipped config and are tracked below.

## Fixed

- **[BLOCKER] `page.cuh` was missing from the install** (Codex P1).
  `prefill.cuh` calls `paged_kv_t::protective_get_{k,v}_offset` (the B2 split), which stock
  `page.cuh` does not define. `apply_patches.sh`, the `Dockerfile`, and `patches/` only shipped 3
  files → a **cold JIT compile of the primary nvfp4 kernel fails**. (The benchmarks didn't surface it
  because they reused a warm JIT cache; FlashInfer's JIT hash doesn't cover `.cuh`/`.cu` content.)
  **Fix:** `page.cuh` added to `apply_patches.sh`, `Dockerfile`, `patches/` (now a 4-diff set), and
  the `step37-up-eval.sh` mount. **Re-validated end-to-end** with a truly empty JIT cache (the prior
  root-owned cache was wiped via a throwaway container, so the kernel really recompiled — FlashInfer's
  JIT hash ignores `.cuh` content): the nvfp4 kernel compiled from scratch (page.cuh + the new A1
  `static_assert`) and the SM120 server reached READY — **GPU KV cache 3,046,467 tokens / 23.24×
  @ util 0.92** — and smoke-passed ("capital of France" → "Paris"; coherent free-form generation).
- **[minor] in-kernel V-SF de-swizzle had no compile guard for head_dim** (Opus-A #1). Added
  `static_assert(SF_COLS % 4 == 0, …)` in `prefill.cuh` so an unsupported `head_dim_vo` (not a
  multiple of 64) **fails to compile** instead of silently reading a V-SF word across a token-row
  boundary. head_dim 128 (Step-3.7) is unaffected.
- **[minor] doc accuracy** (Opus-C #1–4): quality penalty widened to **+0.01–0.04 nats (≈4–15×
  floor)** to include the 8k-MTP worst case (was "+0.01–0.02 / 4–10×"); decode-speed range corrected
  to **~91–100%** (conc4 is 91.2%, was "92–100% / equal-to-higher"); DESIGN headline ratio aligned to
  **1.78×** (was "≈1.79×"); the MTP-off baseline (96.0 tok/s) given a labeled provenance block in
  `bench_cap_speed_raw.txt`.

## Tracked (unreachable in the shipped symmetric/fp8 config — addressed when the mixed path is wired)

These are all in the **dormant** independent-K/V (mixed) serving surface or exotic shapes the shipped
config never hits. Mixed serving is an explicit follow-up (see `KV_QUALITY_STUDY.md`); fold these in
when wiring it, with end-to-end tests:

- **[Codex P2] mixed-cache permute drops the block dim** — `vllm/.../flashinfer.py` `_split_*` permute
  helper decrements *every* retained dim after removing the legacy KV axis, yielding `-1` (e.g. NHD →
  `(-1,1,2,3)`). Only dims **after** the removed axis should be decremented. Deterministic bug, but
  only reachable once mixed is selectable at serve time (today it is not — see dormancy guards below).
- **[Opus-A #2] `block_size % 4 == 0` is assumed, not checked** — the in-kernel V de-swizzle groups
  tokens by `entry_idx & ~3u` within a page; a non-multiple-of-4 block size would corrupt. vLLM's
  default block_size 16 is safe. Add a runtime guard on the SM120 NVFP4 path.
- **[Opus-A #3] ragged/contiguous `produce_kv_sf<true>` reads V-SF linearly** (no de-swizzle) — only
  correct if fed already-linear V-SF. Not reached today (vLLM uses the paged wrappers; cascade off).
- **[Opus-A #4] cascade attention would pass no `kv_cache_sf`** — inert (`use_cascade_attention()`
  returns False unconditionally + a dtype short-circuit). Would break NVFP4-KV if re-enabled upstream.
- **[Opus-B F4] dead B1 (+5.5%) V-SF cache machinery** (`_fa2_update_sf_cache`,
  `_nvfp4_v_sf_fill_kernel`, `_NVFP4_SF_SCRATCH`) is present but has **zero call sites** under B2 —
  harmless, a cleanup candidate.

## Verified clean (no change needed)

- **Symmetric NVFP4 / fp8 / auto paths: no regression.** The `DTypeKV`→`DTypeK`/`DTypeV` split is
  bit-identical when K==V (trailing template defaults; legacy `paged_kv_t` ctors set V strides equal
  to K; SF reads are SFINAE-guarded). fp8/auto compile the new code out entirely (Opus-B F6/F7).
- **Mixed is truly dormant at serve time.** Three independent guards reject every mixed-intent
  dtype string before any mixed code runs: the upstream `CacheDType` Literal (argparse choices) + the
  Pydantic validator + `supported_kv_cache_dtypes`; the backend ignores `VLLM_KV_DTYPE_K/V`
  (Opus-B F1–F3). The repo ships no `cache.py` patch, so mixed cannot be selected.
- **MTP × B2 is correct.** `supports_spec_as_decode=False`; the q_len>1 verify goes through the FA2
  paged-prefill kernel and the q_len=1 draft through the captured decode, both hitting the same
  swizzled interleaved cache with no stale intermediate (Opus-A). Empirically: clean MTP=1 capacity +
  speed bench and a 6-config MTP{0,1} quality study.
