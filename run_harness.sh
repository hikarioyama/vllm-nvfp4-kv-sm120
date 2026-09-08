#!/usr/bin/env bash
# run_harness.sh — compile the modified flashinfer (prefill.cuh + codegen) in a throwaway
# container with a FRESH JIT cache and run a numeric harness. Used for the B2 dev loop.
#   usage: run_harness.sh [harness.py] [--keep-jit]
# Default harness = B2 (in-kernel V de-swizzle). Fresh JIT each run unless --keep-jit, because
# FlashInfer's JIT hash does NOT include prefill.cuh content, so source edits need a clean cache.
set -uo pipefail
HARNESS="${1:-/home/hikari/nvfp4-kv-sm120/harness/h_layout_b2.py}"
KEEP_JIT="${2:-}"
IMAGE="vllm/vllm-openai:stepfun37"
FI_PREFILL="/home/hikari/bench/step37/fi-src/flashinfer/attn/prefill.cuh"
FI_UTILS="/home/hikari/bench/step37/fi-src/flashinfer/jit/attention/utils.py"
JIT="/home/hikari/bench/step37/jit-harness"
T_PREFILL="/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/attention/prefill.cuh"
T_UTILS="/usr/local/lib/python3.12/dist-packages/flashinfer/jit/attention/utils.py"

[[ -f "$HARNESS" ]] || { echo "!! no harness: $HARNESS" >&2; exit 1; }
if [[ "$KEEP_JIT" != "--keep-jit" ]]; then rm -rf "$JIT"; fi
mkdir -p "$JIT"

echo "→ harness=$HARNESS  jit=$JIT (fresh=$([[ "$KEEP_JIT" == "--keep-jit" ]] && echo no || echo yes)) — first compile ~3-8min"
docker run --rm --gpus all --ipc=host --shm-size=8g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_NVFP4_DEBUG="${VLLM_NVFP4_DEBUG:-}" \
  --entrypoint python3 \
  -v "$FI_PREFILL":"$T_PREFILL":ro \
  -v "$FI_UTILS":"$T_UTILS":ro \
  -v "$JIT":/root/.cache/flashinfer \
  -v "$HARNESS":/h.py:ro \
  "$IMAGE" /h.py
