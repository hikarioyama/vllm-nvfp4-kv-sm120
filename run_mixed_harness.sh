#!/usr/bin/env bash
# run_mixed_harness — compile the mixed-dev flashinfer (wrappers + kernels + JIT templates)
# in a throwaway container with a FRESH JIT cache and run a numeric harness.
#   usage: run_mixed_harness.sh [harness.py]   (default = h_layout_mixed.py)
set -uo pipefail
HARNESS="${1:-/home/hikari/bench/step37/mixed-dev/harness/h_layout_mixed.py}"
SRC=/home/hikari/bench/step37/mixed-dev/fi-src/flashinfer
IMAGE="vllm/vllm-openai:stepfun37"
JIT="/home/hikari/bench/step37/jit-mixed-dev"
P=/usr/local/lib/python3.12/dist-packages/flashinfer
# KEEP_JIT=1 reuses the compiled kernel — safe for Python-only (wrapper) edits, since
# FlashInfer's JIT hash does NOT cover .cuh/.cu content (kernel edits need a fresh cache).
[[ "${KEEP_JIT:-0}" == "1" ]] || rm -rf "$JIT"
mkdir -p "$JIT"
[[ -f "$HARNESS" ]] || { echo "!! no harness: $HARNESS" >&2; exit 1; }

echo "→ mixed harness=$HARNESS  (fresh JIT compile ~5-8min)"
docker run --rm --gpus all --ipc=host --shm-size=8g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_NVFP4_DEBUG="${VLLM_NVFP4_DEBUG:-}" \
  --entrypoint python3 \
  -v "$SRC/attn/prefill.cuh":"$P/data/include/flashinfer/attention/prefill.cuh":ro \
  -v "$SRC/attn/page.cuh":"$P/data/include/flashinfer/page.cuh":ro \
  -v "$SRC/decode.py":"$P/decode.py":ro \
  -v "$SRC/prefill.py":"$P/prefill.py":ro \
  -v "$SRC/utils.py":"$P/utils.py":ro \
  -v "$SRC/jit/attention/utils.py":"$P/jit/attention/utils.py":ro \
  -v "$SRC/jit/attention/modules.py":"$P/jit/attention/modules.py":ro \
  -v "$SRC/jit_csrc/batch_prefill.cu":"$P/data/csrc/batch_prefill.cu":ro \
  -v "$SRC/jit_csrc/batch_decode.cu":"$P/data/csrc/batch_decode.cu":ro \
  -v "$SRC/jit_csrc/batch_prefill_customize_config.jinja":"$P/data/csrc/batch_prefill_customize_config.jinja":ro \
  -v "$SRC/jit_csrc/batch_decode_customize_config.jinja":"$P/data/csrc/batch_decode_customize_config.jinja":ro \
  -v "$JIT":/root/.cache/flashinfer \
  -v "$HARNESS":/h.py:ro \
  "$IMAGE" /h.py
