#!/usr/bin/env bash
# run.sh — launch a patched vLLM image with NVFP4 KV on SM120.
# Assumes you built the image with the Dockerfile (patches baked in). Persists the FlashInfer
# JIT cache to the host so the explicit-stride kernel only compiles once.
#
#   MODEL_DIR=/path/to/model GPU_UTIL=0.88 IMAGE=vllm-nvfp4-kv-sm120 ./run.sh
set -uo pipefail

IMAGE="${IMAGE:-vllm-nvfp4-kv-sm120}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR=/path/to/model}"
SERVED="${SERVED:-model}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.88}"          # see README: +5.5% SF cache is unaccounted; keep below fp8 util
MAX_LEN="${MAX_LEN:-32768}"
TP="${TP:-1}"
NAME="${NAME:-vllm-nvfp4-kv-sm120}"
JIT="${JIT:-$HOME/.cache/vllm-nvfp4-kv-jit}"   # host dir to persist the FlashInfer JIT cache
EXTRA=("$@")                                    # pass through any extra vllm serve flags

mkdir -p "$JIT"
docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "→ launching $SERVED NVFP4-KV on :$PORT (util=$GPU_UTIL tp=$TP)"
docker run -d --name "$NAME" \
  --gpus all --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$MODEL_DIR":/model:ro \
  -v "$JIT":/root/.cache/flashinfer \
  --network host \
  "$IMAGE" \
  /model \
    --served-model-name "$SERVED" \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size "$TP" \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    "${EXTRA[@]}" >/dev/null

echo "→ started container '$NAME'. Tail logs: docker logs -f $NAME"
echo "  first start JIT-compiles the kernel (~minutes); watch for 'GPU KV cache size: N tokens'."
