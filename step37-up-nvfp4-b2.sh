#!/usr/bin/env bash
# step37-up-nvfp4-b2 — Step3.7-Flash with NVFP4 KV cache + B2 IN-KERNEL V DE-SWIZZLE.
# vs v2 (explicit-stride): the +5.5% contiguous V-SF cache and the Triton de-swizzle fill
# are GONE — the fa2 kernel de-swizzles V-SF in place on read. SF overhead +5.5% -> +0%,
# so the profiler is accurate and gpu-util can rise to fp8's ceiling -> pool ~1.78x fp8.
#
# Dev server on :8002 (production :8001 stays the chosen mode). Mounts the SAME modified
# sources (now carrying B2) + a fresh B2 JIT cache (recompiles the de-swizzle kernel).
set -uo pipefail

PORT="${PORT:-8002}"
SERVED="step3p7"
MODEL_DIR="/home/hikari/models/Step-3.7-Flash-NVFP4"
DRAFT_DIR="/home/hikari/models/Step-3.7-Flash-MTP-draft"
IMAGE="vllm/vllm-openai:stepfun37"
NAME="${NAME:-vllm-step37-b2}"
PATCH="/home/hikari/bench/step37/nvfp4-fa2-patch/flashinfer.py"
FI_PREFILL_CUH="/home/hikari/bench/step37/fi-src/flashinfer/attn/prefill.cuh"
FI_UTILS_PY="/home/hikari/bench/step37/fi-src/flashinfer/jit/attention/utils.py"
JIT="/home/hikari/bench/step37/jit-cache-b2"
MAX_LEN="${MAX_LEN:-131072}"
GPU_UTIL="${GPU_UTIL:-0.90}"   # B2 has no +5.5% hidden cache -> push toward fp8's 0.92
MAXSEQS="${MAXSEQS:-32}"
CGSIZES="${CGSIZES:-1,2,4,8,16}"
K="${K:-1}"
BASE="http://127.0.0.1:${PORT}"
SPEC="{\"method\":\"mtp\",\"model\":\"/draft\",\"num_speculative_tokens\":${K}}"
CCFG="{\"cudagraph_capture_sizes\":[${CGSIZES}]}"

T_PREFILL="/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/attention/prefill.cuh"
T_UTILS="/usr/local/lib/python3.12/dist-packages/flashinfer/jit/attention/utils.py"
T_VLLM="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py"

if curl -fsS --max-time 3 "${BASE}/v1/models" 2>/dev/null | grep -q "\"${SERVED}\""; then
  echo "→ already up (${SERVED} @ :${PORT}) ✓"; exit 0
fi
for f in "$MODEL_DIR" "$DRAFT_DIR"; do [[ -d "$f" ]] || { echo "!! missing dir: $f" >&2; exit 1; }; done
for f in "$PATCH" "$FI_PREFILL_CUH" "$FI_UTILS_PY"; do [[ -f "$f" ]] || { echo "!! missing file: $f" >&2; exit 1; }; done
[[ "${FRESH_JIT:-1}" == "1" ]] && rm -rf "$JIT"
mkdir -p "$JIT"

echo "→ launching ${SERVED} NVFP4-KV(B2 in-kernel deswz) + MTP (util=${GPU_UTIL} max-len=${MAX_LEN} max-seqs=${MAXSEQS}) … JIT recompile first run ~5-10min"
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
  --gpus all --ipc=host --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo \
  -e NCCL_DEBUG=WARN -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_NVFP4_DEBUG="${VLLM_NVFP4_DEBUG:-}" \
  -v "$MODEL_DIR":/model:ro \
  -v "$DRAFT_DIR":/draft:ro \
  -v "$JIT":/root/.cache/flashinfer \
  -v "$PATCH":"$T_VLLM":ro \
  -v "$FI_PREFILL_CUH":"$T_PREFILL":ro \
  -v "$FI_UTILS_PY":"$T_UTILS":ro \
  --network host \
  "$IMAGE" \
  /model \
    --served-model-name "$SERVED" \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --disable-custom-all-reduce \
    --quantization modelopt \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAXSEQS" \
    --reasoning-parser step3p5 \
    --tool-call-parser step3p5 \
    --enable-auto-tool-choice \
    --async-scheduling \
    --compilation-config "$CCFG" \
    --speculative-config "$SPEC" >/dev/null

for _ in $(seq 1 240); do
  log="$(docker logs "$NAME" 2>&1)"
  if grep -q "Application startup complete" <<<"$log"; then
    kv="$(grep -oE "GPU KV cache size: [0-9,]+ tokens" <<<"$log" | tail -1)"
    conc="$(grep -oE "Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" <<<"$log" | tail -1)"
    echo "→ READY: ${SERVED} NVFP4-KV(B2) / ${kv:-KV ?} / ${conc:-} / MTP K=${K} ✓"
    echo "   ${BASE}/v1  (util ${GPU_UTIL}, max_model_len ${MAX_LEN})"
    exit 0
  fi
  if grep -Eq "Traceback \(most recent call last\)|EngineCore (failed|encountered)|CUDA error|CUDA out of memory|torch\.OutOfMemoryError|NaN detected|RuntimeError:|ValueError:|AssertionError:|KeyError:" <<<"$log"; then
    echo "!! startup error:" >&2
    grep -iE "error|traceback|nan|out of memory|assert|cache size" <<<"$log" | tail -25 >&2
    exit 1
  fi
  if ! docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME"; then
    echo "!! container exited unexpectedly:" >&2; docker logs --tail 50 "$NAME" 2>&1 | grep -vE "IPv4|Networking" >&2; exit 2
  fi
  sleep 5
done
echo "!! timeout (check: docker logs $NAME)" >&2; exit 124
