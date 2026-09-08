#!/usr/bin/env bash
# step37-up-eval — single launcher for the 8-pattern KV-quality study.
# One server at a time on :8002 (the bench owns both GPUs; production :8001 stays free).
#
#   KVDTYPE = auto | fp8 | nvfp4 | mixed     (bf16 = "auto")
#   MTP     = 0 | 1                          (1 -> --speculative-config MTP K=1)
#
# auto/fp8  -> STOCK vLLM (no patch mounts).
# nvfp4     -> B2 in-kernel V de-swizzle sources from /home/hikari/bench/step37.
# mixed     -> independent-precision (K=fp8/V=nvfp4) sources from .../mixed-dev,
#              selected via VLLM_KV_DTYPE_K / VLLM_KV_DTYPE_V (honored by the patch).
set -uo pipefail

PORT="${PORT:-8002}"
KVDTYPE="${KVDTYPE:-nvfp4}"
MTP="${MTP:-1}"
SERVED="step3p7"
MODEL_DIR="/home/hikari/models/Step-3.7-Flash-NVFP4"
DRAFT_DIR="/home/hikari/models/Step-3.7-Flash-MTP-draft"
IMAGE="vllm/vllm-openai:stepfun37"
NAME="${NAME:-vllm-step37-eval}"
MAX_LEN="${MAX_LEN:-131072}"
MAXSEQS="${MAXSEQS:-32}"
CGSIZES="${CGSIZES:-1,2,4,8,16}"
K="${K:-1}"
BASE="http://127.0.0.1:${PORT}"

T_PREFILL="/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/attention/prefill.cuh"
T_PAGE="/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/page.cuh"
T_UTILS="/usr/local/lib/python3.12/dist-packages/flashinfer/jit/attention/utils.py"
T_VLLM="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py"

# ── resolve per-KVDTYPE config ────────────────────────────────────────────────
PATCHED=0; SRC=""; KVFLAG="$KVDTYPE"; MIXED_ENV=(); DEF_UTIL="0.92"
case "$KVDTYPE" in
  auto)  KVFLAG="auto";  DEF_UTIL="0.92"; PATCHED=0 ;;
  fp8)   KVFLAG="fp8";   DEF_UTIL="0.92"; PATCHED=0 ;;
  nvfp4) KVFLAG="nvfp4"; DEF_UTIL="0.93"; PATCHED=1; SRC="/home/hikari/bench/step37" ;;
  mixed) KVFLAG="nvfp4"; DEF_UTIL="0.91"; PATCHED=1; SRC="/home/hikari/bench/step37/mixed-dev"
         MIXED_ENV=(-e VLLM_KV_DTYPE_K=fp8 -e VLLM_KV_DTYPE_V=nvfp4) ;;
  *) echo "!! bad KVDTYPE=$KVDTYPE (auto|fp8|nvfp4|mixed)" >&2; exit 2 ;;
esac
GPU_UTIL="${GPU_UTIL:-$DEF_UTIL}"
JIT="/home/hikari/bench/step37/jit-cache-eval-${KVDTYPE}"

# ── spec-decode flag (MTP) ────────────────────────────────────────────────────
SPEC_ARGS=()
if [[ "$MTP" == "1" ]]; then
  SPEC="{\"method\":\"mtp\",\"model\":\"/draft\",\"num_speculative_tokens\":${K}}"
  SPEC_ARGS=(--speculative-config "$SPEC")
fi
CCFG="{\"cudagraph_capture_sizes\":[${CGSIZES}]}"

# ── idempotent: already serving on this port? ─────────────────────────────────
if curl -fsS --max-time 3 "${BASE}/v1/models" 2>/dev/null | grep -q "\"${SERVED}\""; then
  echo "→ already up (${SERVED} @ :${PORT}) ✓ — rm first if you need a different config"; exit 0
fi
for f in "$MODEL_DIR" "$DRAFT_DIR"; do [[ -d "$f" ]] || { echo "!! missing dir: $f" >&2; exit 1; }; done

# ── patch mounts (nvfp4 / mixed only) ─────────────────────────────────────────
MOUNTS=()
if [[ "$PATCHED" == "1" ]]; then
  PATCH="$SRC/nvfp4-fa2-patch/flashinfer.py"
  FI_PREFILL_CUH="$SRC/fi-src/flashinfer/attn/prefill.cuh"
  FI_PAGE_CUH="$SRC/fi-src/flashinfer/attn/page.cuh"
  FI_UTILS_PY="$SRC/fi-src/flashinfer/jit/attention/utils.py"
  for f in "$PATCH" "$FI_PREFILL_CUH" "$FI_PAGE_CUH" "$FI_UTILS_PY"; do
    [[ -f "$f" ]] || { echo "!! missing patch file: $f" >&2; exit 1; }
  done
  [[ "${FRESH_JIT:-1}" == "1" ]] && rm -rf "$JIT"; mkdir -p "$JIT"
  # page.cuh is REQUIRED: prefill.cuh calls paged_kv_t::protective_get_{k,v}_offset (B2 split),
  # which stock page.cuh does not define — a cold JIT compile fails without this mount.
  MOUNTS=(-v "$JIT":/root/.cache/flashinfer
          -v "$PATCH":"$T_VLLM":ro
          -v "$FI_PREFILL_CUH":"$T_PREFILL":ro
          -v "$FI_PAGE_CUH":"$T_PAGE":ro
          -v "$FI_UTILS_PY":"$T_UTILS":ro)
fi

echo "→ launching ${SERVED} KV=${KVDTYPE}(flag=${KVFLAG}) MTP=${MTP} util=${GPU_UTIL} max-len=${MAX_LEN}"
[[ "$PATCHED" == "1" ]] && echo "   patched sources: $SRC  (JIT recompile first run ~5-10min)"
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
  --gpus all --ipc=host --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo \
  -e NCCL_DEBUG=WARN -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_NVFP4_DEBUG="${VLLM_NVFP4_DEBUG:-}" \
  "${MIXED_ENV[@]}" \
  -v "$MODEL_DIR":/model:ro \
  -v "$DRAFT_DIR":/draft:ro \
  "${MOUNTS[@]}" \
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
    --kv-cache-dtype "$KVFLAG" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAXSEQS" \
    --reasoning-parser step3p5 \
    --tool-call-parser step3p5 \
    --enable-auto-tool-choice \
    --async-scheduling \
    --compilation-config "$CCFG" \
    "${SPEC_ARGS[@]}" >/dev/null

for _ in $(seq 1 300); do
  log="$(docker logs "$NAME" 2>&1)"
  if grep -q "Application startup complete" <<<"$log"; then
    kv="$(grep -oE "GPU KV cache size: [0-9,]+ tokens" <<<"$log" | tail -1)"
    conc="$(grep -oE "Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" <<<"$log" | tail -1)"
    echo "→ READY: KV=${KVDTYPE} MTP=${MTP} / ${kv:-KV ?} / ${conc:-} ✓"
    echo "   ${BASE}/v1  (util ${GPU_UTIL}, max_model_len ${MAX_LEN})"
    exit 0
  fi
  if grep -Eq "Traceback \(most recent call last\)|EngineCore (failed|encountered)|CUDA error|CUDA out of memory|torch\.OutOfMemoryError|NaN detected|RuntimeError:|ValueError:|AssertionError:|KeyError:" <<<"$log"; then
    echo "!! startup error (KV=${KVDTYPE} MTP=${MTP}):" >&2
    grep -iE "error|traceback|nan|out of memory|assert|cache size" <<<"$log" | tail -25 >&2
    exit 1
  fi
  if ! docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME"; then
    echo "!! container exited unexpectedly:" >&2; docker logs --tail 50 "$NAME" 2>&1 | grep -vE "IPv4|Networking" >&2; exit 2
  fi
  sleep 5
done
echo "!! timeout (check: docker logs $NAME)" >&2; exit 124
