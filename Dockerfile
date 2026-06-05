# Build a vLLM image with SM120 NVFP4-KV support baked in.
#   docker build -t vllm-nvfp4-kv-sm120 --build-arg BASE_IMAGE=<your-vllm-image> .
# BASE_IMAGE must contain vLLM 0.1.dev16944 + FlashInfer 0.6.11.post2 (the pinned versions).
ARG BASE_IMAGE=vllm/vllm-openai:latest
FROM ${BASE_IMAGE}

# Copy the modified files over their site-packages locations. src/ mirrors the layout under
# /usr/local/lib/python3.12/dist-packages. Adjust SITE if your base uses a different path.
ARG SITE=/usr/local/lib/python3.12/dist-packages
COPY src/flashinfer/data/include/flashinfer/attention/prefill.cuh ${SITE}/flashinfer/data/include/flashinfer/attention/prefill.cuh
COPY src/flashinfer/data/include/flashinfer/page.cuh              ${SITE}/flashinfer/data/include/flashinfer/page.cuh
COPY src/flashinfer/jit/attention/utils.py                        ${SITE}/flashinfer/jit/attention/utils.py
COPY src/vllm/v1/attention/backends/flashinfer.py                 ${SITE}/vllm/v1/attention/backends/flashinfer.py

# The FA2 NVFP4 kernel is JIT-compiled on first use into
#   $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/<ver>/cached_ops   (default base = $HOME).
# Persist it by mounting a host dir there at runtime (see run.sh) so restarts are fast.
