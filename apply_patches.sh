#!/usr/bin/env bash
# apply_patches.sh — install the SM120 NVFP4-KV modifications into the active Python env.
# Copies the 3 modified files over their site-packages locations. Refuses to run on a
# version mismatch (the patches are pinned to specific vLLM / FlashInfer releases).
set -euo pipefail

WANT_VLLM="0.1.dev16944"
WANT_FI="0.6.11.post2"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"

SITE="$($PY -c 'import site; print(site.getsitepackages()[0])')"
echo "site-packages: $SITE"

ver() { $PY -c "import $1,sys; print(getattr($1,'__version__','?'))" 2>/dev/null || echo MISSING; }
V_VLLM="$(ver vllm)"; V_FI="$(ver flashinfer)"
echo "detected: vllm=$V_VLLM  flashinfer=$V_FI   (expected vllm=$WANT_VLLM flashinfer=$WANT_FI)"

if [[ "${FORCE:-0}" != "1" ]]; then
  [[ "$V_VLLM" == "$WANT_VLLM" ]] || { echo "!! vllm version mismatch; re-base patches or set FORCE=1" >&2; exit 1; }
  [[ "$V_FI"   == "$WANT_FI"   ]] || { echo "!! flashinfer version mismatch; re-base patches or set FORCE=1" >&2; exit 1; }
fi

backup_and_copy() {  # $1 = relative path under site-packages
  local rel="$1" srcf="$HERE/src/$1" dstf="$SITE/$1"
  [[ -f "$srcf" ]] || { echo "!! missing src: $srcf" >&2; exit 1; }
  [[ -f "$dstf" ]] || { echo "!! target not found (version layout changed?): $dstf" >&2; exit 1; }
  cp -n "$dstf" "$dstf.nvfp4kv.bak" 2>/dev/null && echo "  backup: $dstf.nvfp4kv.bak" || echo "  (backup exists, keeping)"
  cp "$srcf" "$dstf"
  echo "  patched: $rel"
}

echo "applying:"
backup_and_copy flashinfer/data/include/flashinfer/attention/prefill.cuh
backup_and_copy flashinfer/jit/attention/utils.py
backup_and_copy vllm/v1/attention/backends/flashinfer.py

echo "done. NOTE: clear the FlashInfer JIT cache so the new kernel recompiles:"
echo "  rm -rf \"\${HOME}/.cache/flashinfer\"   # real JIT dir = \$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/<ver>/cached_ops"
echo "To revert: restore the *.nvfp4kv.bak files."
