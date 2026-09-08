#!/usr/bin/env bash
# Guarded installer for the vLLM 0.27.1 / FlashInfer 0.6.16.post3 SM120 port.
set -euo pipefail

readonly WANT_VLLM="0.27.1"
readonly WANT_FLASHINFER="0.6.16.post3"
readonly BACKUP_SUFFIX=".nvfp4kv-sm120-vllm-0.27.1.bak"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PATCH_FILE="$SCRIPT_DIR/patches/vllm-0.27.1-flashinfer-0.6.16.post3/sm120-fa2-nvfp4-kv.patch"
readonly PY="${PYTHON:-python3}"

MODE="--check"
if (($# > 1)); then
  echo "usage: PYTHON=/path/to/python $0 [--check|--apply|--restore]" >&2
  exit 64
fi
if (($# == 1)); then
  MODE="$1"
fi
case "$MODE" in
  --check | --apply | --restore) ;;
  -h | --help)
    echo "usage: PYTHON=/path/to/python $0 [--check|--apply|--restore]"
    exit 0
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    echo "usage: PYTHON=/path/to/python $0 [--check|--apply|--restore]" >&2
    exit 64
    ;;
esac

command -v "$PY" >/dev/null 2>&1 || {
  echo "nvfp4-kv: Python executable not found: $PY" >&2
  exit 69
}
command -v sha256sum >/dev/null 2>&1 || {
  echo "nvfp4-kv: sha256sum is required" >&2
  exit 69
}
[[ -f "$PATCH_FILE" ]] || {
  echo "nvfp4-kv: patch artifact is missing: $PATCH_FILE" >&2
  exit 66
}

# Read package metadata without importing vLLM, Torch, or FlashInfer. This keeps
# --check CPU-only and avoids initializing CUDA in the target environment.
if ! META="$($PY - <<'PY'
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

try:
    vllm = distribution("vllm")
    flashinfer = distribution("flashinfer-python")
except PackageNotFoundError as exc:
    raise SystemExit(f"missing distribution: {exc.name}") from None

vllm_root = Path(vllm.locate_file("vllm")).resolve()
flashinfer_root = Path(flashinfer.locate_file("flashinfer")).resolve()
print("\t".join((vllm.version, flashinfer.version, str(vllm_root), str(flashinfer_root))))
PY
)"; then
  echo "nvfp4-kv: could not inspect the selected Python environment" >&2
  exit 69
fi

IFS=$'\t' read -r DETECTED_VLLM DETECTED_FLASHINFER VLLM_ROOT FLASHINFER_ROOT <<<"$META"
printf 'python:      %s\n' "$PY"
printf 'vLLM:        %s (%s)\n' "$DETECTED_VLLM" "$VLLM_ROOT"
printf 'FlashInfer:  %s (%s)\n' "$DETECTED_FLASHINFER" "$FLASHINFER_ROOT"

if [[ "$DETECTED_VLLM" != "$WANT_VLLM" || "$DETECTED_FLASHINFER" != "$WANT_FLASHINFER" ]]; then
  printf 'UNSUPPORTED: this port requires vLLM %s and FlashInfer %s exactly.\n' \
    "$WANT_VLLM" "$WANT_FLASHINFER" >&2
  exit 65
fi

readonly -a RELATIVE_PATHS=(
  "flashinfer/data/include/flashinfer/attention/prefill.cuh"
  "vllm/v1/attention/backends/flashinfer.py"
  "vllm/v1/worker/gpu/attn_utils.py"
  "vllm/v1/worker/gpu_model_runner.py"
)
readonly -a TARGET_PATHS=(
  "$FLASHINFER_ROOT/data/include/flashinfer/attention/prefill.cuh"
  "$VLLM_ROOT/v1/attention/backends/flashinfer.py"
  "$VLLM_ROOT/v1/worker/gpu/attn_utils.py"
  "$VLLM_ROOT/v1/worker/gpu_model_runner.py"
)
readonly -a ORIGINAL_HASHES=(
  "2e5927bdc0d36ddb393cb4fab68c2e958d65d5b4b0085c969f7cfa777ecdfb5b"
  "758acb584792328cb20b05373ec8cab93aee76b48562baf1b40f8f4a55c5d054"
  "5f889b6694eb83cdfa1f2d099e646e2c985d5213a68168edc821038de58946a0"
  "993f2c926012241bcb8cc7f568c8e82968e4d81ca125641c831ecf501aac02ac"
)
readonly -a PATCHED_HASHES=(
  "d9ec99f19c5f4ee24e5ac124a7555975dd490bfef2e7af1c236a81070b627906"
  "b55796a9ea9e526c986a3cccbc495acbbeace928ffae7051973c95dd0aece21c"
  "076cb25d89bb967aeb3026c5c5a4c3cc84d5258d89be456759ed9fcfcb9d1343"
  "051a1cc66e6f4df2c4eb41ad861c03ed1236a53fbf25cbe9c1aa79238c38ec8a"
)

declare -a FILE_STATES=()
STATE=""

inspect_targets() {
  local original_count=0 patched_count=0 unknown_count=0
  local i target digest label
  FILE_STATES=()

  for i in "${!TARGET_PATHS[@]}"; do
    target="${TARGET_PATHS[$i]}"
    if [[ ! -f "$target" ]]; then
      label="UNKNOWN"
      ((unknown_count += 1))
    else
      digest="$(sha256sum -- "$target" | awk '{print $1}')"
      if [[ "$digest" == "${ORIGINAL_HASHES[$i]}" ]]; then
        label="ORIGINAL"
        ((original_count += 1))
      elif [[ "$digest" == "${PATCHED_HASHES[$i]}" ]]; then
        label="PATCHED"
        ((patched_count += 1))
      else
        label="UNKNOWN"
        ((unknown_count += 1))
      fi
    fi
    FILE_STATES+=("$label")
    printf '  %-9s %s\n' "$label" "$target"
  done

  if ((unknown_count > 0)); then
    STATE="UNSUPPORTED"
  elif ((original_count == ${#TARGET_PATHS[@]})); then
    STATE="READY"
  elif ((patched_count == ${#TARGET_PATHS[@]})); then
    STATE="APPLIED"
  else
    STATE="PARTIAL"
  fi
  printf 'state:       %s\n' "$STATE"
}

inspect_targets

if [[ "$MODE" == "--check" ]]; then
  case "$STATE" in
    READY)
      echo "All four files match the supported originals; --apply is safe to run."
      exit 0
      ;;
    APPLIED)
      echo "All four files already contain this patch."
      exit 0
      ;;
    PARTIAL)
      echo "PARTIAL: originals and patched files are mixed; restore before applying." >&2
      exit 2
      ;;
    *)
      echo "UNSUPPORTED: at least one source hash or path differs; no changes were made." >&2
      exit 3
      ;;
  esac
fi

if [[ "$MODE" == "--restore" ]]; then
  if [[ "$STATE" == "READY" ]]; then
    echo "All four files are already at their original hashes."
    exit 0
  fi
  if [[ "$STATE" == "UNSUPPORTED" ]]; then
    echo "UNSUPPORTED: refusing to overwrite an unrecognized source state." >&2
    exit 3
  fi

  for i in "${!TARGET_PATHS[@]}"; do
    backup="${TARGET_PATHS[$i]}$BACKUP_SUFFIX"
    [[ -f "$backup" ]] || {
      echo "restore refused: backup is missing: $backup" >&2
      exit 4
    }
    digest="$(sha256sum -- "$backup" | awk '{print $1}')"
    [[ "$digest" == "${ORIGINAL_HASHES[$i]}" ]] || {
      echo "restore refused: backup hash is not the supported original: $backup" >&2
      exit 4
    }
  done

  for i in "${!TARGET_PATHS[@]}"; do
    target="${TARGET_PATHS[$i]}"
    backup="$target$BACKUP_SUFFIX"
    replacement="$target.nvfp4kv-restore.$$"
    cp -- "$backup" "$replacement"
    chmod --reference="$target" "$replacement" 2>/dev/null || true
    mv -f -- "$replacement" "$target"
    printf 'restored:    %s\n' "$target"
  done
  echo "Restore complete; backups were retained."
  exit 0
fi

if [[ "$STATE" == "APPLIED" ]]; then
  echo "All four files already contain this patch; nothing to do."
  exit 0
fi
if [[ "$STATE" != "READY" ]]; then
  echo "$STATE: refusing to patch anything except the four exact originals." >&2
  exit 3
fi
command -v patch >/dev/null 2>&1 || {
  echo "nvfp4-kv: the POSIX patch utility is required for --apply" >&2
  exit 69
}

PATCH_TMP="$(mktemp -d "${TMPDIR:-/tmp}/nvfp4kv-sm120.XXXXXX")"
ROLLBACK_REQUIRED=0
cleanup() {
  local rc=$?
  trap - EXIT
  if ((ROLLBACK_REQUIRED)); then
    echo "nvfp4-kv: installation failed; restoring all four original files" >&2
    for i in "${!TARGET_PATHS[@]}"; do
      target="${TARGET_PATHS[$i]}"
      backup="$target$BACKUP_SUFFIX"
      [[ -f "$backup" ]] && cp -- "$backup" "$target"
      rm -f -- "$target.nvfp4kv-new.$$"
    done
  fi
  if [[ -n "${PATCH_TMP:-}" && -d "$PATCH_TMP" ]]; then
    rm -rf -- "$PATCH_TMP"
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

for i in "${!TARGET_PATHS[@]}"; do
  staged="$PATCH_TMP/${RELATIVE_PATHS[$i]}"
  mkdir -p -- "$(dirname -- "$staged")"
  cp -- "${TARGET_PATHS[$i]}" "$staged"
done

patch --batch --forward --directory="$PATCH_TMP" -p1 <"$PATCH_FILE"

for i in "${!TARGET_PATHS[@]}"; do
  staged="$PATCH_TMP/${RELATIVE_PATHS[$i]}"
  digest="$(sha256sum -- "$staged" | awk '{print $1}')"
  [[ "$digest" == "${PATCHED_HASHES[$i]}" ]] || {
    echo "nvfp4-kv: staged output hash mismatch: $staged" >&2
    exit 5
  }
done

"$PY" -m py_compile \
  "$PATCH_TMP/vllm/v1/attention/backends/flashinfer.py" \
  "$PATCH_TMP/vllm/v1/worker/gpu/attn_utils.py" \
  "$PATCH_TMP/vllm/v1/worker/gpu_model_runner.py"

for i in "${!TARGET_PATHS[@]}"; do
  target="${TARGET_PATHS[$i]}"
  backup="$target$BACKUP_SUFFIX"
  if [[ -f "$backup" ]]; then
    digest="$(sha256sum -- "$backup" | awk '{print $1}')"
    [[ "$digest" == "${ORIGINAL_HASHES[$i]}" ]] || {
      echo "nvfp4-kv: existing backup has an unexpected hash: $backup" >&2
      exit 6
    }
  else
    cp -- "$target" "$backup"
  fi
done

ROLLBACK_REQUIRED=1
for i in "${!TARGET_PATHS[@]}"; do
  target="${TARGET_PATHS[$i]}"
  staged="$PATCH_TMP/${RELATIVE_PATHS[$i]}"
  replacement="$target.nvfp4kv-new.$$"
  cp -- "$staged" "$replacement"
  chmod --reference="$target" "$replacement" 2>/dev/null || true
  mv -f -- "$replacement" "$target"
  printf 'patched:     %s\n' "$target"
done
ROLLBACK_REQUIRED=0

echo "Patch applied and hashes verified."
echo "Before serving, rotate FlashInfer's cached_ops directory so this header is recompiled."
echo "Then add: --kv-cache-dtype nvfp4"
