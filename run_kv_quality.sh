#!/usr/bin/env bash
# run_kv_quality — orchestrate the 8-pattern KV-cache quality study end to end.
#
# For each (KV config × MTP) it: launches the eval server on :8002, runs a Gate0
# coherence smoke, then the 3 sensitivity harnesses (ppl / divergence / ruler_hard)
# against the SAME persisted prompts, and tears the server down. bf16-off goes first
# (the gold reference); a second bf16-off reload provides the noise-floor control.
# Finally compare_kv_quality.py emits the bf16-off-referenced Δ matrix.
#
#   PATTERNS : space list of "KVDTYPE:tag:MTP"  (default = the 6 Part1-independent ones)
#              add  "mixed:mixed:0 mixed:mixed:1"  once the mixed kernel is validated.
#   PYBIN    : python with `datasets` (for the one-time Pile prompt build).
set -uo pipefail

ROOT=/home/hikari/bench/step37
HARNESS=$ROOT/harness
LAUNCH=$ROOT/step37-up-eval.sh
SWAP=/home/hikari/bench/swap_bench.py
RESULTS=/home/hikari/bench/results
PYBIN="${PYBIN:-/home/hikari/eigenself/.venv/bin/python}"
PORT="${PORT:-8002}"
NAME="${NAME:-vllm-step37-eval}"

PATTERNS="${PATTERNS:-auto:bf16:0 fp8:fp8:0 nvfp4:nvfp4:0 auto:bf16:1 fp8:fp8:1 nvfp4:nvfp4:1}"
# NOTE: prompt_logprobs materializes an O(n_tokens x vocab) fp32 log_softmax. At
# vocab~128k that is ~0.48 MB/token/GPU, so 32k~15GB / 128k~61GB. 128k is infeasible;
# the long-context degradation signal comes from RULER(128k)+divergence(64k) instead.
PPL_CTX="${PPL_CTX:-2000,8000,32000,64000}"   # 64k proven feasible at util 0.70; 128k is not
RULER_CTX="${RULER_CTX:-4000,16000,64000,128000}"
RULER_DEPTHS="${RULER_DEPTHS:-0.25,0.75}"
RULER_TRIALS="${RULER_TRIALS:-6}"
RULER_CONC="${RULER_CONC:-8}"     # eval server is ours; accuracy is conc-invariant
RUN_CONTROL="${RUN_CONTROL:-1}"

# reuse the JIT cache within a kvdtype (compile once per nvfp4/mixed, not per MTP)
export FRESH_JIT=0
# quality bench needs NO big KV pool (conc=1); cap util low so the prompt_logprobs
# log_softmax spike (~15GB at 32k) has free VRAM and the engine doesn't OOM-die.
export GPU_UTIL="${GPU_UTIL:-0.70}"
mkdir -p "$RESULTS"
rm -rf "$ROOT"/jit-cache-eval-* 2>/dev/null || true
stop_server(){ docker rm -f "$NAME" >/dev/null 2>&1 || true; sleep 3; }
log(){ echo -e "\n\033[1;36m[kvq $(date +%H:%M:%S)] $*\033[0m"; }
alive(){ curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q step3p7; }

run_harnesses(){  # $1=fulltag ; returns 1 if the server died mid-pattern
  local ft="$1"
  log "Gate0 smoke ($ft) [informational — reasoning model trips the heuristic]"
  "$PYBIN" "$SWAP" --port "$PORT" --model step3p7 --label "kvq_$ft" --phase gate \
       --out "$RESULTS/kvq_gate0_$ft.json" || echo "  (gate0 had issues — continuing)"
  alive || { echo "!! server DEAD after gate0 ($ft) — skipping rest"; return 1; }
  log "PPL ($ft)"
  "$PYBIN" "$HARNESS/ppl.py"        --port "$PORT" --tag "$ft" --ctx "$PPL_CTX"   || echo "  ppl failed"
  alive || { echo "!! server DEAD after PPL ($ft) — skipping divergence/ruler"; return 1; }
  log "divergence ($ft)"
  "$PYBIN" "$HARNESS/divergence.py" --port "$PORT" --tag "$ft"                    || echo "  divergence failed"
  alive || { echo "!! server DEAD after divergence ($ft) — skipping ruler"; return 1; }
  log "ruler_hard ($ft)"
  "$PYBIN" "$HARNESS/ruler_hard.py" --port "$PORT" --tag "$ft" --ctx "$RULER_CTX" \
       --depths "$RULER_DEPTHS" --trials "$RULER_TRIALS" --concurrency "$RULER_CONC" || echo "  ruler failed"
  alive || { echo "!! server DEAD after ruler ($ft)"; return 1; }
  return 0
}

# ── 0. free both GPUs (drop any pre-existing eval / b2 server) ────────────────
log "freeing GPUs (stopping vllm-step37-b2 / $NAME)"
docker rm -f vllm-step37-b2 >/dev/null 2>&1 || true
stop_server

# ── 1. main patterns ──────────────────────────────────────────────────────────
for spec in $PATTERNS; do
  IFS=: read -r kv tag mtp <<<"$spec"
  ft="${tag}_mtp${mtp}"
  log "PATTERN $ft  (KVDTYPE=$kv MTP=$mtp)"
  if ! KVDTYPE="$kv" MTP="$mtp" PORT="$PORT" NAME="$NAME" bash "$LAUNCH"; then
    echo "!! launch failed for $ft — skipping"; stop_server; continue
  fi
  if ! run_harnesses "$ft"; then
    echo "!! PATTERN $ft did NOT complete cleanly (server died mid-pattern)."
    [[ "$ft" == "bf16_mtp0" ]] && echo "!! ^ this is the GOLD baseline — comparison will be invalid. Investigate before trusting results."
  fi
  stop_server
done

# ── 2. noise-floor control: a second independent bf16-off load ────────────────
if [[ "$RUN_CONTROL" == "1" ]]; then
  log "CONTROL: second bf16-off load (self ΔPPL/divergence ≈ 0 expected)"
  if KVDTYPE=auto MTP=0 PORT="$PORT" NAME="$NAME" bash "$LAUNCH"; then
    "$PYBIN" "$HARNESS/ppl.py"        --port "$PORT" --tag bf16ctrl_mtp0 --ctx "$PPL_CTX" || true
    "$PYBIN" "$HARNESS/divergence.py" --port "$PORT" --tag bf16ctrl_mtp0 || true
    stop_server
  fi
fi

# ── 3. aggregate ──────────────────────────────────────────────────────────────
log "aggregating -> kv_quality_compare.md"
"$PYBIN" "$HARNESS/compare_kv_quality.py" || echo "compare failed"
log "DONE. results in $RESULTS (kvq_*_*.json, kv_quality_compare.md)"
