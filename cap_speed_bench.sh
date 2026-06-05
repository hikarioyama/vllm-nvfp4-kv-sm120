#!/usr/bin/env bash
# cap_speed_bench — refresh README headline numbers for B2, PRODUCTION config (MTP K=1).
# For each (KVDTYPE, util): launch via step37-up-eval.sh with MTP=1, capture the GPU KV
# cache size + max concurrency from the READY line, run a single+conc decode bench, kill.
# max-len 131072, warm JIT reused (FRESH_JIT=0). Speculative decode = MTP, K=1.
set -uo pipefail
cd /home/hikari/bench/step37
OUT=/home/hikari/bench/results/cap_speed.txt
NAME=vllm-step37-eval
MTP="${MTP:-1}"
: > "$OUT"
log(){ echo "$@" | tee -a "$OUT"; }

run_one(){
  local kv="$1" util="$2"
  log ""
  log "########## KV=$kv util=$util MTP=$MTP ##########"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  local ready
  ready="$(KVDTYPE=$kv MTP=$MTP GPU_UTIL=$util FRESH_JIT=0 PORT=8002 \
            bash /home/hikari/nvfp4-kv-sm120/step37-up-eval.sh 2>&1)"
  log "$ready"
  if ! grep -q "READY:" <<<"$ready"; then
    log "!! launch FAILED for KV=$kv util=$util MTP=$MTP — skipping bench"
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    return 1
  fi
  for _ in $(seq 1 30); do
    curl -fsS --max-time 3 http://127.0.0.1:8002/v1/models 2>/dev/null | grep -q step3p7 && break
    sleep 3
  done
  log "--- decode speed (single + conc 1,2,4 aggregate, MTP=$MTP) ---"
  python3 /home/hikari/bench/step37/bench_client.py 8002 "cap_${kv}_${util}_mtp${MTP}" 2>&1 | tee -a "$OUT"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  sleep 3
}

run_one fp8   0.92
run_one nvfp4 0.92
run_one nvfp4 0.93

log ""
log "########## DONE (MTP=$MTP) ##########"
log "summary (KV pool tokens @ util):"
grep -E "READY:" "$OUT" | sed 's/^/  /'
