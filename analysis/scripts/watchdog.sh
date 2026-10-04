#!/usr/bin/env bash
# Self-healing supervisor for the two long-running analysis pipelines.
# Every 2 min: if a pipeline's process is gone but its work is unfinished,
# relaunch it (both pipelines are resume-safe). Logs to watchdog.log.
set -u
BASE="$(cd "$(dirname "$0")/../.." && pwd)"
ANA=$BASE/analysis
VSE=$BASE
LOG=$ANA/outputs/watchdog.log
P2_RETRIES=0
P3_RETRIES=0
MAX_RETRIES=6

log() { echo "[watchdog $(date +%H:%M:%S)] $*" >> "$LOG"; }
log "started"

p2_done() { [ "$(wc -l < $ANA/outputs/diverse_pool/safety.jsonl 2>/dev/null || echo 0)" -ge 3000 ]; }
p3_done() {
  for f in eta20_rdpo eta20_ipo eta10_rdpo eta10_ipo eta0_rdpo eta0_ipo; do
    [ -f "$VSE/results/eval/structured_unsafe_${f}.json" ] || return 1
  done
  return 0
}

while true; do
  FREE_GB=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "$FREE_GB" -lt 1 ]; then
    log "LOW DISK: ${FREE_GB}G free - pausing relaunches"
    sleep 120; continue
  fi

  # ---- P2: gemma safety scoring ----
  if ! p2_done; then
    if ! pgrep -f "precompute_diverse.py --role safety" >/dev/null; then
      if [ "$P2_RETRIES" -lt "$MAX_RETRIES" ]; then
        P2_RETRIES=$((P2_RETRIES+1))
        log "P2 safety not running (rows=$(wc -l < $ANA/outputs/diverse_pool/safety.jsonl 2>/dev/null || echo 0)); relaunch attempt $P2_RETRIES"
        echo "[p2] safety WATCHDOG relaunch $(date)" >> $ANA/outputs/p2_progress.log
        cd $ANA/scripts && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          CUDA_VISIBLE_DEVICES=0 nohup python precompute_diverse.py --role safety --n_rows 3000 \
          >> $ANA/outputs/p2_progress.log 2>&1 &
      else
        log "P2 exceeded max retries"
      fi
    fi
  fi

  # ---- P3: rdpo/ipo grid ----
  if ! p3_done; then
    if ! pgrep -f "src.train_dpo|src.evaluate_model" >/dev/null; then
      if [ "$P3_RETRIES" -lt "$MAX_RETRIES" ]; then
        P3_RETRIES=$((P3_RETRIES+1))
        log "P3 not running; relaunch attempt $P3_RETRIES"
        GPU=1 nohup bash $ANA/scripts/run_p3_robust_baselines.sh >/dev/null 2>&1 &
      else
        log "P3 exceeded max retries"
      fi
    fi
  fi

  if p2_done && p3_done; then
    log "both pipelines complete; watchdog exiting"
    exit 0
  fi
  sleep 120
done
