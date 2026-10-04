#!/usr/bin/env bash
# P2: score the 3000-pair subsample with the cross-backbone verifier pool.
# Resume-safe: precompute_diverse.py skips already-scored row ids.
set -uo pipefail
cd "$(dirname "$0")"
LOG="$(cd ../outputs && pwd)/p2_progress.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for role in policy factuality helpfulness safety; do
  echo "[p2] $role start $(date)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 python precompute_diverse.py --role $role --n_rows 3000 \
    >> "$LOG" 2>&1 || echo "[p2] ROLE FAILED: $role" >> "$LOG"
done
echo "[p2] ALL DONE $(date)" >> "$LOG"
