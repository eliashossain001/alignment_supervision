#!/usr/bin/env bash
# P3: rDPO (TRL loss_type=robust, given the TRUE corruption rate -- oracle-informed,
# i.e. favorable to the baseline) and IPO baselines on unfiltered (raw) data.
# Runs serially on one GPU; train then eval per config.
set -uo pipefail
cd "$(dirname "$0")/../.."

GPU=${GPU:-1}
LOG="$PWD/analysis/outputs/p3_progress.log"
echo "[p3] start $(date)" >> "$LOG"

run_one () {
  local eta=$1 loss=$2 ls=$3 tag=$4
  local train_file="data/filtered/hh_train_structured_unsafe_eta${eta}/raw.train.jsonl"
  local out="results/checkpoints/structured_unsafe_eta${eta}/${tag}"
  local eval_json="results/eval/structured_unsafe_eta${eta}_${tag}.json"
  if [[ -f "$out/train_status.json" ]]; then
    echo "[p3] skip train $tag eta$eta (exists)" >> "$LOG"
  else
    echo "[p3] train $tag eta$eta start $(date)" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU python3 -m src.train_dpo \
      --train_file "$train_file" --output_dir "$out" \
      --loss_type "$loss" --label_smoothing "$ls" \
      >> "$LOG" 2>&1 || { echo "[p3] TRAIN FAILED $tag eta$eta" >> "$LOG"; return 1; }
  fi
  if [[ -f "$eval_json" ]]; then
    echo "[p3] skip eval $tag eta$eta (exists)" >> "$LOG"
  else
    echo "[p3] eval $tag eta$eta start $(date)" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU python3 -m src.evaluate_model \
      --model_dir "$out" \
      --hh_test data/processed/hh_test.jsonl \
      --refusal_eval data/raw/safety_refusal_eval.jsonl \
      --out_json "$eval_json" --max_samples 500 \
      >> "$LOG" 2>&1 || { echo "[p3] EVAL FAILED $tag eta$eta" >> "$LOG"; return 1; }
  fi
  echo "[p3] done $tag eta$eta $(date)" >> "$LOG"
}

# Ordered by priority: eta=20 first (headline corruption level), then eta=10,
# then eta=0. rDPO gets the TRUE eta as smoothing (oracle-informed, favors baseline).
run_one 20 robust 0.2 rdpo
run_one 20 ipo 0.0 ipo
run_one 10 robust 0.1 rdpo
run_one 10 ipo 0.0 ipo
run_one 0  robust 0.1 rdpo
run_one 0  ipo 0.0 ipo

echo "[p3] ALL DONE $(date)" >> "$LOG"
