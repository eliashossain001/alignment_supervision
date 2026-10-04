# Run configuration and reproducibility record

This document records every seed,
checkpoint identifier, and hyperparameter used to produce the reported pilot
results, together with the file and line from which each value is taken. Values
were read directly from the repository. Where a value is not set anywhere in the
repository, that is stated rather than inferred.

Files read for this document:

`scripts/run_pilot_hh.sh`, `configs/dpo_config.yaml`,
`configs/corruption_config.yaml`, `src/train_dpo.py`,
`src/train_noise_aware_dpo.py`, `src/evaluate_model.py`, `src/load_datasets.py`,
`src/create_corruption.py`, `analysis/scripts/run_p3_robust_baselines.sh`,
`analysis/scripts/dependence.py`, `analysis/scripts/e2_correlation.py`,
`analysis/scripts/p2_diverse_analysis.py`, `analysis/scripts/precompute_diverse.py`,
`src/batched_backend.py`, `src/precompute_verifiers.py`,
`data/corrupted/*.summary.json`, `requirements.txt`.

## 1. Seeds

| Seed | Value | Mechanism | File and line |
|---|---|---|---|
| Dataset subsample seed | 42 | `rng = random.Random(args.seed)`, then `rng.shuffle(train)` and `train[:train_size]`; the same generator is reused for the validation shuffle | `src/load_datasets.py` lines 64 and 67, shuffles at lines 72 and 75; CLI default at line 64; passed as `--seed "$SEED"` with `SEED="${SEED:-42}"` at `scripts/run_pilot_hh.sh` lines 14 and 35 |
| Corruption seed | 42 | `rng = random.Random(seed)` inside `apply_corruption`, drawn once per eligible row via `rng.random() < eta` | `src/create_corruption.py` line 65, draws at lines 87, 98, 108; CLI default at line 158; config value `seed: 42` at `configs/corruption_config.yaml` line 3; passed as `--seed "$SEED"` at `scripts/run_pilot_hh.sh` line 46 |
| Verifier inference determinism | no seed set | Determinism comes from greedy decoding, `do_sample=False`; no `torch.manual_seed` call exists in `src/batched_backend.py`, `src/precompute_verifiers.py`, `src/verifiers.py`, or `agents/local_llm_backend.py` | `src/batched_backend.py` line 84 |
| Cross-backbone subsample seed | 42 | `rng = random.Random(args.seed)`, then `rng.sample(range(len(rows)), args.n_rows)` over `data/processed/hh_train.jsonl`, indices sorted | `analysis/scripts/precompute_diverse.py` lines 83 and 90 to 93 |
| Training seed | 42 | `DPOConfig(..., seed=args.seed)` | `src/train_dpo.py` line 107, CLI default at line 51; config value `seed: 42` at `configs/dpo_config.yaml` line 12; passed as `--seed "$SEED"` at `scripts/run_pilot_hh.sh` line 122 |
| Evaluation seed | not applicable, no stochastic step | Preference scoring is a deterministic forward pass; refusal generation uses `do_sample=False` | `src/evaluate_model.py` lines 72 to 94 and line 127 |
| Bootstrap seed (correlation CIs) | 42 at both call sites | `bootstrap_correlation(E, n_boot=1000, seed=42, shrunk=True)`; the function's own default is 20260601 and is overridden by both callers | `analysis/scripts/e2_correlation.py` line 35 and `analysis/scripts/p2_diverse_analysis.py` line 68; function definition and default at `analysis/scripts/dependence.py` line 50, generator constructed at line 55 |

The bootstrap resamples examples, not training runs. The docstring at
`analysis/scripts/dependence.py` lines 52 to 53 states this: it "quantifies
sampling variability over examples, not variation across model-training seeds".

## 2. Models

| Item | Value | File and line |
|---|---|---|
| DPO base model | `Qwen/Qwen2.5-1.5B-Instruct` | `configs/dpo_config.yaml` line 4; `src/train_dpo.py` line 44; `scripts/run_pilot_hh.sh` line 15 |
| Verifier backbone (Pool A) | `Qwen/Qwen2.5-1.5B-Instruct`, defaulted to the DPO base model | `configs/verifier_config.yaml` line 3; `scripts/run_pilot_hh.sh` line 16 (`VERIFIER_MODEL="${VERIFIER_MODEL:-$BASE_MODEL}"`) |
| Evaluation base model | `Qwen/Qwen2.5-1.5B-Instruct` | `src/evaluate_model.py` line 158; passed explicitly at `scripts/run_pilot_hh.sh` line 150 |
| Reference model for DPO | none passed; the frozen base is used because a `peft_config` is supplied | `src/train_dpo.py` line 120 (`ref_model=None`) with the inline comment "use frozen base when peft_config is provided" |

## 3. LoRA configuration

| Parameter | Value | File and line |
|---|---|---|
| `r` | 16 | `configs/dpo_config.yaml` line 13; `src/train_dpo.py` line 52 |
| `lora_alpha` | 32 | `configs/dpo_config.yaml` line 14; `src/train_dpo.py` line 53 |
| `lora_dropout` | 0.05 | `configs/dpo_config.yaml` line 15; `src/train_dpo.py` line 54 |
| `target_modules` | `["q_proj", "k_proj", "v_proj", "o_proj"]` | `src/train_dpo.py` line 96 |
| `task_type` | `TaskType.CAUSAL_LM` | `src/train_dpo.py` line 92 |
| `inference_mode` | `False` | `src/train_dpo.py` line 93 |
| `bias`, `modules_to_save` | not set, PEFT defaults apply | `src/train_dpo.py` lines 90 to 97 |

The target modules cover the attention projections only. The MLP projections
(`gate_proj`, `up_proj`, `down_proj`) are not adapted. `target_modules` is not
exposed as a CLI argument and is not present in `configs/dpo_config.yaml`; it is
hard-coded.

## 4. DPO optimisation hyperparameters

| Parameter | Value | File and line |
|---|---|---|
| `beta` | 0.1 | `configs/dpo_config.yaml` line 9; `src/train_dpo.py` line 49; `scripts/run_pilot_hh.sh` line 21 |
| Learning rate | 5e-6 | `configs/dpo_config.yaml` line 8; `src/train_dpo.py` line 48; `scripts/run_pilot_hh.sh` line 20 |
| Per-device train batch size | 2 | `configs/dpo_config.yaml` line 6; `src/train_dpo.py` line 46; `scripts/run_pilot_hh.sh` line 18 |
| Gradient accumulation steps | 8 | `configs/dpo_config.yaml` line 7; `src/train_dpo.py` line 47; `scripts/run_pilot_hh.sh` line 19 |
| Effective batch size | 16 | `configs/dpo_config.yaml` line 7 inline comment "effective batch = 16"; each run occupies a single GPU, so no data-parallel multiplier applies |
| Epochs | 1.0 | `configs/dpo_config.yaml` line 5; `src/train_dpo.py` line 45; `scripts/run_pilot_hh.sh` line 17 |
| `loss_type` | `sigmoid` for the main methods | `src/train_dpo.py` line 57 |
| `label_smoothing` | 0.0 for raw, single, consensus_k3, oracle; 0.1 for noise_aware | `configs/dpo_config.yaml` lines 19 to 23; CLI default at `src/train_dpo.py` line 55; `--label_smoothing 0.1` injected for noise_aware at `scripts/run_pilot_hh.sh` line 114 and as a forced default at `src/train_noise_aware_dpo.py` lines 25 to 26 |
| `max_length` | 1024 | `configs/dpo_config.yaml` line 10; `src/train_dpo.py` line 50; set on `DPOConfig` at line 106 |
| `max_prompt_length` | 512 in config, but never passed to `DPOConfig` | value present at `configs/dpo_config.yaml` line 11; absent from the `DPOConfig` construction at `src/train_dpo.py` lines 99 to 116 and absent from the argument parser at lines 41 to 60. The effective value is the TRL `DPOConfig` default, not 512 |
| Precision | bfloat16 weights and `bf16=True` training | `src/train_dpo.py` line 85 (`"dtype": torch.bfloat16`) and line 112 (`bf16=True`) |
| Device placement | `device_map="auto"`, one visible GPU per run | `src/train_dpo.py` line 86; `CUDA_VISIBLE_DEVICES=$GPU_IDX` at `scripts/run_pilot_hh.sh` line 118 |
| `logging_steps` | 20 | `src/train_dpo.py` line 108 |
| `save_strategy` | `"no"` (only the final adapter is saved by `trainer.save_model`) | `src/train_dpo.py` line 109, save at line 127 |
| `eval_strategy` | `"no"` (no in-training evaluation) | `src/train_dpo.py` line 110 |
| `report_to` | `"none"` | `src/train_dpo.py` line 111 |
| `remove_unused_columns` | `False` | `src/train_dpo.py` line 113 |
| Optimizer | not set anywhere in the repository | The `DPOConfig` construction at `src/train_dpo.py` lines 99 to 116 sets no `optim`, `lr_scheduler_type`, `warmup_steps`, `warmup_ratio`, `weight_decay`, `adam_beta1`, `adam_beta2`, or `max_grad_norm`. All of these fall through to the Hugging Face `TrainingArguments` defaults of the installed library version |

Regarding the optimizer, the installed environment in the working directory
reports `transformers 5.5.3`, `trl 0.29.1`, `peft 0.19.1`, `torch 2.13.0+cu130`.
Under that installed version the inherited defaults are `optim=adamw_torch_fused`,
`lr_scheduler_type=linear`, `warmup_steps=0`, `warmup_ratio=None`,
`weight_decay=0.0`, `adam_beta1=0.9`, `adam_beta2=0.999`, `max_grad_norm=1.0`.
These are library defaults observed in the current environment, not values
recorded by the repository. `requirements.txt` specifies only lower bounds
(`transformers>=4.30.0`, `datasets>=2.12.0`, `trl>=0.4.0`, `torch>=2.0.0`) and
does not list `peft` at all, although `src/train_dpo.py` line 23 imports it. The
exact library versions in effect at the time the reported checkpoints were
trained are not recorded anywhere in the repository and are therefore not
recoverable from it; the files checked for a lockfile or version record were
`requirements.txt` and the per-run `train_status.json` schema at
`src/train_dpo.py` lines 129 to 147, which records hyperparameters but no library
versions.

## 5. Data sizes and corruption sweep

| Item | Value | File and line |
|---|---|---|
| Train subsample size | 10,000 pairs | `scripts/run_pilot_hh.sh` line 11 (`TRAIN_SIZE="${TRAIN_SIZE:-10000}"`); default also at `src/load_datasets.py` line 61 |
| Dev subsample size | 1,000 pairs | `scripts/run_pilot_hh.sh` line 12; `src/load_datasets.py` line 62 |
| Test subsample size | 1,000 pairs | `scripts/run_pilot_hh.sh` line 13; `src/load_datasets.py` line 63 |
| Corruption regime | `structured_unsafe` | `scripts/run_pilot_hh.sh` line 25; regime definition at `src/create_corruption.py` lines 92 to 101 |
| Nominal eta values | 0.0, 0.10, 0.20 | `scripts/run_pilot_hh.sh` line 24; `configs/corruption_config.yaml` line 4 |
| Methods trained | raw, noise_aware, single, consensus_k3, oracle | `scripts/run_pilot_hh.sh` line 26 |
| Additional robust baselines | rdpo (`loss_type=robust`), ipo (`loss_type=ipo`) | `analysis/scripts/run_p3_robust_baselines.sh` lines 42 to 47 |
| Consensus rule | k = 3 of n = 4, single-verifier baseline uses safety | `scripts/run_pilot_hh.sh` line 87; `configs/verifier_config.yaml` lines 8 to 11 |
| GPUs used | 2 by default, round-robin over training and evaluation runs | `scripts/run_pilot_hh.sh` line 27, scheduling at lines 118 to 129 and 149 to 158 |

The robust baselines in `analysis/scripts/run_p3_robust_baselines.sh` invoke
`src/train_dpo` without `--seed`, `--epochs`, `--batch_size`, `--grad_accum`,
`--lr`, or `--beta` (lines 21 to 24), so they run at the argument-parser defaults
of `src/train_dpo.py`, which coincide with the values in
`configs/dpo_config.yaml`: seed 42, 1.0 epoch, batch size 2, gradient
accumulation 8, learning rate 5e-6, beta 0.1. rDPO is given the true corruption
rate as its smoothing value (0.2 at eta = 20 percent, 0.1 at eta = 10 percent),
which the script header describes as oracle-informed and therefore favourable to
the baseline (`analysis/scripts/run_p3_robust_baselines.sh` lines 2 and 40 to 46).
At eta = 0 the rDPO run is given a smoothing value of 0.1 rather than 0.0
(line 46), so at that one setting the smoothing does not equal the true
corruption rate.

## 6. Evaluation configuration

| Item | Value | File and line |
|---|---|---|
| `--max_samples` | 500 | `scripts/run_pilot_hh.sh` lines 22 and 154; `analysis/scripts/run_p3_robust_baselines.sh` line 34 |
| Preference-evaluation sample size | first 500 rows of `data/processed/hh_test.jsonl` (which contains 1,000 rows) | truncation at `src/evaluate_model.py` lines 173 to 174; confirmed by `"n": 500` in every file under `results/eval/` |
| Refusal-evaluation sample size | first 500 rows of `data/raw/safety_refusal_eval.jsonl`, comprising 250 unsafe and 250 benign | truncation at `src/evaluate_model.py` lines 179 to 180; confirmed by `"unsafe_n": 250` and `"benign_n": 250` in the files under `results/eval/` |
| Preference metric | fraction of pairs where the mean per-token log probability of `chosen` exceeds that of `rejected`, plus the mean margin | `src/evaluate_model.py` lines 97 to 118, log probability computation at lines 72 to 94 |
| Preference-scoring `max_length` | 1024 tokens, with left truncation of the prompt when the concatenation overflows | `src/evaluate_model.py` line 163, truncation logic at lines 79 to 83 |
| Refusal decoding | greedy, `do_sample=False` | `src/evaluate_model.py` line 127 |
| Refusal `max_new_tokens` | 64 | `src/evaluate_model.py` lines 122 and 164 |
| Refusal detection | substring match of the generated text, lower-cased, against a fixed 15-pattern list | `src/evaluate_model.py` lines 38 to 49 |
| Evaluation precision | bfloat16 | `src/evaluate_model.py` line 62 |
| Adapter handling | if `adapter_config.json` is present in `--model_dir`, the base model is loaded and the LoRA adapter stacked with `PeftModel.from_pretrained` | `src/evaluate_model.py` lines 61 to 67 |

Evaluation is fully deterministic: the preference metric is a forward pass with
no sampling, and refusal generation uses greedy decoding. No random seed is set
in `src/evaluate_model.py`, and none is needed.

## 7. Verifier stage configuration (summary)

Full detail is in `discussion/verifier_prompts.md`. The values relevant to a
reproduction attempt are:

| Item | Value | File and line |
|---|---|---|
| Backbone | `Qwen/Qwen2.5-1.5B-Instruct` | `configs/verifier_config.yaml` line 3 |
| Threshold | 0.7 | `configs/verifier_config.yaml` line 4; `scripts/run_pilot_hh.sh` line 64 |
| Decoding | greedy, `do_sample=False` | `src/batched_backend.py` line 84 |
| `max_new_tokens` | 96 | `src/batched_backend.py` line 27; `src/precompute_verifiers.py` line 82 |
| Input truncation | 2048 tokens | `src/batched_backend.py` line 75 |
| Precision | float16 on CUDA | `src/batched_backend.py` line 51 |
| Batch composition | four role prompts for one direction of one row | `src/precompute_verifiers.py` lines 108 to 117 |
| Sharding | row index modulo shard count, 2 shards | `src/precompute_verifiers.py` lines 90 to 92; `scripts/run_pilot_hh.sh` lines 27 and 56 to 65 |
| Parse-failure fallback | score 0.5, which fails the 0.7 threshold and therefore counts as a reject vote | `src/precompute_verifiers.py` lines 63 to 66 and 69 |

## 8. Explicitly stated reproducibility facts

### 8.1 Single training seed, no multi-seed downstream uncertainty

The training seed used for all reported downstream runs is a single seed, 42.
This is set once as `SEED="${SEED:-42}"` (`scripts/run_pilot_hh.sh` line 14) and
passed to `src/train_dpo` as `--seed "$SEED"` (line 122), where it reaches
`DPOConfig(seed=args.seed)` (`src/train_dpo.py` line 107). The robust baselines
in `analysis/scripts/run_p3_robust_baselines.sh` do not pass `--seed` at all and
therefore use the parser default, which is also 42 (`src/train_dpo.py` line 51).

There is currently no multi-seed downstream uncertainty. No script in the
repository sweeps the training seed: `scripts/run_pilot_hh.sh` iterates over eta
and method only (lines 96 to 131), and
`analysis/scripts/run_p3_robust_baselines.sh` enumerates six fixed
(eta, loss) configurations (lines 42 to 47). Every reported downstream number is
therefore a single-seed point estimate, and the reported confidence intervals on
verifier dependence are bootstraps over examples rather than over training runs
(`analysis/scripts/dependence.py` lines 50 to 66).

### 8.2 Deterministic dataset regeneration, verified against published counts

The dataset construction is deterministic. It was re-derived on 2026-07-31 from
the Hugging Face dataset a DPO-format mirror of `Anthropic/hh-rlhf`, whose train split contains
152,715 rows and whose validation split contains 8,037 rows. These counts were
confirmed against the materialised raw files in the working tree:
`data/raw/dpo_hh_train.jsonl` has 152,715 lines and `data/raw/dpo_hh_val.jsonl`
has 8,037 lines. Those two files are the inputs read by `src/load_datasets.py`
lines 69 and 70.

The regeneration reproduces the published realized corruption counts exactly.
The verification artifacts are the corruption summary files written by
`src/create_corruption.py` lines 136 to 145 and 172 to 174:

| File | `input_rows` | `eligible_rows` | `corrupted_rows` | `nominal_eta` | `effective_eta` | `seed` |
|---|---|---|---|---|---|---|
| `data/corrupted/hh_train_structured_unsafe_eta0.summary.json` | 10000 | 0 | 0 | 0.0 | 0.0 | 42 |
| `data/corrupted/hh_train_structured_unsafe_eta10.summary.json` | 10000 | 10000 | 986 | 0.1 | 0.0986 | 42 |
| `data/corrupted/hh_train_structured_unsafe_eta20.summary.json` | 10000 | 10000 | 2031 | 0.2 | 0.2031 | 42 |

The published counts of 986 corrupted rows at nominal eta = 10 percent (effective
eta 0.0986) and 2,031 corrupted rows at nominal eta = 20 percent (effective eta
0.2031) are reproduced exactly. At eta = 0 the corruption branch is short
circuited by the `eta <= 0.0` guard (`src/create_corruption.py` line 81), which is
why `eligible_rows` is 0 for that file rather than 10,000.

Determinism holds because both stages draw from `random.Random` seeded with 42
and consume draws in a fixed row order: the subsample shuffle at
`src/load_datasets.py` line 72 and the per-row Bernoulli draw at
`src/create_corruption.py` line 98. Under `structured_unsafe` a draw is consumed
only for rows whose `user_choice` is `"A"` (line 96), which on the canonically
oriented HH data is every row, since `src/load_datasets.py` line 52 assigns
`user_choice: "A"` to all rows.

One qualification, stated for completeness: the download step that materialises
`data/raw/dpo_hh_train.jsonl` and `data/raw/dpo_hh_val.jsonl` from the Hugging
Face dataset is not implemented by any script in the repository. A search of all
Python, shell, YAML, and Markdown files for `load_dataset`, `huggingface`, and the
dataset identifier returns only the consumption sites in `src/load_datasets.py`
and the prose note in `README.md` line 42 that raw artifacts are regenerated from
public sources. `README.md` line 46 instructs the user to place an `HF_TOKEN` in
`.env`, and `.env.example` exists for that purpose, but the fetch itself is a
manual step. The two raw files are present in the working tree with the row counts
given above.

### 8.3 Known reproducibility gap: the custom refusal evaluation file

`data/raw/safety_refusal_eval.jsonl` is a custom evaluation file of 250 unsafe
and 250 benign prompts, each row carrying a `prompt` field and a `label` field
whose value is either `unsafe` or `benign` (`src/evaluate_model.py` lines 138 to
146). It is referenced as the `--refusal_eval` argument at
`scripts/run_pilot_hh.sh` line 152 and
`analysis/scripts/run_p3_robust_baselines.sh` line 33, it is documented in the
module docstring at `src/evaluate_model.py` lines 11 to 13, and its absolute path
is recorded in the `refusal_eval_file` field of all 21 result files under
`results/eval/`.

The file is not currently present in the repository. `data/raw/` contains only
`dpo_hh_train.jsonl` and `dpo_hh_val.jsonl`. The file is not recoverable from
public sources: it is a custom-authored prompt set, not a redistribution of a
published benchmark, and no generation script for it exists in the repository.
A search across all Python, shell, JSON, YAML, and Markdown files for
`safety_refusal_eval` returns only the consumption sites named above and the
recorded paths inside `results/eval/*.json`; no producer is present.

This is a known reproducibility gap. Its consequence is bounded and specific: the
two refusal metrics, `unsafe_refusal_rate` and `benign_refusal_rate`
(`src/evaluate_model.py` lines 147 to 152), cannot be recomputed from the
repository as it stands. The preference metrics `preference_acc` and
`mean_margin` are unaffected, because they are computed from
`data/processed/hh_test.jsonl`, which is present and is regenerated
deterministically by `src/load_datasets.py` at seed 42. Every filtering,
correlation, threshold-sweep, and rejection-audit result is likewise unaffected,
because those depend only on `data/processed/hh_train.jsonl` and the precomputed
verifier scores.
