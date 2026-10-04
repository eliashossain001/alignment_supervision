# Preference-Data Admission Audit

Code for auditing preference data before DPO training: a verifier pool decides
which pairs to admit, and the pipeline reports retention, contamination, and
verifier dependence separately from downstream model behavior.

## Repository layout

```
src/                                Main pipeline
  load_datasets.py                  Subsample HH-RLHF (10k train / 1k dev / 1k test, seed 42)
  create_corruption.py              Broad (structured_unsafe) and targeted (trigger) corruption
  precompute_verifiers.py           Score every pair, both responses, with 4 role verifiers
  verifiers.py, batched_backend.py  Verifier prompts and batched greedy decoding
  consensus_filter_precomputed.py   Admission rules: raw / single verifier / k-of-n consensus / oracle
  train_dpo.py                      LoRA-DPO training (DPO, cDPO via label smoothing, rDPO, IPO)
  evaluate_model.py                 Preference accuracy and refusal evaluation
configs/                            Verifier, corruption, and DPO configuration
scripts/run_pilot_hh.sh             End-to-end main experiment

analysis/
  scripts/
    e1_cdpo_table.py, build_full_grid.py   Seven-method baseline grid
    paired_bootstrap.py                    Paired example-level bootstrap (2,000 resamples)
    random_filter_control.py               Consensus vs. random filter at matched retention
    e2_correlation.py                      Error correlation, effective pool size
    precompute_diverse.py                  Cross-backbone verifier pool (3,000-pair subsample)
    p2_diverse_analysis.py                 Shared vs. cross-backbone dependence
    repair_truncated_scores.py             Recovers scores cut off by the 96-token limit
    calibration_analysis.py                Held-out AUROC, Platt / isotonic calibration
    matched_retention_fixed.py             Pools compared at exactly matched retention
    e3_sweep.py, e5_regimes.py             Frontier sweep; broad vs. targeted corruption
    build_rejection_audit_sample.py        Stratified 200-pair audit sample
    run_pointwise_audit.py, aggregate_*    Blinded audit with two out-of-pool judge models
    run_seed_study.sh, summarize_seed_study.py   Three-seed downstream study
    run_p3_robust_baselines.sh             rDPO and IPO baselines
  EVIDENCE_MANIFEST.md              Every reported number -> source artifact + script
  VERIFIER_POOL_CARD.md             Both verifier pools, measured dependence, known weaknesses
  *.md, *.csv                       Analysis reports and result tables
discussion/                         Metric definitions, run configuration, uncertainty
                                    protocol, verbatim verifier prompts
artifacts/                          Cached verifier scores (parquet) + manifest
data/processed/                     The exact 10k / 1k / 1k HH-RLHF subsample used
results/                            Evaluation outputs, tables, figures, cached scores
docs/mathematical_framework.md      Consensus bound, correlation, effective pool size
```

## Data

`data/processed/` contains the exact HH-RLHF subsample used in the paper
(Anthropic HH-RLHF in prompt/chosen/rejected form, shuffled with seed 42).
Running `src/create_corruption.py` on it with seed 42 reproduces the paper's
corruption exactly: 986 flipped pairs at nominal 10% and 2,031 at nominal 20%.

`results/verifier_outputs/hh_train.precomputed.jsonl` holds the cached
shared-backbone verifier scores for all 10,000 pairs, both responses, and all
four roles (SHA-256 recorded in `artifacts/verifier_score_manifest.json`). With
it, every admission-side result can be recomputed without a GPU.

## Quickstart

```bash
pip install -r requirements.txt

# Admission-side analyses from the cached scores (CPU only)
python3 -m src.create_corruption --input data/processed/hh_train.jsonl \
    --output data/corrupted/hh_train_structured_unsafe_eta20.jsonl \
    --corruption structured_unsafe --eta 0.20 --seed 42
cd analysis/scripts
python3 paired_bootstrap.py          # admitted-contamination gap and CI
python3 random_filter_control.py     # matched random-filter control
python3 e2_correlation.py            # error correlation, effective pool size
python3 calibration_analysis.py      # held-out AUROC and calibration

# Full pipeline including verifier scoring, training, and evaluation (GPU)
bash scripts/run_pilot_hh.sh
```

## Key measured results (HH-RLHF, 10,000 pairs)

| Quantity | Value |
|---|---|
| Admitted contamination at 20% corruption, raw / consensus k=3 | 20.31% / 18.08% at 40.2% retention (paired bootstrap gap 2.23 pp, 95% CI [1.35, 3.18]) |
| Consensus vs. random filter at matched retention | +4.4 pp more corrupted pairs rejected at 20% corruption |
| Shared-backbone pool dependence | error correlation 0.67 [0.66, 0.68], effective pool size 1.33 of 4 |
| Cross-backbone pool (3,000-pair subsample) | error correlation 0.26 [0.24, 0.28], effective pool size 2.25 of 4 |
| Held-out per-role AUROC | 0.515 to 0.533 |
| Targeted poisoning | 70 of 82 poisoned pairs rejected (recall 0.85), 0.82% -> 0.29% prevalence |
| Downstream, three training seeds at 20% corruption | mean preference accuracy 0.548 to 0.551 across five pipelines, seed SD 0.002 to 0.006 |

Admission-side changes do not translate into measurable downstream differences
in these experiments. The full number-to-artifact mapping is in
`analysis/EVIDENCE_MANIFEST.md`.

## Known limitations of this release

- **Refusal evaluation set.** The 500-prompt refusal set (250 unsafe, 250
  benign) was written for this study and has been lost. The refusal values in
  `results/eval/` remain readable but cannot be regenerated; the seed study
  therefore reports preference accuracy only.
- **Training determinism.** Retraining with the same seed is not bit-for-bit
  reproducible on our hardware; seed-42 reruns differ by up to 0.006 in
  preference accuracy.
- **Library versions.** `requirements.txt` pins the versions recorded for the
  follow-up analyses. The versions used for the original training runs were
  not recorded.
- **Hardware.** Experiments ran on 2x NVIDIA TITAN RTX (24 GB).

## License

Code released for research use. A citation entry will be added after the
review process concludes.
