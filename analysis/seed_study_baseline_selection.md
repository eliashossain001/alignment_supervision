# Selection of the robust-objective baseline for the multi-seed study

This document records the choice of the single objective-only baseline carried
into the multi-seed study, and it is written before any seed-study run is
launched. The selection uses only results that already existed at the time of
writing. No outcome produced by the seed study was consulted.

## Selection rule, declared in advance

The objective-only method with the best mean rank across the existing
nominal eta = 20% preference accuracy and unsafe refusal rate is selected. Both
metrics are ranked with higher values ranked better: preference accuracy is a
correctness measure, and the unsafe refusal rate counts refusals of unsafe
prompts, so a higher value indicates safer behaviour. Tied values receive the
average of the tied ranks.

## Candidate values

All three values are taken from `analysis/full_baseline_grid.csv`, which is
derived from the per-run evaluation artifacts in `results/eval/`. These are
single-seed values from the original pilot, all at training seed 42.

| Method | Preference accuracy | Unsafe refusal rate |
|---|---|---|
| cDPO (label smoothing 0.1) | 0.554 | 0.416 |
| rDPO | 0.550 | 0.396 |
| IPO | 0.550 | 0.424 |

## Ranking

| Method | Rank on preference accuracy | Rank on unsafe refusal | Mean rank |
|---|---|---|---|
| cDPO | 1.0 | 2.0 | **1.50** |
| IPO | 2.5 | 1.0 | 1.75 |
| rDPO | 2.5 | 3.0 | 2.75 |

## Selected method

cDPO, implemented as the DPO objective with label smoothing 0.1, is selected. It
attains the best mean rank under the declared rule. The margin over IPO is one
quarter of a rank position and rests on differences of 0.004 in preference
accuracy and 0.008 in unsafe refusal, both of which are single-seed values
without uncertainty estimates. The selection should therefore be read as
"cDPO is not worse than the alternatives on the existing evidence" rather than
as a finding that cDPO is the strongest robust objective.

Because cDPO was already the configured baseline, the driver is retained
unchanged. This also avoids the reproduction risk flagged in the launch
instructions: cDPO is produced by `src/train_dpo.py` with `--label_smoothing 0.1`
and otherwise published hyperparameters, so its training implementation is
reproducible exactly. Substituting rDPO or IPO would have required verifying a
different loss path, and `analysis/scripts/run_p3_robust_baselines.sh` line 46
additionally applies a smoothing of 0.1 to the eta = 0 rDPO run, which is an
inconsistency that would have needed resolution first.

No second robust objective is added. The launch instruction is explicit that a
second objective must not jeopardise completing three seeds for the primary five
configurations, and the measured wall-time bracket for the existing 15 runs does
not leave room for six more.

## Reproducibility of this selection

The three candidate values can be re-derived with:

```
python3 -c "import csv; [print(r['method_key'], r['preference_acc'], r['unsafe_refusal']) for r in csv.DictReader(open('analysis/full_baseline_grid.csv')) if r['eta_nominal_pct']=='20' and r['method_key'] in ('noise_aware','rdpo','ipo')]"
```

The method key `noise_aware` is the repository's internal name for cDPO.
