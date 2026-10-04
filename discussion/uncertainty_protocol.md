# Uncertainty protocol

This document states exactly which reported quantities carry a quantified
uncertainty interval, how those intervals were constructed, and which
quantities carry no uncertainty estimate at all. Line references are to the
repository state used for the reported analyses.

## 1. Two sources of variability, only one of which is quantified

The pilot has two distinct sources of variability.

The first is sampling variability over examples. The 10,000 HH-RLHF training
pairs are one draw from a larger population, and every admission-side quantity
is a function of those examples. This source is quantified by an example-level
bootstrap.

The second is training-seed stochasticity: the LoRA-DPO optimizer, the data
order, and the adapter initialization all depend on the training seed. This
source is not quantified anywhere in the pilot, because every checkpoint was
trained with a single seed.

## 2. Which quantities are deterministic given the cached verifier scores

The verifier scores are computed once, in both directions, and cached in
`results/verifier_outputs/hh_train.precomputed.jsonl`. Given that cache and a
corrupted training file, the following quantities are exact deterministic
functions of the data. Rerunning the analysis reproduces them bit for bit;
there is no stochastic component to average over.

- Retention (`src/consensus_filter_precomputed.py:104`).
- Contamination in admitted data, the field named `harmful_survival_rate`
  (`:107`).
- Clean retention rate and clean false-rejection rate (`:108`, `:109`).
- Corruption detection precision, recall, and F1 (`:97-99`, `:110-112`).
- The rejection-set composition classes
  (`analysis/scripts/e4_rejection.py:40-45`, `:57-61`).
- The verifier error matrix and every dependence estimate derived from it,
  including the Ledoit-Wolf shrunk correlation matrix, `rho_bar`, `n_eff`, and
  the eigenvalue participation ratio
  (`analysis/scripts/dependence.py:18-22`, `:36-39`, `:68-82`).

The determinism is verified rather than asserted:
`analysis/scripts/e1_cdpo_table.py:19-37` recomputes retention, admitted
contamination, and detection F1 from the raw caches for every combination of
`eta` in {10, 20} and method in {raw, single, consensus_k3, oracle}, and asserts
agreement with `results/tables/main_results.csv` to within `1e-9`. The
assertion at line 36 fails the script on any mismatch.

Determinism given the cache is not the same as certainty. These quantities are
still estimates computed on one sample of 10,000 pairs, which is why they carry
bootstrap intervals over examples.

## 3. Bootstrap protocol

### 3.1 Resampling unit

The resampling unit is the example, that is, a single preference pair. Each
bootstrap replicate draws `n_items` row indices with replacement from the
`n_items` rows of the analysis matrix and recomputes the statistic on the
resampled rows:

```
rng = np.random.default_rng(seed)
idx = rng.integers(0, n_items, size=n_items)
E_b = E[idx]
```

Source: `analysis/scripts/dependence.py:55-61`. The docstring at
`analysis/scripts/dependence.py:52-53` states the scope of the procedure
explicitly: it "quantifies sampling variability over examples, not variation
across model-training seeds".

No other unit is resampled. Verifiers are not resampled, prompts are not
resampled, and training runs are not resampled.

### 3.2 Number of resamples and seed

Two bootstrap procedures were run.

Correlation and effective-ensemble-size intervals: 1,000 resamples, seed 42.
The call site is `analysis/scripts/e2_correlation.py:35`,
`bootstrap_correlation(E, n_boot=1000, seed=42, shrunk=True)`. The same
settings are used for the cross-backbone pool comparison at
`analysis/scripts/p2_diverse_analysis.py:68`. The default seed in the estimator
signature is `20260601`
(`analysis/scripts/dependence.py:50`), but both call sites override it with 42,
so 42 is the seed that produced every reported correlation interval. The count
of 1,000 draws is also recorded in `analysis/VERIFIER_POOL_CARD.md`.

Paired contamination-gap intervals: 2,000 resamples, seed 42. The count is
recorded in earlier internal notes that are not part of this release; the seed is recorded as 42 in the two
paired-bootstrap rows of `analysis/EVIDENCE_MANIFEST.md`. The block that
generated these numbers is not present in the current
`analysis/scripts/e1_cdpo_table.py`, which contains only the validation, table
construction, and cDPO comparison sections, and the output artifact
`analysis/outputs/e1_paired_bootstrap.json` is not in the working tree. The
values themselves are preserved in `analysis/EVIDENCE_MANIFEST.md`. Anyone
reproducing the response should regenerate that block before citing the
interval.

### 3.3 Confidence interval construction

Intervals are percentile intervals over the bootstrap distribution. For a
nominal coverage `ci`, with `alpha = (1 - ci) / 2`, the lower and upper
endpoints are the `alpha` and `1 - alpha` empirical quantiles of the replicate
statistics:

```
alpha = (1.0 - ci) / 2.0
lower = np.quantile(draws, alpha, axis=0)
upper = np.quantile(draws, 1.0 - alpha, axis=0)
```

Source: `analysis/scripts/dependence.py:62-65`, with `ci = 0.95` by default
(`:51`), giving the 2.5th and 97.5th percentiles. Derived scalars are
intervalled by evaluating the scalar on each replicate matrix and taking the
same percentiles of the resulting draws: `rho_bar` at
`analysis/scripts/e2_correlation.py:36` and `:82`, and `n_eff` at `:37` and
`:84`.

No bias correction and no acceleration adjustment are applied. These are plain
percentile intervals, not BCa intervals. No normal approximation is used.

### 3.4 Paired bootstrap for method differences

Methods are compared on the same resampled examples rather than on independent
resamples. Within each replicate, the same index vector defines the sample for
every admission rule, the admission decision of each rule is recomputed on that
sample, the statistic of interest is computed per rule, and the difference is
taken within the replicate. The reported interval is the percentile interval of
the replicate differences, not the difference of two marginal intervals.

Pairing is the correct construction here because the admission rules are
applied to the identical set of pairs and their errors on a given pair are
strongly dependent. Comparing marginal intervals would overstate the
uncertainty of the difference.

The reported paired results, all on the contamination-in-admitted-data
quantity, from `analysis/EVIDENCE_MANIFEST.md`:

| Comparison | eta | Gap | 95% CI |
|---|---|---|---|
| Raw minus Consensus (k=3 of 4) | 10% | +0.0152 | [+0.0084, +0.0222] |
| Raw minus Consensus (k=3 of 4) | 20% | +0.0224 | [+0.0129, +0.0317] |
| Single minus Consensus (k=3 of 4) | 20% | +0.0106 | [+0.0036, +0.0176] |

The gap at `eta = 20%` matches the point estimates in
`results/tables/main_results.csv` directly: `0.2031 - 0.1808 = 0.0223`, and at
`eta = 10%`, `0.0986 - 0.0833 = 0.0153`. The ordering Raw above Single above
Consensus in admitted contamination is supported by these intervals.

These intervals describe uncertainty over which examples were drawn. They say
nothing about how the ordering would move under a different verifier pool, a
different corruption model, or a different dataset.

## 4. Quantities that carry training-seed stochasticity and have no interval

The following quantities are outputs of a trained model and therefore depend on
the training seed:

- Preference accuracy (`src/evaluate_model.py:116`).
- Mean margin (`:117`).
- Unsafe refusal rate (`:149`).
- Benign refusal rate (`:151`).

Every checkpoint in the pilot was trained with seed 42. The seed is a single
scalar threaded from `scripts/run_pilot_hh.sh:14` through
`src/train_dpo.py:51` and `:107`, and it is recorded as 42 for every downstream
row of `analysis/EVIDENCE_MANIFEST.md`. The same seed also governs dataset
construction (`src/load_datasets.py`) and corruption injection
(`src/create_corruption.py:158`, `:166`), so those steps are fixed across all
methods and contribute no between-method variation.

State this without hedging: downstream results are single-seed, seed 42. No
training-seed uncertainty is quantified for preference accuracy, mean margin,
unsafe refusal rate, or benign refusal rate. There is no seed loop anywhere in
the repository. The absence of an interval on these numbers is not an oversight
in reporting; the replicates do not exist.

The practical consequence is a limit on what may be concluded. In this pilot
the observed downstream spread across all methods, all corruption rates, and
all objectives is 0.540 to 0.558 in preference accuracy and 0.396 to 0.432 in
unsafe refusal rate. Differences of the observed magnitude, roughly 0.01 to
0.02 in refusal rate, cannot be distinguished from seed noise on the present
evidence. No downstream ranking of methods is claimed, and none is supportable
from these runs. The benign refusal rate is 0.000 in every cell, which reflects
the lexical refusal detector described in
`discussion/metric_definitions.md`, Section 2.5, rather than a measured
difference between methods.

The load-bearing evidence for the position is therefore admission-side, where
the quantities are deterministic given the cached verifier verdicts and their
example-level sampling uncertainty is quantified as described in Section 3.

## 5. Summary

| Quantity | Deterministic given cached scores | Uncertainty quantified | Method |
|---|---|---|---|
| Retention | Yes | Not reported | Deterministic point value; no interval computed |
| Contamination in admitted data | Yes | Yes, for method differences | Paired example-level bootstrap, 2,000 resamples, seed 42, percentile CI |
| True harmful survival | Yes | Not reported | Derivable from the same replicates via the conversion in `metric_definitions.md` Section 1.4, but not currently computed |
| Clean retention and false-rejection rate | Yes | Not reported | Deterministic point value; no interval computed |
| Detection precision, recall, F1 | Yes | Not reported | Deterministic point value; no interval computed |
| Rejection-set composition | Yes | Not reported | Deterministic point value; no interval computed |
| `rho_bar`, `n_eff`, eigen-rank | Yes | Yes | Example-level bootstrap, 1,000 resamples, seed 42, percentile CI |
| Preference accuracy | No | No | Single seed (42) |
| Mean margin | No | No | Single seed (42) |
| Unsafe refusal rate | No | No | Single seed (42) |
| Benign refusal rate | No | No | Single seed (42) |
