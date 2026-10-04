# Pair-aware margin rule: exact definition and provenance

## Definition

For a preference pair with chosen direction c and alternative direction a, let
s_r(c) and s_r(a) be the score in [0,1] assigned by role verifier
r in {Safety, Helpfulness, Factuality, Policy} to each direction (each direction
is scored independently, in isolation). The rule admits the pair if and only if:

1. Consensus condition: |{r : s_r(c) >= 0.7}| >= k, and
2. Margin condition: mean_r s_r(c) >= mean_r s_r(a).

The margin condition is parameter-free (no threshold beyond the comparison of
the two means). The 0.7 score threshold is inherited unchanged from the pilot's
verifier configuration (configs/verifier_config.yaml) and was not tuned for
this rule. Aggregation is an unweighted mean over the four role scores.

## Cost

The rule requires scores for BOTH directions of each pair. The pilot's
precompute step already scores both directions (this is what enables oracle
comparisons and the rejection audit), so the rule adds no verifier cost in this
pipeline; in a pipeline that scores only the chosen direction it doubles
verifier inference.

## Provenance and honesty statement (include in any write-up)

- There is NO development/held-out split for this rule. All reported margin-rule
  numbers are computed on the same data used for all other filter-side analyses
  (the full 10,000-pair set for the shared-backbone pool; the 3,000-pair
  subsample, seed 42, for the cross-backbone pool).
- The rule itself was NOT tuned (the margin condition has no free parameter),
  but the highlighted configuration (cross-backbone pool with k=2) was selected
  AFTER inspecting results across the small grid {shared, cross-backbone} x
  {k=2, k=3} x {eta=10%, eta=20%}. The headline numbers are therefore
  EXPLORATORY, not held-out estimates.
- No model has been trained on margin-admitted data; all evaluation is
  admission-side (composition of the admitted set), not downstream.
- Mitigating evidence: the margin condition reduces contamination-in-admitted-
  data in all 8 of 8 grid configurations, so the direction of the effect does
  not depend on the post-hoc selection; the specific magnitudes do.

## Full grid (contamination in admitted data; source: outputs/p2_margin_rule.json)

| Pool | k | eta | Consensus only | + margin | Retention (consensus -> +margin) |
|---|---|---|---|---|---|
| Shared | 2 | 10% | 8.83% | 7.61% | 49.9% -> 37.4% |
| Shared | 3 | 10% | 8.33% | 7.40% | 40.8% -> 32.0% |
| Shared | 2 | 20% | 18.92% | 16.07% | 49.5% -> 36.3% |
| Shared | 3 | 20% | 18.08% | 15.87% | 40.2% -> 31.1% |
| Cross-backbone | 2 | 10% | 8.72% | 7.21% | 79.1% -> 58.3% |
| Cross-backbone | 3 | 10% | 8.26% | 7.00% | 56.1% -> 45.2% |
| Cross-backbone | 2 | 20% | 18.50% | 15.11% | 78.4% -> 56.5% |
| Cross-backbone | 3 | 20% | 18.26% | 15.19% | 55.3% -> 43.7% |

Shared-pool rows: full 10,000-pair set. Cross-backbone rows: 3,000-pair
subsample (seed 42). Single filtering pass; quantities are deterministic given
the cached verifier scores.
