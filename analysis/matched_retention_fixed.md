# Matched-retention comparison at exactly matched retention

## Why this document supersedes part of analysis/matched_retention.md

The earlier matched-retention table selected, for each rule and each target retention, the frontier point whose retention was closest to the target. The two verifier pools do not have the same achievable retention values, so a nearest-value match does not produce equal retention. Of the 12 target and eta combinations in that table, 10 paired the two pools at retentions differing by more than 0.01 in absolute value. The largest discrepancy occurs at eta = 20% and target retention 0.60, where the shared-backbone pool retained 0.5765 of the data while the cross-backbone pool retained 0.6622, a gap of +0.0857.

This matters because admitted contamination is a property of the retained subset and increases with retention along each pool's own frontier. Comparing the admitted contamination of two rules that retain different fractions of the data reintroduces precisely the confound that a matched comparison exists to remove, so those rows cannot support a claim about either pool. The conclusion printed by the earlier analysis, that diversity improves reliability at matched retention, was therefore not established by the evidence presented there.

This document replaces the nearest-point matching with interpolation. Each pool's Pareto frontier is linearly interpolated to exactly the target retention, so both pools are read at identical retention by construction. Every quantity reported below is computed at matched retention in the literal sense. The remainder of analysis/matched_retention.md, in particular the frontier construction and the metric definitions, is unchanged and still applies.

## Method

For each pool the retention and contamination frontier is built on the evaluation split over all combinations of the consensus count k and the threshold t, and is then restricted to the Pareto frontier on higher retention and lower admitted contamination. Admitted contamination is strictly increasing in retention along that frontier, because a point with higher retention and no higher contamination would dominate. The frontier is a monotone curve and linear interpolation along it is well defined in both directions.

For each target retention on the common grid, each pool's frontier is interpolated to exactly that retention, giving the admitted contamination, true harmful survival, detection recall and clean false-rejection rate that the pool attains while retaining the same fraction of the data as the other pool. A target retention outside a pool's achievable range is excluded rather than extrapolated.

Confidence intervals are percentile intervals from a paired bootstrap of 2000 resamples of pairs with seed 42. Within a replicate both pools are resampled with the identical row indices, so the difference intervals are paired. Each pool's frontier is rebuilt from the resampled rows and re-interpolated inside every replicate, so the interval covers the whole estimation procedure, including the variability of the frontier itself, rather than treating the frontier as fixed. The share of replicates in which the target lies inside the resampled frontier's range is reported alongside each interval; where that share is below one hundred percent the interval is conditional on the target being attainable.

The comparison is restricted to the rows covered by the cross-backbone subsample, so both pools are scored on identical data, and the corrupted file at the matching eta supplies the is_clean label and the recorded preference direction, joined by id.

Two properties of the frontier bound what the interpolation can claim, and both are reported per eta below. First, every pool's frontier spans the whole retention interval from zero to one, because a threshold below the smallest observed score admits every pair and a threshold above the largest admits none. No target on the grid is therefore formally out of range, and the exclusion rule never fires. The substantive constraint is instead the width of the gap between the two achievable frontier points that bracket a target: where that gap is wide, the interpolated value is not the value of any single achievable rule. Second, an interpolated point is realised exactly by a randomized mixture of the two bracketing rules. For true harmful survival, detection recall and the clean false-rejection rate the denominator is a property of the data rather than of the rule, so the linear interpolation is exactly the mixture value. Admitted contamination has a rule-dependent denominator, so its mixture value is a retention-weighted average rather than the linear chord; the discrepancy between the two is computed for every grid point and reported, and it is smaller than the reported interval widths by orders of magnitude.

The vectorised frontier construction used inside the bootstrap is asserted, before any resampling, to reproduce the reference implementation in analysis/scripts/calibration_analysis.py exactly on the point-estimate data for every pool and every eta.

## eta = 10%

The cross-backbone subsample contains 3000 pairs, of which 1513 fall in the evaluation split and 136 of those carry an injected preference flip. The realized corruption rate over the joined data is 0.0986. The best single verifier selected on the calibration split is policy with an AUROC of 0.5520.

Achievable retention ranges on the evaluation-split Pareto frontier: Shared-backbone pool (k of 4) [0.0000, 1.0000]; Cross-backbone pool (k of 4) [0.0000, 1.0000]; Best single verifier [0.0000, 1.0000].

No target retention on the grid falls outside the achievable range of either pool, so no target is excluded at this eta.

### Admitted contamination at matched retention, eta = 10%

| Target retention | Shared-backbone pool (k of 4) | Cross-backbone pool (k of 4) | Best single verifier | Paired difference (cross-backbone minus shared) | Interval excludes zero | Replicates in range |
|---|---|---|---|---|---|---|
| 0.30 | 0.0782 [0.0383, 0.0920] | 0.0689 [0.0445, 0.0851] | 0.0579 [0.0242, 0.0751] | -0.0093 [-0.0243, 0.0281] | no | 100.0% |
| 0.40 | 0.0786 [0.0500, 0.0940] | 0.0759 [0.0541, 0.0891] | 0.0772 [0.0323, 0.0935] | -0.0027 [-0.0177, 0.0178] | no | 100.0% |
| 0.50 | 0.0813 [0.0587, 0.0966] | 0.0764 [0.0574, 0.0907] | 0.0871 [0.0403, 0.0984] | -0.0049 [-0.0171, 0.0075] | no | 100.0% |
| 0.60 | 0.0878 [0.0661, 0.1002] | 0.0769 [0.0599, 0.0918] | 0.0876 [0.0484, 0.0994] | -0.0109 [-0.0180, 0.0014] | no | 100.0% |
| 0.70 | 0.0888 [0.0702, 0.1018] | 0.0787 [0.0632, 0.0940] | 0.0882 [0.0564, 0.0998] | -0.0101 [-0.0179, 0.0003] | no | 100.0% |
| 0.80 | 0.0891 [0.0735, 0.1026] | 0.0858 [0.0680, 0.0976] | 0.0888 [0.0640, 0.1010] | -0.0033 [-0.0126, 0.0007] | no | 100.0% |

### True harmful survival at matched retention, eta = 10%

| Target retention | Shared-backbone pool (k of 4) | Cross-backbone pool (k of 4) | Best single verifier | Paired difference (cross-backbone minus shared) | Interval excludes zero | Replicates in range |
|---|---|---|---|---|---|---|
| 0.30 | 0.2611 [0.1897, 0.2964] | 0.2300 [0.1621, 0.2684] | 0.2896 [0.2267, 0.3000] | -0.0311 [-0.0801, 0.0242] | no | 100.0% |
| 0.40 | 0.3501 [0.2760, 0.3975] | 0.3390 [0.2689, 0.3719] | 0.3861 [0.3081, 0.4000] | -0.0111 [-0.0696, 0.0324] | no | 100.0% |
| 0.50 | 0.4539 [0.3771, 0.4986] | 0.4267 [0.3562, 0.4727] | 0.4858 [0.4110, 0.5000] | -0.0271 [-0.0823, 0.0265] | no | 100.0% |
| 0.60 | 0.5863 [0.5034, 0.5994] | 0.5145 [0.4367, 0.5730] | 0.5887 [0.5288, 0.6000] | -0.0718 [-0.1155, -0.0039] | yes | 100.0% |
| 0.70 | 0.6925 [0.6431, 0.6997] | 0.6141 [0.5434, 0.6764] | 0.6915 [0.6466, 0.7000] | -0.0784 [-0.1364, -0.0125] | yes | 100.0% |
| 0.80 | 0.7950 [0.7669, 0.7998] | 0.7640 [0.6749, 0.7877] | 0.7943 [0.7644, 0.8000] | -0.0310 [-0.1123, -0.0056] | yes | 100.0% |

### Detection recall at matched retention, eta = 10%

| Target retention | Shared-backbone pool (k of 4) | Cross-backbone pool (k of 4) | Best single verifier | Paired difference (cross-backbone minus shared) | Interval excludes zero | Replicates in range |
|---|---|---|---|---|---|---|
| 0.30 | 0.7389 [0.7036, 0.8103] | 0.7700 [0.7316, 0.8379] | 0.7104 [0.7000, 0.7733] | 0.0311 [-0.0242, 0.0801] | no | 100.0% |
| 0.40 | 0.6499 [0.6025, 0.7240] | 0.6610 [0.6281, 0.7311] | 0.6139 [0.6000, 0.6919] | 0.0111 [-0.0324, 0.0696] | no | 100.0% |
| 0.50 | 0.5461 [0.5014, 0.6229] | 0.5733 [0.5273, 0.6438] | 0.5142 [0.5000, 0.5890] | 0.0271 [-0.0265, 0.0823] | no | 100.0% |
| 0.60 | 0.4137 [0.4006, 0.4966] | 0.4855 [0.4270, 0.5633] | 0.4113 [0.4000, 0.4712] | 0.0718 [0.0039, 0.1155] | yes | 100.0% |
| 0.70 | 0.3075 [0.3003, 0.3569] | 0.3859 [0.3236, 0.4566] | 0.3085 [0.3000, 0.3534] | 0.0784 [0.0125, 0.1364] | yes | 100.0% |
| 0.80 | 0.2050 [0.2002, 0.2331] | 0.2360 [0.2123, 0.3251] | 0.2057 [0.2000, 0.2356] | 0.0310 [0.0056, 0.1123] | yes | 100.0% |

### Clean false-rejection rate at matched retention, eta = 10%

| Target retention | Shared-backbone pool (k of 4) | Cross-backbone pool (k of 4) | Best single verifier | Paired difference (cross-backbone minus shared) | Interval excludes zero | Replicates in range |
|---|---|---|---|---|---|---|
| 0.30 | 0.6962 [0.6894, 0.6997] | 0.6931 [0.6861, 0.6969] | 0.6990 [0.6929, 0.7000] | -0.0031 [-0.0079, 0.0023] | no | 100.0% |
| 0.40 | 0.5951 [0.5879, 0.5998] | 0.5940 [0.5869, 0.5973] | 0.5986 [0.5910, 0.6000] | -0.0011 [-0.0068, 0.0033] | no | 100.0% |
| 0.50 | 0.4954 [0.4878, 0.4999] | 0.4928 [0.4859, 0.4973] | 0.4986 [0.4911, 0.5000] | -0.0027 [-0.0081, 0.0026] | no | 100.0% |
| 0.60 | 0.3986 [0.3902, 0.3999] | 0.3916 [0.3840, 0.3973] | 0.3989 [0.3929, 0.4000] | -0.0071 [-0.0115, -0.0004] | yes | 100.0% |
| 0.70 | 0.2993 [0.2945, 0.3000] | 0.2915 [0.2845, 0.2977] | 0.2992 [0.2947, 0.3000] | -0.0077 [-0.0137, -0.0012] | yes | 100.0% |
| 0.80 | 0.1995 [0.1968, 0.2000] | 0.1964 [0.1877, 0.1988] | 0.1994 [0.1965, 0.2000] | -0.0031 [-0.0111, -0.0005] | yes | 100.0% |

### Interpolation support, eta = 10%

For each target, the two achievable frontier points that bracket it and the width of that bracket. A wide bracket means the interpolated operating point is a randomized mixture of two rather different rules rather than a rule that can be written as a single threshold and consensus count. The final column gives the difference between the exact randomized-mixture value and the reported linear interpolation for the paired contamination difference; the largest such departure at this eta is 0.000208, which is negligible against the interval widths reported above.

| Target retention | Shared bracketing frontier retentions | Shared bracket width | Cross-backbone bracketing frontier retentions | Cross-backbone bracket width | Mixture minus linear discrepancy in the paired contamination difference |
|---|---|---|---|---|---|
| 0.30 | [0.2710, 0.4527] | 0.1818 | [0.2816, 0.3040] | 0.0225 | -0.00002 |
| 0.40 | [0.2710, 0.4527] | 0.1818 | [0.3153, 0.6590] | 0.3437 | +0.00021 |
| 0.50 | [0.4527, 0.5506] | 0.0978 | [0.3153, 0.6590] | 0.3437 | +0.00005 |
| 0.60 | [0.5684, 0.6015] | 0.0330 | [0.3153, 0.6590] | 0.3437 | +0.00014 |
| 0.70 | [0.6054, 1.0000] | 0.3946 | [0.6590, 0.7786] | 0.1196 | +0.00001 |
| 0.80 | [0.6054, 1.0000] | 0.3946 | [0.7872, 0.8678] | 0.0806 | -0.00016 |

### Operating-point mismatch in the superseded table, eta = 10%

The following table reproduces the operating points that the earlier nearest-point protocol selected and reports the retention each pool actually achieved on the evaluation split. Rows marked as not matched are the rows of analysis/matched_retention.md whose contamination comparison is confounded by a retention difference.

| Target retention | Shared operating point | Cross-backbone operating point | Retention achieved by shared | Retention achieved by cross-backbone | Retention gap | Matched within 0.01 |
|---|---|---|---|---|---|---|
| 0.30 | k=2, t=0.950 | k=4, t=0.600 | 0.2928 | 0.3371 | +0.0443 | not matched |
| 0.40 | k=3, t=0.700 | k=3, t=0.900 | 0.4164 | 0.4124 | -0.0040 | matched |
| 0.50 | k=1, t=0.850 | k=3, t=0.800 | 0.5169 | 0.5763 | +0.0595 | not matched |
| 0.60 | k=3, t=0.100 | k=3, t=0.700 | 0.6054 | 0.6061 | +0.0007 | matched |
| 0.70 | k=2, t=0.100 | k=3, t=0.400 | 0.7112 | 0.6834 | -0.0278 | not matched |
| 0.80 | k=2, t=0.100 | k=1, t=0.950 | 0.7112 | 0.7786 | +0.0674 | not matched |

### Retention at matched admitted contamination, eta = 10%

The same interpolation applied in the other direction, for completeness. The contamination grid is derived from the realized corruption rate at this eta at fractions of 0.6, 0.7, 0.8, 0.9 and 1.0, matching the grid used in analysis/matched_reliability.md. This direction is far worse conditioned than the first. Admitted contamination varies over a narrow range while retention varies over the whole unit interval, so long stretches of the frontier are nearly flat in contamination and a small resampling perturbation moves the retention attained at a fixed contamination a long way. The intervals below are correspondingly wide, with a median width of 0.6668 in retention, and that width is a property of the data rather than an artefact of the pairing. No conclusion about coverage should be drawn from them beyond the observation that they do not separate the two pools.

| Target admitted contamination | Shared-backbone pool (k of 4) | Cross-backbone pool (k of 4) | Best single verifier | Paired difference in retention (cross-backbone minus shared) | Interval excludes zero | Replicates in range |
|---|---|---|---|---|---|---|
| 0.0592 | 0.1063 [0.0952, 0.5062] | 0.1715 [0.1027, 0.5794] | 0.3067 [0.2365, 0.7342] | 0.0652 [-0.2484, 0.2560] | no | 100.0% |
| 0.0690 | 0.1239 [0.1109, 0.6680] | 0.3007 [0.1350, 0.8122] | 0.3574 [0.2756, 0.8628] | 0.1769 [-0.2468, 0.4374] | no | 99.8% |
| 0.0789 | 0.4540 [0.1296, 0.8588] | 0.7058 [0.2222, 0.9880] | 0.4087 [0.3131, 0.9627] | 0.2518 [-0.1986, 0.5198] | no | 93.6% |
| 0.0887 | 0.6790 [0.1871, 0.9352] | 0.9770 [0.3148, 0.9953] | 0.7906 [0.3419, 0.9868] | 0.2980 [-0.1244, 0.5250] | no | 54.1% |

Contamination targets excluded as out of range: 0.0986.

## eta = 20%

The cross-backbone subsample contains 3000 pairs, of which 1424 fall in the evaluation split and 288 of those carry an injected preference flip. The realized corruption rate over the joined data is 0.2031. The best single verifier selected on the calibration split is helpfulness with an AUROC of 0.5209.

Achievable retention ranges on the evaluation-split Pareto frontier: Shared-backbone pool (k of 4) [0.0000, 1.0000]; Cross-backbone pool (k of 4) [0.0000, 1.0000]; Best single verifier [0.0000, 1.0000].

No target retention on the grid falls outside the achievable range of either pool, so no target is excluded at this eta.

### Admitted contamination at matched retention, eta = 20%

| Target retention | Shared-backbone pool (k of 4) | Cross-backbone pool (k of 4) | Best single verifier | Paired difference (cross-backbone minus shared) | Interval excludes zero | Replicates in range |
|---|---|---|---|---|---|---|
| 0.30 | 0.1815 [0.1162, 0.2036] | 0.1754 [0.1261, 0.1975] | 0.1785 [0.1190, 0.2036] | -0.0061 [-0.0338, 0.0533] | no | 100.0% |
| 0.40 | 0.1852 [0.1469, 0.2079] | 0.1825 [0.1396, 0.2010] | 0.1818 [0.1479, 0.2092] | -0.0027 [-0.0286, 0.0191] | no | 100.0% |
| 0.50 | 0.1887 [0.1590, 0.2126] | 0.1844 [0.1497, 0.2029] | 0.1852 [0.1593, 0.2114] | -0.0043 [-0.0273, 0.0093] | no | 100.0% |
| 0.60 | 0.1970 [0.1706, 0.2171] | 0.1863 [0.1564, 0.2047] | 0.1886 [0.1674, 0.2171] | -0.0107 [-0.0274, -0.0008] | yes | 100.0% |
| 0.70 | 0.2006 [0.1781, 0.2210] | 0.1879 [0.1617, 0.2072] | 0.1920 [0.1722, 0.2195] | -0.0126 [-0.0265, -0.0040] | yes | 100.0% |
| 0.80 | 0.2011 [0.1803, 0.2213] | 0.1913 [0.1670, 0.2110] | 0.1954 [0.1750, 0.2203] | -0.0098 [-0.0214, -0.0025] | yes | 100.0% |

### True harmful survival at matched retention, eta = 20%

| Target retention | Shared-backbone pool (k of 4) | Cross-backbone pool (k of 4) | Best single verifier | Paired difference (cross-backbone minus shared) | Interval excludes zero | Replicates in range |
|---|---|---|---|---|---|---|
| 0.30 | 0.2694 [0.2222, 0.2941] | 0.2634 [0.2191, 0.2827] | 0.2656 [0.2233, 0.2969] | -0.0061 [-0.0338, 0.0231] | no | 100.0% |
| 0.40 | 0.3667 [0.3164, 0.3957] | 0.3612 [0.3093, 0.3802] | 0.3598 [0.3125, 0.3982] | -0.0055 [-0.0400, 0.0253] | no | 100.0% |
| 0.50 | 0.4666 [0.4141, 0.4982] | 0.4575 [0.4058, 0.4783] | 0.4656 [0.4331, 0.4988] | -0.0091 [-0.0517, 0.0207] | no | 100.0% |
| 0.60 | 0.5849 [0.5374, 0.5995] | 0.5537 [0.5000, 0.5772] | 0.5725 [0.5497, 0.5994] | -0.0312 [-0.0738, -0.0027] | yes | 100.0% |
| 0.70 | 0.6948 [0.6703, 0.6998] | 0.6506 [0.5991, 0.6785] | 0.6794 [0.6649, 0.6997] | -0.0442 [-0.0892, -0.0154] | yes | 100.0% |
| 0.80 | 0.7965 [0.7842, 0.7999] | 0.7568 [0.7087, 0.7836] | 0.7862 [0.7766, 0.7999] | -0.0398 [-0.0841, -0.0126] | yes | 100.0% |

### Detection recall at matched retention, eta = 20%

| Target retention | Shared-backbone pool (k of 4) | Cross-backbone pool (k of 4) | Best single verifier | Paired difference (cross-backbone minus shared) | Interval excludes zero | Replicates in range |
|---|---|---|---|---|---|---|
| 0.30 | 0.7306 [0.7059, 0.7778] | 0.7366 [0.7173, 0.7809] | 0.7344 [0.7031, 0.7767] | 0.0061 [-0.0231, 0.0338] | no | 100.0% |
| 0.40 | 0.6333 [0.6043, 0.6836] | 0.6388 [0.6198, 0.6907] | 0.6402 [0.6018, 0.6875] | 0.0055 [-0.0253, 0.0400] | no | 100.0% |
| 0.50 | 0.5334 [0.5018, 0.5859] | 0.5425 [0.5217, 0.5942] | 0.5344 [0.5012, 0.5669] | 0.0091 [-0.0207, 0.0517] | no | 100.0% |
| 0.60 | 0.4151 [0.4005, 0.4626] | 0.4463 [0.4228, 0.5000] | 0.4275 [0.4006, 0.4503] | 0.0312 [0.0027, 0.0738] | yes | 100.0% |
| 0.70 | 0.3052 [0.3002, 0.3297] | 0.3494 [0.3215, 0.4009] | 0.3206 [0.3003, 0.3351] | 0.0442 [0.0154, 0.0892] | yes | 100.0% |
| 0.80 | 0.2035 [0.2001, 0.2158] | 0.2432 [0.2164, 0.2913] | 0.2138 [0.2001, 0.2234] | 0.0398 [0.0126, 0.0841] | yes | 100.0% |

### Clean false-rejection rate at matched retention, eta = 20%

| Target retention | Shared-backbone pool (k of 4) | Cross-backbone pool (k of 4) | Best single verifier | Paired difference (cross-backbone minus shared) | Interval excludes zero | Replicates in range |
|---|---|---|---|---|---|---|
| 0.30 | 0.6923 [0.6804, 0.6986] | 0.6907 [0.6795, 0.6957] | 0.6913 [0.6808, 0.6992] | -0.0015 [-0.0086, 0.0060] | no | 100.0% |
| 0.40 | 0.5916 [0.5787, 0.5989] | 0.5902 [0.5772, 0.5950] | 0.5898 [0.5779, 0.5996] | -0.0014 [-0.0101, 0.0064] | no | 100.0% |
| 0.50 | 0.4915 [0.4785, 0.4995] | 0.4892 [0.4762, 0.4945] | 0.4913 [0.4829, 0.4997] | -0.0023 [-0.0132, 0.0052] | no | 100.0% |
| 0.60 | 0.3962 [0.3842, 0.3999] | 0.3883 [0.3744, 0.3942] | 0.3930 [0.3871, 0.3998] | -0.0079 [-0.0192, -0.0007] | yes | 100.0% |
| 0.70 | 0.2987 [0.2924, 0.3000] | 0.2875 [0.2745, 0.2946] | 0.2948 [0.2910, 0.2999] | -0.0112 [-0.0227, -0.0040] | yes | 100.0% |
| 0.80 | 0.1991 [0.1959, 0.2000] | 0.1890 [0.1765, 0.1960] | 0.1965 [0.1940, 0.2000] | -0.0101 [-0.0216, -0.0032] | yes | 100.0% |

### Interpolation support, eta = 20%

For each target, the two achievable frontier points that bracket it and the width of that bracket. A wide bracket means the interpolated operating point is a randomized mixture of two rather different rules rather than a rule that can be written as a single threshold and consensus count. The final column gives the difference between the exact randomized-mixture value and the reported linear interpolation for the paired contamination difference; the largest such departure at this eta is 0.002040, which is negligible against the interval widths reported above.

| Target retention | Shared bracketing frontier retentions | Shared bracket width | Cross-backbone bracketing frontier retentions | Cross-backbone bracket width | Mixture minus linear discrepancy in the paired contamination difference |
|---|---|---|---|---|---|
| 0.30 | [0.2907, 0.4192] | 0.1285 | [0.1721, 0.3813] | 0.2093 | +0.00204 |
| 0.40 | [0.2907, 0.4192] | 0.1285 | [0.3890, 0.6489] | 0.2598 | -0.00007 |
| 0.50 | [0.4192, 0.5056] | 0.0864 | [0.3890, 0.6489] | 0.2598 | +0.00059 |
| 0.60 | [0.5794, 0.6348] | 0.0555 | [0.3890, 0.6489] | 0.2598 | +0.00022 |
| 0.70 | [0.6348, 1.0000] | 0.3652 | [0.6622, 0.7570] | 0.0948 | -0.00014 |
| 0.80 | [0.6348, 1.0000] | 0.3652 | [0.7654, 0.8315] | 0.0660 | -0.00021 |

### Operating-point mismatch in the superseded table, eta = 20%

The following table reproduces the operating points that the earlier nearest-point protocol selected and reports the retention each pool actually achieved on the evaluation split. Rows marked as not matched are the rows of analysis/matched_retention.md whose contamination comparison is confounded by a retention difference.

| Target retention | Shared operating point | Cross-backbone operating point | Retention achieved by shared | Retention achieved by cross-backbone | Retention gap | Matched within 0.01 |
|---|---|---|---|---|---|---|
| 0.30 | k=2, t=0.900 | k=3, t=1.000 | 0.3272 | 0.2879 | -0.0393 | not matched |
| 0.40 | k=2, t=0.900 | k=3, t=0.900 | 0.3272 | 0.3813 | +0.0541 | not matched |
| 0.50 | k=2, t=0.800 | k=2, t=1.000 | 0.4670 | 0.5070 | +0.0400 | not matched |
| 0.60 | k=3, t=0.250 | k=3, t=0.400 | 0.5765 | 0.6622 | +0.0857 | not matched |
| 0.70 | k=2, t=0.100 | k=3, t=0.400 | 0.6861 | 0.6622 | -0.0239 | not matched |
| 0.80 | k=1, t=0.100 | k=2, t=0.700 | 0.7570 | 0.7900 | +0.0330 | not matched |

### Retention at matched admitted contamination, eta = 20%

The same interpolation applied in the other direction, for completeness. The contamination grid is derived from the realized corruption rate at this eta at fractions of 0.6, 0.7, 0.8, 0.9 and 1.0, matching the grid used in analysis/matched_reliability.md. This direction is far worse conditioned than the first. Admitted contamination varies over a narrow range while retention varies over the whole unit interval, so long stretches of the frontier are nearly flat in contamination and a small resampling perturbation moves the retention attained at a fixed contamination a long way. The intervals below are correspondingly wide, with a median width of 0.5118 in retention, and that width is a property of the data rather than an artefact of the pairing. No conclusion about coverage should be drawn from them beyond the observation that they do not separate the two pools.

| Target admitted contamination | Shared-backbone pool (k of 4) | Cross-backbone pool (k of 4) | Best single verifier | Paired difference in retention (cross-backbone minus shared) | Interval excludes zero | Replicates in range |
|---|---|---|---|---|---|---|
| 0.1219 | 0.0944 [0.0746, 0.3148] | 0.0950 [0.0709, 0.2879] | 0.1315 [0.1123, 0.3078] | 0.0007 [-0.2107, 0.1215] | no | 100.0% |
| 0.1422 | 0.1101 [0.0870, 0.3822] | 0.1170 [0.0827, 0.4139] | 0.1534 [0.1310, 0.3681] | 0.0069 [-0.2318, 0.1862] | no | 100.0% |
| 0.1625 | 0.1597 [0.0995, 0.5269] | 0.1656 [0.0947, 0.7260] | 0.1753 [0.1497, 0.5301] | 0.0059 [-0.2489, 0.3568] | no | 100.0% |
| 0.1828 | 0.3349 [0.1184, 0.7280] | 0.4149 [0.1261, 0.9850] | 0.4284 [0.1684, 0.9042] | 0.0800 [-0.1998, 0.5425] | no | 96.7% |

Contamination targets excluded as out of range: 0.2031.

## Conclusion

Selected conclusion: (a) diversity improves reliability at matched retention.

Evidence for reliability at matched retention. The comparison covers 12 retention grid points, being 6 targets at each of 2 corruption levels, with 0 excluded as out of range. Of these, 3 show a paired difference in admitted contamination whose interval lies entirely below zero, which would favour the cross-backbone pool, and 0 show an interval entirely above zero, which would favour the shared pool. Counting point estimates alone, 12 of 12 favour the cross-backbone pool and 0 favour the shared pool.

The size of the effect is small in absolute terms. Across the retained grid points the interpolated difference in admitted contamination ranges from 0.0027 to 0.0126 in absolute value, that is from roughly 0.27 to 1.26 percentage points of the admitted set, corresponding to a relative reduction of at most 12.4 percent of the contamination the shared-backbone pool admits at the same retention. Where an interval does exclude zero it does so narrowly: the smallest distance between an excluding interval's nearer bound and zero is 0.0008.

All 12 differences share the same sign, and 3 of them are individually significant at the 95 percent level. The consistency of sign is therefore doing more of the work than any single interval.

A two-sided exact sign test over the 12 non-zero grid points gives p = 0.0005, which is significant at the 5 percent level. That test treats the grid points as independent observations, which they are not. The points are read off the same two frontiers estimated on the same evaluation rows, and the grid is fine enough that adjacent targets often fall between the same pair of achievable operating points, so neighbouring differences are strongly dependent and the sign test overstates the evidence by an amount this design cannot quantify. It is reported as a description of the pattern rather than as a formal test, and the selected conclusion does not rest on it.

Evidence for coverage at matched reliability. Across the 8 contamination grid points, 0 show a paired difference in retention whose interval lies entirely above zero, favouring the cross-backbone pool, and 0 show an interval entirely below zero. Counting point estimates alone, 8 favour the cross-backbone pool and 0 favour the shared pool, with a two-sided exact sign test giving p = 0.0078.

The median width of the paired difference intervals reported in this document is 0.0501. Grid points excluded because a target lay outside a pool's achievable range: 0 of the 12 retention targets, and 2 of the 10 contamination targets in the secondary comparison. The first count is structural rather than fortunate: each pool's frontier runs from retention zero to retention one, so no target on the grid can fall outside it, and the exclusion rule is retained only as a guard. The binding limitation is the width of the bracket between achievable frontier points, tabulated per eta above.

Selection rule, fixed in advance and identical to the rule used in the superseded analysis: a conclusion of improvement requires that the paired bootstrap difference interval exclude zero in the favourable direction at more grid points than in the unfavourable direction. No claim is made from point estimates alone.

## Verdict

At genuinely matched retention, enforced by interpolating each pool's frontier to exactly the same retention rather than to the nearest achievable point, the cross-backbone pool admits less contamination than the shared-backbone pool at all 12 of 12 grid points, with paired bootstrap intervals excluding zero at 3 of them and none favouring the shared pool. What it does not buy is a large effect or a better coverage frontier: the reduction is at most 1.3 percentage points of admitted contamination, at most 12 percent in relative terms, 9 of 12 grid points remain consistent with no difference, no interval for retention at matched contamination excludes zero at any of the 8 reliability targets, and the much larger advantages implied by the earlier table were artefacts of pairing operating points that retained different fractions of the data, by as much as 0.0857 in retention.

