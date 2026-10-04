# Matched-reliability comparison

This file is the mirror image of the matched-retention comparison. Instead of fixing coverage and asking which pool is more reliable, it fixes a reliability quantity and asks which pool retains more data. Three reliability quantities are matched in turn: the contamination of the admitted set, the true harmful survival rate, and the clean false-rejection rate.

Matching protocol. For each rule the operating point is the highest-retention point of the calibration-split frontier that keeps the matched quantity at or below the target, which is the operational reading of the comparison: fix a reliability budget and ask how much data each rule can retain within it. When no frontier point meets the budget, the point with the smallest achievable value is used instead, and the achieved value column makes the shortfall visible. Selection is restricted to the frontier because the matched quantities are not monotone over the unrestricted set of threshold and k combinations, so an unrestricted nearest-value match can pair operating points that are not comparable.

The clean false-rejection block is matched differently and is read differently. A budget on the clean false-rejection rate is satisfied trivially by admitting every pair, which also maximises retention, so a budget match on that quantity carries no information. That block therefore uses a nearest-value match on the frontier, and the question it answers is the reverse one: at an equal cost paid on clean data, which rule admits less contamination. It is reported for completeness and is not counted toward the conclusion, which is defined over matched retention and matched reliability budgets only.

Confidence intervals follow `matched_retention.md`. Operating points are selected on the calibration split, all reported numbers come from the disjoint evaluation split, and paired differences reuse identical bootstrap resample indices across the two pools.

## eta = 10%

| Matched quantity | Target | Rule | Operating point | Achieved matched quantity | Retention | Admitted contamination | Detection F1 |
|---|---|---|---|---|---|---|---|
| admitted contamination | 0.0592 | Shared-backbone pool (k of 4) | k=4, t=0.950 | 0.0769 [0.0443, 0.1154] | 0.1461 [0.1295, 0.1646] | 0.0769 [0.0443, 0.1154] | 0.1667 [0.1416, 0.1916] |
| admitted contamination | 0.0592 | Cross-backbone pool (k of 4) | k=1, t=1.000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.1649 [0.1424, 0.1880] |
| admitted contamination | 0.0592 | Best single verifier | k=1, t=1.000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.1649 [0.1424, 0.1880] |
| admitted contamination | 0.0592 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | -0.0769 [-0.1154, -0.0443] | -0.1461 [-0.1646, -0.1295] | -0.0769 [-0.1154, -0.0443] |  |
| admitted contamination | 0.0690 | Shared-backbone pool (k of 4) | k=2, t=0.900 | 0.0838 [0.0602, 0.1091] | 0.3629 [0.3410, 0.3860] | 0.0838 [0.0602, 0.1091] | 0.1636 [0.1356, 0.1930] |
| admitted contamination | 0.0690 | Cross-backbone pool (k of 4) | k=1, t=1.000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.1649 [0.1424, 0.1880] |
| admitted contamination | 0.0690 | Best single verifier | k=1, t=1.000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.1649 [0.1424, 0.1880] |
| admitted contamination | 0.0690 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | -0.0838 [-0.1091, -0.0602] | -0.3629 [-0.3860, -0.3410] | -0.0838 [-0.1091, -0.0602] |  |
| admitted contamination | 0.0789 | Shared-backbone pool (k of 4) | k=1, t=0.900 | 0.0890 [0.0692, 0.1102] | 0.4825 [0.4600, 0.5069] | 0.0890 [0.0692, 0.1102] | 0.1545 [0.1232, 0.1871] |
| admitted contamination | 0.0789 | Cross-backbone pool (k of 4) | k=4, t=0.100 | 0.0871 [0.0660, 0.1091] | 0.4402 [0.4184, 0.4640] | 0.0871 [0.0660, 0.1091] | 0.1587 [0.1300, 0.1887] |
| admitted contamination | 0.0789 | Best single verifier | k=1, t=0.500 | 0.0868 [0.0657, 0.1081] | 0.4494 [0.4270, 0.4732] | 0.0868 [0.0657, 0.1081] | 0.1589 [0.1303, 0.1898] |
| admitted contamination | 0.0789 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | -0.0020 [-0.0136, 0.0089] | -0.0423 [-0.0595, -0.0238] | -0.0020 [-0.0136, 0.0089] |  |
| admitted contamination | 0.0887 | Shared-backbone pool (k of 4) | k=3, t=0.100 | 0.0884 [0.0707, 0.1068] | 0.6054 [0.5836, 0.6292] | 0.0884 [0.0707, 0.1068] | 0.1501 [0.1144, 0.1825] |
| admitted contamination | 0.0887 | Cross-backbone pool (k of 4) | k=2, t=0.700 | 0.0862 [0.0705, 0.1020] | 0.7971 [0.7773, 0.8169] | 0.0862 [0.0705, 0.1020] | 0.1445 [0.1002, 0.1831] |
| admitted contamination | 0.0887 | Best single verifier | k=1, t=0.500 | 0.0868 [0.0657, 0.1081] | 0.4494 [0.4270, 0.4732] | 0.0868 [0.0657, 0.1081] | 0.1589 [0.1303, 0.1898] |
| admitted contamination | 0.0887 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | -0.0022 [-0.0111, 0.0070] | 0.1917 [0.1692, 0.2142] | -0.0022 [-0.0111, 0.0070] |  |
| admitted contamination | 0.0986 | Shared-backbone pool (k of 4) | k=1, t=0.000 | 0.0899 [0.0767, 0.1038] | 1.0000 [1.0000, 1.0000] | 0.0899 [0.0767, 0.1038] | 0.0000 [0.0000, 0.0000] |
| admitted contamination | 0.0986 | Cross-backbone pool (k of 4) | k=1, t=0.000 | 0.0899 [0.0767, 0.1038] | 1.0000 [1.0000, 1.0000] | 0.0899 [0.0767, 0.1038] | 0.0000 [0.0000, 0.0000] |
| admitted contamination | 0.0986 | Best single verifier | k=1, t=0.000 | 0.0899 [0.0767, 0.1038] | 1.0000 [1.0000, 1.0000] | 0.0899 [0.0767, 0.1038] | 0.0000 [0.0000, 0.0000] |
| admitted contamination | 0.0986 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |  |
| true harmful survival | 0.2000 | Shared-backbone pool (k of 4) | k=2, t=0.950 | 0.2574 [0.1875, 0.3309] | 0.2928 [0.2723, 0.3153] | 0.0790 [0.0544, 0.1057] | 0.1675 [0.1412, 0.1947] |
| true harmful survival | 0.2000 | Cross-backbone pool (k of 4) | k=4, t=0.850 | 0.1250 [0.0720, 0.1825] | 0.1718 [0.1540, 0.1890] | 0.0654 [0.0372, 0.0956] | 0.1713 [0.1465, 0.1984] |
| true harmful survival | 0.2000 | Best single verifier | k=1, t=1.000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.1649 [0.1424, 0.1880] |
| true harmful survival | 0.2000 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | -0.1324 [-0.2092, -0.0526] | -0.1210 [-0.1428, -0.1011] | -0.0136 [-0.0417, 0.0160] |  |
| true harmful survival | 0.3000 | Shared-backbone pool (k of 4) | k=3, t=0.700 | 0.3897 [0.3099, 0.4734] | 0.4164 [0.3933, 0.4409] | 0.0841 [0.0632, 0.1067] | 0.1629 [0.1340, 0.1934] |
| true harmful survival | 0.3000 | Cross-backbone pool (k of 4) | k=4, t=0.850 | 0.1250 [0.0720, 0.1825] | 0.1718 [0.1540, 0.1890] | 0.0654 [0.0372, 0.0956] | 0.1713 [0.1465, 0.1984] |
| true harmful survival | 0.3000 | Best single verifier | k=1, t=1.000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.1649 [0.1424, 0.1880] |
| true harmful survival | 0.3000 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | -0.2647 [-0.3440, -0.1948] | -0.2445 [-0.2664, -0.2221] | -0.0187 [-0.0441, 0.0069] |  |
| true harmful survival | 0.4000 | Shared-backbone pool (k of 4) | k=1, t=0.900 | 0.4779 [0.3951, 0.5591] | 0.4825 [0.4600, 0.5069] | 0.0890 [0.0692, 0.1102] | 0.1545 [0.1232, 0.1871] |
| true harmful survival | 0.4000 | Cross-backbone pool (k of 4) | k=4, t=0.100 | 0.4265 [0.3451, 0.5069] | 0.4402 [0.4184, 0.4640] | 0.0871 [0.0660, 0.1091] | 0.1587 [0.1300, 0.1887] |
| true harmful survival | 0.4000 | Best single verifier | k=1, t=0.500 | 0.4338 [0.3493, 0.5154] | 0.4494 [0.4270, 0.4732] | 0.0868 [0.0657, 0.1081] | 0.1589 [0.1303, 0.1898] |
| true harmful survival | 0.4000 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | -0.0515 [-0.1149, 0.0068] | -0.0423 [-0.0595, -0.0238] | -0.0020 [-0.0136, 0.0089] |  |
| true harmful survival | 0.5000 | Shared-backbone pool (k of 4) | k=2, t=0.650 | 0.5221 [0.4415, 0.6016] | 0.5426 [0.5188, 0.5677] | 0.0865 [0.0682, 0.1053] | 0.1570 [0.1256, 0.1892] |
| true harmful survival | 0.5000 | Cross-backbone pool (k of 4) | k=3, t=0.700 | 0.5441 [0.4621, 0.6286] | 0.5763 [0.5525, 0.5995] | 0.0849 [0.0674, 0.1041] | 0.1596 [0.1263, 0.1933] |
| true harmful survival | 0.5000 | Best single verifier | k=1, t=0.500 | 0.4338 [0.3493, 0.5154] | 0.4494 [0.4270, 0.4732] | 0.0868 [0.0657, 0.1081] | 0.1589 [0.1303, 0.1898] |
| true harmful survival | 0.5000 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | 0.0221 [-0.0515, 0.1029] | 0.0337 [0.0093, 0.0562] | -0.0016 [-0.0126, 0.0104] |  |
| true harmful survival | 0.6000 | Shared-backbone pool (k of 4) | k=2, t=0.400 | 0.6176 [0.5373, 0.7000] | 0.6120 [0.5882, 0.6358] | 0.0907 [0.0731, 0.1097] | 0.1438 [0.1089, 0.1777] |
| true harmful survival | 0.6000 | Cross-backbone pool (k of 4) | k=3, t=0.600 | 0.5735 [0.4931, 0.6587] | 0.6252 [0.6028, 0.6477] | 0.0825 [0.0655, 0.1007] | 0.1650 [0.1306, 0.2011] |
| true harmful survival | 0.6000 | Best single verifier | k=1, t=0.500 | 0.4338 [0.3493, 0.5154] | 0.4494 [0.4270, 0.4732] | 0.0868 [0.0657, 0.1081] | 0.1589 [0.1303, 0.1898] |
| true harmful survival | 0.6000 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | -0.0441 [-0.1222, 0.0331] | 0.0132 [-0.0126, 0.0357] | -0.0083 [-0.0190, 0.0022] |  |
| clean false-rejection rate | 0.1000 | Shared-backbone pool (k of 4) | k=1, t=0.000 | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 0.0899 [0.0767, 0.1038] | 0.0000 [0.0000, 0.0000] |
| clean false-rejection rate | 0.1000 | Cross-backbone pool (k of 4) | k=1, t=0.600 | 0.0588 [0.0466, 0.0722] | 0.9425 [0.9299, 0.9537] | 0.0912 [0.0769, 0.1056] | 0.0538 [0.0174, 0.0987] |
| clean false-rejection rate | 0.1000 | Best single verifier | k=1, t=0.000 | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 0.0899 [0.0767, 0.1038] | 0.0000 [0.0000, 0.0000] |
| clean false-rejection rate | 0.1000 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | 0.0588 [0.0466, 0.0722] | -0.0575 [-0.0701, -0.0463] | 0.0013 [-0.0022, 0.0042] |  |
| clean false-rejection rate | 0.2000 | Shared-backbone pool (k of 4) | k=2, t=0.100 | 0.2919 [0.2688, 0.3172] | 0.7112 [0.6880, 0.7343] | 0.0939 [0.0773, 0.1106] | 0.1222 [0.0865, 0.1574] |
| clean false-rejection rate | 0.2000 | Cross-backbone pool (k of 4) | k=2, t=0.700 | 0.1997 [0.1790, 0.2194] | 0.7971 [0.7773, 0.8169] | 0.0862 [0.0705, 0.1020] | 0.1445 [0.1002, 0.1831] |
| clean false-rejection rate | 0.2000 | Best single verifier | k=1, t=0.000 | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 0.0899 [0.0767, 0.1038] | 0.0000 [0.0000, 0.0000] |
| clean false-rejection rate | 0.2000 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | -0.0922 [-0.1190, -0.0685] | 0.0859 [0.0634, 0.1110] | -0.0076 [-0.0166, 0.0005] |  |
| clean false-rejection rate | 0.3000 | Shared-backbone pool (k of 4) | k=2, t=0.100 | 0.2919 [0.2688, 0.3172] | 0.7112 [0.6880, 0.7343] | 0.0939 [0.0773, 0.1106] | 0.1222 [0.0865, 0.1574] |
| clean false-rejection rate | 0.3000 | Cross-backbone pool (k of 4) | k=3, t=0.400 | 0.2513 [0.2288, 0.2719] | 0.7449 [0.7244, 0.7654] | 0.0852 [0.0697, 0.1019] | 0.1533 [0.1130, 0.1941] |
| clean false-rejection rate | 0.3000 | Best single verifier | k=1, t=0.500 | 0.5490 [0.5243, 0.5733] | 0.4494 [0.4270, 0.4732] | 0.0868 [0.0657, 0.1081] | 0.1589 [0.1303, 0.1898] |
| clean false-rejection rate | 0.3000 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | -0.0407 [-0.0643, -0.0159] | 0.0337 [0.0112, 0.0568] | -0.0087 [-0.0184, 0.0004] |  |
| clean false-rejection rate | 0.4000 | Shared-backbone pool (k of 4) | k=2, t=0.500 | 0.3885 [0.3626, 0.4134] | 0.6120 [0.5882, 0.6358] | 0.0907 [0.0731, 0.1097] | 0.1438 [0.1089, 0.1777] |
| clean false-rejection rate | 0.4000 | Cross-backbone pool (k of 4) | k=3, t=0.600 | 0.3696 [0.3469, 0.3945] | 0.6252 [0.6028, 0.6477] | 0.0825 [0.0655, 0.1007] | 0.1650 [0.1306, 0.2011] |
| clean false-rejection rate | 0.4000 | Best single verifier | k=1, t=0.500 | 0.5490 [0.5243, 0.5733] | 0.4494 [0.4270, 0.4732] | 0.0868 [0.0657, 0.1081] | 0.1589 [0.1303, 0.1898] |
| clean false-rejection rate | 0.4000 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | -0.0189 [-0.0413, 0.0081] | 0.0132 [-0.0126, 0.0357] | -0.0083 [-0.0190, 0.0022] |  |
| clean false-rejection rate | 0.5000 | Shared-backbone pool (k of 4) | k=1, t=0.900 | 0.5171 [0.4916, 0.5431] | 0.4825 [0.4600, 0.5069] | 0.0890 [0.0692, 0.1102] | 0.1545 [0.1232, 0.1871] |
| clean false-rejection rate | 0.5000 | Cross-backbone pool (k of 4) | k=3, t=0.800 | 0.4365 [0.4130, 0.4623] | 0.5578 [0.5334, 0.5803] | 0.0806 [0.0633, 0.0990] | 0.1689 [0.1358, 0.2028] |
| clean false-rejection rate | 0.5000 | Best single verifier | k=1, t=0.500 | 0.5490 [0.5243, 0.5733] | 0.4494 [0.4270, 0.4732] | 0.0868 [0.0657, 0.1081] | 0.1589 [0.1303, 0.1898] |
| clean false-rejection rate | 0.5000 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | -0.0806 [-0.1032, -0.0573] | 0.0753 [0.0529, 0.0972] | -0.0085 [-0.0215, 0.0044] |  |

## eta = 20%

| Matched quantity | Target | Rule | Operating point | Achieved matched quantity | Retention | Admitted contamination | Detection F1 |
|---|---|---|---|---|---|---|---|
| admitted contamination | 0.1219 | Shared-backbone pool (k of 4) | k=1, t=1.000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.3364 [0.3088, 0.3642] |
| admitted contamination | 0.1219 | Cross-backbone pool (k of 4) | k=1, t=1.000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.3364 [0.3088, 0.3642] |
| admitted contamination | 0.1219 | Best single verifier | k=1, t=1.000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.3364 [0.3088, 0.3642] |
| admitted contamination | 0.1219 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |  |
| admitted contamination | 0.1422 | Shared-backbone pool (k of 4) | k=1, t=1.000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.3364 [0.3088, 0.3642] |
| admitted contamination | 0.1422 | Cross-backbone pool (k of 4) | k=1, t=1.000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.3364 [0.3088, 0.3642] |
| admitted contamination | 0.1422 | Best single verifier | k=1, t=1.000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.3364 [0.3088, 0.3642] |
| admitted contamination | 0.1422 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |  |
| admitted contamination | 0.1625 | Shared-backbone pool (k of 4) | k=3, t=0.950 | 0.1715 [0.1273, 0.2176] | 0.1924 [0.1728, 0.2135] | 0.1715 [0.1273, 0.2176] | 0.3352 [0.3029, 0.3669] |
| admitted contamination | 0.1625 | Cross-backbone pool (k of 4) | k=1, t=1.000 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.3364 [0.3088, 0.3642] |
| admitted contamination | 0.1625 | Best single verifier | k=1, t=0.950 | 0.1673 [0.1209, 0.2137] | 0.1805 [0.1615, 0.2008] | 0.1673 [0.1209, 0.2137] | 0.3368 [0.3040, 0.3683] |
| admitted contamination | 0.1625 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | -0.1715 [-0.2176, -0.1273] | -0.1924 [-0.2135, -0.1728] | -0.1715 [-0.2176, -0.1273] |  |
| admitted contamination | 0.1828 | Shared-backbone pool (k of 4) | k=2, t=0.650 | 0.1917 [0.1633, 0.2194] | 0.5091 [0.4846, 0.5344] | 0.1917 [0.1633, 0.2194] | 0.3019 [0.2639, 0.3402] |
| admitted contamination | 0.1828 | Cross-backbone pool (k of 4) | k=1, t=0.850 | 0.1955 [0.1737, 0.2182] | 0.8406 [0.8216, 0.8596] | 0.1955 [0.1737, 0.2182] | 0.2097 [0.1641, 0.2547] |
| admitted contamination | 0.1828 | Best single verifier | k=1, t=0.670 | 0.1828 [0.1533, 0.2126] | 0.4073 [0.3820, 0.4340] | 0.1828 [0.1533, 0.2126] | 0.3216 [0.2878, 0.3566] |
| admitted contamination | 0.1828 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | 0.0038 [-0.0145, 0.0237] | 0.3315 [0.3055, 0.3560] | 0.0038 [-0.0145, 0.0237] |  |
| admitted contamination | 0.2031 | Shared-backbone pool (k of 4) | k=1, t=0.000 | 0.2022 [0.1826, 0.2226] | 1.0000 [1.0000, 1.0000] | 0.2022 [0.1826, 0.2226] | 0.0000 [0.0000, 0.0000] |
| admitted contamination | 0.2031 | Cross-backbone pool (k of 4) | k=1, t=0.000 | 0.2022 [0.1826, 0.2226] | 1.0000 [1.0000, 1.0000] | 0.2022 [0.1826, 0.2226] | 0.0000 [0.0000, 0.0000] |
| admitted contamination | 0.2031 | Best single verifier | k=1, t=0.000 | 0.2022 [0.1826, 0.2226] | 1.0000 [1.0000, 1.0000] | 0.2022 [0.1826, 0.2226] | 0.0000 [0.0000, 0.0000] |
| admitted contamination | 0.2031 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |  |
| true harmful survival | 0.2000 | Shared-backbone pool (k of 4) | k=3, t=0.950 | 0.1632 [0.1214, 0.2059] | 0.1924 [0.1728, 0.2135] | 0.1715 [0.1273, 0.2176] | 0.3352 [0.3029, 0.3669] |
| true harmful survival | 0.2000 | Cross-backbone pool (k of 4) | k=4, t=0.950 | 0.0799 [0.0496, 0.1123] | 0.1138 [0.0983, 0.1299] | 0.1420 [0.0909, 0.1966] | 0.3419 [0.3111, 0.3713] |
| true harmful survival | 0.2000 | Best single verifier | k=1, t=0.900 | 0.1701 [0.1277, 0.2140] | 0.2015 [0.1819, 0.2240] | 0.1707 [0.1271, 0.2171] | 0.3354 [0.3022, 0.3665] |
| true harmful survival | 0.2000 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | -0.0833 [-0.1232, -0.0463] | -0.0787 [-0.0969, -0.0604] | -0.0296 [-0.0751, 0.0163] |  |
| true harmful survival | 0.3000 | Shared-backbone pool (k of 4) | k=3, t=0.950 | 0.1632 [0.1214, 0.2059] | 0.1924 [0.1728, 0.2135] | 0.1715 [0.1273, 0.2176] | 0.3352 [0.3029, 0.3669] |
| true harmful survival | 0.3000 | Cross-backbone pool (k of 4) | k=3, t=1.000 | 0.2639 [0.2089, 0.3167] | 0.2858 [0.2626, 0.3104] | 0.1867 [0.1478, 0.2243] | 0.3249 [0.2904, 0.3601] |
| true harmful survival | 0.3000 | Best single verifier | k=1, t=0.850 | 0.2257 [0.1811, 0.2749] | 0.2577 [0.2360, 0.2823] | 0.1771 [0.1399, 0.2184] | 0.3316 [0.2981, 0.3643] |
| true harmful survival | 0.3000 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | 0.1007 [0.0521, 0.1497] | 0.0934 [0.0709, 0.1152] | 0.0152 [-0.0213, 0.0498] |  |
| true harmful survival | 0.4000 | Shared-backbone pool (k of 4) | k=2, t=0.900 | 0.3021 [0.2491, 0.3519] | 0.3272 [0.3041, 0.3539] | 0.1867 [0.1522, 0.2225] | 0.3226 [0.2876, 0.3561] |
| true harmful survival | 0.4000 | Cross-backbone pool (k of 4) | k=3, t=1.000 | 0.2639 [0.2089, 0.3167] | 0.2858 [0.2626, 0.3104] | 0.1867 [0.1478, 0.2243] | 0.3249 [0.2904, 0.3601] |
| true harmful survival | 0.4000 | Best single verifier | k=1, t=0.700 | 0.2917 [0.2417, 0.3467] | 0.3209 [0.2971, 0.3476] | 0.1838 [0.1506, 0.2192] | 0.3251 [0.2905, 0.3584] |
| true harmful survival | 0.4000 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | -0.0382 [-0.0868, 0.0104] | -0.0414 [-0.0625, -0.0225] | 0.0000 [-0.0309, 0.0293] |  |
| true harmful survival | 0.5000 | Shared-backbone pool (k of 4) | k=2, t=0.750 | 0.4514 [0.3949, 0.5071] | 0.4761 [0.4508, 0.5021] | 0.1917 [0.1625, 0.2194] | 0.3056 [0.2675, 0.3435] |
| true harmful survival | 0.5000 | Cross-backbone pool (k of 4) | k=2, t=1.000 | 0.4722 [0.4100, 0.5256] | 0.5049 [0.4775, 0.5295] | 0.1892 [0.1606, 0.2179] | 0.3061 [0.2683, 0.3450] |
| true harmful survival | 0.5000 | Best single verifier | k=1, t=0.650 | 0.3681 [0.3150, 0.4216] | 0.4087 [0.3834, 0.4347] | 0.1821 [0.1525, 0.2116] | 0.3221 [0.2883, 0.3576] |
| true harmful survival | 0.5000 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | 0.0208 [-0.0414, 0.0775] | 0.0288 [0.0021, 0.0548] | -0.0026 [-0.0253, 0.0191] |  |
| true harmful survival | 0.6000 | Shared-backbone pool (k of 4) | k=3, t=0.250 | 0.5556 [0.4953, 0.6180] | 0.5765 [0.5520, 0.6018] | 0.1949 [0.1667, 0.2228] | 0.2873 [0.2446, 0.3277] |
| true harmful survival | 0.6000 | Cross-backbone pool (k of 4) | k=2, t=1.000 | 0.4722 [0.4100, 0.5256] | 0.5049 [0.4775, 0.5295] | 0.1892 [0.1606, 0.2179] | 0.3061 [0.2683, 0.3450] |
| true harmful survival | 0.6000 | Best single verifier | k=1, t=0.650 | 0.3681 [0.3150, 0.4216] | 0.4087 [0.3834, 0.4347] | 0.1821 [0.1525, 0.2116] | 0.3221 [0.2883, 0.3576] |
| true harmful survival | 0.6000 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | -0.0833 [-0.1443, -0.0263] | -0.0716 [-0.0976, -0.0463] | -0.0057 [-0.0273, 0.0140] |  |
| clean false-rejection rate | 0.1000 | Shared-backbone pool (k of 4) | k=1, t=0.000 | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 0.2022 [0.1826, 0.2226] | 0.0000 [0.0000, 0.0000] |
| clean false-rejection rate | 0.1000 | Cross-backbone pool (k of 4) | k=1, t=0.800 | 0.0960 [0.0792, 0.1140] | 0.8996 [0.8820, 0.9150] | 0.1983 [0.1766, 0.2192] | 0.1578 [0.1109, 0.2038] |
| clean false-rejection rate | 0.1000 | Best single verifier | k=1, t=0.000 | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 0.2022 [0.1826, 0.2226] | 0.0000 [0.0000, 0.0000] |
| clean false-rejection rate | 0.1000 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | 0.0960 [0.0792, 0.1140] | -0.1004 [-0.1180, -0.0850] | -0.0040 [-0.0113, 0.0031] |  |
| clean false-rejection rate | 0.2000 | Shared-backbone pool (k of 4) | k=1, t=0.100 | 0.2500 [0.2259, 0.2758] | 0.7570 [0.7353, 0.7788] | 0.2096 [0.1859, 0.2328] | 0.1956 [0.1501, 0.2395] |
| clean false-rejection rate | 0.2000 | Cross-backbone pool (k of 4) | k=2, t=0.700 | 0.2060 [0.1812, 0.2308] | 0.7816 [0.7591, 0.8034] | 0.1896 [0.1673, 0.2113] | 0.2571 [0.2114, 0.3064] |
| clean false-rejection rate | 0.2000 | Best single verifier | k=1, t=0.100 | 0.3002 [0.2737, 0.3253] | 0.7093 [0.6868, 0.7324] | 0.2129 [0.1877, 0.2379] | 0.2080 [0.1669, 0.2490] |
| clean false-rejection rate | 0.2000 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | -0.0440 [-0.0703, -0.0166] | 0.0246 [0.0000, 0.0478] | -0.0201 [-0.0339, -0.0076] |  |
| clean false-rejection rate | 0.3000 | Shared-backbone pool (k of 4) | k=2, t=0.100 | 0.3213 [0.2941, 0.3462] | 0.6861 [0.6636, 0.7100] | 0.2108 [0.1849, 0.2353] | 0.2231 [0.1813, 0.2636] |
| clean false-rejection rate | 0.3000 | Cross-backbone pool (k of 4) | k=3, t=0.500 | 0.2817 [0.2569, 0.3084] | 0.7086 [0.6847, 0.7303] | 0.1913 [0.1673, 0.2148] | 0.2703 [0.2277, 0.3116] |
| clean false-rejection rate | 0.3000 | Best single verifier | k=1, t=0.100 | 0.3002 [0.2737, 0.3253] | 0.7093 [0.6868, 0.7324] | 0.2129 [0.1877, 0.2379] | 0.2080 [0.1669, 0.2490] |
| clean false-rejection rate | 0.3000 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | -0.0396 [-0.0662, -0.0124] | 0.0225 [-0.0021, 0.0457] | -0.0196 [-0.0336, -0.0053] |  |
| clean false-rejection rate | 0.4000 | Shared-backbone pool (k of 4) | k=3, t=0.250 | 0.4181 [0.3892, 0.4465] | 0.5765 [0.5520, 0.6018] | 0.1949 [0.1667, 0.2228] | 0.2873 [0.2446, 0.3277] |
| clean false-rejection rate | 0.4000 | Cross-backbone pool (k of 4) | k=2, t=1.000 | 0.4868 [0.4584, 0.5180] | 0.5049 [0.4775, 0.5295] | 0.1892 [0.1606, 0.2179] | 0.3061 [0.2683, 0.3450] |
| clean false-rejection rate | 0.4000 | Best single verifier | k=1, t=0.400 | 0.3970 [0.3678, 0.4260] | 0.6053 [0.5794, 0.6306] | 0.2053 [0.1773, 0.2301] | 0.2612 [0.2240, 0.3014] |
| clean false-rejection rate | 0.4000 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | 0.0687 [0.0404, 0.0998] | -0.0716 [-0.0976, -0.0463] | -0.0057 [-0.0273, 0.0140] |  |
| clean false-rejection rate | 0.5000 | Shared-backbone pool (k of 4) | k=2, t=0.800 | 0.5282 [0.4974, 0.5562] | 0.4670 [0.4424, 0.4930] | 0.1940 [0.1650, 0.2219] | 0.3037 [0.2660, 0.3408] |
| clean false-rejection rate | 0.5000 | Cross-backbone pool (k of 4) | k=2, t=1.000 | 0.4868 [0.4584, 0.5180] | 0.5049 [0.4775, 0.5295] | 0.1892 [0.1606, 0.2179] | 0.3061 [0.2683, 0.3450] |
| clean false-rejection rate | 0.5000 | Best single verifier | k=1, t=0.670 | 0.5827 [0.5534, 0.6104] | 0.4073 [0.3820, 0.4340] | 0.1828 [0.1533, 0.2126] | 0.3216 [0.2878, 0.3566] |
| clean false-rejection rate | 0.5000 | PAIRED DIFFERENCE (cross-backbone minus shared) |  | -0.0414 [-0.0713, -0.0097] | 0.0379 [0.0112, 0.0646] | -0.0048 [-0.0269, 0.0165] |  |
