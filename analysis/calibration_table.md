# Verifier calibration diagnostics

All calibrators are fitted on the calibration split and every number in this file is computed on the disjoint evaluation split. The split is 50/50, stratified by the corruption label, with seed 20260731. The identifier lists and their SHA-256 digests are recorded in `analysis/outputs/calibration_splits.json`.

The calibration target is the binary label of whether a synthetic preference flip was injected into the pair, that is `is_clean == False` in the corrupted file for the given eta. The score used for a pair is the score on the recorded-preference direction. A low score is intended to indicate a corrupted pair, therefore the predictor of the positive class in the ROC and precision-recall computations is the NEGATED score.

Probability convention. The raw method treats `1 - score` as the predicted probability of corruption. Platt scaling is a logistic regression fitted on the raw score. Isotonic regression is fitted on the raw score with clipping outside the observed range. Temperature scaling is not attempted: the verifier scores are floats parsed out of model-emitted text rather than logits, so there is no pre-softmax quantity that a temperature could rescale. This is a limitation of the measurement, not an oversight.

Expected calibration error uses 10 equal-width bins on the interval [0, 1]. Bin b covers [b/10, (b+1)/10), and the final bin is closed at 1.0. Empty bins contribute nothing to the sum. The reported value is the sample-weighted mean absolute gap between the mean predicted probability and the observed corruption frequency inside each bin.

The operating point columns describe the reject decision, where rejecting a pair is a prediction that the pair is corrupted. At the original threshold a pair is rejected when its score falls below 0.7. The calibrated threshold is the value that maximises corruption-detection F1 on the calibration split, applied unchanged to the evaluation split.

## eta = 10%

| Pool | Role | Method | AUROC | AUPRC | Brier | ECE | Prec@0.7 | Rec@0.7 | F1@0.7 | FPR@0.7 | FNR@0.7 | Calibrated threshold | F1 at calibrated threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shared | safety | raw | 0.525 | 0.104 | 0.3961 | 0.3892 | 0.108 | 0.444 | 0.174 | 0.400 | 0.556 | 0.900 | 0.175 |
| shared | safety | platt | 0.525 | 0.104 | 0.0888 | 0.0014 | 0.108 | 0.444 | 0.174 | 0.400 | 0.556 | 0.092 | 0.175 |
| shared | safety | isotonic | 0.525 | 0.104 | 0.0889 | 0.0105 | 0.108 | 0.444 | 0.174 | 0.400 | 0.556 | 0.087 | 0.175 |
| shared | helpfulness | raw | 0.525 | 0.104 | 0.3793 | 0.4200 | 0.103 | 0.686 | 0.180 | 0.650 | 0.314 | 0.900 | 0.180 |
| shared | helpfulness | platt | 0.525 | 0.104 | 0.0889 | 0.0057 | 0.103 | 0.686 | 0.180 | 0.650 | 0.314 | 0.088 | 0.180 |
| shared | helpfulness | isotonic | 0.525 | 0.104 | 0.0889 | 0.0030 | 0.103 | 0.686 | 0.180 | 0.650 | 0.314 | 0.096 | 0.180 |
| shared | factuality | raw | 0.515 | 0.101 | 0.4058 | 0.4250 | 0.104 | 0.560 | 0.175 | 0.528 | 0.440 | 0.950 | 0.183 |
| shared | factuality | platt | 0.515 | 0.101 | 0.0889 | 0.0101 | 0.104 | 0.560 | 0.175 | 0.528 | 0.440 | 0.086 | 0.183 |
| shared | factuality | isotonic | 0.515 | 0.101 | 0.0889 | 0.0076 | 0.104 | 0.560 | 0.175 | 0.528 | 0.440 | 0.068 | 0.183 |
| shared | policy | raw | 0.523 | 0.103 | 0.5439 | 0.5449 | 0.106 | 0.609 | 0.181 | 0.561 | 0.391 | 0.600 | 0.181 |
| shared | policy | platt | 0.523 | 0.103 | 0.0888 | 0.0060 | 0.106 | 0.609 | 0.181 | 0.561 | 0.391 | 0.095 | 0.181 |
| shared | policy | isotonic | 0.523 | 0.103 | 0.0888 | 0.0062 | 0.106 | 0.609 | 0.181 | 0.561 | 0.391 | 0.111 | 0.181 |
| diverse | safety | raw | 0.539 | 0.096 | 0.1927 | 0.1849 | 0.104 | 0.206 | 0.138 | 0.175 | 0.794 | 1.000 | 0.174 |
| diverse | safety | platt | 0.539 | 0.096 | 0.0818 | 0.0059 | 0.104 | 0.206 | 0.138 | 0.175 | 0.794 | 0.093 | 0.174 |
| diverse | safety | isotonic | 0.539 | 0.096 | 0.0818 | 0.0060 | 0.104 | 0.206 | 0.138 | 0.175 | 0.794 | 0.082 | 0.165 |
| diverse | helpfulness | raw | 0.522 | 0.096 | 0.1722 | 0.1879 | 0.088 | 0.243 | 0.129 | 0.248 | 0.757 | 0.900 | 0.166 |
| diverse | helpfulness | platt | 0.522 | 0.096 | 0.0818 | 0.0059 | 0.088 | 0.243 | 0.129 | 0.248 | 0.757 | 0.088 | 0.165 |
| diverse | helpfulness | isotonic | 0.522 | 0.096 | 0.0818 | 0.0056 | 0.088 | 0.243 | 0.129 | 0.248 | 0.757 | 0.087 | 0.166 |
| diverse | factuality | raw | 0.562 | 0.105 | 0.2306 | 0.2848 | 0.108 | 0.566 | 0.182 | 0.460 | 0.434 | 0.600 | 0.181 |
| diverse | factuality | platt | 0.562 | 0.105 | 0.0815 | 0.0052 | 0.108 | 0.566 | 0.182 | 0.460 | 0.434 | 0.082 | 0.183 |
| diverse | factuality | isotonic | 0.562 | 0.105 | 0.0816 | 0.0050 | 0.108 | 0.566 | 0.182 | 0.460 | 0.434 | 0.094 | 0.181 |
| diverse | policy | raw | 0.506 | 0.091 | 0.5403 | 0.5422 | 0.092 | 0.566 | 0.158 | 0.553 | 0.434 | 0.600 | 0.158 |
| diverse | policy | platt | 0.506 | 0.091 | 0.0821 | 0.0164 | 0.092 | 0.566 | 0.158 | 0.553 | 0.434 | 0.087 | 0.158 |
| diverse | policy | isotonic | 0.506 | 0.091 | 0.0821 | 0.0174 | 0.092 | 0.566 | 0.158 | 0.553 | 0.434 | 0.113 | 0.158 |

## eta = 20%

| Pool | Role | Method | AUROC | AUPRC | Brier | ECE | Prec@0.7 | Rec@0.7 | F1@0.7 | FPR@0.7 | FNR@0.7 | Calibrated threshold | F1 at calibrated threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shared | safety | raw | 0.529 | 0.216 | 0.4058 | 0.3741 | 0.224 | 0.451 | 0.299 | 0.399 | 0.549 | 1.000 | 0.338 |
| shared | safety | platt | 0.529 | 0.216 | 0.1617 | 0.0074 | 0.224 | 0.451 | 0.299 | 0.399 | 0.549 | 0.193 | 0.338 |
| shared | safety | isotonic | 0.529 | 0.216 | 0.1617 | 0.0001 | 0.224 | 0.451 | 0.299 | 0.399 | 0.549 | 0.184 | 0.338 |
| shared | helpfulness | raw | 0.533 | 0.215 | 0.3791 | 0.3630 | 0.215 | 0.705 | 0.329 | 0.657 | 0.295 | 0.950 | 0.337 |
| shared | helpfulness | platt | 0.533 | 0.215 | 0.1616 | 0.0063 | 0.215 | 0.705 | 0.329 | 0.657 | 0.295 | 0.189 | 0.336 |
| shared | helpfulness | isotonic | 0.533 | 0.215 | 0.1616 | 0.0006 | 0.215 | 0.705 | 0.329 | 0.657 | 0.295 | 0.165 | 0.337 |
| shared | factuality | raw | 0.523 | 0.212 | 0.4052 | 0.3823 | 0.215 | 0.571 | 0.312 | 0.533 | 0.429 | 0.950 | 0.338 |
| shared | factuality | platt | 0.523 | 0.212 | 0.1617 | 0.0020 | 0.215 | 0.571 | 0.312 | 0.533 | 0.429 | 0.191 | 0.338 |
| shared | factuality | isotonic | 0.523 | 0.212 | 0.1616 | 0.0027 | 0.215 | 0.571 | 0.312 | 0.533 | 0.429 | 0.161 | 0.338 |
| shared | policy | raw | 0.523 | 0.212 | 0.5276 | 0.5157 | 0.216 | 0.617 | 0.320 | 0.570 | 0.383 | 1.000 | 0.338 |
| shared | policy | platt | 0.523 | 0.212 | 0.1617 | 0.0033 | 0.216 | 0.617 | 0.320 | 0.570 | 0.383 | 0.191 | 0.324 |
| shared | policy | isotonic | 0.523 | 0.212 | 0.1617 | 0.0012 | 0.216 | 0.617 | 0.320 | 0.570 | 0.383 | 0.186 | 0.322 |
| diverse | safety | raw | 0.533 | 0.218 | 0.2591 | 0.2381 | 0.253 | 0.226 | 0.239 | 0.169 | 0.774 | 1.000 | 0.336 |
| diverse | safety | platt | 0.533 | 0.218 | 0.1612 | 0.0127 | 0.253 | 0.226 | 0.239 | 0.169 | 0.774 | 0.180 | 0.336 |
| diverse | safety | isotonic | 0.533 | 0.218 | 0.1612 | 0.0120 | 0.253 | 0.226 | 0.239 | 0.169 | 0.774 | 0.171 | 0.336 |
| diverse | helpfulness | raw | 0.521 | 0.217 | 0.2396 | 0.2088 | 0.208 | 0.274 | 0.237 | 0.264 | 0.726 | 1.000 | 0.336 |
| diverse | helpfulness | platt | 0.521 | 0.217 | 0.1614 | 0.0122 | 0.208 | 0.274 | 0.237 | 0.264 | 0.726 | 0.185 | 0.317 |
| diverse | helpfulness | isotonic | 0.521 | 0.217 | 0.1612 | 0.0127 | 0.208 | 0.274 | 0.237 | 0.264 | 0.726 | 0.187 | 0.237 |
| diverse | factuality | raw | 0.550 | 0.228 | 0.2726 | 0.2845 | 0.224 | 0.542 | 0.317 | 0.476 | 0.458 | 1.000 | 0.336 |
| diverse | factuality | platt | 0.550 | 0.228 | 0.1607 | 0.0128 | 0.224 | 0.542 | 0.317 | 0.476 | 0.458 | 0.168 | 0.336 |
| diverse | factuality | isotonic | 0.550 | 0.228 | 0.1607 | 0.0130 | 0.224 | 0.542 | 0.317 | 0.476 | 0.458 | 0.161 | 0.336 |
| diverse | policy | raw | 0.508 | 0.205 | 0.5401 | 0.5269 | 0.206 | 0.597 | 0.307 | 0.582 | 0.403 | 1.000 | 0.336 |
| diverse | policy | platt | 0.508 | 0.205 | 0.1615 | 0.0118 | 0.206 | 0.597 | 0.307 | 0.582 | 0.403 | 0.177 | 0.336 |
| diverse | policy | isotonic | 0.508 | 0.205 | 0.1615 | 0.0117 | 0.206 | 0.597 | 0.307 | 0.582 | 0.403 | 0.176 | 0.317 |
