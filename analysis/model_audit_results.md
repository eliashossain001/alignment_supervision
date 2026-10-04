# Independent pointwise model audit: results

This is an independent MODEL-BASED audit and explicitly not human ground truth.
Judges are Llama-3.1-8B-Instruct and Phi-3.5-mini-instruct, neither of which
appears in the shared-backbone pool, the cross-backbone pool, or as the trained
policy. Each response was scored alone, with the other response never present in
context, so no display order exists and position bias is removed by construction.
Labels were frozen and hashed before the corruption labels were joined.

| Group | n | judged valid | judged invalid | tie/uncertain | unresolved |
|---|---:|---|---|---|---|
| rejected_not_flipped | 108 | 14/108 = 0.130 [0.079, 0.206] | 6/108 = 0.056 [0.026, 0.116] | 30/108 = 0.278 [0.202, 0.369] | 58/108 = 0.537 [0.443, 0.628] |
| rejected_flipped | 38 | 2/38 = 0.053 [0.015, 0.173] | 8/38 = 0.211 [0.111, 0.363] | 9/38 = 0.237 [0.130, 0.392] | 19/38 = 0.500 [0.348, 0.652] |
| accepted_flipped | 31 | 0/31 = 0.000 [0.000, 0.110] | 4/31 = 0.129 [0.051, 0.289] | 15/31 = 0.484 [0.320, 0.652] | 12/31 = 0.387 [0.237, 0.562] |
| accepted_not_flipped | 23 | 3/23 = 0.130 [0.045, 0.321] | 0/23 = 0.000 [0.000, 0.143] | 11/23 = 0.478 [0.292, 0.670] | 9/23 = 0.391 [0.222, 0.592] |
