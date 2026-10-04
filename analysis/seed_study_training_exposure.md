# Training exposure across methods in the multi-seed study

Filtering methods train on fewer examples than unfiltered methods, because
rejecting pairs is what they do. Every method receives one epoch over its own
retained dataset with identical optimization settings, which means the methods do
not receive an identical number of optimizer steps. This document states the
disparity explicitly so that it can be read directly rather than inferred.

The design is deliberately unchanged. The multi-seed study must reproduce the
pilot's configuration, and equalising optimizer steps would produce runs that are
not comparable to the published single-seed results. The disparity is therefore
disclosed rather than removed.

## Identical settings across all configurations

All runs use the base model `Qwen/Qwen2.5-1.5B-Instruct`, one epoch, per-device
batch size 2, gradient accumulation 8, learning rate 5e-6, beta 0.1, maximum
sequence length 1024, and LoRA with rank 16, alpha 32 and dropout 0.05. The only
deliberate difference between configurations is the training file and, for cDPO,
a label smoothing of 0.1. These values match the argument defaults in
`src/train_dpo.py` and the pilot environment defaults in
`scripts/run_pilot_hh.sh`.

## Resulting exposure at nominal eta = 20%

Micro-batches are `ceil(pairs / 2)` and optimizer steps are
`floor(micro_batches / 8)`.

| Method | Training pairs | Micro-batches | Optimizer steps | Relative exposure vs raw |
|---|---:|---:|---:|---:|
| Raw DPO | 10000 | 5000 | 625 | 1.000 |
| cDPO (label smoothing 0.1) | 10000 | 5000 | 625 | 1.000 |
| Oracle (ground truth) | 7969 | 3985 | 498 | 0.797 |
| Single verifier (safety) | 5905 | 2953 | 369 | 0.591 |
| Consensus (k = 3 of 4) | 4020 | 2010 | 251 | 0.402 |

The consensus configuration receives 40.2% of the optimizer steps that Raw DPO
receives. This is a direct consequence of its retention of 0.4020 and is not an
independent design choice.

## Why this matters for interpretation

A reviewer may reasonably argue that any downstream difference between a
filtered method and Raw DPO reflects reduced training exposure rather than the
effect of verification. On the present design that confound cannot be separated,
because retention and exposure are the same quantity. Two consequences follow.

First, no downstream comparison in this study can attribute an effect to
verification rather than to exposure. Any claim of the form "consensus filtering
changes downstream behaviour" is confounded with "consensus filtering trains on
40% as many optimizer steps".

Second, the confound has a direction that is worth stating. Reduced exposure
would ordinarily be expected to move a filtered model toward its initialisation,
which for a preference-alignment run means weaker adherence to the preference
signal in either direction. It does not obviously favour the filtered methods.

The appropriate resolution is an exposure-matched control, in which filtered and
unfiltered methods receive the same number of optimizer steps, either by
subsampling the unfiltered set to the filtered set's size or by training the
filtered set for proportionally more epochs. That control is not run here for the
reason given above, that this study must reproduce the pilot configuration, and
it is named in the response as an open item rather than presented as completed.

## Note on the study's scope

The dataset split and the corruption realization are fixed across all runs. The
subsample is drawn by `src/load_datasets.py` at seed 42 and the corruption is
injected by `src/create_corruption.py` at seed 42, producing 2,031 corrupted
rows at nominal eta = 20%. Only the training seed varies. This study therefore
estimates training-seed uncertainty alone. It does not estimate uncertainty over
corruption draws, over dataset subsamples, or over verifier scoring, and it must
not be described as though it did.
