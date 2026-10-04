# Verifier pool card (the artifact the paper proposes, instantiated)

## Pool A: shared backbone (pilot pool)

| Field | Value |
|---|---|
| Roles | Safety, Helpfulness, Factuality, Policy (role-specific system prompts, shared instruction format; full prompts in src/verifiers.py) |
| Backbone (all roles) | Qwen2.5-1.5B-Instruct, fp16, greedy decoding, 96 new tokens |
| Score threshold | 0.7 (all roles; no per-role calibration) |
| Scored data | 10,000 HH-RLHF training pairs, BOTH response directions |
| Error definition | error = admits a corrupted pair OR rejects a clean pair (relative to injected-corruption labels) |
| Correlation estimator | Ledoit-Wolf shrunk error correlation; 1,000-draw bootstrap over examples |
| Measured dependence (eta=20%) | rho-bar 0.67, 95% CI [0.66, 0.68]; effective ensemble size n_eff 1.33 of 4, CI [1.32, 1.35]; top eigenvalue 3.01 of 4 |
| Known weaknesses | Safety role has the highest false-admit rate on structured-unsafe corruption (55.6%); rationales are frequently unreliable at this model scale |

## Pool B: cross-backbone

| Field | Value |
|---|---|
| Roles and prompts | identical to Pool A |
| Safety backbone | Gemma-2-9B-it, 4-bit NF4 quantization (system prompt folded into the user turn because the Gemma chat template does not accept a system role) |
| Helpfulness backbone | Mistral-7B-Instruct-v0.3, fp16 |
| Factuality backbone | Qwen3-4B-Instruct-2507, fp16 |
| Policy backbone | Qwen2.5-1.5B-Instruct, fp16 (retained from Pool A for continuity) |
| Score threshold | 0.7 (unchanged) |
| Scored data | 3,000-pair subsample of the same 10,000 pairs (uniform, seed 42), BOTH directions |
| Measured dependence (eta=20%) | rho-bar 0.26, CI [0.24, 0.28]; n_eff 2.25 of 4; top eigenvalue 1.81 of 4 |
| Known weaknesses | The two largest backbones (Gemma-2-9B-it safety, Mistral-7B helpfulness) are more permissive: higher false-admit on corrupted pairs (78.0% and 73.3%) with much lower false-reject on clean pairs (18.1% and 25.5%); their pairwise error correlation (0.48) is the largest remaining off-diagonal entry |

## Notes

- Family caveat: Pool B spans three model families (Gemma, Mistral, Qwen), with
  two Qwen generations (Qwen3-4B and Qwen2.5-1.5B); it is "cross-backbone", not
  fully "one family per role".
- All dependence quantities are computed under the injected-corruption labels of
  the structured-unsafe regime; they are properties of the pool on this
  data and threat model, not universal constants.
