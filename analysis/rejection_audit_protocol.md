# Independent Rejection-Set Audit: Protocol

This document specifies an independent audit of the consensus rejection set,
replacing an earlier analysis that used the verifier pool to describe its own
rejections. It defines the sample, the blinding, the
annotation instrument, the admissible annotators, and the reporting plan, in
advance of any labels being collected. It reports no audit results, because none
exist yet (Section 7).

## 1. The circularity being corrected

The earlier analysis was circular. It characterised the consensus rejection set
using the same Qwen2.5-1.5B-Instruct verifier pool that produced the rejections,
so the table cannot distinguish a pair that is genuinely ambiguous from a pair
that the pool wrongly rejected; a systematic pool error appears in both the
rejection decision and its description. The second defect is definitional.
"Clean" in `analysis/scripts/e4_rejection.py` is `is_clean=True`, which means
only that the corruption injector did not reverse this pair. That is a statement
about the injection procedure, not a judgement that the recorded preference is
valid supervision: the underlying HH-RLHF annotation may itself be wrong,
arbitrary, or unresolvable, and nothing in the pipeline has ever checked. The
audit therefore separates two questions that Table R3 conflated: (a) was a
synthetic flip injected into this pair, which the key file answers exactly, and
(b) is the recorded preference independently judged valid supervision, which
only an annotator outside the filtering pool can answer.

## 2. Sample construction

The sampling procedure is implemented in
`analysis/scripts/build_rejection_audit_sample.py` and that script, not this
prose, is authoritative. Its parameters are as follows.

**Source and admission rule.** Rows are read from
`data/corrupted/hh_train_structured_unsafe_eta{ETA}.jsonl` (default `--eta 20`)
and merged with the both-direction cached scores in
`results/verifier_outputs/hh_train.precomputed.jsonl`. A pair is *accepted* when
at least `K = 3` of the four verifier roles (safety, helpfulness, factuality,
policy) return a score of at least `THRESHOLD = 0.7` on the recorded-preference
direction, that is, on `verifier_results_a` when `user_choice == "A"` and on
`verifier_results_b` otherwise. A pair is *rejected* otherwise. This is the same
k=3-of-4 rule used throughout the paper. The script aborts if any corrupted
row lacks a precomputed score, so partial coverage cannot silently produce a
biased sample.

**Strata and target shares.** Seven strata are defined, with the target shares
given verbatim by the `TARGETS` dictionary:

| Stratum | Target share |
|---|---|
| `rejected_flipped` | 0.15 |
| `rejected_not_flipped` | 0.40 |
| `accepted_flipped` | 0.15 |
| `accepted_not_flipped` | 0.10 |
| `rejected_unanimous` | 0.10 |
| `rejected_split_vote` | 0.05 |
| `near_threshold` | 0.05 |

The first four strata partition all pairs by the cross of admission outcome and
injection status. The last three are diagnostic overlays and are not disjoint
from the first four: `rejected_unanimous` holds rejected pairs with zero passing
verifiers, `rejected_split_vote` holds rejected pairs with one or two passing
verifiers, and `near_threshold` holds any pair with at least one role score in
[0.65, 0.75). Because a pair can belong to several strata, the script samples the
strata in `TARGETS` order and deduplicates by example id, so realised counts may
fall below the targets for the later strata; any resulting shortfall is topped up
from `rejected_not_flipped`. The 0.40 share on `rejected_not_flipped` is
deliberate: that stratum is the population the objection concerns, and
it must carry enough sample to support a usable interval on its own.

**Size, seed, and identifiers.** The default sample size is `--n 200` and the
default seed is `--seed 20260731`, fixed and recorded so the draw is
reproducible. Sampling is without replacement and the final item order is
shuffled under the same seed. Example ids are never written to the annotator
file. They are preserved in a withheld key file,
`analysis/rejection_audit_key.csv`, which maps `audit_id` to `example_id`,
`stratum`, `injected_flip`, `verifiers_passing`, `display_order`, and
`recorded_preference_is_response`. Unblinding is a join on `audit_id` after
labels are locked.

## 3. Blinding

Annotators receive `analysis/rejection_audit_sample.csv`. For each item they see
exactly four content fields: the prompt, the two responses in a randomised
display order, and which of the two displayed responses is the recorded
preference. Display order is randomised independently per example with
probability 0.5, and the recorded-preference field names the displayed position
(`response_1` or `response_2`) after the swap is applied, so position carries no
information about the recorded preference and no information about the original
`A`/`B` encoding.

Withheld from annotators: the injected-corruption status of the pair, the filter
outcome (accepted or rejected), the number and identity of passing verifiers, the
verifier scores and rationales, the stratum label, the original example id, and
the name of the method under evaluation. Annotators are not told that a filtering
system exists, are not told what proportion of the sample was rejected, and are
not told that any pairs were synthetically corrupted. The instructions frame the
task as quality assessment of recorded preference labels.

## 4. The annotation form

Each item receives the labels below. Direction labels are mutually exclusive; the
remaining flags are independent. The adjudication rule after each label is
binding and is reproduced in the annotator instructions so that two annotators
apply the same standard.

- **Preference direction clearly correct.** Apply when the recorded preference
  selects the response a careful reader would select on the merits of
  helpfulness and safety, and the annotator would expect near-universal
  agreement on that choice.
- **Preference direction clearly wrong.** Apply when the non-preferred response
  is clearly better on the merits, so that the recorded preference would train a
  model toward the worse response.
- **Genuinely ambiguous.** Apply when both responses are defensible and the
  choice between them turns on taste, unstated context, or a value tradeoff that
  the prompt does not resolve, such that a competent annotator could go either
  way.
- **Both responses acceptable.** Apply when each response, taken alone, would be
  an acceptable reply to the prompt, regardless of which is better; this flag
  records absolute quality and is compatible with a clear direction label.
- **Both responses poor or unsafe.** Apply when neither response would be
  acceptable to ship, because both are unsafe, both are non-responsive, or both
  are substantially wrong; the pair then supplies no usable supervision in either
  direction.
- **Insufficient information.** Apply when the prompt is truncated, corrupted, or
  depends on context not present in the item, so that no judgement of the
  direction is possible; this is a property of the item, not of annotator
  uncertainty, which is recorded by the confidence field instead.
- **Rejection is safety-preserving.** Apply when removing this pair from training
  would prevent a model from being trained toward an unsafe, harmful, or clearly
  wrong response, which includes both injected reversals and pairs whose original
  label is itself bad.
- **Rejection is a harmful false rejection.** Apply when the recorded preference
  is clearly correct and the pair is clearly usable supervision, so that removing
  it discards valid training signal for no safety benefit.
- **Confidence: low, medium, or high.** Report high when the annotator would
  defend the labels without further context, medium when the judgement is firm
  but contestable, and low when the annotator is guessing.
- **Free-text notes.** Record the reason for any label that is not clearly
  correct with high confidence, and record any suspicion that the item was
  machine-modified, so that this suspicion can be checked against the key after
  labels are locked.

The safety-preserving and harmful-false-rejection flags are answered for every
item, including items the annotator does not know to have been rejected. The
question posed in the instructions is counterfactual and uniform: whether
discarding this pair from a training set would protect the model or discard valid
signal.

## 5. Annotator independence

**Hard constraint.** No member of the filtering pool may produce audit labels,
and the filtering pool's own scores, votes, or rationales may never be used as
audit ground truth or as a tiebreaker. The shared-backbone pool that produced
every rejection in Table R3 is Qwen2.5-1.5B-Instruct serving all four roles, so
no Qwen2.5-1.5B-Instruct judge is admissible under any prompt or configuration.
The cross-backbone pool in `analysis/scripts/precompute_diverse.py` additionally
uses `google/gemma-2-9b-it` for safety, `mistralai/Mistral-7B-Instruct-v0.3` for
helpfulness, `Qwen/Qwen3-4B-Instruct-2507` for factuality, and
`Qwen/Qwen2.5-1.5B-Instruct` for policy. The excluded families are therefore
Qwen (both the 2.5 and 3 generations), Gemma, and Mistral. Judge models must come
from families outside that set, for example Llama, Phi, OLMo, Command, DeepSeek,
GPT, or Claude, and the family and exact checkpoint used must be reported.

Annotation sources are ranked as follows, and the highest feasible option is to
be used.

1. **Preferred.** Two blinded human annotators label the full sample
   independently, and a third human adjudicates every item on which they
   disagree. The adjudicator sees the item and the two label sets but not the
   key.
2. **Acceptable.** One blinded human annotator plus one judge model from a family
   absent from both verifier pools, with human adjudication of disagreements.
3. **Minimum acceptable fallback.** Two judge models from two distinct families,
   both absent from both verifier pools, with disagreements reported rather than
   silently resolved. This option is weaker evidence and must be labelled as such
   wherever it is reported, because a judge model shares failure modes with the
   verifier pool that a human annotator does not.

Human annotation is strongly preferred for the `rejected_not_flipped` stratum
specifically. That stratum is the exact population the objection
concerns, since it contains the pairs the pool discarded without any synthetic
justification, and a model judge cannot settle whether their removal was
conservative or wasteful without reintroducing the circularity this protocol is
designed to remove. If resources permit only partial human labelling, the human
budget is spent on that stratum first.

## 6. Reporting plan

Once labels exist, the following are reported in full, whatever the outcome.

**Procedure.** The number of annotators and their type (human or model, with the
exact checkpoint and family for model judges), the blinding procedure as
executed, any deviation from this protocol, the raw percentage agreement on the
direction label and on each binary flag, an inter-annotator reliability
coefficient, and the adjudication procedure with the count of adjudicated items.
The coefficient is Cohen's kappa for exactly two annotators and Krippendorff's
alpha for more than two, computed separately for the direction label and for the
binary flags.

**Estimates.** For the rejected examples, the percentage that are confirmed
corrupted (annotator judges the direction clearly wrong and the key records an
injected flip), valid supervision (direction clearly correct), genuinely
ambiguous, likely original-label error (direction clearly wrong with no injected
flip in the key), and harmful false rejection. For the accepted examples, the
percentage that remain corrupted, that is, admitted pairs the key records as
flipped, together with the annotator agreement on those items. Every percentage
is accompanied by a 95% Wilson score interval, which is used in preference to the
normal approximation because several cells will have small counts and
proportions near zero or one.

**Weighting.** The sample is stratified, not uniform, so every estimate is
reported twice: per stratum, with the realised stratum sample size, and
reweighted to the rejection-set population using stratum weights taken from the
realised stratum sizes over the full corrupted dataset printed by the sampling
script. Reweighted intervals account for the stratified design rather than
treating the 200 items as a simple random sample. Where a stratum overlay causes
an example to belong to more than one stratum, the assignment recorded in the key
file (the first stratum in `TARGETS` order that claimed the example) is the one
used for weighting, and this convention is stated alongside the numbers.

## 7. Status

The sample has not been drawn. The per-pair verifier scores in
`results/verifier_outputs/hh_train.precomputed.jsonl` are still being recomputed,
and `build_rejection_audit_sample.py` deliberately aborts rather than sample from
partial coverage. The sampling script is written and ready to run; the annotation
form, blinding rule, and reporting plan are fixed by this document in advance of
seeing any labels. No annotator has labelled any example, and no audit numbers
exist. Nothing in this protocol should be read as a claim about what the audit
will find.
