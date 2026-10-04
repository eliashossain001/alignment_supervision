"""E4: what does consensus (k=3 of 4) actually reject?

Uses both-direction cached scores to categorize every rejected pair:
  truly_corrupted        - is_clean=False (the filter did its job)
  clean, pair_unusable   - BOTH directions fail consensus: no admissible
                           supervision in the pair at all (chosen response is
                           itself low-quality/unsafe per the verifiers)
  clean, direction_flip  - chosen direction fails but the OTHER direction would
                           pass: verifiers disagree with the human label
                           (ambiguous or possibly mislabeled pair)
  clean, split_vote      - 2 of 4 pass on chosen direction (disagreement zone)
Also: which verifier drives rejections, unanimous vs partial, safety-specific
failures, and concrete examples with rationales.
"""
import json

import numpy as np
from common import (OUT, TAB, ROLES, clean_mask, keep_consensus, load_corrupted,
                    load_precomputed, savefig, score_matrix, set_style,
                    TEXT_PRIMARY, TEXT_SECONDARY)

set_style()
pre = load_precomputed()
K, T = 3, 0.7
summary = {}
examples = []

for eta in (10, 20):
    rows = load_corrupted("structured_unsafe", eta)
    s_chosen = score_matrix(rows, pre, "chosen")
    s_other = score_matrix(rows, pre, "other")
    ic = clean_mask(rows)
    keep = keep_consensus(s_chosen, k=K, threshold=T)
    keep_other = keep_consensus(s_other, k=K, threshold=T)
    rej = ~keep
    n_rej = int(rej.sum())
    npass = (s_chosen >= T).sum(axis=1)

    cat = np.full(len(rows), "", dtype=object)
    cat[rej & ~ic] = "truly_corrupted"
    cat[rej & ic & ~keep_other] = "clean_pair_unusable"       # neither direction passes
    cat[rej & ic & keep_other] = "clean_direction_flip"       # verifiers prefer the other direction
    # further split the unusable set by vote pattern on the chosen direction
    unanimous = rej & ic & ~keep_other & (npass == 0)
    split = rej & ic & (npass == 2)

    # which verifier drives rejection (fails while others pass) among clean rejects
    fails = s_chosen < T
    driver = {r: float(fails[rej & ic, j].mean()) for j, r in enumerate(ROLES)}

    # safety-check: among clean rejected, how often does safety fail the CHOSEN
    # (=originally human-preferred, safer) response?
    safety_fail_chosen = float(fails[rej & ic, ROLES.index("safety")].mean())

    d = dict(
        n=len(rows), n_rejected=n_rej,
        frac_truly_corrupted=round(float((rej & ~ic).sum()) / n_rej, 4),
        frac_clean_pair_unusable=round(float((rej & ic & ~keep_other).sum()) / n_rej, 4),
        frac_clean_direction_flip=round(float((rej & ic & keep_other).sum()) / n_rej, 4),
        frac_clean_unanimous_zero_pass=round(float(unanimous.sum()) / n_rej, 4),
        frac_clean_split_2of4=round(float(split.sum()) / n_rej, 4),
        clean_reject_fail_rate_by_verifier={k: round(v, 4) for k, v in driver.items()},
        safety_fail_on_chosen_among_clean_rejects=round(safety_fail_chosen, 4),
        recall_on_corrupted=round(float((rej & ~ic).sum() / (~ic).sum()), 4),
        mean_chosen_score_clean_kept=round(float(s_chosen[keep & ic].mean()), 4),
        mean_chosen_score_clean_rejected=round(float(s_chosen[rej & ic].mean()), 4),
        mean_chosen_score_corrupted=round(float(s_chosen[~ic].mean()), 4),
    )
    summary[f"eta{eta}"] = d
    print(f"eta={eta}: rejected={n_rej} | corrupted {100*d['frac_truly_corrupted']:.1f}% | "
          f"clean-unusable {100*d['frac_clean_pair_unusable']:.1f}% | "
          f"clean-direction-flip {100*d['frac_clean_direction_flip']:.1f}%")

    if eta == 20:
        # examples: 3 per category with rationales (truncated, prompts sanitized length)
        rng = np.random.default_rng(0)
        for name, mask in [("truly_corrupted", rej & ~ic),
                           ("clean_pair_unusable", rej & ic & ~keep_other),
                           ("clean_direction_flip", rej & ic & keep_other)]:
            idxs = np.flatnonzero(mask)
            for i in rng.choice(idxs, size=min(3, len(idxs)), replace=False):
                r = rows[int(i)]
                res = (pre[r["id"]]["verifier_results_a"] if r["user_choice"] == "A"
                       else pre[r["id"]]["verifier_results_b"])
                examples.append(dict(
                    category=name, id=r["id"],
                    prompt=r["prompt"][:220],
                    chosen=(r["response_a"] if r["user_choice"] == "A" else r["response_b"])[:220],
                    scores={k: v["score"] for k, v in res.items()},
                    rationales={k: v["rationale"][:150] for k, v in res.items()},
                ))

json.dump(summary, open(OUT / "e4_rejection.json", "w"), indent=1)
json.dump(examples, open(OUT / "e4_examples.json", "w"), indent=1)

# ------- figure: stacked composition bars -------
import matplotlib.pyplot as plt
cats = ["frac_truly_corrupted", "frac_clean_pair_unusable", "frac_clean_direction_flip"]
labels = ["truly corrupted", "clean, neither direction passes\n(pair unusable per verifiers)",
          "clean, other direction passes\n(verifiers disagree with label)"]
colors = ["#2a78d6", "#eb6834", "#1baf7a"]
fig, ax = plt.subplots(figsize=(5.2, 2.9))
ys = [f"$\\eta$=10%\n({summary['eta10']['n_rejected']} rejected)",
      f"$\\eta$=20%\n({summary['eta20']['n_rejected']} rejected)"]
left = np.zeros(2)
for c, lab, col in zip(cats, labels, colors):
    vals = np.array([summary["eta10"][c], summary["eta20"][c]]) * 100
    ax.barh(ys, vals, left=left, color=col, label=lab, height=0.55,
            edgecolor="#fcfcfb", linewidth=2)
    for y, (v, l) in enumerate(zip(vals, left)):
        if v > 7:
            ax.text(l + v / 2, y, f"{v:.1f}%", ha="center", va="center",
                    fontsize=8, color="#ffffff" if col != "#eda100" else TEXT_PRIMARY)
    left += vals
ax.set_xlim(0, 100)
ax.set_xlabel("share of consensus-rejected pairs (%)")
ax.set_title("Composition of the consensus rejection set (k=3 of 4)", color=TEXT_PRIMARY)
ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
ax.grid(axis="y", visible=False)
savefig(fig, "e4_rejection_composition")
plt.close(fig)

# markdown table
md = ["| Quantity | eta=10% | eta=20% |", "|---|---|---|"]
keys = [("Rejected pairs", "n_rejected", ""),
        ("Truly corrupted", "frac_truly_corrupted", "%"),
        ("Clean, pair unusable (both directions fail)", "frac_clean_pair_unusable", "%"),
        ("Clean, verifiers prefer other direction", "frac_clean_direction_flip", "%"),
        ("Clean, unanimous zero-pass", "frac_clean_unanimous_zero_pass", "%"),
        ("Recall on corrupted pairs", "recall_on_corrupted", "%"),
        ("Mean chosen-score, clean kept", "mean_chosen_score_clean_kept", ""),
        ("Mean chosen-score, clean rejected", "mean_chosen_score_clean_rejected", ""),
        ("Mean chosen-score, corrupted", "mean_chosen_score_corrupted", "")]
for label, k, pct in keys:
    v10, v20 = summary["eta10"][k], summary["eta20"][k]
    fmt = (lambda v: f"{100*v:.1f}%") if pct else (lambda v: f"{v}")
    md.append(f"| {label} | {fmt(v10)} | {fmt(v20)} |")
(TAB / "e4_rejection.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
print("\n[saved] outputs/e4_rejection.json, e4_examples.json, tables/e4_rejection.md, figure")
