"""E5: filter-side behaviour across the three corruption regimes.

All from cached both-direction verdicts (zero GPU): for each regime x eta,
recompute which direction is 'chosen' after corruption, look up that direction's
verdicts, and apply the same four admission rules.
Also reports detection recall on corrupted pairs, which is the regime-sensitive
quantity (structured flips put a low-scoring response in the chosen slot;
random flips do so only half the time... on HH both directions matter).
"""
import json

import numpy as np
from common import (OUT, TAB, clean_mask, filter_metrics, keep_consensus,
                    keep_single, load_corrupted, load_precomputed, savefig,
                    score_matrix, set_style, TEXT_PRIMARY, TEXT_SECONDARY)

set_style()
pre = load_precomputed()
REGIMES = ["structured_unsafe", "random_flip", "coordinated_poisoning"]
# NOTE (verified): on HH as loaded every pair is oriented user_choice='A', so
# random_flip and structured_unsafe realize the IDENTICAL corrupted dataset
# (same RNG stream, same eligible set); they differ only in metadata. On a
# quality-ordered pair dataset any label flip is directional. The realizable
# contrast is therefore broad (uniform) vs targeted (coordinated) corruption.
REGIME_LABEL = {"structured_unsafe": "Broad flips (uniform over pairs)",
                "random_flip": "Random flip (== broad on HH; see note)",
                "coordinated_poisoning": "Coordinated poisoning (targeted)"}
PLOT_REGIMES = ["structured_unsafe", "coordinated_poisoning"]

res = {}
for reg in REGIMES:
    for eta in (10, 20):
        rows = load_corrupted(reg, eta)
        scores = score_matrix(rows, pre)
        ic = clean_mask(rows)
        entry = {"effective_corruption": round(float((~ic).mean()), 4),
                 "n_corrupted": int((~ic).sum())}
        for method, keep in [
            ("raw", np.ones(len(rows), dtype=bool)),
            ("single", keep_single(scores)),
            ("consensus_k3", keep_consensus(scores, k=3)),
        ]:
            m = filter_metrics(keep, ic)
            entry[method] = {k: round(v, 4) for k, v in m.items()
                             if k in ("retention", "harmful_survival",
                                      "det_f1", "det_recall", "clean_retention")}
        res[f"{reg}_eta{eta}"] = entry
        c = entry["consensus_k3"]
        print(f"{reg:22s} eta={eta}: eff_corr={entry['effective_corruption']:.4f} "
              f"consensus: retention={c['retention']:.3f} harm={c['harmful_survival']:.4f} "
              f"recall={c['det_recall']:.3f} F1={c['det_f1']:.3f}")

json.dump(res, open(OUT / "e5_regimes.json", "w"), indent=1)

# relative contamination reduction: (raw_harm - cons_harm)/raw_harm
md = ["| Regime | eta | Eff. corr. | Raw harm | Cons. harm | Rel. reduction | Recall | Det. F1 |",
      "|---|---|---|---|---|---|---|---|"]
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5.4, 3.0))
colors = {"structured_unsafe": "#2a78d6", "random_flip": "#eb6834",
          "coordinated_poisoning": "#1baf7a"}
x = np.arange(2)
w = 0.32
for i, reg in enumerate(PLOT_REGIMES):
    reds, recs = [], []
    for eta in (10, 20):
        e = res[f"{reg}_eta{eta}"]
        raw_h = e["raw"]["harmful_survival"]
        con_h = e["consensus_k3"]["harmful_survival"]
        rel = (raw_h - con_h) / raw_h if raw_h > 0 else 0.0
        reds.append(100 * rel)
        recs.append(e["consensus_k3"]["det_recall"])
        md.append(f"| {REGIME_LABEL[reg]} | {eta}% | {100*e['effective_corruption']:.2f}% | "
                  f"{100*raw_h:.2f}% | {100*con_h:.2f}% | {100*rel:.1f}% | "
                  f"{e['consensus_k3']['det_recall']:.3f} | {e['consensus_k3']['det_f1']:.3f} |")
    ax.bar(x + (i - 0.5) * w, reds, w, color=colors[reg], label=REGIME_LABEL[reg],
           edgecolor="#fcfcfb", linewidth=1.5)
    for xi, v in zip(x + (i - 0.5) * w, reds):
        ax.text(xi, v + 0.6, f"{v:.0f}%", ha="center", fontsize=8, color=TEXT_PRIMARY)
ax.set_xticks(x, ["$\\eta$ = 10%", "$\\eta$ = 20%"])
ax.set_ylabel("relative reduction in contamination (%)\n(consensus k=3 vs raw)")
ax.set_title("Consensus filtering vs corruption structure", color=TEXT_PRIMARY)
ax.legend(fontsize=8)
ax.grid(axis="x", visible=False)
savefig(fig, "e5_regime_comparison")
plt.close(fig)

(TAB / "e5_regimes.md").write_text("\n".join(md) + "\n")
print()
print("\n".join(md))
