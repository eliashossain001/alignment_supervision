"""E3: k-of-n and score-threshold sweep from cached continuous scores.

Produces the retention vs harmful-survival trade-off frontier per eta, marks the
paper's operating point (k=3, t=0.7), and extracts Pareto-optimal points.
"""
import json

import numpy as np
from common import (OUT, TAB, TEXT_PRIMARY, TEXT_SECONDARY, clean_mask,
                    filter_metrics, load_corrupted, load_precomputed, savefig,
                    score_matrix, set_style)

set_style()
pre = load_precomputed()

THRESHOLDS = np.round(np.arange(0.40, 0.91, 0.05), 2)
KS = [1, 2, 3, 4]
K_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}

all_records = []
for eta in (10, 20):
    rows = load_corrupted("structured_unsafe", eta)
    scores = score_matrix(rows, pre)
    ic = clean_mask(rows)
    for k in KS:
        for t in THRESHOLDS:
            keep = (scores >= t).sum(axis=1) >= k
            m = filter_metrics(keep, ic)
            m.update(eta=eta, k=k, threshold=float(t))
            all_records.append(m)

json.dump(all_records, open(OUT / "e3_sweep.json", "w"), indent=1)

# Pareto frontier: maximize retention, minimize harmful survival
def pareto(recs):
    pts = sorted(recs, key=lambda r: (-r["retention"], r["harmful_survival"]))
    frontier, best_harm = [], np.inf
    for r in sorted(recs, key=lambda r: -r["retention"]):
        if r["harmful_survival"] < best_harm - 1e-12:
            frontier.append(r)
            best_harm = r["harmful_survival"]
    return frontier

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
for ax, eta in zip(axes, (10, 20)):
    recs = [r for r in all_records if r["eta"] == eta]
    raw_harm = eta / 100 * (0.986 if eta == 10 else 1.0155)  # effective rates
    raw_harm = {10: 0.0986, 20: 0.2031}[eta]
    for k in KS:
        rk = sorted((r for r in recs if r["k"] == k), key=lambda r: r["threshold"])
        ax.plot([r["retention"] for r in rk], [100 * r["harmful_survival"] for r in rk],
                color=K_COLORS[k], marker="o", markersize=3.5, linewidth=1.6,
                label=f"k={k} of 4")
    ax.axhline(100 * raw_harm, color=TEXT_SECONDARY, linestyle="--", linewidth=1.1)
    ax.annotate("Raw DPO (no filter)", xy=(0.05, 100 * raw_harm), fontsize=8,
                color=TEXT_SECONDARY, va="bottom")
    # paper operating point: k=3, t=0.7
    p = next(r for r in recs if r["k"] == 3 and abs(r["threshold"] - 0.7) < 1e-9)
    ax.scatter([p["retention"]], [100 * p["harmful_survival"]], s=70,
               facecolor="none", edgecolor=TEXT_PRIMARY, linewidth=1.4, zorder=6)
    ax.annotate("paper's operating\npoint (k=3, t=0.70)",
                xy=(p["retention"], 100 * p["harmful_survival"]),
                xytext=(0.04, 100 * p["harmful_survival"] + (0.55 if eta == 10 else 1.1)),
                fontsize=8, color=TEXT_PRIMARY,
                arrowprops=dict(arrowstyle="-", color=TEXT_SECONDARY, lw=0.8))
    ax.set_xlabel("retention")
    ax.set_ylabel("harmful survival (%)")
    ax.set_title(f"$\\eta$ = {eta}%", color=TEXT_PRIMARY)
    ax.set_xlim(0, 1.02)
axes[1].legend(loc="lower right", fontsize=8, title="admission rule", title_fontsize=8)
fig.suptitle("Retention vs harmful survival across k and score threshold "
             "(each curve: t = 0.40 ... 0.90)", fontsize=10, color=TEXT_PRIMARY)
savefig(fig, "e3_retention_vs_survival")
plt.close(fig)

# print frontier + recommended operating points
md = ["| eta | k | t | Retention | Harm. surv. | Det. F1 |", "|---|---|---|---|---|---|"]
for eta in (10, 20):
    recs = [r for r in all_records if r["eta"] == eta]
    front = pareto(recs)
    print(f"\n== Pareto frontier eta={eta}% (retention up, harm down) ==")
    for r in front:
        line = (f"k={r['k']} t={r['threshold']:.2f} retention={r['retention']:.3f} "
                f"harm={100*r['harmful_survival']:.2f}% F1={r['det_f1']:.3f}")
        print(line)
        md.append(f"| {eta}% | {r['k']} | {r['threshold']:.2f} | {r['retention']:.3f} | "
                  f"{100*r['harmful_survival']:.2f}% | {r['det_f1']:.3f} |")
    # comparison: points dominating the paper's choice
    p = next(r for r in recs if r["k"] == 3 and abs(r["threshold"] - 0.7) < 1e-9)
    dom = [r for r in recs if r["retention"] > p["retention"] + 1e-9
           and r["harmful_survival"] <= p["harmful_survival"] + 1e-9]
    dom = sorted(dom, key=lambda r: -r["retention"])[:5]
    print(f"-- points dominating paper's (k=3,t=0.7): retention={p['retention']:.3f}, "
          f"harm={100*p['harmful_survival']:.2f}%")
    for r in dom:
        print(f"   k={r['k']} t={r['threshold']:.2f} retention={r['retention']:.3f} "
              f"harm={100*r['harmful_survival']:.2f}% F1={r['det_f1']:.3f}")

(TAB / "e3_pareto.md").write_text("\n".join(md) + "\n")
print("\n[saved] outputs/e3_sweep.json, tables/e3_pareto.md, figures/e3_retention_vs_survival")
