"""P2 analysis: shared-backbone pool vs cross-backbone pool on the SAME 3000-pair
subsample (seed 42), same role prompts, same threshold, same corruption masks.

Compares: error correlation (shrunk R, rho_bar, n_eff + bootstrap CIs),
per-verifier error rates, and filter-side metrics (k=3-of-4 consensus) under
structured_unsafe eta in {10, 20}.
"""
import json
import random
from pathlib import Path

import numpy as np
from common import (OUT, TAB, ROLES, VSE, clean_mask, filter_metrics,
                    keep_consensus, load_corrupted, load_precomputed, savefig,
                    set_style, TEXT_PRIMARY, TEXT_SECONDARY)
from dependence import (bootstrap_correlation, compute_error_matrix,
                        correlation_shrunk, effective_eig_rank,
                        effective_size, mean_off_diagonal)

set_style()
DIVERSE_DIR = OUT / "diverse_pool"
N_ROWS, SEED = 3000, 42

# reproduce the subsample id set
rows_all = [json.loads(l) for l in open(VSE / "data/processed/hh_train.jsonl") if l.strip()]
rng = random.Random(SEED)
idx = sorted(rng.sample(range(len(rows_all)), N_ROWS))
sub_ids = {rows_all[i]["id"] for i in idx}

# diverse-pool verdicts: role -> id -> {'a': result, 'b': result}
diverse = {}
for role in ROLES:
    diverse[role] = {}
    for line in open(DIVERSE_DIR / f"{role}.jsonl"):
        r = json.loads(line)
        diverse[role][r["id"]] = r
    missing = sub_ids - set(diverse[role])
    assert not missing, f"{role}: {len(missing)} missing rows"

pre = load_precomputed()
results = {}

def pool_matrices(rows, pool):
    """(N,4) pass-votes for the chosen direction under a pool ('shared'|'diverse')."""
    V = np.zeros((len(rows), 4), dtype=int)
    S = np.zeros((len(rows), 4))
    for i, r in enumerate(rows):
        d = "a" if r["user_choice"] == "A" else "b"
        for j, role in enumerate(ROLES):
            if pool == "shared":
                res = pre[r["id"]][f"verifier_results_{d}"][role]
            else:
                res = diverse[role][r["id"]][f"result_{d}"]
            V[i, j] = int(res["passed"])
            S[i, j] = res["score"]
    return V, S

for eta in (10, 20):
    rows = [r for r in load_corrupted("structured_unsafe", eta) if r["id"] in sub_ids]
    ic = clean_mask(rows)
    gold = ic.astype(int)
    entry = {}
    for pool in ("shared", "diverse"):
        V, S = pool_matrices(rows, pool)
        M = np.ones_like(V)
        E, _ = compute_error_matrix(V, M, gold)
        R, _ = correlation_shrunk(E)
        boot = bootstrap_correlation(E, n_boot=1000, seed=42, shrunk=True)
        rho_d = np.array([mean_off_diagonal(s) for s in boot.samples])
        neff_d = np.array([effective_size(s) for s in boot.samples])
        per_v = {}
        for j, role in enumerate(ROLES):
            per_v[role] = dict(
                false_pass_on_corrupted=round(float(E[~ic, j].mean()), 4) if (~ic).any() else None,
                false_reject_on_clean=round(float(E[ic, j].mean()), 4),
            )
        m3 = filter_metrics((V.sum(axis=1) >= 3), ic)
        m2 = filter_metrics((V.sum(axis=1) >= 2), ic)
        entry[pool] = dict(
            rho_bar=round(mean_off_diagonal(R), 4),
            rho_bar_ci95=[round(float(np.quantile(rho_d, q)), 4) for q in (0.025, 0.975)],
            n_eff=round(effective_size(R), 3),
            n_eff_ci95=[round(float(np.quantile(neff_d, q)), 3) for q in (0.025, 0.975)],
            eig_rank=round(effective_eig_rank(R), 3),
            R=np.round(R, 3).tolist(),
            per_verifier=per_v,
            consensus_k3={k: round(v, 4) for k, v in m3.items()
                          if k in ("retention", "harmful_survival", "det_f1",
                                   "det_recall", "clean_retention")},
            consensus_k2={k: round(v, 4) for k, v in m2.items()
                          if k in ("retention", "harmful_survival", "det_f1",
                                   "det_recall", "clean_retention")},
        )
        print(f"eta={eta} {pool:8s}: rho_bar={entry[pool]['rho_bar']} "
              f"CI {entry[pool]['rho_bar_ci95']} n_eff={entry[pool]['n_eff']} | "
              f"k3: ret={entry[pool]['consensus_k3']['retention']:.3f} "
              f"harm={entry[pool]['consensus_k3']['harmful_survival']:.4f} "
              f"F1={entry[pool]['consensus_k3']['det_f1']:.3f} "
              f"recall={entry[pool]['consensus_k3']['det_recall']:.3f}")
    results[f"eta{eta}"] = entry

json.dump(results, open(OUT / "p2_diverse_pool.json", "w"), indent=1)

# ---- figure: paired heatmaps + rho comparison ----
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
blues = LinearSegmentedColormap.from_list("seqblue", ["#fcfcfb", "#cde2fb", "#9ec5f4",
                                                      "#6da7ec", "#3987e5", "#256abf",
                                                      "#184f95", "#0d366b"])
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
for ax, pool, title in [(axes[0], "shared", "Shared backbone (paper's pool)"),
                        (axes[1], "diverse", "Cross-backbone pool (new)")]:
    e = results["eta20"][pool]
    Rm = np.array(e["R"])
    im = ax.imshow(Rm, cmap=blues, vmin=0, vmax=1)
    ax.set_xticks(range(4), [r.capitalize() for r in ROLES], rotation=30, ha="right")
    if ax is axes[0]:
        ax.set_yticks(range(4), [r.capitalize() for r in ROLES])
    else:
        ax.set_yticks(range(4), [""] * 4)
    ax.set_title(f"{title}\n$\\bar\\rho$={e['rho_bar']:.2f}, $n_{{eff}}$={e['n_eff']:.1f} of 4",
                 color=TEXT_PRIMARY, fontsize=9)
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            v = Rm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if v > 0.55 else TEXT_PRIMARY)
fig.colorbar(im, ax=axes, shrink=0.85, label="error correlation")
fig.suptitle("Verifier failure correlation: shared vs cross-backbone pools "
             "($\\eta$=20%, same 3000 pairs)", fontsize=10, color=TEXT_PRIMARY)
savefig(fig, "p2_shared_vs_diverse_heatmap")
plt.close(fig)

# markdown summary
md = ["| eta | Pool | rho_bar [95% CI] | n_eff | Retention | Harm. surv. | Recall | Det. F1 |",
      "|---|---|---|---|---|---|---|---|"]
for eta in (10, 20):
    for pool in ("shared", "diverse"):
        e = results[f"eta{eta}"][pool]
        c = e["consensus_k3"]
        md.append(f"| {eta}% | {pool} | {e['rho_bar']:.3f} [{e['rho_bar_ci95'][0]:.3f}, "
                  f"{e['rho_bar_ci95'][1]:.3f}] | {e['n_eff']:.2f} | {c['retention']:.3f} | "
                  f"{100*c['harmful_survival']:.2f}% | {c['det_recall']:.3f} | {c['det_f1']:.3f} |")
(TAB / "p2_diverse.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
