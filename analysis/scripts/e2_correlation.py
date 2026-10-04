"""E2: verifier failure-correlation analysis on cached verdicts.

Error definition (matches the paper's analyze_verifier_correlation.py):
a verifier errs on a pair if it approves a corrupted pair (false pass) or
rejects a clean pair (false reject). All computed on the chosen direction of
the structured_unsafe corrupted sets.
"""
import json

import numpy as np
from common import (FIG, OUT, ROLES, SEQ_BLUES, TEXT_PRIMARY, TEXT_SECONDARY,
                    clean_mask, filter_metrics, keep_consensus, load_corrupted,
                    load_precomputed, savefig, score_matrix, set_style)

from dependence import (bootstrap_correlation, correlation_pearson,
                        correlation_shrunk, compute_error_matrix,
                        effective_eig_rank, effective_size,
                        mean_off_diagonal, neff_curve, corrfilter_score)

set_style()
pre = load_precomputed()
results = {}

for eta in (10, 20):
    rows = load_corrupted("structured_unsafe", eta)
    scores = score_matrix(rows, pre)
    ic = clean_mask(rows)
    V = (scores >= 0.7).astype(int)          # vote: 1 = pass/admit
    M = np.ones_like(V)
    gold = ic.astype(int)                    # verifier should pass clean, fail corrupted
    E, _ = compute_error_matrix(V, M, gold)

    phi = correlation_pearson(E)
    R_shrunk, shrink = correlation_shrunk(E)
    boot = bootstrap_correlation(E, n_boot=1000, seed=42, shrunk=True)
    rho_draws = np.array([mean_off_diagonal(s) for s in boot.samples])
    neff_draws = np.array([effective_size(s) for s in boot.samples])

    # per-verifier error decomposition
    per_verifier = {}
    for j, role in enumerate(ROLES):
        fp = float(E[~ic.astype(bool)[: len(E)], j].mean()) if (~ic).any() else 0.0
        fn = float(E[ic.astype(bool)[: len(E)], j].mean())
        per_verifier[role] = dict(
            false_pass_on_corrupted=round(fp, 4),
            false_reject_on_clean=round(fn, 4),
            total_error=round(float(E[:, j].mean()), 4),
            pass_rate=round(float(V[:, j].mean()), 4),
        )

    # leave-one-out: remove one verifier, run k=2-of-3 (majority) and k=3-of-3
    loo = {}
    for j, role in enumerate(ROLES):
        keep_idx = [i for i in range(4) if i != j]
        sub = scores[:, keep_idx]
        trio_E = E[:, keep_idx]
        entry = {"rho_bar_trio": round(mean_off_diagonal(correlation_shrunk(trio_E)[0]), 4)}
        for k in (2, 3):
            keep = (sub >= 0.7).sum(axis=1) >= k
            m = filter_metrics(keep, ic)
            entry[f"k{k}of3"] = {kk: round(vv, 4) for kk, vv in m.items()
                                 if kk in ("retention", "harmful_survival", "det_f1")}
        loo[role] = entry

    # correlation-adjusted consensus (alpha_subset), retention-matched to k=3-of-4
    cf = corrfilter_score(V, M, R_shrunk, np.ones(len(V), dtype=int))
    alpha = np.where(np.isnan(cf.score), -np.inf, cf.score)
    kc = keep_consensus(scores, k=3)
    n_keep = int(kc.sum())
    thr = np.sort(alpha)[::-1][n_keep - 1]
    keep_alpha = alpha >= thr
    m_alpha = filter_metrics(keep_alpha, ic)
    m_cons = filter_metrics(kc, ic)

    results[f"eta{eta}"] = dict(
        n=len(rows),
        phi_mean_offdiag=round(mean_off_diagonal(phi), 4),
        phi_matrix=np.round(phi, 3).tolist(),
        shrunk_matrix=np.round(R_shrunk, 3).tolist(),
        shrinkage=round(float(shrink), 4),
        rho_bar=round(mean_off_diagonal(R_shrunk), 4),
        rho_bar_ci95=[round(float(np.quantile(rho_draws, q)), 4) for q in (0.025, 0.975)],
        n_eff=round(effective_size(R_shrunk), 3),
        n_eff_ci95=[round(float(np.quantile(neff_draws, q)), 3) for q in (0.025, 0.975)],
        eig_rank=round(effective_eig_rank(R_shrunk), 3),
        eigenvalues=np.round(np.linalg.eigvalsh(R_shrunk)[::-1], 3).tolist(),
        rho_proxy_paper_style=round(4 / effective_size(R_shrunk), 3),
        per_verifier=per_verifier,
        leave_one_out=loo,
        alpha_subset_vs_consensus=dict(
            matched_retention=round(m_alpha["retention"], 4),
            consensus=dict(harmful_survival=round(m_cons["harmful_survival"], 4),
                           det_f1=round(m_cons["det_f1"], 4)),
            alpha_subset=dict(harmful_survival=round(m_alpha["harmful_survival"], 4),
                              det_f1=round(m_alpha["det_f1"], 4)),
        ),
    )
    print(f"eta={eta}: phi_mean={results[f'eta{eta}']['phi_mean_offdiag']}, "
          f"rho_bar={results[f'eta{eta}']['rho_bar']} CI {results[f'eta{eta}']['rho_bar_ci95']}, "
          f"n_eff={results[f'eta{eta}']['n_eff']} CI {results[f'eta{eta}']['n_eff_ci95']}, "
          f"eig_rank={results[f'eta{eta}']['eig_rank']}")

json.dump(results, open(OUT / "e2_correlation.json", "w"), indent=1)

# ---------------- figures ----------------
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

blues = LinearSegmentedColormap.from_list("seqblue", ["#fcfcfb"] + SEQ_BLUES)
r20 = np.array(results["eta20"]["shrunk_matrix"])
phi20 = np.array(results["eta20"]["phi_matrix"])

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
for ax, Rm, title in [(axes[0], phi20, "Pearson $\\phi$ (error indicators)"),
                      (axes[1], r20, "Ledoit-Wolf shrunk $\\hat{R}$")]:
    im = ax.imshow(Rm, cmap=blues, vmin=0, vmax=1)
    ax.set_xticks(range(4), [r.capitalize() for r in ROLES], rotation=30, ha="right")
    if ax is axes[0]:
        ax.set_yticks(range(4), [r.capitalize() for r in ROLES])
    else:
        ax.set_yticks(range(4), [""] * 4)
    ax.set_title(title, color=TEXT_PRIMARY)
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            v = Rm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if v > 0.55 else TEXT_PRIMARY)
fig.colorbar(im, ax=axes, shrink=0.85, label="error correlation")
fig.suptitle(f"Shared-backbone verifier pool, $\\eta$=20% "
             f"($\\bar\\rho$={results['eta20']['rho_bar']:.2f}, "
             f"$n_{{eff}}$={results['eta20']['n_eff']:.1f} of 4)",
             fontsize=10, color=TEXT_PRIMARY)
savefig(fig, "e2_correlation_heatmap")
plt.close(fig)

# n_eff saturation curve at measured rho_bar
fig, ax = plt.subplots(figsize=(3.6, 3.0))
rho = results["eta20"]["rho_bar"]
curve = neff_curve(rho, max_n=16)
ax.plot(range(1, 17), curve, color="#2a78d6")
ax.plot(range(1, 17), range(1, 17), color=TEXT_SECONDARY, linewidth=1.2,
        linestyle="--")
ax.annotate("independent ($n_{eff}=n$)", xy=(9, 9), xytext=(5.2, 12.6),
            fontsize=8, color=TEXT_SECONDARY)
ax.annotate(f"measured $\\bar\\rho$={rho:.2f}\nceiling $1/\\bar\\rho$={1/rho:.1f}",
            xy=(12, curve[11]), xytext=(8.5, 4.6), fontsize=8, color="#2a78d6")
ax.axhline(1 / rho, color="#2a78d6", linewidth=1.0, linestyle=":")
ax.scatter([4], [results["eta20"]["n_eff"]], s=42, color="#eb6834", zorder=5)
ax.annotate("pilot pool (n=4)", xy=(4, results["eta20"]["n_eff"]),
            xytext=(4.6, 1.4), fontsize=8, color="#eb6834")
ax.set_xlabel("nominal verifiers $n$")
ax.set_ylabel("effective independent verifiers $n_{eff}$")
ax.set_title("Correlation caps the value of adding verifiers", color=TEXT_PRIMARY)
savefig(fig, "e2_neff_saturation")
plt.close(fig)

print("\n[done] outputs/e2_correlation.json + 2 figures")
