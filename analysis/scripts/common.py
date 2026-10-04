"""Shared loaders + metrics for the extended analyses.

All analyses run over the cached both-direction verifier verdicts
(results/verifier_outputs/hh_train.precomputed.jsonl) so nothing here needs a GPU.
Metric definitions match the pilot pipeline (validated against
results/tables/main_results.csv in e1_validate.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VSE = ROOT  # flattened layout: data/ and results/ live at repo root
ANA = ROOT / "analysis"
FIG = ANA / "figures"
TAB = ANA / "tables"
OUT = ANA / "outputs"
for d in (FIG, TAB, OUT):
    d.mkdir(parents=True, exist_ok=True)

ROLES = ["safety", "helpfulness", "factuality", "policy"]

# --- plotting style (light mode, print-quality) -----------------------------
PALETTE = {  # fixed categorical order, one hue per method everywhere
    "raw": "#2a78d6",        # blue
    "noise_aware": "#eb6834",  # orange
    "single": "#1baf7a",     # aqua
    "consensus_k3": "#eda100",  # yellow
    "oracle": "#e87ba4",     # magenta
}
METHOD_LABEL = {
    "raw": "Raw DPO",
    "noise_aware": "cDPO (label smoothing 0.1)",
    "single": "Single verifier (safety)",
    "consensus_k3": "Consensus (k=3 of 4)",
    "oracle": "Oracle (ground truth)",
}
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TEXT_PRIMARY, TEXT_SECONDARY, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": TEXT_SECONDARY, "axes.labelcolor": TEXT_PRIMARY,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
        "axes.grid": True, "axes.axisbelow": True,
        "legend.frameon": False, "lines.linewidth": 2.0,
    })


def savefig(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    print(f"[fig] {FIG / name}.pdf/.png")


# --- data -------------------------------------------------------------------

def load_precomputed() -> dict:
    """id -> row with verifier_results_a / verifier_results_b."""
    pre = {}
    with open(VSE / "results/verifier_outputs/hh_train.precomputed.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                pre[r["id"]] = r
    return pre


def load_corrupted(regime: str, eta: int):
    """Rows of the corrupted training file for (regime, eta)."""
    path = VSE / f"data/corrupted/hh_train_{regime}_eta{eta}.jsonl"
    return [json.loads(l) for l in open(path) if l.strip()]


def chosen_results(row: dict, pre: dict) -> dict:
    """Verifier verdicts for the row's current chosen direction."""
    p = pre[row["id"]]
    return p["verifier_results_a"] if row["user_choice"] == "A" else p["verifier_results_b"]


def other_results(row: dict, pre: dict) -> dict:
    p = pre[row["id"]]
    return p["verifier_results_b"] if row["user_choice"] == "A" else p["verifier_results_a"]


def score_matrix(rows, pre, direction="chosen") -> np.ndarray:
    """(N, 4) continuous scores for the chosen (or other) direction."""
    fn = chosen_results if direction == "chosen" else other_results
    return np.array([[fn(r, pre)[role]["score"] for role in ROLES] for r in rows])


def clean_mask(rows) -> np.ndarray:
    return np.array([bool(r.get("is_clean", True)) for r in rows])


# --- filter metrics (paper definitions) -------------------------------------

def filter_metrics(keep: np.ndarray, is_clean: np.ndarray) -> dict:
    """keep: bool (N,) admitted; is_clean: bool (N,). Matches aggregate_results.py."""
    N = len(keep)
    kept = keep.sum()
    corrupted = ~is_clean
    n_corrupt = corrupted.sum()
    retention = kept / N
    harmful_survival = (keep & corrupted).sum() / kept if kept else 0.0
    clean_retention = (keep & is_clean).sum() / is_clean.sum()
    false_rejection = (~keep & is_clean).sum() / is_clean.sum()
    # detection: "reject" predicts "corrupted"
    tp = (~keep & corrupted).sum()
    fp = (~keep & is_clean).sum()
    fn = (keep & corrupted).sum()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(retention=retention, harmful_survival=harmful_survival,
                clean_retention=clean_retention, false_rejection=false_rejection,
                det_precision=prec, det_recall=rec, det_f1=f1,
                n=N, n_kept=int(kept), n_corrupted=int(n_corrupt))


def keep_consensus(scores: np.ndarray, k: int, threshold: float = 0.7) -> np.ndarray:
    return (scores >= threshold).sum(axis=1) >= k


def keep_single(scores: np.ndarray, role: str = "safety", threshold: float = 0.7) -> np.ndarray:
    return scores[:, ROLES.index(role)] >= threshold
