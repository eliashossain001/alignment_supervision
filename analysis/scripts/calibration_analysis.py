#!/usr/bin/env python3
"""Calibration and matched-operating-point analysis.

This script addresses two concerns:

  (i)  applying one fixed score threshold of 0.7 to heterogeneous verifier
       models is not a calibrated comparison;
  (ii) the cross-backbone ("diverse") pool may have improved coverage rather
       than verification reliability.

This script answers both. It calibrates every verifier against the ground-truth
corruption label with disjoint fit and evaluation splits, then compares the
shared-backbone pool, the cross-backbone pool, and the best single verifier at
MATCHED retention and at MATCHED reliability, with paired bootstrap confidence
intervals. It finally prints which of five pre-registered conclusions the
computed intervals support.

Every reported number is computed from the input files. Nothing is hard coded.

Typical use:

  python analysis/scripts/calibration_analysis.py \
      --precomputed results/verifier_outputs/hh_train.precomputed.jsonl \
      --eta 10 20

Shard files may be passed instead of (or in addition to) the merged file:

  python analysis/scripts/calibration_analysis.py \
      --precomputed results/verifier_outputs/hh_train.precomputed.shard0.jsonl \
                    results/verifier_outputs/hh_train.precomputed.shard1.jsonl \
      --eta 20 --outdir /tmp/smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required for headless runs
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

ROLES = ["safety", "helpfulness", "factuality", "policy"]
PAPER_THRESHOLD = 0.7
PAPER_K = 3
N_POOL = 4
SPLIT_SEED = 20260731
BOOT_SEED = 42
N_BOOT = 1000
N_ECE_BINS = 10
PARSE_FAILURE_SCORE = 0.5  # src/precompute_verifiers.py:65 on a parse exception

# Repository house palette (analysis/scripts/common.py). Fixed hue order, one
# hue per admission rule, never cycled. Line style and marker carry the same
# identity as a secondary encoding so the series remain separable without color.
RULE_STYLE = {
    "shared_pool": dict(color="#2a78d6", ls="-", marker="o", label="Shared-backbone pool (k of 4)"),
    "diverse_pool": dict(color="#eb6834", ls="--", marker="s", label="Cross-backbone pool (k of 4)"),
    "best_single": dict(color="#1baf7a", ls=":", marker="^", label="Best single verifier"),
}
METHOD_STYLE = {
    "raw": dict(color="#2a78d6", ls="-", marker="o", label="Raw score (1 - score)"),
    "platt": dict(color="#eb6834", ls="--", marker="s", label="Platt scaling"),
    "isotonic": dict(color="#1baf7a", ls=":", marker="^", label="Isotonic regression"),
}
TEXT_PRIMARY, TEXT_SECONDARY, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def set_style() -> None:
    matplotlib.rcParams.update({
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


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def read_jsonl(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A shard still being written can end in a truncated line.
                continue
    return rows


def load_precomputed(paths) -> dict:
    """id -> {verifier_results_a, verifier_results_b} for the shared pool."""
    pre = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise SystemExit(f"[fatal] precomputed file not found: {p}")
        n_before = len(pre)
        for r in read_jsonl(p):
            if "verifier_results_a" in r and "verifier_results_b" in r:
                pre[r["id"]] = r
        print(f"[load] {p.name}: {len(pre) - n_before} new ids (total {len(pre)})")
    if not pre:
        raise SystemExit("[fatal] no usable rows in the precomputed input")
    return pre


def load_corrupted(corrupted_dir: Path, regime: str, eta: int):
    path = corrupted_dir / f"hh_train_{regime}_eta{eta}.jsonl"
    if not path.exists():
        raise SystemExit(f"[fatal] corrupted file not found: {path}")
    return read_jsonl(path)


def load_diverse(diverse_dir: Path):
    """role -> {id: {result_a, result_b}}, only for roles that exist on disk."""
    out = {}
    if not diverse_dir.exists():
        return out
    for role in ROLES:
        p = diverse_dir / f"{role}.jsonl"
        if not p.exists():
            continue
        per_id = {r["id"]: r for r in read_jsonl(p) if "result_a" in r and "result_b" in r}
        if per_id:
            out[role] = per_id
    return out


# --------------------------------------------------------------------------
# Frame construction
# --------------------------------------------------------------------------

def build_frame(corrupt_rows, pre, diverse):
    """Join corruption labels to verifier scores on the recorded direction.

    Returns a dict of aligned numpy arrays. The score used for a pair is the
    score of the direction the recorded preference points at, that is
    verifier_results_a when user_choice == "A" and verifier_results_b
    otherwise. A LOW score is evidence of corruption, so the predictor of the
    positive class ("corrupted") is the NEGATED score.
    """
    ids, corrupted, choice = [], [], []
    shared_chosen, shared_other = [], []
    div_roles = sorted(diverse.keys(), key=ROLES.index)
    div_chosen = []
    have_full_diverse = len(div_roles) == N_POOL

    for r in corrupt_rows:
        rid = r["id"]
        p = pre.get(rid)
        if p is None:
            continue
        if have_full_diverse and not all(rid in diverse[role] for role in div_roles):
            in_div = False
        else:
            in_div = have_full_diverse
        a_side = r["user_choice"] == "A"
        vr = p["verifier_results_a"] if a_side else p["verifier_results_b"]
        vo = p["verifier_results_b"] if a_side else p["verifier_results_a"]
        ids.append(rid)
        corrupted.append(not bool(r.get("is_clean", True)))
        choice.append(r["user_choice"])
        shared_chosen.append([float(vr[role]["score"]) for role in ROLES])
        shared_other.append([float(vo[role]["score"]) for role in ROLES])
        if in_div:
            key = "result_a" if a_side else "result_b"
            div_chosen.append([float(diverse[role][rid][key]["score"]) for role in ROLES])
        else:
            div_chosen.append([np.nan] * N_POOL)

    frame = dict(
        ids=np.array(ids),
        corrupted=np.array(corrupted, dtype=bool),
        choice=np.array(choice),
        shared=np.array(shared_chosen, dtype=float),
        shared_other=np.array(shared_other, dtype=float),
        diverse=np.array(div_chosen, dtype=float),
        diverse_roles=div_roles,
        diverse_complete=have_full_diverse,
    )
    frame["diverse_mask"] = np.isfinite(frame["diverse"]).all(axis=1)
    return frame


def stratified_split(labels: np.ndarray, seed: int):
    """Disjoint 50/50 calibration and evaluation index sets, stratified."""
    rng = np.random.default_rng(seed)
    calib = np.zeros(len(labels), dtype=bool)
    for value in (False, True):
        idx = np.flatnonzero(labels == value)
        rng.shuffle(idx)
        calib[idx[: len(idx) // 2]] = True
    return calib, ~calib


def sha256_of_ids(ids) -> str:
    payload = "\n".join(sorted(map(str, ids))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# Calibration diagnostics
# --------------------------------------------------------------------------

def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y.astype(float)) ** 2))


def ece_equal_width(y: np.ndarray, p: np.ndarray, n_bins: int = N_ECE_BINS):
    """Expected calibration error with n_bins EQUAL-WIDTH bins on [0, 1].

    Bin b covers [b/n_bins, (b+1)/n_bins), the last bin closes at 1.0. Empty
    bins contribute nothing. Returns (ece, per-bin table).
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    total = len(p)
    ece, table = 0.0, []
    for b in range(n_bins):
        m = idx == b
        n_b = int(m.sum())
        if n_b == 0:
            table.append(dict(bin=b, lo=edges[b], hi=edges[b + 1], n=0,
                              mean_pred=np.nan, frac_pos=np.nan))
            continue
        mean_pred = float(p[m].mean())
        frac_pos = float(y[m].mean())
        ece += (n_b / total) * abs(mean_pred - frac_pos)
        table.append(dict(bin=b, lo=edges[b], hi=edges[b + 1], n=n_b,
                          mean_pred=mean_pred, frac_pos=frac_pos))
    return float(ece), table


def operating_point(reject: np.ndarray, corrupted: np.ndarray) -> dict:
    """Detection metrics for the decision "reject == predict corrupted"."""
    clean = ~corrupted
    n_rej = int(reject.sum())
    n_corrupt = int(corrupted.sum())
    n_clean = int(clean.sum())
    tp = int((reject & corrupted).sum())
    fp = int((reject & clean).sum())
    fn = int((~reject & corrupted).sum())
    prec = tp / n_rej if n_rej else 0.0
    rec = tp / n_corrupt if n_corrupt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(precision=prec, recall=rec, f1=f1,
                fpr=fp / n_clean if n_clean else 0.0,
                fnr=fn / n_corrupt if n_corrupt else 0.0,
                n_rejected=n_rej, n_corrupt=n_corrupt, n_clean=n_clean)


def best_f1_threshold(scores: np.ndarray, corrupted: np.ndarray):
    """Threshold t maximising detection F1 when the rule is "reject if s < t".

    Candidates are the distinct observed scores plus a value above the maximum,
    which covers every distinct decision the rule can make.
    """
    cands = np.unique(np.round(scores, 8))
    cands = np.concatenate([cands, [cands[-1] + 1e-6]])
    best_t, best_f1 = float(cands[0]), -1.0
    for t in cands:
        op = operating_point(scores < t, corrupted)
        if op["f1"] > best_f1:
            best_t, best_f1 = float(t), op["f1"]
    return best_t, best_f1


def fit_calibrators(s_cal: np.ndarray, y_cal: np.ndarray):
    """Fit Platt scaling and isotonic regression on the CALIBRATION split only.

    Both map a raw verifier score to P(corrupted). Temperature scaling is NOT
    applicable here and is not attempted: the scores are floats parsed out of
    verifier-generated text, not logits of a classifier head, so there is no
    pre-softmax quantity to divide by a temperature.
    """
    out = {}
    if len(np.unique(y_cal)) < 2 or len(np.unique(s_cal)) < 2:
        return out
    lr = LogisticRegression(solver="lbfgs", C=1e6, max_iter=1000)
    lr.fit(s_cal.reshape(-1, 1), y_cal.astype(int))
    out["platt"] = lambda s: lr.predict_proba(s.reshape(-1, 1))[:, 1]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing="auto",
                            out_of_bounds="clip")
    iso.fit(s_cal, y_cal.astype(float))
    out["isotonic"] = lambda s: np.asarray(iso.predict(s), dtype=float)
    out["_iso_increasing"] = bool(iso.increasing_)
    out["_platt_coef"] = float(lr.coef_[0][0])
    out["_platt_intercept"] = float(lr.intercept_[0])
    return out


def per_verifier_diagnostics(frame, calib, evalm, eta, role_idx, role, pool_name,
                             scores_all):
    """All single-verifier diagnostics for one role, one eta, one pool."""
    s = scores_all[:, role_idx]
    y = frame["corrupted"]
    s_cal, y_cal = s[calib], y[calib]
    s_ev, y_ev = s[evalm], y[evalm]
    rows = []
    if len(np.unique(y_ev)) < 2:
        print(f"[warn] {pool_name}/{role}/eta{eta}: evaluation split has a single "
              f"class, discrimination metrics are undefined and are reported as NaN")

    def auc_pair(y_, s_):
        if len(np.unique(y_)) < 2:
            return float("nan"), float("nan")
        return (float(roc_auc_score(y_, -s_)),
                float(average_precision_score(y_, -s_)))

    auroc, auprc = auc_pair(y_ev, s_ev)

    cals = fit_calibrators(s_cal, y_cal)
    probs = {"raw": 1.0 - s_ev}
    if "platt" in cals:
        probs["platt"] = cals["platt"](s_ev)
        probs["isotonic"] = cals["isotonic"](s_ev)

    # Paper operating point: pass iff score >= 0.7, so reject iff score < 0.7.
    op_paper = operating_point(s_ev < PAPER_THRESHOLD, y_ev)
    # Calibrated threshold: chosen on the calibration split only.
    t_star, f1_cal = best_f1_threshold(s_cal, y_cal)
    op_star = operating_point(s_ev < t_star, y_ev)

    for method, p in probs.items():
        b = brier(y_ev, p)
        e, _ = ece_equal_width(y_ev, p)
        if method == "raw":
            m_t_star, m_op_star = t_star, op_star
        else:
            # Threshold on the calibrated probability of corruption, again
            # selected on the calibration split only.
            p_cal = cals[method](s_cal)
            cands = np.unique(np.round(p_cal, 8))
            best_t, best_f1 = float(cands[0]), -1.0
            for t in np.concatenate([cands, [cands[-1] + 1e-6]]):
                op = operating_point(p_cal >= t, y_cal)
                if op["f1"] > best_f1:
                    best_t, best_f1 = float(t), op["f1"]
            m_t_star = best_t
            m_op_star = operating_point(cals[method](s_ev) >= best_t, y_ev)
        rows.append(dict(
            pool=pool_name, eta=eta, role=role, method=method,
            n_calib=int(calib.sum()), n_eval=int(evalm.sum()),
            n_eval_corrupt=int(y_ev.sum()),
            auroc=auroc, auprc=auprc, brier=b, ece=e,
            brier_raw=brier(y_ev, probs["raw"]),
            ece_raw=ece_equal_width(y_ev, probs["raw"])[0],
            op070_precision=op_paper["precision"], op070_recall=op_paper["recall"],
            op070_f1=op_paper["f1"], op070_fpr=op_paper["fpr"],
            op070_fnr=op_paper["fnr"],
            cal_threshold=m_t_star,
            cal_precision=m_op_star["precision"], cal_recall=m_op_star["recall"],
            cal_f1=m_op_star["f1"], cal_fpr=m_op_star["fpr"],
            cal_fnr=m_op_star["fnr"],
            calib_f1_at_threshold=f1_cal if method == "raw" else float("nan"),
        ))
    return rows, probs, y_ev


# --------------------------------------------------------------------------
# Admission rules, frontier, matched comparisons
# --------------------------------------------------------------------------

def rule_metrics(keep: np.ndarray, corrupted: np.ndarray) -> dict:
    """Definitions transcribed from discussion/metric_definitions.md."""
    n = len(keep)
    n_kept = int(keep.sum())
    n_rej = n - n_kept
    corrupt = corrupted
    clean = ~corrupted
    n_corrupt = int(corrupt.sum())
    n_clean = int(clean.sum())
    kept_corrupt = int((keep & corrupt).sum())
    rej_corrupt = n_corrupt - kept_corrupt
    rej_clean = int((~keep & clean).sum())
    prec = rej_corrupt / n_rej if n_rej else 0.0
    rec = rej_corrupt / n_corrupt if n_corrupt else 0.0
    return dict(
        retention=n_kept / n if n else 0.0,
        admitted_contamination=kept_corrupt / n_kept if n_kept else 0.0,
        true_harmful_survival=kept_corrupt / n_corrupt if n_corrupt else 0.0,
        det_precision=prec, det_recall=rec,
        det_f1=2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
        clean_false_rejection=rej_clean / n_clean if n_clean else 0.0,
    )


MATCH_METRICS = ["retention", "admitted_contamination", "true_harmful_survival",
                 "det_precision", "det_recall", "det_f1", "clean_false_rejection"]


def rule_metrics_boot(keep: np.ndarray, corrupted: np.ndarray, boot_idx: np.ndarray):
    """Vectorised metrics over bootstrap resamples of PAIRS (rows)."""
    k = keep[boot_idx]          # (B, n)
    c = corrupted[boot_idx]
    n = k.shape[1]
    n_kept = k.sum(1).astype(float)
    n_rej = n - n_kept
    n_corrupt = c.sum(1).astype(float)
    n_clean = n - n_corrupt
    kept_corrupt = (k & c).sum(1).astype(float)
    rej_corrupt = n_corrupt - kept_corrupt
    rej_clean = ((~k) & (~c)).sum(1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        retention = n_kept / n
        contamination = np.where(n_kept > 0, kept_corrupt / np.maximum(n_kept, 1), 0.0)
        survival = np.where(n_corrupt > 0, kept_corrupt / np.maximum(n_corrupt, 1), 0.0)
        prec = np.where(n_rej > 0, rej_corrupt / np.maximum(n_rej, 1), 0.0)
        rec = np.where(n_corrupt > 0, rej_corrupt / np.maximum(n_corrupt, 1), 0.0)
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
        fr = np.where(n_clean > 0, rej_clean / np.maximum(n_clean, 1), 0.0)
    return dict(retention=retention, admitted_contamination=contamination,
                true_harmful_survival=survival, det_precision=prec,
                det_recall=rec, det_f1=f1, clean_false_rejection=fr)


def ci(values: np.ndarray):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def keep_pool(scores: np.ndarray, k: int, t: float) -> np.ndarray:
    return (scores >= t).sum(axis=1) >= k


def build_frontier(scores: np.ndarray, corrupted: np.ndarray, ks):
    """All (k, t) operating points with their metrics on the given rows."""
    cands = np.unique(np.round(scores, 8))
    cands = np.concatenate([cands, [cands[-1] + 1e-6]])
    points = []
    for k in ks:
        for t in cands:
            keep = keep_pool(scores, k, float(t))
            m = rule_metrics(keep, corrupted)
            m.update(k=int(k), t=float(t))
            points.append(m)
    return points


def pareto_frontier(points):
    """Points not dominated on (higher retention, lower contamination)."""
    front = []
    for p in points:
        dominated = any(
            (q["retention"] >= p["retention"]) and
            (q["admitted_contamination"] <= p["admitted_contamination"]) and
            ((q["retention"] > p["retention"]) or
             (q["admitted_contamination"] < p["admitted_contamination"]))
            for q in points)
        if not dominated:
            front.append(p)
    front.sort(key=lambda p: p["retention"])
    dedup, seen = [], set()
    for p in front:
        key = round(p["retention"], 10)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    return dedup


def interp_on_frontier(front, target_retention, metric):
    if not front:
        return float("nan")
    xs = np.array([p["retention"] for p in front])
    ys = np.array([p[metric] for p in front])
    if target_retention < xs.min() or target_retention > xs.max():
        return float("nan")
    return float(np.interp(target_retention, xs, ys))


def select_point(points, key, target, prefer=None):
    """Operating point whose `key` is closest to `target`.

    Selection is restricted to the Pareto frontier on (higher retention, lower
    admitted contamination). The restriction matters: over the unrestricted set
    of (k, t) points the matched quantity is not monotone, so a nearest-value
    match can land on a degenerate point with very low retention and the two
    pools end up compared at operating points that are not comparable. On the
    frontier every matched quantity is near-monotone in retention, so the match
    is stable. Ties are broken by `prefer`, a (metric, direction) pair with
    direction "min" or "max". Selection uses the CALIBRATION split only.
    """
    front = pareto_frontier(points)
    pool = front if front else points
    best, best_cost = None, None
    for p in pool:
        d = abs(p[key] - target)
        tie = 0.0
        if prefer is not None:
            m, direction = prefer
            tie = p[m] if direction == "min" else -p[m]
        cost = (round(d, 10), tie)
        if best_cost is None or cost < best_cost:
            best, best_cost = p, cost
    return best


def select_point_constrained(points, key, target):
    """Highest-retention frontier point satisfying `key <= target`.

    This is the operational reading of a matched-reliability comparison: fix a
    reliability budget and ask how much data each rule can retain without
    exceeding it. When no frontier point meets the budget the point with the
    smallest value of `key` is returned instead, and the achieved value
    reported in the tables makes the shortfall visible.
    """
    front = pareto_frontier(points)
    pool = front if front else points
    feasible = [p for p in pool if p[key] <= target + 1e-12]
    if feasible:
        return max(feasible, key=lambda p: (p["retention"], -p[key]))
    return min(pool, key=lambda p: p[key])


# --------------------------------------------------------------------------
# Parse-failure audit
# --------------------------------------------------------------------------

def parse_failure_audit(frame, eta, pools):
    """Counts of scores exactly equal to the parse-failure sentinel 0.5.

    src/precompute_verifiers.py:65 assigns score = 0.5 whenever the verifier's
    JSON cannot be parsed. Since 0.5 < 0.7, every parse failure is silently
    converted into a REJECT vote. This audit quantifies how many admission
    decisions that sentinel actually drives.
    """
    rows = []
    for pool_name, scores in pools.items():
        if scores is None:
            continue
        ok = np.isfinite(scores).all(axis=1)
        s = scores[ok]
        corrupted = frame["corrupted"][ok]
        if len(s) == 0:
            continue
        is_sentinel = s == PARSE_FAILURE_SCORE
        for j, role in enumerate(ROLES):
            rows.append(dict(pool=pool_name, eta=eta, role=role,
                             n_scored=int(len(s)),
                             n_exact_half=int(is_sentinel[:, j].sum()),
                             frac_exact_half=float(is_sentinel[:, j].mean())))
        # Rule-level dependence of rejections on the sentinel.
        for rule_name, k in (("consensus_k3_t0.7", PAPER_K), ("single_safety_t0.7", 1)):
            if rule_name.startswith("single"):
                sub = s[:, [ROLES.index("safety")]]
                sub_sent = is_sentinel[:, [ROLES.index("safety")]]
            else:
                sub, sub_sent = s, is_sentinel
            keep = keep_pool(sub, k, PAPER_THRESHOLD)
            counterfactual = np.where(sub_sent, 1.0, sub)
            keep_cf = keep_pool(counterfactual, k, PAPER_THRESHOLD)
            rejected = ~keep
            flipped = rejected & keep_cf
            n_rej = int(rejected.sum())
            rows.append(dict(pool=pool_name, eta=eta, role=f"[rule] {rule_name}",
                             n_scored=int(len(s)), n_exact_half=int(flipped.sum()),
                             frac_exact_half=float(flipped.sum() / n_rej) if n_rej else 0.0,
                             n_rejected=n_rej,
                             n_rejections_sentinel_driven=int(flipped.sum()),
                             frac_rejections_sentinel_driven=(
                                 float(flipped.sum() / n_rej) if n_rej else 0.0),
                             corrupt_share_of_flipped=(
                                 float(corrupted[flipped].mean()) if flipped.any() else float("nan"))))
    return rows


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def plot_reliability(y_ev, probs, role, eta, pool_name, figdir):
    """Reliability diagram with a bin-occupancy panel.

    The occupancy panel is not decoration. When a calibrator maps every score
    to approximately the base rate, the reliability curve degenerates to one or
    two points and the diagram alone does not show why. The occupancy panel
    makes the concentration of the predicted probabilities explicit.
    """
    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(3.8, 4.2), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.12))
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, lw=1.0, ls="-", alpha=0.5,
            zorder=1, label="Perfect calibration")
    base_rate = float(y_ev.mean())
    ax.axhline(base_rate, color=TEXT_SECONDARY, lw=0.8, ls=(0, (1, 3)), zorder=1)
    ax.annotate(f"base rate {base_rate:.3f}", xy=(0.55, base_rate),
                xytext=(0, 4), textcoords="offset points", ha="left",
                fontsize=6.5, color=TEXT_SECONDARY)
    width = 1.0 / N_ECE_BINS
    for i, (method, p) in enumerate(probs.items()):
        _, table = ece_equal_width(y_ev, p)
        xs = [r["mean_pred"] for r in table if r["n"] > 0]
        ys = [r["frac_pos"] for r in table if r["n"] > 0]
        st = METHOD_STYLE[method]
        ax.plot(xs, ys, color=st["color"], ls=st["ls"], marker=st["marker"],
                ms=4.5, label=st["label"], zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        centers = np.array([(r["lo"] + r["hi"]) / 2 for r in table])
        fracs = np.array([r["n"] for r in table], dtype=float) / max(len(p), 1)
        off = (i - (len(probs) - 1) / 2) * (width / len(probs))
        axh.bar(centers + off, fracs, width=width / len(probs) * 0.9,
                color=st["color"], linewidth=0.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Observed corruption frequency")
    ax.set_title(f"{role}, eta = {eta}% ({pool_name})", loc="left")
    ax.legend(fontsize=7, loc="upper left")
    axh.set_xlabel("Predicted P(corrupted)")
    axh.set_ylabel("Share of\npairs", fontsize=7)
    axh.tick_params(labelsize=7)
    name = figdir / f"reliability_{pool_name}_{role}_eta{eta}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{name}.{ext}")
    plt.close(fig)
    return f"{name}.png"


def plot_frontiers(fronts, eta, figdir, suffix=""):
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    for rule, front in fronts.items():
        if not front:
            continue
        st = RULE_STYLE[rule]
        xs = [p["retention"] for p in front]
        ys = [p["admitted_contamination"] for p in front]
        ax.plot(xs, ys, color=st["color"], ls=st["ls"], marker=st["marker"],
                ms=4.0, label=st["label"], markeredgecolor=SURFACE,
                markeredgewidth=1.0)
    ax.set_xlabel("Retention (fraction of pairs admitted)")
    ax.set_ylabel("Contamination of admitted data")
    ax.set_title(f"Retention and contamination frontier, eta = {eta}%", loc="left")
    ax.legend(fontsize=7)
    name = figdir / f"frontier_contamination_eta{eta}{suffix}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{name}.{ext}")
    plt.close(fig)
    return f"{name}.png"


# --------------------------------------------------------------------------
# Markdown helpers
# --------------------------------------------------------------------------

def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def fmt_ci(point, lo, hi, nd=4):
    if not np.isfinite(point):
        return "n/a"
    return f"{point:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precomputed", nargs="+",
                    default=[str(ROOT / "results/verifier_outputs/hh_train.precomputed.jsonl")],
                    help="merged precomputed file, or one or more shard files")
    ap.add_argument("--corrupted-dir", default=str(ROOT / "data/corrupted"))
    ap.add_argument("--diverse-dir", default=str(ROOT / "analysis/outputs/diverse_pool"))
    ap.add_argument("--regime", default="structured_unsafe")
    ap.add_argument("--eta", nargs="+", type=int, default=[10, 20])
    ap.add_argument("--outdir", default=str(ROOT / "analysis"),
                    help="root for calibration_results.csv, the .md tables, "
                         "outputs/calibration_splits.json and calibration_figures/")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--boot-seed", type=int, default=BOOT_SEED)
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--retention-grid", nargs="+", type=float,
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--contamination-grid", nargs="+", type=float, default=None,
                    help="matched-reliability targets; default is derived from "
                         "the realized corruption rate of each eta")
    args = ap.parse_args()

    set_style()
    outdir = Path(args.outdir)
    figdir = outdir / "calibration_figures"
    (outdir / "outputs").mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Calibration analysis")
    print("=" * 78)
    print("Calibration target: the binary label is whether a synthetic preference "
          "flip was injected into the pair (is_clean == False) at the given eta.")
    print("The verifier score used for a pair is the score on the RECORDED-"
          "preference direction. A LOW score is meant to indicate a corrupted "
          "pair, so for ROC and PR purposes the predictor of the positive class "
          "'corrupted' is the NEGATED score.")
    print("Temperature scaling is NOT attempted anywhere in this analysis: the "
          "verifier scores are floats parsed out of model-emitted text, not "
          "logits, so there is no pre-softmax quantity to rescale. Only Platt "
          "scaling and isotonic regression are fitted.")
    print("All calibrators are fitted on the CALIBRATION split only; every "
          "diagnostic below is computed on the disjoint EVALUATION split.")
    print("-" * 78)

    pre = load_precomputed(args.precomputed)
    diverse = load_diverse(Path(args.diverse_dir))
    missing_roles = [r for r in ROLES if r not in diverse]
    if missing_roles:
        print(f"[diverse] roles present: {sorted(diverse)} ; missing: {missing_roles}")
        print("[diverse] the cross-backbone pool is INCOMPLETE, so every "
              "cross-backbone section is skipped in this run. The shared-pool "
              "and single-verifier sections are unaffected.")
    else:
        n_ids = min(len(diverse[r]) for r in ROLES)
        print(f"[diverse] all four roles present, {n_ids} ids scored by every role")

    csv_rows, audit_rows = [], []
    splits_payload = {"split_seed": args.split_seed, "stratified_by": "corruption label",
                      "fraction_calibration": 0.5, "etas": {}}
    retention_md, reliability_md = [], []
    conclusion_evidence = {"reliability_at_matched_retention": [],
                           "coverage_at_matched_reliability": []}
    table_md_sections = []
    reliability_figs, frontier_figs = [], []

    for eta in args.eta:
        print("=" * 78)
        print(f"eta = {eta}%")
        print("=" * 78)
        corrupt_rows = load_corrupted(Path(args.corrupted_dir), args.regime, eta)
        frame = build_frame(corrupt_rows, pre, diverse)
        n = len(frame["ids"])
        if n == 0:
            print(f"[warn] no ids shared between the corrupted file and the "
                  f"precomputed scores at eta={eta}; skipping")
            continue
        y = frame["corrupted"]
        realized_eta = float(y.mean())
        print(f"[data] {n} pairs joined on id; realized corruption rate "
              f"{realized_eta:.4f} ({int(y.sum())} corrupted)")

        calib, evalm = stratified_split(y, args.split_seed)
        cal_ids = sorted(frame["ids"][calib].tolist())
        ev_ids = sorted(frame["ids"][evalm].tolist())
        splits_payload["etas"][str(eta)] = dict(
            n=int(n), n_calibration=len(cal_ids), n_evaluation=len(ev_ids),
            n_corrupt=int(y.sum()), realized_eta=realized_eta,
            calibration_ids=cal_ids, evaluation_ids=ev_ids,
            calibration_sha256=sha256_of_ids(cal_ids),
            evaluation_sha256=sha256_of_ids(ev_ids),
            disjoint=bool(set(cal_ids).isdisjoint(ev_ids)))
        print(f"[split] calibration {len(cal_ids)} ids, evaluation {len(ev_ids)} ids, "
              f"disjoint={set(cal_ids).isdisjoint(ev_ids)}")

        # ---------------- per-verifier diagnostics ----------------
        print("-" * 78)
        print(f"Per-verifier calibration diagnostics, eta = {eta}% "
              f"(evaluation split, {int(evalm.sum())} pairs)")
        pools_for_diag = {"shared": frame["shared"]}
        if frame["diverse_complete"] and frame["diverse_mask"].any():
            pools_for_diag["diverse"] = frame["diverse"]
        for pool_name, scores_all in pools_for_diag.items():
            usable = np.isfinite(scores_all).all(axis=1)
            for j, role in enumerate(ROLES):
                cal_m = calib & usable
                ev_m = evalm & usable
                if cal_m.sum() < 10 or ev_m.sum() < 10:
                    print(f"[warn] {pool_name}/{role}: fewer than 10 usable rows "
                          f"in a split, skipping")
                    continue
                rows, probs, y_ev = per_verifier_diagnostics(
                    frame, cal_m, ev_m, eta, j, role, pool_name, scores_all)
                csv_rows.extend(rows)
                reliability_figs.append(
                    plot_reliability(y_ev, probs, role, eta, pool_name, figdir))
                r0 = rows[0]
                print(f"  {pool_name:8s} {role:12s} AUROC {fmt(r0['auroc'],3)}  "
                      f"AUPRC {fmt(r0['auprc'],3)}  Brier(raw) {fmt(r0['brier_raw'],3)}  "
                      f"ECE(raw) {fmt(r0['ece_raw'],3)}  "
                      f"F1@0.7 {fmt(r0['op070_f1'],3)}  "
                      f"t* {fmt(r0['cal_threshold'],3)} -> F1 {fmt(r0['cal_f1'],3)}")

        # markdown section for this eta
        sec_rows = []
        for r in csv_rows:
            if r["eta"] != eta:
                continue
            sec_rows.append([
                r["pool"], r["role"], r["method"], fmt(r["auroc"], 3), fmt(r["auprc"], 3),
                fmt(r["brier"], 4), fmt(r["ece"], 4),
                fmt(r["op070_precision"], 3), fmt(r["op070_recall"], 3),
                fmt(r["op070_f1"], 3), fmt(r["op070_fpr"], 3), fmt(r["op070_fnr"], 3),
                fmt(r["cal_threshold"], 3), fmt(r["cal_f1"], 3)])
        table_md_sections.append((eta, md_table(
            ["Pool", "Role", "Method", "AUROC", "AUPRC", "Brier", "ECE",
             "Prec@0.7", "Rec@0.7", "F1@0.7", "FPR@0.7", "FNR@0.7",
             "Calibrated threshold", "F1 at calibrated threshold"], sec_rows)))

        # ---------------- parse-failure audit ----------------
        audit_rows.extend(parse_failure_audit(
            frame, eta,
            {"shared": frame["shared"],
             "diverse": frame["diverse"] if frame["diverse_complete"] else None}))

        # ---------------- matched comparisons ----------------
        if not frame["diverse_complete"] or frame["diverse_mask"].sum() < 20:
            msg = (f"eta = {eta}%: the cross-backbone pool does not yet cover all "
                   f"four roles with a usable number of rows, so the matched-"
                   f"retention and matched-reliability comparisons are skipped. "
                   f"Roles present: {sorted(diverse) if diverse else 'none'}.")
            print(f"[skip] {msg}")
            retention_md.append(("skip", eta, msg))
            reliability_md.append(("skip", eta, msg))
            continue

        sub = frame["diverse_mask"]
        sub_cal = calib & sub
        sub_ev = evalm & sub
        print("-" * 78)
        print(f"Matched comparisons on the {int(sub.sum())} pairs covered by the "
              f"cross-backbone subsample ({int(sub_cal.sum())} calibration, "
              f"{int(sub_ev.sum())} evaluation). Both pools are compared on "
              f"exactly these rows.")

        # best single verifier chosen on the calibration split of the subsample
        best_role, best_auc = None, -np.inf
        for j, role in enumerate(ROLES):
            s = frame["shared"][sub_cal][:, j]
            yy = frame["corrupted"][sub_cal]
            if len(np.unique(yy)) < 2:
                continue
            a = roc_auc_score(yy, -s)
            if a > best_auc:
                best_role, best_auc = role, a
        if best_role is None:
            best_role, best_auc = "safety", float("nan")
        print(f"[single] best single verifier on the calibration split: "
              f"{best_role} (AUROC {best_auc:.4f})")

        rule_scores_cal = {
            "shared_pool": frame["shared"][sub_cal],
            "diverse_pool": frame["diverse"][sub_cal],
            "best_single": frame["shared"][sub_cal][:, [ROLES.index(best_role)]],
        }
        rule_scores_ev = {
            "shared_pool": frame["shared"][sub_ev],
            "diverse_pool": frame["diverse"][sub_ev],
            "best_single": frame["shared"][sub_ev][:, [ROLES.index(best_role)]],
        }
        y_cal_sub = frame["corrupted"][sub_cal]
        y_ev_sub = frame["corrupted"][sub_ev]

        points_cal, fronts_cal = {}, {}
        for rule, sc in rule_scores_cal.items():
            ks = range(1, N_POOL + 1) if sc.shape[1] > 1 else [1]
            points_cal[rule] = build_frontier(sc, y_cal_sub, ks)
            fronts_cal[rule] = pareto_frontier(points_cal[rule])
        points_ev = {}
        for rule, sc in rule_scores_ev.items():
            ks = range(1, N_POOL + 1) if sc.shape[1] > 1 else [1]
            points_ev[rule] = build_frontier(sc, y_ev_sub, ks)
        fronts_ev = {r: pareto_frontier(p) for r, p in points_ev.items()}
        frontier_figs.append(plot_frontiers(fronts_ev, eta, figdir))

        rng = np.random.default_rng(args.boot_seed)
        n_ev = int(sub_ev.sum())
        boot_idx = rng.integers(0, n_ev, size=(args.n_boot, n_ev))

        def evaluate_selected(sel):
            """Point estimates, CIs and bootstrap draws on the evaluation split."""
            out = {}
            for rule, p in sel.items():
                if p is None:
                    out[rule] = None
                    continue
                sc = rule_scores_ev[rule]
                keep = keep_pool(sc, p["k"], p["t"])
                pt = rule_metrics(keep, y_ev_sub)
                draws = rule_metrics_boot(keep, y_ev_sub, boot_idx)
                out[rule] = dict(point=pt, draws=draws, k=p["k"], t=p["t"])
            return out

        def paired_diff(res, metric, a="diverse_pool", b="shared_pool"):
            if res.get(a) is None or res.get(b) is None:
                return None
            d = res[a]["draws"][metric] - res[b]["draws"][metric]
            point = res[a]["point"][metric] - res[b]["point"][metric]
            lo, hi = ci(d)
            return dict(point=point, lo=lo, hi=hi,
                        excludes_zero=bool(np.isfinite(lo) and np.isfinite(hi)
                                           and (lo > 0 or hi < 0)))

        # ---- 4. matched retention ----
        print("-" * 78)
        print(f"Matched-retention comparison, eta = {eta}%")
        rows_ret = []
        for target in args.retention_grid:
            sel = {r: select_point(points_cal[r], "retention", target,
                                   prefer=("admitted_contamination", "min"))
                   for r in points_cal}
            res = evaluate_selected(sel)
            for rule in ["shared_pool", "diverse_pool", "best_single"]:
                if res[rule] is None:
                    continue
                pt, dr = res[rule]["point"], res[rule]["draws"]
                interp = interp_on_frontier(fronts_ev[rule], target,
                                            "admitted_contamination")
                rows_ret.append([
                    fmt(target, 2), RULE_STYLE[rule]["label"],
                    f"k={res[rule]['k']}, t={res[rule]['t']:.3f}",
                    fmt_ci(pt["retention"], *ci(dr["retention"])),
                    fmt_ci(pt["admitted_contamination"], *ci(dr["admitted_contamination"])),
                    fmt_ci(pt["true_harmful_survival"], *ci(dr["true_harmful_survival"])),
                    fmt_ci(pt["det_precision"], *ci(dr["det_precision"])),
                    fmt_ci(pt["det_recall"], *ci(dr["det_recall"])),
                    fmt_ci(pt["det_f1"], *ci(dr["det_f1"])),
                    fmt_ci(pt["clean_false_rejection"], *ci(dr["clean_false_rejection"])),
                    fmt(interp, 4)])
            diffs = {m: paired_diff(res, m) for m in MATCH_METRICS}
            if diffs["admitted_contamination"] is not None:
                d = diffs["admitted_contamination"]
                conclusion_evidence["reliability_at_matched_retention"].append(
                    dict(eta=eta, target=target, metric="admitted_contamination", **d))
                rows_ret.append([
                    fmt(target, 2), "PAIRED DIFFERENCE (cross-backbone minus shared)", "",
                    fmt_ci(diffs["retention"]["point"], diffs["retention"]["lo"],
                           diffs["retention"]["hi"]),
                    fmt_ci(d["point"], d["lo"], d["hi"]),
                    fmt_ci(diffs["true_harmful_survival"]["point"],
                           diffs["true_harmful_survival"]["lo"],
                           diffs["true_harmful_survival"]["hi"]),
                    fmt_ci(diffs["det_precision"]["point"], diffs["det_precision"]["lo"],
                           diffs["det_precision"]["hi"]),
                    fmt_ci(diffs["det_recall"]["point"], diffs["det_recall"]["lo"],
                           diffs["det_recall"]["hi"]),
                    fmt_ci(diffs["det_f1"]["point"], diffs["det_f1"]["lo"],
                           diffs["det_f1"]["hi"]),
                    fmt_ci(diffs["clean_false_rejection"]["point"],
                           diffs["clean_false_rejection"]["lo"],
                           diffs["clean_false_rejection"]["hi"]), ""])
                print(f"  retention {target:.2f}: contamination difference "
                      f"{d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] "
                      f"{'(excludes zero)' if d['excludes_zero'] else '(includes zero)'}")
        retention_md.append(("table", eta, md_table(
            ["Target retention", "Rule", "Operating point", "Retention",
             "Admitted contamination", "True harmful survival", "Detection precision",
             "Detection recall", "Detection F1", "Clean false-rejection rate",
             "Contamination interpolated at target"], rows_ret)))

        # ---- 5. matched reliability ----
        print("-" * 78)
        print(f"Matched-reliability comparison, eta = {eta}%")
        if args.contamination_grid is not None:
            cont_grid = list(args.contamination_grid)
        else:
            cont_grid = [round(realized_eta * f, 4) for f in (0.6, 0.7, 0.8, 0.9, 1.0)]
        surv_grid = [0.2, 0.3, 0.4, 0.5, 0.6]
        fpr_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
        rows_rel = []
        for key, grid, label, mode in (
                ("admitted_contamination", cont_grid, "admitted contamination", "budget"),
                ("true_harmful_survival", surv_grid, "true harmful survival", "budget"),
                ("clean_false_rejection", fpr_grid, "clean false-rejection rate", "nearest")):
            # A "budget" match asks how much data a rule retains without exceeding
            # a reliability budget. That framing is vacuous for the clean
            # false-rejection rate, because admitting every pair trivially
            # satisfies any budget on it while maximising retention. For that
            # quantity the match is therefore a nearest-value match on the
            # frontier, and the compared quantity is reliability rather than
            # coverage: at an equal cost paid on clean data, which rule admits
            # less contamination.
            compared = "retention" if mode == "budget" else "admitted_contamination"
            for target in grid:
                if mode == "budget":
                    sel = {r: select_point_constrained(points_cal[r], key, target)
                           for r in points_cal}
                else:
                    sel = {r: select_point(points_cal[r], key, target,
                                           prefer=("admitted_contamination", "min"))
                           for r in points_cal}
                res = evaluate_selected(sel)
                for rule in ["shared_pool", "diverse_pool", "best_single"]:
                    if res[rule] is None:
                        continue
                    pt, dr = res[rule]["point"], res[rule]["draws"]
                    rows_rel.append([
                        label, fmt(target, 4), RULE_STYLE[rule]["label"],
                        f"k={res[rule]['k']}, t={res[rule]['t']:.3f}",
                        fmt_ci(pt[key], *ci(dr[key])),
                        fmt_ci(pt["retention"], *ci(dr["retention"])),
                        fmt_ci(pt["admitted_contamination"],
                               *ci(dr["admitted_contamination"])),
                        fmt_ci(pt["det_f1"], *ci(dr["det_f1"]))])
                d_cmp = paired_diff(res, compared)
                d_key = paired_diff(res, key)
                if d_cmp is None:
                    continue
                d_ret = paired_diff(res, "retention")
                d_con = paired_diff(res, "admitted_contamination")
                rows_rel.append([
                    label, fmt(target, 4),
                    "PAIRED DIFFERENCE (cross-backbone minus shared)", "",
                    fmt_ci(d_key["point"], d_key["lo"], d_key["hi"]),
                    fmt_ci(d_ret["point"], d_ret["lo"], d_ret["hi"]),
                    fmt_ci(d_con["point"], d_con["lo"], d_con["hi"]), ""])
                degenerate = (res["shared_pool"]["point"]["retention"] == 0.0
                              and res["diverse_pool"]["point"]["retention"] == 0.0)
                if mode == "budget" and not degenerate:
                    conclusion_evidence["coverage_at_matched_reliability"].append(
                        dict(eta=eta, target=target, metric="retention",
                             matched=key, **d_ret))
                print(f"  {label} {target:.4f}: "
                      f"{'retention' if mode == 'budget' else 'contamination'} "
                      f"difference {d_cmp['point']:+.4f} "
                      f"[{d_cmp['lo']:+.4f}, {d_cmp['hi']:+.4f}] "
                      f"{'(excludes zero)' if d_cmp['excludes_zero'] else '(includes zero)'}")
                if degenerate:
                    print("    (both pools admit nothing at this budget, so the "
                          "comparison is degenerate and is excluded from the "
                          "conclusion evidence)")
                if mode == "nearest":
                    print("    (matched on cost paid to clean data; this block is "
                          "reported but is not counted toward the conclusion, which "
                          "is defined over matched retention and matched reliability "
                          "budgets only)")
        reliability_md.append(("table", eta, md_table(
            ["Matched quantity", "Target", "Rule", "Operating point",
             "Achieved matched quantity", "Retention", "Admitted contamination",
             "Detection F1"], rows_rel)))

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    splits_path = outdir / "outputs" / "calibration_splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits_payload, f, indent=2)
    print("-" * 78)
    print(f"[wrote] {splits_path}")

    df = pd.DataFrame(csv_rows)
    csv_path = outdir / "calibration_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[wrote] {csv_path} ({len(df)} rows)")

    audit_df = pd.DataFrame(audit_rows)
    audit_path = outdir / "parse_failure_audit.csv"
    if len(audit_df):
        audit_df.to_csv(audit_path, index=False)
        print(f"[wrote] {audit_path} ({len(audit_df)} rows)")

    # ---- calibration_table.md ----
    lines = ["# Verifier calibration diagnostics", "",
             "All calibrators are fitted on the calibration split and every number "
             "in this file is computed on the disjoint evaluation split. The split "
             f"is 50/50, stratified by the corruption label, with seed "
             f"{args.split_seed}. The identifier lists and their SHA-256 digests are "
             "recorded in `analysis/outputs/calibration_splits.json`.", "",
             "The calibration target is the binary label of whether a synthetic "
             "preference flip was injected into the pair, that is `is_clean == False` "
             "in the corrupted file for the given eta. The score used for a pair is "
             "the score on the recorded-preference direction. A low score is intended "
             "to indicate a corrupted pair, therefore the predictor of the positive "
             "class in the ROC and precision-recall computations is the NEGATED "
             "score.", "",
             "Probability convention. The raw method treats `1 - score` as the "
             "predicted probability of corruption. Platt scaling is a logistic "
             "regression fitted on the raw score. Isotonic regression is fitted on "
             "the raw score with clipping outside the observed range. Temperature "
             "scaling is not attempted: the verifier scores are floats parsed out of "
             "model-emitted text rather than logits, so there is no pre-softmax "
             "quantity that a temperature could rescale. This is a limitation of the "
             "measurement, not an oversight.", "",
             f"Expected calibration error uses {N_ECE_BINS} equal-width bins on the "
             "interval [0, 1]. Bin b covers [b/10, (b+1)/10), and the final bin is "
             "closed at 1.0. Empty bins contribute nothing to the sum. The reported "
             "value is the sample-weighted mean absolute gap between the mean "
             "predicted probability and the observed corruption frequency inside "
             "each bin.", "",
             "The operating point columns describe the reject decision, where "
             f"rejecting a pair is a prediction that the pair is corrupted. At the "
             f"original threshold a pair is rejected when its score falls below "
             f"{PAPER_THRESHOLD}. The calibrated threshold is the value that "
             "maximises corruption-detection F1 on the calibration split, applied "
             "unchanged to the evaluation split.", ""]
    for eta, section in table_md_sections:
        lines += [f"## eta = {eta}%", "", section, ""]
    if not table_md_sections:
        lines += ["No diagnostics were produced for the requested eta values.", ""]
    (outdir / "calibration_table.md").write_text("\n".join(lines))
    print(f"[wrote] {outdir / 'calibration_table.md'}")

    # ---- matched_retention.md ----
    lines = ["# Matched-retention comparison", "",
             "This file tests whether verifier diversity improves reliability at matched operating points. Three "
             "admission rules are compared: the shared-backbone pool with a k of 4 "
             "consensus, the cross-backbone pool with a k of 4 consensus, and the "
             "best single verifier. For every rule the threshold and, where "
             "applicable, the value of k are swept to build the full retention and "
             "contamination frontier.", "",
             "Matching protocol. The operating point of each rule is selected on the "
             "calibration split as the frontier point whose retention is closest to "
             "the target, with ties broken toward lower admitted contamination. "
             "Selection is restricted to the Pareto frontier on higher retention and "
             "lower contamination, because over the unrestricted set of threshold and "
             "k combinations the retention is achieved by many dominated points and "
             "the two pools would be paired at operating points that are not "
             "comparable. That fixed "
             "operating point is then applied to the disjoint evaluation split, where "
             "all reported quantities and confidence intervals are computed. The "
             "achieved retention is reported alongside the target so that any residual "
             "mismatch is visible. The final column additionally gives the admitted "
             "contamination obtained by linear interpolation of the evaluation-split "
             "frontier at exactly the target retention.", "",
             f"Confidence intervals are percentile bootstrap intervals from "
             f"{args.n_boot} resamples of PAIRS with seed {args.boot_seed}. Paired "
             "differences reuse the identical resample indices for both pools, so the "
             "difference intervals are paired rather than independent.", "",
             "The comparison is restricted to the rows covered by the cross-backbone "
             "subsample, so both pools are scored on identical data.", ""]
    for kind, eta, body in retention_md:
        lines += [f"## eta = {eta}%", "", body, ""]
    if not retention_md:
        lines += ["No matched-retention comparison was produced.", ""]
    (outdir / "matched_retention.md").write_text("\n".join(lines))
    print(f"[wrote] {outdir / 'matched_retention.md'}")

    # ---- matched_reliability.md ----
    lines = ["# Matched-reliability comparison", "",
             "This file is the mirror image of the matched-retention comparison. "
             "Instead of fixing coverage and asking which pool is more reliable, it "
             "fixes a reliability quantity and asks which pool retains more data. "
             "Three reliability quantities are matched in turn: the contamination of "
             "the admitted set, the true harmful survival rate, and the clean "
             "false-rejection rate.", "",
             "Matching protocol. For each rule the operating point is the "
             "highest-retention point of the calibration-split frontier that keeps "
             "the matched quantity at or below the target, which is the operational "
             "reading of the comparison: fix a reliability budget and ask how much "
             "data each rule can retain within it. When no frontier point meets the "
             "budget, the point with the smallest achievable value is used instead, "
             "and the achieved value column makes the shortfall visible. Selection "
             "is restricted to the frontier because the matched quantities are not "
             "monotone over the unrestricted set of threshold and k combinations, so "
             "an unrestricted nearest-value match can pair operating points that are "
             "not comparable.", "",
             "The clean false-rejection block is matched differently and is read "
             "differently. A budget on the clean false-rejection rate is satisfied "
             "trivially by admitting every pair, which also maximises retention, so "
             "a budget match on that quantity carries no information. That block "
             "therefore uses a nearest-value match on the frontier, and the question "
             "it answers is the reverse one: at an equal cost paid on clean data, "
             "which rule admits less contamination. It is reported for completeness "
             "and is not counted toward the conclusion, which is defined over "
             "matched retention and matched reliability budgets only.", "",
             "Confidence intervals follow `matched_retention.md`. Operating points "
             "are selected on the calibration split, all reported numbers come from "
             "the disjoint evaluation split, and paired differences reuse identical "
             "bootstrap resample indices across the two pools.", ""]
    for kind, eta, body in reliability_md:
        lines += [f"## eta = {eta}%", "", body, ""]
    if not reliability_md:
        lines += ["No matched-reliability comparison was produced.", ""]
    (outdir / "matched_reliability.md").write_text("\n".join(lines))
    print(f"[wrote] {outdir / 'matched_reliability.md'}")

    # ------------------------------------------------------------------
    # Parse-failure audit report
    # ------------------------------------------------------------------
    print("=" * 78)
    print("Parse-failure audit")
    print("=" * 78)
    print(f"src/precompute_verifiers.py assigns score = {PARSE_FAILURE_SCORE} whenever "
          f"the verifier's JSON output cannot be parsed. Because "
          f"{PARSE_FAILURE_SCORE} < {PAPER_THRESHOLD}, every parse failure is "
          f"silently converted into a REJECT vote.")
    if not audit_rows:
        print("[warn] no rows available for the parse-failure audit")
    for r in audit_rows:
        if r["role"].startswith("[rule]"):
            print(f"  {r['pool']:8s} eta={r['eta']:<3d} {r['role']:28s} "
                  f"rejected {r['n_rejected']}, of which "
                  f"{r['n_rejections_sentinel_driven']} "
                  f"({r['frac_rejections_sentinel_driven']*100:.2f}%) would have been "
                  f"admitted had every {PARSE_FAILURE_SCORE} vote passed")
        else:
            print(f"  {r['pool']:8s} eta={r['eta']:<3d} {r['role']:12s} "
                  f"scores exactly {PARSE_FAILURE_SCORE}: {r['n_exact_half']}/"
                  f"{r['n_scored']} ({r['frac_exact_half']*100:.3f}%)")

    # ------------------------------------------------------------------
    # Conclusion selection
    # ------------------------------------------------------------------
    print("=" * 78)
    print("Conclusion")
    print("=" * 78)
    rel = conclusion_evidence["reliability_at_matched_retention"]
    cov = conclusion_evidence["coverage_at_matched_reliability"]
    if not rel and not cov:
        conclusion = ("(e) inconclusive: the cross-backbone pool is not yet complete, "
                      "so no paired comparison between the two pools could be computed "
                      "in this run.")
        detail = "No paired difference intervals were available."
    else:
        # Reliability improves if the cross-backbone pool admits LESS
        # contamination at matched retention, with an interval excluding zero.
        rel_better = [d for d in rel if d["excludes_zero"] and d["hi"] < 0]
        rel_worse = [d for d in rel if d["excludes_zero"] and d["lo"] > 0]
        # Coverage improves if the cross-backbone pool retains MORE data at
        # matched contamination, with an interval excluding zero.
        cov_better = [d for d in cov if d["excludes_zero"] and d["lo"] > 0]
        cov_worse = [d for d in cov if d["excludes_zero"] and d["hi"] < 0]
        widths = [d["hi"] - d["lo"] for d in rel + cov
                  if np.isfinite(d["hi"]) and np.isfinite(d["lo"])]
        median_width = float(np.median(widths)) if widths else float("nan")
        improves_reliability = len(rel_better) > len(rel_worse) and len(rel_better) > 0
        improves_coverage = len(cov_better) > len(cov_worse) and len(cov_better) > 0
        if improves_reliability and improves_coverage:
            conclusion = ("(c) both: at matched retention the cross-backbone pool "
                          "admits significantly less contamination, and at matched "
                          "contamination it retains significantly more data.")
        elif improves_reliability:
            conclusion = ("(a) diversity improves reliability at matched retention.")
        elif improves_coverage:
            conclusion = ("(b) diversity improves coverage at matched reliability.")
        elif median_width < 0.05:
            conclusion = ("(d) diversity reduces measured correlation without "
                          "improving practical operating points: no paired difference "
                          "interval excludes zero, and the intervals are tight enough "
                          "to rule out a materially better operating point.")
        else:
            conclusion = ("(e) inconclusive: no paired difference interval excludes "
                          "zero, and the intervals are too wide to distinguish a small "
                          "real effect from no effect.")
        detail = (f"Matched-retention contamination differences: "
                  f"{len(rel_better)} of {len(rel)} favour the cross-backbone pool "
                  f"with an interval excluding zero, {len(rel_worse)} favour the "
                  f"shared pool. Matched-contamination retention differences: "
                  f"{len(cov_better)} of {len(cov)} favour the cross-backbone pool, "
                  f"{len(cov_worse)} favour the shared pool. Median paired interval "
                  f"width {median_width:.4f}.")
    print(f"SUPPORTED CONCLUSION: {conclusion}")
    print(detail)
    print("Selection rule: a conclusion of improvement requires that the paired "
          "bootstrap difference interval exclude zero in the favourable direction at "
          "more grid points than in the unfavourable direction. No claim is made from "
          "point estimates alone.")

    print("=" * 78)
    print(f"[figures] {len(reliability_figs)} reliability diagrams and "
          f"{len(frontier_figs)} frontier figures written to {figdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
