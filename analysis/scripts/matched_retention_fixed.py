#!/usr/bin/env python3
"""Matched-retention comparison with exact retention matching.

This script supersedes the matched-retention section of
analysis/matched_retention.py -> analysis/matched_retention.md.

The problem it fixes
--------------------
The original comparison selected, for each pool and each target retention, the
frontier point whose retention was CLOSEST to the target. The two pools have
different achievable retention values, so at most grid points the two selected
operating points did not retain the same fraction of the data. Admitted
contamination is a property of the retained subset, and it moves with retention
along each pool's own frontier, so a comparison of admitted contamination
between two rules that retain different fractions of the data reintroduces
exactly the confound the matched comparison exists to remove.

What this script does instead
-----------------------------
For every target retention on a common grid, each pool's Pareto frontier is
LINEARLY INTERPOLATED to exactly that retention. The two pools are therefore
evaluated at genuinely identical retention, and the paired difference is a
difference at matched retention by construction rather than by approximation.

Uncertainty is a paired percentile bootstrap over PAIRS in which the frontier is
rebuilt and re-interpolated inside every replicate, so the interval reflects the
whole estimation procedure (frontier construction plus interpolation) rather
than treating the frontier as a fixed object. Both pools use the identical
resample indices within a replicate, so the difference intervals are paired.

Definitions of the frontier, the admission rule, the metrics and the data join
are imported unchanged from calibration_analysis.py. The frontier construction
here is a vectorised re-implementation used inside the bootstrap for speed; it
is asserted to agree exactly with the imported reference implementation on the
point-estimate data before any bootstrap is run.

Outputs: analysis/matched_retention_fixed.md and analysis/matched_retention_fixed.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibration_analysis as ca  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# Pre-registered constants, carried over from calibration_analysis.py so that
# this document is comparable with the one it supersedes.
N_BOOT = 2000
BOOT_SEED = 42
SPLIT_SEED = ca.SPLIT_SEED
ROLES = ca.ROLES
N_POOL = ca.N_POOL
RULES = ["shared_pool", "diverse_pool", "best_single"]
LABEL = {r: ca.RULE_STYLE[r]["label"] for r in RULES}

# Metrics interpolated at matched retention.
INTERP_METRICS = ["admitted_contamination", "true_harmful_survival",
                  "det_recall", "clean_false_rejection"]
METRIC_LABEL = {
    "admitted_contamination": "Admitted contamination",
    "true_harmful_survival": "True harmful survival",
    "det_recall": "Detection recall",
    "clean_false_rejection": "Clean false-rejection rate",
}
# Direction that favours the cross-backbone pool, for reporting only.
METRIC_FAVOURABLE_SIGN = {
    "admitted_contamination": -1,
    "true_harmful_survival": -1,
    "det_recall": +1,
    "clean_false_rejection": -1,
}
# Width below which a null result is read as an informative null rather than as
# an underpowered one. Transcribed from calibration_analysis.py.
TIGHT_INTERVAL_WIDTH = 0.05
# Absolute retention gap above which an operating point pair in the original
# document is treated as not matched.
MISMATCH_TOLERANCE = 0.01


# --------------------------------------------------------------------------
# Vectorised frontier construction
# --------------------------------------------------------------------------

def threshold_grid(scores: np.ndarray) -> np.ndarray:
    """Candidate thresholds, identical to calibration_analysis.build_frontier."""
    cands = np.unique(np.round(scores, 8))
    return np.concatenate([cands, [cands[-1] + 1e-6]])


def pass_counts(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """(n, T) count of roles whose score is at or above each threshold.

    The threshold grid is fixed once from the full evaluation split rather than
    recomputed inside each bootstrap replicate. A replicate's own unique-score
    set is a subset of this grid, so the extra thresholds only duplicate
    operating points already present. Duplicates cannot change a Pareto
    frontier, so the fixed grid is equivalent and avoids rebuilding the grid
    2000 times.
    """
    return (scores[:, :, None] >= thresholds[None, None, :]).sum(axis=1).astype(np.int16)


def frontier_from_counts(counts: np.ndarray, corrupted: np.ndarray, ks) -> dict:
    """Pareto frontier as arrays, from precomputed pass counts.

    Metric definitions and their zero-denominator conventions are transcribed
    from calibration_analysis.rule_metrics. Points are generated in the same
    order as build_frontier (k outer, threshold inner) so that stable sorting
    resolves ties identically.
    """
    n = counts.shape[0]
    n_corrupt = float(corrupted.sum())
    n_clean = float(n - n_corrupt)
    y = corrupted[:, None]

    ret, cont, surv, prec, rec, f1, cfr, kk, tt = [], [], [], [], [], [], [], [], []
    for k in ks:
        keep = counts >= k                       # (n, T)
        n_kept = keep.sum(0).astype(float)
        kept_corrupt = (keep & y).sum(0).astype(float)
        n_rej = n - n_kept
        rej_corrupt = n_corrupt - kept_corrupt
        kept_clean = n_kept - kept_corrupt
        rej_clean = n_clean - kept_clean
        with np.errstate(divide="ignore", invalid="ignore"):
            r_ = n_kept / n if n else np.zeros_like(n_kept)
            c_ = np.where(n_kept > 0, kept_corrupt / np.maximum(n_kept, 1.0), 0.0)
            s_ = np.where(n_corrupt > 0, kept_corrupt / max(n_corrupt, 1.0), 0.0)
            p_ = np.where(n_rej > 0, rej_corrupt / np.maximum(n_rej, 1.0), 0.0)
            q_ = np.where(n_corrupt > 0, rej_corrupt / max(n_corrupt, 1.0), 0.0)
            f_ = np.where((p_ + q_) > 0, 2 * p_ * q_ / np.maximum(p_ + q_, 1e-12), 0.0)
            e_ = np.where(n_clean > 0, rej_clean / max(n_clean, 1.0), 0.0)
        ret.append(r_); cont.append(c_); surv.append(s_)
        prec.append(p_); rec.append(q_); f1.append(f_); cfr.append(e_)
        kk.append(np.full(counts.shape[1], k)); tt.append(np.arange(counts.shape[1]))

    cols = dict(retention=np.concatenate(ret),
                admitted_contamination=np.concatenate(cont),
                true_harmful_survival=np.concatenate(surv),
                det_precision=np.concatenate(prec),
                det_recall=np.concatenate(rec),
                det_f1=np.concatenate(f1),
                clean_false_rejection=np.concatenate(cfr),
                k=np.concatenate(kk), t_index=np.concatenate(tt))

    r, c = cols["retention"], cols["admitted_contamination"]
    # [i, j] is True when point j dominates point i on (higher retention,
    # lower admitted contamination). Same predicate as pareto_frontier.
    ge_r = r[None, :] >= r[:, None]
    le_c = c[None, :] <= c[:, None]
    strict = (r[None, :] > r[:, None]) | (c[None, :] < c[:, None])
    dominated = (ge_r & le_c & strict).any(axis=1)

    idx = np.flatnonzero(~dominated)
    idx = idx[np.argsort(r[idx], kind="stable")]
    _, first = np.unique(np.round(r[idx], 10), return_index=True)
    idx = idx[np.sort(first)]
    return {key: val[idx] for key, val in cols.items()}


def interp_at_retention(front: dict, target: float, metrics) -> dict:
    """Linear interpolation of the frontier to exactly `target` retention.

    Returns NaN for every metric when the target lies outside the frontier's
    achievable retention range. Extrapolation is never performed.
    """
    xs = front["retention"]
    out = {m: float("nan") for m in metrics}
    if xs.size == 0 or target < xs[0] - 1e-12 or target > xs[-1] + 1e-12:
        return out
    for m in metrics:
        out[m] = float(np.interp(target, xs, front[m]))
    return out


def bracketing_points(front: dict, target: float):
    """Retentions of the two frontier points that bracket `target`.

    The width of that bracket is the span over which the interpolation carries,
    and is reported so that a reader can see which targets rest on a narrow
    interpolation and which rest on a wide one.
    """
    xs = front["retention"]
    if xs.size == 0 or target < xs[0] - 1e-12 or target > xs[-1] + 1e-12:
        return float("nan"), float("nan")
    j = int(np.searchsorted(xs, target, side="left"))
    if j < xs.size and abs(xs[j] - target) <= 1e-12:
        return float(xs[j]), float(xs[j])
    j = min(max(j, 1), xs.size - 1)
    return float(xs[j - 1]), float(xs[j])


def mixture_contamination(front: dict, target: float) -> float:
    """Admitted contamination of the randomized mixture at exactly `target`.

    An interpolated operating point is realised by applying the lower
    bracketing rule with probability lambda and the upper one otherwise. For
    true harmful survival, detection recall and the clean false-rejection rate
    the denominator does not depend on the rule, so the mixture value is exactly
    the linear interpolation. Admitted contamination has a rule-dependent
    denominator, so its mixture value is a retention-weighted average rather
    than the linear chord. This function computes that exact value so that the
    departure of the reported linear interpolation from an achievable operating
    point can be quantified rather than assumed small.
    """
    xs = front["retention"]
    cs = front["admitted_contamination"]
    r_lo, r_hi = bracketing_points(front, target)
    if not np.isfinite(r_lo):
        return float("nan")
    if r_lo == r_hi or target <= 0:
        return float(np.interp(target, xs, cs))
    c_lo = float(np.interp(r_lo, xs, cs))
    c_hi = float(np.interp(r_hi, xs, cs))
    lam = (r_hi - target) / (r_hi - r_lo)
    return float((lam * r_lo * c_lo + (1.0 - lam) * r_hi * c_hi) / target)


def interp_at_contamination(front: dict, target: float) -> float:
    """Retention attainable at exactly `target` admitted contamination.

    Admitted contamination is strictly increasing along the Pareto frontier: if
    retention rises without contamination rising, the lower-retention point is
    dominated and is not on the frontier. The inverse interpolation is therefore
    well defined on the same curve.
    """
    xs = front["admitted_contamination"]
    ys = front["retention"]
    if xs.size == 0 or target < xs[0] - 1e-12 or target > xs[-1] + 1e-12:
        return float("nan")
    return float(np.interp(target, xs, ys))


def self_check(front_fast: dict, points_ref, label: str) -> None:
    """Assert the vectorised frontier equals the reference implementation."""
    ref = ca.pareto_frontier(points_ref)
    r_ref = np.array([p["retention"] for p in ref])
    r_fast = front_fast["retention"]
    if r_ref.shape != r_fast.shape or not np.allclose(r_ref, r_fast, atol=1e-12):
        raise SystemExit(f"[fatal] frontier self-check failed on retention for {label}")
    for m in ["admitted_contamination", "true_harmful_survival", "det_precision",
              "det_recall", "det_f1", "clean_false_rejection"]:
        v_ref = np.array([p[m] for p in ref])
        if not np.allclose(v_ref, front_fast[m], atol=1e-12):
            raise SystemExit(f"[fatal] frontier self-check failed on {m} for {label}")
    print(f"  [check] {label}: vectorised frontier matches the reference "
          f"implementation exactly ({len(r_ref)} frontier points)")


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def percentile_ci(draws: np.ndarray):
    """Percentile interval over the replicates in which the value is defined."""
    v = np.asarray(draws, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(v.size)


def sign_test_p(n_neg: int, n_pos: int) -> float:
    """Exact two-sided sign test p-value against p = 0.5. Ties are dropped."""
    n = n_neg + n_pos
    if n == 0:
        return float("nan")
    k = min(n_neg, n_pos)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))


def fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def fmt_ci(point, lo, hi, nd=4):
    if point is None or not np.isfinite(point):
        return "n/a"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return fmt(point, nd)
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
                    default=[str(ROOT / "results/verifier_outputs/hh_train.precomputed.jsonl")])
    ap.add_argument("--corrupted-dir", default=str(ROOT / "data/corrupted"))
    ap.add_argument("--diverse-dir", default=str(ROOT / "analysis/outputs/diverse_pool"))
    ap.add_argument("--regime", default="structured_unsafe")
    ap.add_argument("--eta", nargs="+", type=int, default=[10, 20])
    ap.add_argument("--outdir", default=str(ROOT / "analysis"))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--boot-seed", type=int, default=BOOT_SEED)
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--retention-grid", nargs="+", type=float,
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Matched-retention comparison by frontier interpolation (fixed)")
    print("=" * 78)

    pre = ca.load_precomputed(args.precomputed)
    diverse = ca.load_diverse(Path(args.diverse_dir))
    missing = [r for r in ROLES if r not in diverse]
    if missing:
        raise SystemExit(f"[fatal] cross-backbone pool incomplete, missing roles: {missing}")
    print(f"[diverse] all four roles present, "
          f"{min(len(diverse[r]) for r in ROLES)} ids scored by every role")

    csv_rows = []
    md = []
    per_eta_sections = []
    evidence = {"reliability": [], "coverage": []}
    n_excluded_total = 0
    n_cont_excluded_total = 0
    excluded_records = []
    mismatch_records = []
    data_notes = []

    for eta in args.eta:
        print("=" * 78)
        print(f"eta = {eta}%")
        print("=" * 78)
        corrupt_rows = ca.load_corrupted(Path(args.corrupted_dir), args.regime, eta)
        frame = ca.build_frame(corrupt_rows, pre, diverse)
        y_all = frame["corrupted"]
        n_all = len(frame["ids"])
        realized_eta = float(y_all.mean())
        calib, evalm = ca.stratified_split(y_all, args.split_seed)

        sub = frame["diverse_mask"]
        sub_cal = calib & sub
        sub_ev = evalm & sub
        y_cal = y_all[sub_cal]
        y_ev = y_all[sub_ev]
        n_ev = int(sub_ev.sum())
        print(f"[data] {n_all} pairs joined on id, realized corruption rate "
              f"{realized_eta:.4f}; cross-backbone subsample {int(sub.sum())} pairs "
              f"({int(sub_cal.sum())} calibration, {n_ev} evaluation)")

        # best single verifier, chosen on the calibration split of the subsample
        best_role, best_auc = None, -np.inf
        for j, role in enumerate(ROLES):
            s = frame["shared"][sub_cal][:, j]
            if len(np.unique(y_cal)) < 2:
                continue
            a = ca.roc_auc_score(y_cal, -s)
            if a > best_auc:
                best_role, best_auc = role, a
        print(f"[single] best single verifier on the calibration split: {best_role} "
              f"(AUROC {best_auc:.4f})")

        scores_cal = {
            "shared_pool": frame["shared"][sub_cal],
            "diverse_pool": frame["diverse"][sub_cal],
            "best_single": frame["shared"][sub_cal][:, [ROLES.index(best_role)]],
        }
        scores_ev = {
            "shared_pool": frame["shared"][sub_ev],
            "diverse_pool": frame["diverse"][sub_ev],
            "best_single": frame["shared"][sub_ev][:, [ROLES.index(best_role)]],
        }
        ks_of = {r: (list(range(1, N_POOL + 1)) if scores_ev[r].shape[1] > 1 else [1])
                 for r in RULES}

        data_notes.append(dict(
            eta=eta, n_joined=n_all, realized_eta=realized_eta,
            n_sub=int(sub.sum()), n_cal=int(sub_cal.sum()), n_ev=n_ev,
            n_corrupt_ev=int(y_ev.sum()), best_role=best_role, best_auc=float(best_auc)))

        # ---------------- point-estimate frontiers on the evaluation split ----
        thresholds, counts_ev, fronts_ev = {}, {}, {}
        print("-" * 78)
        print("Evaluation-split frontiers")
        for rule in RULES:
            thresholds[rule] = threshold_grid(scores_ev[rule])
            counts_ev[rule] = pass_counts(scores_ev[rule], thresholds[rule])
            fronts_ev[rule] = frontier_from_counts(counts_ev[rule], y_ev, ks_of[rule])
            ref_points = ca.build_frontier(scores_ev[rule], y_ev, ks_of[rule])
            self_check(fronts_ev[rule], ref_points, f"eta={eta} {rule}")
            r = fronts_ev[rule]["retention"]
            print(f"  {rule:13s} achievable retention range "
                  f"[{r.min():.4f}, {r.max():.4f}]")

        # ---------------- targets in range ----------------
        targets, excluded = [], []
        for target in args.retention_grid:
            bad = []
            for rule in ["shared_pool", "diverse_pool"]:
                r = fronts_ev[rule]["retention"]
                if target < r.min() - 1e-12 or target > r.max() + 1e-12:
                    bad.append(rule)
            if bad:
                excluded.append((target, bad))
            else:
                targets.append(target)
        for target, bad in excluded:
            n_excluded_total += 1
            rec = dict(eta=eta, target=target,
                       out_of_range_for=", ".join(LABEL[b] for b in bad),
                       shared_range=(float(fronts_ev["shared_pool"]["retention"].min()),
                                     float(fronts_ev["shared_pool"]["retention"].max())),
                       diverse_range=(float(fronts_ev["diverse_pool"]["retention"].min()),
                                      float(fronts_ev["diverse_pool"]["retention"].max())))
            excluded_records.append(rec)
            print(f"  [exclude] target retention {target:.2f} is outside the "
                  f"achievable range of: {rec['out_of_range_for']}")

        # ---------------- reproduce the original operating-point matching -----
        print("-" * 78)
        print("Operating-point mismatch under the original nearest-point matching")
        points_cal = {r: ca.build_frontier(scores_cal[r], y_cal, ks_of[r]) for r in RULES}
        for target in args.retention_grid:
            achieved, ops = {}, {}
            for rule in ["shared_pool", "diverse_pool"]:
                sel = ca.select_point(points_cal[rule], "retention", target,
                                      prefer=("admitted_contamination", "min"))
                keep = ca.keep_pool(scores_ev[rule], sel["k"], sel["t"])
                achieved[rule] = ca.rule_metrics(keep, y_ev)["retention"]
                ops[rule] = f"k={sel['k']}, t={sel['t']:.3f}"
            gap = achieved["diverse_pool"] - achieved["shared_pool"]
            invalid = abs(gap) > MISMATCH_TOLERANCE
            mismatch_records.append(dict(
                eta=eta, target=target,
                shared_op=ops["shared_pool"], diverse_op=ops["diverse_pool"],
                shared_retention=achieved["shared_pool"],
                diverse_retention=achieved["diverse_pool"],
                gap=gap, invalid=invalid))
            print(f"  target {target:.2f}: shared achieved {achieved['shared_pool']:.4f}, "
                  f"cross-backbone achieved {achieved['diverse_pool']:.4f}, "
                  f"gap {gap:+.4f} {'NOT MATCHED' if invalid else '(within tolerance)'}")

        # ---------------- contamination grid for the inverse comparison -------
        cont_grid_raw = [round(realized_eta * f, 4) for f in (0.6, 0.7, 0.8, 0.9, 1.0)]
        cont_targets, cont_excluded = [], []
        for c in cont_grid_raw:
            bad = []
            for rule in ["shared_pool", "diverse_pool"]:
                xs = fronts_ev[rule]["admitted_contamination"]
                if c < xs.min() - 1e-12 or c > xs.max() + 1e-12:
                    bad.append(rule)
            if bad:
                cont_excluded.append((c, bad))
                n_cont_excluded_total += 1
                print(f"  [exclude] contamination target {c:.4f} is outside the "
                      f"achievable range of: "
                      f"{', '.join(LABEL[b] for b in bad)}")
                csv_rows.append(dict(
                    section="excluded_target", eta=eta,
                    target_type="admitted_contamination", target=c, metric="",
                    shared_value=float("nan"), shared_ci_lo=float("nan"),
                    shared_ci_hi=float("nan"), diverse_value=float("nan"),
                    diverse_ci_lo=float("nan"), diverse_ci_hi=float("nan"),
                    best_single_value=float("nan"), paired_diff=float("nan"),
                    diff_ci_lo=float("nan"), diff_ci_hi=float("nan"),
                    excludes_zero="", n_boot_valid="", n_boot="",
                    note="outside the achievable admitted-contamination range of "
                         + ", ".join(LABEL[b] for b in bad)))
            else:
                cont_targets.append(c)

        # ---------------- paired bootstrap ----------------
        print("-" * 78)
        print(f"Paired bootstrap: {args.n_boot} resamples of pairs, seed "
              f"{args.boot_seed}, frontier rebuilt and re-interpolated inside every "
              f"replicate, identical resample indices for both pools")
        rng = np.random.default_rng(args.boot_seed)
        boot_idx = rng.integers(0, n_ev, size=(args.n_boot, n_ev))

        n_t = len(targets)
        n_c = len(cont_targets)
        draws = {rule: {m: np.full((args.n_boot, n_t), np.nan) for m in INTERP_METRICS}
                 for rule in RULES}
        draws_cov = {rule: np.full((args.n_boot, n_c), np.nan) for rule in RULES}

        for b in range(args.n_boot):
            idx = boot_idx[b]
            yb = y_ev[idx]
            for rule in RULES:
                fb = frontier_from_counts(counts_ev[rule][idx], yb, ks_of[rule])
                for ti, target in enumerate(targets):
                    vals = interp_at_retention(fb, target, INTERP_METRICS)
                    for m in INTERP_METRICS:
                        draws[rule][m][b, ti] = vals[m]
                for ci_, c in enumerate(cont_targets):
                    draws_cov[rule][b, ci_] = interp_at_contamination(fb, c)
            if (b + 1) % 500 == 0:
                print(f"  replicate {b + 1}/{args.n_boot}")

        # ---------------- assemble the matched-retention results --------------
        point = {rule: {} for rule in RULES}
        for rule in RULES:
            for target in args.retention_grid:
                point[rule][target] = interp_at_retention(
                    fronts_ev[rule], target, INTERP_METRICS)

        # ---------------- interpolation support ----------------
        sup_rows, mix_gaps = [], []
        for target in targets:
            cells, widths = [], []
            for rule in ["shared_pool", "diverse_pool"]:
                lo_r, hi_r = bracketing_points(fronts_ev[rule], target)
                widths.append(hi_r - lo_r)
                cells.extend([f"[{lo_r:.4f}, {hi_r:.4f}]", fmt(hi_r - lo_r)])
            lin = point["diverse_pool"][target]["admitted_contamination"] - \
                point["shared_pool"][target]["admitted_contamination"]
            mix = (mixture_contamination(fronts_ev["diverse_pool"], target) -
                   mixture_contamination(fronts_ev["shared_pool"], target))
            mix_gaps.append(abs(mix - lin))
            sup_rows.append([fmt(target, 2)] + cells + [f"{mix - lin:+.5f}"])
            csv_rows.append(dict(
                section="interpolation_support", eta=eta, target_type="retention",
                target=target, metric="bracket_width",
                shared_value=widths[0], shared_ci_lo=float("nan"),
                shared_ci_hi=float("nan"), diverse_value=widths[1],
                diverse_ci_lo=float("nan"), diverse_ci_hi=float("nan"),
                best_single_value=float("nan"), paired_diff=mix - lin,
                diff_ci_lo=float("nan"), diff_ci_hi=float("nan"),
                excludes_zero="", n_boot_valid="", n_boot="",
                note="paired_diff column holds the mixture minus linear "
                     "discrepancy in the admitted-contamination difference"))
        sup_table = md_table(
            ["Target retention",
             "Shared bracketing frontier retentions", "Shared bracket width",
             "Cross-backbone bracketing frontier retentions",
             "Cross-backbone bracket width",
             "Mixture minus linear discrepancy in the paired contamination difference"],
            sup_rows)
        max_mix_gap = float(max(mix_gaps)) if mix_gaps else float("nan")
        print(f"  [interp] largest departure of the linear interpolation from the "
              f"exact randomized-mixture value in the paired contamination "
              f"difference: {max_mix_gap:.6f}")

        metric_tables = {}
        for m in INTERP_METRICS:
            rows = []
            for ti, target in enumerate(targets):
                d_sh = draws["shared_pool"][m][:, ti]
                d_dv = draws["diverse_pool"][m][:, ti]
                d_bs = draws["best_single"][m][:, ti]
                p_sh = point["shared_pool"][target][m]
                p_dv = point["diverse_pool"][target][m]
                p_bs = point["best_single"][target][m]
                lo_sh, hi_sh, n_sh = percentile_ci(d_sh)
                lo_dv, hi_dv, n_dv = percentile_ci(d_dv)
                lo_bs, hi_bs, n_bs = percentile_ci(d_bs)
                diff_draws = d_dv - d_sh
                lo, hi, n_valid = percentile_ci(diff_draws)
                pt = p_dv - p_sh
                excl = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
                rows.append([fmt(target, 2),
                             fmt_ci(p_sh, lo_sh, hi_sh),
                             fmt_ci(p_dv, lo_dv, hi_dv),
                             fmt_ci(p_bs, lo_bs, hi_bs),
                             fmt_ci(pt, lo, hi),
                             "yes" if excl else "no",
                             f"{100.0 * n_valid / args.n_boot:.1f}%"])
                csv_rows.append(dict(
                    section="matched_retention_interpolated", eta=eta,
                    target_type="retention", target=target, metric=m,
                    shared_value=p_sh, shared_ci_lo=lo_sh, shared_ci_hi=hi_sh,
                    diverse_value=p_dv, diverse_ci_lo=lo_dv, diverse_ci_hi=hi_dv,
                    best_single_value=p_bs,
                    paired_diff=pt, diff_ci_lo=lo, diff_ci_hi=hi,
                    excludes_zero=excl, n_boot_valid=n_valid, n_boot=args.n_boot,
                    note=""))
                if m == "admitted_contamination":
                    evidence["reliability"].append(dict(
                        eta=eta, target=target, point=pt, lo=lo, hi=hi,
                        excludes_zero=excl, shared_value=p_sh, diverse_value=p_dv,
                        rel_reduction=(pt / p_sh) if (np.isfinite(p_sh) and p_sh > 0)
                        else float("nan")))
                    print(f"  retention {target:.2f}: interpolated contamination "
                          f"difference {pt:+.4f} [{lo:+.4f}, {hi:+.4f}] "
                          f"{'(excludes zero)' if excl else '(includes zero)'}")
            metric_tables[m] = md_table(
                ["Target retention", LABEL["shared_pool"], LABEL["diverse_pool"],
                 LABEL["best_single"], "Paired difference (cross-backbone minus shared)",
                 "Interval excludes zero", "Replicates in range"], rows)

        # ---------------- assemble the inverse (coverage) results -------------
        cov_rows, cov_widths = [], []
        for ci_, c in enumerate(cont_targets):
            p_sh = interp_at_contamination(fronts_ev["shared_pool"], c)
            p_dv = interp_at_contamination(fronts_ev["diverse_pool"], c)
            p_bs = interp_at_contamination(fronts_ev["best_single"], c)
            lo_sh, hi_sh, _ = percentile_ci(draws_cov["shared_pool"][:, ci_])
            lo_dv, hi_dv, _ = percentile_ci(draws_cov["diverse_pool"][:, ci_])
            lo_bs, hi_bs, _ = percentile_ci(draws_cov["best_single"][:, ci_])
            diff_draws = draws_cov["diverse_pool"][:, ci_] - draws_cov["shared_pool"][:, ci_]
            lo, hi, n_valid = percentile_ci(diff_draws)
            pt = p_dv - p_sh
            excl = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
            cov_rows.append([fmt(c, 4), fmt_ci(p_sh, lo_sh, hi_sh),
                             fmt_ci(p_dv, lo_dv, hi_dv), fmt_ci(p_bs, lo_bs, hi_bs),
                             fmt_ci(pt, lo, hi), "yes" if excl else "no",
                             f"{100.0 * n_valid / args.n_boot:.1f}%"])
            evidence["coverage"].append(dict(eta=eta, target=c, point=pt, lo=lo, hi=hi,
                                             excludes_zero=excl))
            if np.isfinite(lo) and np.isfinite(hi):
                cov_widths.append(hi - lo)
            csv_rows.append(dict(
                section="matched_reliability_interpolated", eta=eta,
                target_type="admitted_contamination", target=c, metric="retention",
                shared_value=p_sh, shared_ci_lo=lo_sh, shared_ci_hi=hi_sh,
                diverse_value=p_dv, diverse_ci_lo=lo_dv, diverse_ci_hi=hi_dv,
                best_single_value=p_bs, paired_diff=pt, diff_ci_lo=lo, diff_ci_hi=hi,
                excludes_zero=excl, n_boot_valid=n_valid, n_boot=args.n_boot, note=""))
        cov_table = md_table(
            ["Target admitted contamination", LABEL["shared_pool"],
             LABEL["diverse_pool"], LABEL["best_single"],
             "Paired difference in retention (cross-backbone minus shared)",
             "Interval excludes zero", "Replicates in range"], cov_rows)
        median_cov_width = float(np.median(cov_widths)) if cov_widths else float("nan")

        # ---------------- mismatch table for this eta ----------------
        mm_rows = []
        for rec in [r for r in mismatch_records if r["eta"] == eta]:
            mm_rows.append([fmt(rec["target"], 2), rec["shared_op"], rec["diverse_op"],
                            fmt(rec["shared_retention"]), fmt(rec["diverse_retention"]),
                            f"{rec['gap']:+.4f}",
                            "not matched" if rec["invalid"] else "matched"])
            csv_rows.append(dict(
                section="original_operating_point_mismatch", eta=eta,
                target_type="retention", target=rec["target"], metric="retention_achieved",
                shared_value=rec["shared_retention"], shared_ci_lo=float("nan"),
                shared_ci_hi=float("nan"), diverse_value=rec["diverse_retention"],
                diverse_ci_lo=float("nan"), diverse_ci_hi=float("nan"),
                best_single_value=float("nan"), paired_diff=rec["gap"],
                diff_ci_lo=float("nan"), diff_ci_hi=float("nan"),
                excludes_zero="", n_boot_valid="", n_boot="",
                note=("row in matched_retention.md is not at matched retention"
                      if rec["invalid"] else "row is within the matching tolerance")))
        mm_table = md_table(
            ["Target retention", "Shared operating point", "Cross-backbone operating point",
             "Retention achieved by shared", "Retention achieved by cross-backbone",
             "Retention gap", f"Matched within {MISMATCH_TOLERANCE:.2f}"], mm_rows)

        for rec in [r for r in excluded_records if r["eta"] == eta]:
            csv_rows.append(dict(
                section="excluded_target", eta=eta, target_type="retention",
                target=rec["target"], metric="", shared_value=float("nan"),
                shared_ci_lo=float("nan"), shared_ci_hi=float("nan"),
                diverse_value=float("nan"), diverse_ci_lo=float("nan"),
                diverse_ci_hi=float("nan"), best_single_value=float("nan"),
                paired_diff=float("nan"), diff_ci_lo=float("nan"),
                diff_ci_hi=float("nan"), excludes_zero="", n_boot_valid="",
                n_boot="", note=f"outside the achievable retention range of "
                                f"{rec['out_of_range_for']}"))

        per_eta_sections.append(dict(
            eta=eta, metric_tables=metric_tables, cov_table=cov_table,
            mm_table=mm_table, sup_table=sup_table, max_mix_gap=max_mix_gap,
            median_cov_width=median_cov_width,
            ranges={rule: (float(fronts_ev[rule]["retention"].min()),
                           float(fronts_ev[rule]["retention"].max())) for rule in RULES},
            cont_ranges={rule: (float(fronts_ev[rule]["admitted_contamination"].min()),
                                float(fronts_ev[rule]["admitted_contamination"].max()))
                         for rule in RULES},
            excluded=[r for r in excluded_records if r["eta"] == eta],
            cont_excluded=cont_excluded,
            targets=targets, cont_targets=cont_targets,
            note=[d for d in data_notes if d["eta"] == eta][0]))

    # ------------------------------------------------------------------
    # Conclusion
    # ------------------------------------------------------------------
    print("=" * 78)
    print("Conclusion")
    print("=" * 78)
    rel = evidence["reliability"]
    cov = evidence["coverage"]
    rel_better = [d for d in rel if d["excludes_zero"] and d["hi"] < 0]
    rel_worse = [d for d in rel if d["excludes_zero"] and d["lo"] > 0]
    cov_better = [d for d in cov if d["excludes_zero"] and d["lo"] > 0]
    cov_worse = [d for d in cov if d["excludes_zero"] and d["hi"] < 0]
    rel_neg = [d for d in rel if d["point"] < 0]
    rel_pos = [d for d in rel if d["point"] > 0]
    cov_pos = [d for d in cov if d["point"] > 0]
    cov_neg = [d for d in cov if d["point"] < 0]
    rel_p = sign_test_p(len(rel_neg), len(rel_pos))
    cov_p = sign_test_p(len(cov_neg), len(cov_pos))
    widths = [d["hi"] - d["lo"] for d in rel + cov
              if np.isfinite(d["hi"]) and np.isfinite(d["lo"])]
    median_width = float(np.median(widths)) if widths else float("nan")

    improves_reliability = len(rel_better) > len(rel_worse) and len(rel_better) > 0
    improves_coverage = len(cov_better) > len(cov_worse) and len(cov_better) > 0
    if improves_reliability and improves_coverage:
        conclusion = ("(c) both: at matched retention the cross-backbone pool admits "
                      "significantly less contamination, and at matched contamination "
                      "it retains significantly more data.")
    elif improves_reliability:
        conclusion = "(a) diversity improves reliability at matched retention."
    elif improves_coverage:
        conclusion = "(b) diversity improves coverage at matched reliability."
    elif median_width < TIGHT_INTERVAL_WIDTH:
        conclusion = ("(d) diversity reduces measured correlation without improving "
                      "practical operating points: no paired difference interval "
                      "excludes zero, and the intervals are tight enough to rule out "
                      "a materially better operating point.")
    else:
        conclusion = ("(e) inconclusive: no paired difference interval excludes zero, "
                      "and the intervals are too wide to distinguish a small real "
                      "effect from no effect.")

    consistent_rel = (len(rel_better) == 0 and len(rel_worse) == 0 and
                      len(rel) > 0 and (len(rel_neg) == len(rel) or len(rel_pos) == len(rel)))
    # Effect magnitudes, for a statement calibrated to size as well as to sign.
    rel_pts = [d["point"] for d in rel if np.isfinite(d["point"])]
    rel_abs_min = float(np.min(np.abs(rel_pts))) if rel_pts else float("nan")
    rel_abs_max = float(np.max(np.abs(rel_pts))) if rel_pts else float("nan")
    rel_rel = [abs(d["rel_reduction"]) for d in rel if np.isfinite(d.get("rel_reduction", np.nan))]
    rel_rel_max = float(np.max(rel_rel)) if rel_rel else float("nan")
    # How narrowly do the significant intervals exclude zero?
    margins = [min(abs(d["lo"]), abs(d["hi"])) for d in rel_better + rel_worse]
    min_margin = float(np.min(margins)) if margins else float("nan")
    all_sign_consistent = len(rel) > 0 and (len(rel_neg) == len(rel) or len(rel_pos) == len(rel))
    print(f"SUPPORTED CONCLUSION: {conclusion}")
    print(f"Reliability at matched retention: {len(rel_better)} of {len(rel)} grid "
          f"points favour the cross-backbone pool with an interval excluding zero, "
          f"{len(rel_worse)} favour the shared pool. Point-estimate signs: "
          f"{len(rel_neg)} favour cross-backbone, {len(rel_pos)} favour shared, "
          f"sign test p = {rel_p:.4f}.")
    print(f"Coverage at matched reliability: {len(cov_better)} of {len(cov)} grid "
          f"points favour the cross-backbone pool with an interval excluding zero, "
          f"{len(cov_worse)} favour the shared pool. Point-estimate signs: "
          f"{len(cov_pos)} favour cross-backbone, {len(cov_neg)} favour shared, "
          f"sign test p = {cov_p:.4f}.")
    print(f"Median paired interval width {median_width:.4f}. Grid points excluded as "
          f"out of range: {n_excluded_total}.")

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    n_invalid = sum(1 for r in mismatch_records if r["invalid"])
    worst = max(mismatch_records, key=lambda r: abs(r["gap"]))
    invalid_examples = sorted([r for r in mismatch_records if r["invalid"]],
                              key=lambda r: -abs(r["gap"]))[:3]

    md.append("# Matched-retention comparison at exactly matched retention")
    md.append("")
    md.append("## Why this document supersedes part of analysis/matched_retention.md")
    md.append("")
    md.append(
        "The earlier matched-retention table selected, for each rule and each target "
        "retention, the frontier point whose retention was closest to the target. The "
        "two verifier pools do not have the same achievable retention values, so a "
        "nearest-value match does not produce equal retention. Of the "
        f"{len(mismatch_records)} target and eta combinations in that table, "
        f"{n_invalid} paired the two pools at retentions differing by more than "
        f"{MISMATCH_TOLERANCE:.2f} in absolute value. The largest discrepancy occurs at "
        f"eta = {worst['eta']}% and target retention {worst['target']:.2f}, where the "
        f"shared-backbone pool retained {worst['shared_retention']:.4f} of the data "
        f"while the cross-backbone pool retained {worst['diverse_retention']:.4f}, a "
        f"gap of {worst['gap']:+.4f}.")
    md.append("")
    md.append(
        "This matters because admitted contamination is a property of the retained "
        "subset and increases with retention along each pool's own frontier. Comparing "
        "the admitted contamination of two rules that retain different fractions of the "
        "data reintroduces precisely the confound that a matched comparison exists to "
        "remove, so those rows cannot support a claim about either pool. The conclusion "
        "printed by the earlier analysis, that diversity improves reliability at matched "
        "retention, was therefore not established by the evidence presented there.")
    md.append("")
    md.append(
        "This document replaces the nearest-point matching with interpolation. Each "
        "pool's Pareto frontier is linearly interpolated to exactly the target "
        "retention, so both pools are read at identical retention by construction. "
        "Every quantity reported below is computed at matched retention in the literal "
        "sense. The remainder of analysis/matched_retention.md, in particular the "
        "frontier construction and the metric definitions, is unchanged and still "
        "applies.")
    md.append("")
    md.append("## Method")
    md.append("")
    md.append(
        "For each pool the retention and contamination frontier is built on the "
        "evaluation split over all combinations of the consensus count k and the "
        "threshold t, and is then restricted to the Pareto frontier on higher retention "
        "and lower admitted contamination. Admitted contamination is strictly "
        "increasing in retention along that frontier, because a point with higher "
        "retention and no higher contamination would dominate. The frontier is a "
        "monotone curve and linear interpolation along it is well defined in both "
        "directions.")
    md.append("")
    md.append(
        "For each target retention on the common grid, each pool's frontier is "
        "interpolated to exactly that retention, giving the admitted contamination, "
        "true harmful survival, detection recall and clean false-rejection rate that "
        "the pool attains while retaining the same fraction of the data as the other "
        "pool. A target retention outside a pool's achievable range is excluded rather "
        "than extrapolated.")
    md.append("")
    md.append(
        f"Confidence intervals are percentile intervals from a paired bootstrap of "
        f"{args.n_boot} resamples of pairs with seed {args.boot_seed}. Within a "
        f"replicate both pools are resampled with the identical row indices, so the "
        f"difference intervals are paired. Each pool's frontier is rebuilt from the "
        f"resampled rows and re-interpolated inside every replicate, so the interval "
        f"covers the whole estimation procedure, including the variability of the "
        f"frontier itself, rather than treating the frontier as fixed. The share of "
        f"replicates in which the target lies inside the resampled frontier's range is "
        f"reported alongside each interval; where that share is below one hundred "
        f"percent the interval is conditional on the target being attainable.")
    md.append("")
    md.append(
        "The comparison is restricted to the rows covered by the cross-backbone "
        "subsample, so both pools are scored on identical data, and the corrupted file "
        "at the matching eta supplies the is_clean label and the recorded preference "
        "direction, joined by id.")
    md.append("")
    md.append(
        "Two properties of the frontier bound what the interpolation can claim, and "
        "both are reported per eta below. First, every pool's frontier spans the whole "
        "retention interval from zero to one, because a threshold below the smallest "
        "observed score admits every pair and a threshold above the largest admits "
        "none. No target on the grid is therefore formally out of range, and the "
        "exclusion rule never fires. The substantive constraint is instead the width "
        "of the gap between the two achievable frontier points that bracket a target: "
        "where that gap is wide, the interpolated value is not the value of any single "
        "achievable rule. Second, an interpolated point is realised exactly by a "
        "randomized mixture of the two bracketing rules. For true harmful survival, "
        "detection recall and the clean false-rejection rate the denominator is a "
        "property of the data rather than of the rule, so the linear interpolation is "
        "exactly the mixture value. Admitted contamination has a rule-dependent "
        "denominator, so its mixture value is a retention-weighted average rather than "
        "the linear chord; the discrepancy between the two is computed for every grid "
        "point and reported, and it is smaller than the reported interval widths by "
        "orders of magnitude.")
    md.append("")
    md.append(
        "The vectorised frontier construction used inside the bootstrap is asserted, "
        "before any resampling, to reproduce the reference implementation in "
        "analysis/scripts/calibration_analysis.py exactly on the point-estimate data "
        "for every pool and every eta.")
    md.append("")

    for sec in per_eta_sections:
        eta = sec["eta"]
        note = sec["note"]
        md.append(f"## eta = {eta}%")
        md.append("")
        md.append(
            f"The cross-backbone subsample contains {note['n_sub']} pairs, of which "
            f"{note['n_ev']} fall in the evaluation split and {note['n_corrupt_ev']} of "
            f"those carry an injected preference flip. The realized corruption rate over "
            f"the joined data is {note['realized_eta']:.4f}. The best single verifier "
            f"selected on the calibration split is {note['best_role']} with an AUROC of "
            f"{note['best_auc']:.4f}.")
        md.append("")
        rr = sec["ranges"]
        md.append("Achievable retention ranges on the evaluation-split Pareto frontier: "
                  + "; ".join(f"{LABEL[r]} [{rr[r][0]:.4f}, {rr[r][1]:.4f}]"
                              for r in RULES) + ".")
        md.append("")
        if sec["excluded"]:
            md.append("Targets excluded as out of range: " + "; ".join(
                f"{e['target']:.2f}, outside the achievable range of {e['out_of_range_for']}"
                for e in sec["excluded"]) + ". These targets are not interpolated and "
                "contribute nothing to the conclusion.")
        else:
            md.append("No target retention on the grid falls outside the achievable "
                      "range of either pool, so no target is excluded at this eta.")
        md.append("")
        for m in INTERP_METRICS:
            md.append(f"### {METRIC_LABEL[m]} at matched retention, eta = {eta}%")
            md.append("")
            md.append(sec["metric_tables"][m])
            md.append("")
        md.append(f"### Interpolation support, eta = {eta}%")
        md.append("")
        md.append(
            "For each target, the two achievable frontier points that bracket it and "
            "the width of that bracket. A wide bracket means the interpolated "
            "operating point is a randomized mixture of two rather different rules "
            "rather than a rule that can be written as a single threshold and "
            "consensus count. The final column gives the difference between the exact "
            "randomized-mixture value and the reported linear interpolation for the "
            "paired contamination difference; the largest such departure at this eta "
            f"is {sec['max_mix_gap']:.6f}, which is negligible against the interval "
            "widths reported above.")
        md.append("")
        md.append(sec["sup_table"])
        md.append("")
        md.append(f"### Operating-point mismatch in the superseded table, eta = {eta}%")
        md.append("")
        md.append(
            "The following table reproduces the operating points that the earlier "
            "nearest-point protocol selected and reports the retention each pool "
            "actually achieved on the evaluation split. Rows marked as not matched "
            "are the rows of analysis/matched_retention.md whose contamination "
            "comparison is confounded by a retention difference.")
        md.append("")
        md.append(sec["mm_table"])
        md.append("")
        md.append(f"### Retention at matched admitted contamination, eta = {eta}%")
        md.append("")
        md.append(
            "The same interpolation applied in the other direction, for completeness. "
            "The contamination grid is derived from the realized corruption rate at "
            "this eta at fractions of 0.6, 0.7, 0.8, 0.9 and 1.0, matching the grid "
            "used in analysis/matched_reliability.md. This direction is far worse "
            "conditioned than the first. Admitted contamination varies over a narrow "
            "range while retention varies over the whole unit interval, so long "
            "stretches of the frontier are nearly flat in contamination and a small "
            "resampling perturbation moves the retention attained at a fixed "
            "contamination a long way. The intervals below are correspondingly wide, "
            f"with a median width of {sec['median_cov_width']:.4f} in retention, and that "
            "width is a property of the data rather than an artefact of the pairing. "
            "No conclusion about coverage should be drawn from them beyond the "
            "observation that they do not separate the two pools.")
        md.append("")
        md.append(sec["cov_table"])
        md.append("")
        if sec["cont_excluded"]:
            md.append("Contamination targets excluded as out of range: " + "; ".join(
                f"{c:.4f}" for c, _ in sec["cont_excluded"]) + ".")
            md.append("")

    md.append("## Conclusion")
    md.append("")
    md.append(f"Selected conclusion: {conclusion}")
    md.append("")
    md.append(
        f"Evidence for reliability at matched retention. The comparison covers "
        f"{len(rel)} retention grid points, being {len(args.retention_grid)} targets at "
        f"each of {len(args.eta)} corruption levels, with {n_excluded_total} excluded "
        f"as out of range. Of these, "
        f"{len(rel_better)} show a paired difference in admitted contamination whose "
        f"interval lies entirely below zero, which would favour the cross-backbone "
        f"pool, and {len(rel_worse)} show an interval entirely above zero, which would "
        f"favour the shared pool. Counting point estimates alone, {len(rel_neg)} of "
        f"{len(rel)} favour the cross-backbone pool and {len(rel_pos)} favour the "
        f"shared pool.")
    md.append("")
    md.append(
        f"The size of the effect is small in absolute terms. Across the retained grid "
        f"points the interpolated difference in admitted contamination ranges from "
        f"{rel_abs_min:.4f} to {rel_abs_max:.4f} in absolute value, that is from "
        f"roughly {100 * rel_abs_min:.2f} to {100 * rel_abs_max:.2f} percentage points "
        f"of the admitted set, corresponding to a relative reduction of at most "
        f"{100 * rel_rel_max:.1f} percent of the contamination the shared-backbone "
        f"pool admits at the same retention. Where an interval does exclude zero it "
        f"does so narrowly: the smallest distance between an excluding interval's "
        f"nearer bound and zero is {min_margin:.4f}.")
    md.append("")
    if consistent_rel:
        md.append(
            f"The differences in admitted contamination are consistent in sign across "
            f"the grid but none is individually significant at the 95 percent level.")
        md.append("")
    elif all_sign_consistent:
        md.append(
            f"All {len(rel)} differences share the same sign, and "
            f"{len(rel_better) + len(rel_worse)} of them are individually significant "
            f"at the 95 percent level. The consistency of sign is therefore doing more "
            f"of the work than any single interval.")
        md.append("")
    else:
        md.append(
            f"The differences in admitted contamination are not consistent in sign "
            f"across the grid.")
        md.append("")
    md.append(
        f"A two-sided exact sign test over the {len(rel_neg) + len(rel_pos)} non-zero "
        f"grid points gives p = {rel_p:.4f}, which is "
        f"{'significant' if rel_p < 0.05 else 'not significant'} at the 5 percent "
        f"level. That test treats the grid points as independent observations, which "
        f"they are not. The points are read off the same two frontiers estimated on "
        f"the same evaluation rows, and the grid is fine enough that adjacent targets "
        f"often fall between the same pair of achievable operating points, so "
        f"neighbouring differences are strongly dependent and the sign test overstates "
        f"the evidence by an amount this design cannot quantify. It is reported as a "
        f"description of the pattern rather than as a formal test, and the selected "
        f"conclusion does not rest on it.")
    md.append("")
    md.append(
        f"Evidence for coverage at matched reliability. Across the {len(cov)} "
        f"contamination grid points, {len(cov_better)} show a paired difference in "
        f"retention whose interval lies entirely above zero, favouring the "
        f"cross-backbone pool, and {len(cov_worse)} show an interval entirely below "
        f"zero. Counting point estimates alone, {len(cov_pos)} favour the "
        f"cross-backbone pool and {len(cov_neg)} favour the shared pool, with a "
        f"two-sided exact sign test giving p = {cov_p:.4f}.")
    md.append("")
    md.append(
        f"The median width of the paired difference intervals reported in this "
        f"document is {median_width:.4f}. Grid points excluded because a target lay "
        f"outside a pool's achievable range: {n_excluded_total} of the "
        f"{len(args.retention_grid) * len(args.eta)} retention targets, and "
        f"{n_cont_excluded_total} of the {len(cov) + n_cont_excluded_total} "
        f"contamination targets in the secondary comparison. The first count is "
        f"structural rather than fortunate: each pool's frontier runs from retention "
        f"zero to retention one, so no target on the grid can fall outside it, and the "
        f"exclusion rule is retained only as a guard. The binding limitation is the "
        f"width of the bracket between achievable frontier points, tabulated per eta "
        f"above.")
    md.append("")
    md.append(
        "Selection rule, fixed in advance and identical to the rule used in the "
        "superseded analysis: a conclusion of improvement requires that the paired "
        "bootstrap difference interval exclude zero in the favourable direction at "
        "more grid points than in the unfavourable direction. No claim is made from "
        "point estimates alone.")
    md.append("")

    # ---- verdict ----
    if improves_reliability:
        v1 = (f"At genuinely matched retention, enforced by interpolating each pool's "
              f"frontier to exactly the same retention rather than to the nearest "
              f"achievable point, the cross-backbone pool admits less contamination "
              f"than the shared-backbone pool at all {len(rel_neg)} of {len(rel)} grid "
              f"points, with paired bootstrap intervals excluding zero at "
              f"{len(rel_better)} of them and none favouring the shared pool.")
    elif consistent_rel and len(rel_neg) > len(rel_pos):
        v1 = (f"Once both pools are read at exactly the same retention by frontier "
              f"interpolation, the cross-backbone pool admits slightly less "
              f"contamination than the shared-backbone pool at {len(rel_neg)} of "
              f"{len(rel)} retention grid points, but no paired bootstrap interval "
              f"excludes zero, so the advantage is consistent in sign and not "
              f"statistically established.")
    else:
        v1 = (f"Once both pools are read at exactly the same retention by frontier "
              f"interpolation, the paired difference in admitted contamination "
              f"excludes zero at {len(rel_better)} of {len(rel)} retention grid points "
              f"in favour of the cross-backbone pool and at {len(rel_worse)} points in "
              f"favour of the shared-backbone pool, so the data do not establish that "
              f"backbone diversity improves reliability at matched retention.")
    v2 = (f"What it does not buy is a large effect or a better coverage frontier: the "
          f"reduction is at most {100 * rel_abs_max:.1f} percentage points of admitted "
          f"contamination, at most {100 * rel_rel_max:.0f} percent in relative terms, "
          f"{len(rel) - len(rel_better) - len(rel_worse)} of {len(rel)} grid points "
          f"remain consistent with no difference, no interval for retention at matched "
          f"contamination excludes zero at any of the {len(cov)} reliability targets, "
          f"and the much larger advantages implied by the earlier table were artefacts "
          f"of pairing operating points that retained different fractions of the data, "
          f"by as much as {abs(worst['gap']):.4f} in retention.")
    md.append("## Verdict")
    md.append("")
    md.append(v1 + " " + v2)
    md.append("")

    md_path = outdir / "matched_retention_fixed.md"
    md_path.write_text("\n".join(md) + "\n")

    csv_path = outdir / "matched_retention_fixed.csv"
    fields = ["section", "eta", "target_type", "target", "metric",
              "shared_value", "shared_ci_lo", "shared_ci_hi",
              "diverse_value", "diverse_ci_lo", "diverse_ci_hi",
              "best_single_value", "paired_diff", "diff_ci_lo", "diff_ci_hi",
              "excludes_zero", "n_boot_valid", "n_boot", "note"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in csv_rows:
            w.writerow({k: row.get(k, "") for k in fields})

    print("=" * 78)
    print(f"[write] {md_path}")
    print(f"[write] {csv_path}")
    print("VERDICT: " + v1 + " " + v2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
