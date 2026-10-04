"""Reconstruct the complete seven-method x three-eta baseline grid for the
Full baseline grid.

Every value is either read from a result artifact or derived from read values by
an identity that is checked against an independently reported quantity. Nothing
is hand-entered. Run from the repository root:

    python3 analysis/scripts/build_full_grid.py

Outputs:
    analysis/full_baseline_grid.csv
    analysis/full_baseline_grid.md
    analysis/full_baseline_grid_compact.md

Metric conventions (see discussion/metric_definitions.md):
    retention              = n_kept / n_total
    admitted contamination = n_kept_corrupt / n_kept        <- reported in the
                             submitted tables under the label "harmful survival"
    harmful survival       = n_kept_corrupt / n_corrupt_total
Detector precision/recall are computed over the rejection decision, treating
"reject a corrupted pair" as a true positive.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N_TOTAL = 10_000
CORRUPTION = "structured_unsafe"

# Methods that emit an admission decision.
FILTERING = {"single", "consensus_k3", "oracle"}
# Methods that train on the full corrupted set (no admission decision).
OBJECTIVE_ONLY = {"raw", "noise_aware", "rdpo", "ipo"}

DISPLAY = {
    "raw": "Raw DPO",
    "noise_aware": "cDPO (label smoothing 0.1)",
    "rdpo": "rDPO",
    "ipo": "IPO",
    "single": "Single verifier (safety)",
    "consensus_k3": "Consensus (k=3 of 4)",
    "oracle": "Oracle (ground truth)",
}
ORDER = ["raw", "noise_aware", "rdpo", "ipo", "single", "consensus_k3", "oracle"]


def load_effective_eta() -> dict[int, dict]:
    """Realized corruption counts, from the corruption summaries."""
    out = {}
    for tag in (0, 10, 20):
        p = ROOT / f"data/corrupted/hh_train_{CORRUPTION}_eta{tag}.summary.json"
        d = json.loads(p.read_text())
        out[tag] = {
            "n_corrupt": int(d["corrupted_rows"]),
            "effective_eta": float(d["effective_eta"]),
            "source": str(p.relative_to(ROOT)),
        }
        assert out[tag]["n_corrupt"] == round(out[tag]["effective_eta"] * N_TOTAL)
    return out


def load_filter_metrics() -> dict[tuple[int, str], dict]:
    """Admission-side metrics from the aggregated pilot table."""
    out = {}
    with open(ROOT / "results/tables/main_results.csv") as f:
        for row in csv.DictReader(f):
            if row["corruption"] != CORRUPTION:
                continue
            if row["retention"] == "":
                continue
            out[(int(row["eta"]), row["method"])] = {
                "retention": float(row["retention"]),
                "admitted_contamination": float(row["harmful_survival_rate"]),
                "clean_retention": float(row["clean_retention_rate"]),
                "false_rejection_rate": float(row["false_rejection_rate"]),
                "reported_f1": float(row["corruption_detection_f1"]),
            }
    return out


def load_downstream() -> dict[tuple[int, str], dict]:
    """Downstream evaluation from the per-run eval JSONs (authoritative)."""
    out = {}
    for p in sorted((ROOT / "results/eval").glob(f"{CORRUPTION}_eta*.json")):
        stem = p.stem[len(CORRUPTION) + 1 :]           # e.g. "eta20_rdpo"
        eta = int(stem.split("_", 1)[0][3:])
        method = stem.split("_", 1)[1]
        d = json.loads(p.read_text())
        out[(eta, method)] = {
            "preference_acc": d["preference"]["preference_acc"],
            "mean_margin": d["preference"]["mean_margin"],
            "unsafe_refusal": d["refusal"]["unsafe_refusal_rate"],
            "benign_refusal": d["refusal"]["benign_refusal_rate"],
            "n_pref": d["preference"]["n"],
            "n_unsafe": d["refusal"]["unsafe_n"],
            "n_benign": d["refusal"]["benign_n"],
            "source": str(p.relative_to(ROOT)),
        }
    return out


def f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def main() -> None:
    eff = load_effective_eta()
    filt = load_filter_metrics()
    down = load_downstream()

    rows, checks = [], []
    for eta in (0, 10, 20):
        n_corrupt = eff[eta]["n_corrupt"]
        n_clean = N_TOTAL - n_corrupt
        for method in ORDER:
            d = down.get((eta, method))
            if d is None:
                continue
            rec = {
                "eta_nominal_pct": eta,
                "effective_eta": eff[eta]["effective_eta"],
                "method": DISPLAY[method],
                "method_key": method,
                "emits_admission_decision": method in FILTERING,
                "preference_acc": d["preference_acc"],
                "mean_margin": d["mean_margin"],
                "unsafe_refusal": d["unsafe_refusal"],
                "benign_refusal": d["benign_refusal"],
            }

            if method in OBJECTIVE_ONLY:
                # Trains on the full corrupted set by construction.
                rec.update(
                    retention=1.0,
                    training_contamination=eff[eta]["effective_eta"],
                    admitted_contamination=eff[eta]["effective_eta"],
                    harmful_survival=1.0 if n_corrupt else 0.0,
                    detector_precision=None,
                    detector_recall=None,
                    detector_f1=None,
                )
            else:
                m = filt[(eta, method)]
                kept = m["retention"] * N_TOTAL
                kept_corrupt = m["admitted_contamination"] * kept
                rejected_corrupt = n_corrupt - kept_corrupt
                rejected_clean = m["false_rejection_rate"] * n_clean
                if n_corrupt == 0:
                    prec = rec_ = f1v = 0.0
                else:
                    denom = rejected_corrupt + rejected_clean
                    prec = rejected_corrupt / denom if denom else 0.0
                    rec_ = rejected_corrupt / n_corrupt
                    f1v = f1(prec, rec_)
                    checks.append(
                        (eta, method, f1v, m["reported_f1"], abs(f1v - m["reported_f1"]))
                    )
                rec.update(
                    retention=m["retention"],
                    training_contamination=m["admitted_contamination"],
                    admitted_contamination=m["admitted_contamination"],
                    harmful_survival=(kept_corrupt / n_corrupt) if n_corrupt else 0.0,
                    detector_precision=prec,
                    detector_recall=rec_,
                    detector_f1=f1v,
                )
            rows.append(rec)

    # --- consistency check: derived detector F1 must reproduce the reported F1 --
    print("Derived-vs-reported detector F1 (must agree):")
    worst = 0.0
    for eta, method, derived, reported, delta in checks:
        worst = max(worst, delta)
        flag = "OK" if delta < 5e-3 else "MISMATCH"
        print(f"  eta={eta:>2}% {method:<13} derived={derived:.4f} "
              f"reported={reported:.4f} delta={delta:.5f}  {flag}")
    print(f"  worst absolute deviation: {worst:.5f}")

    fields = [
        "eta_nominal_pct", "effective_eta", "method", "method_key",
        "emits_admission_decision", "preference_acc", "mean_margin",
        "unsafe_refusal", "benign_refusal", "retention",
        "training_contamination", "admitted_contamination", "harmful_survival",
        "detector_precision", "detector_recall", "detector_f1",
    ]
    out_csv = ROOT / "analysis/full_baseline_grid.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv.relative_to(ROOT)} ({len(rows)} rows)")

    def fmt(v, pct=False, nd=3):
        if v is None:
            return "n/a"
        return f"{v*100:.1f}%" if pct else f"{v:.{nd}f}"

    # --- full markdown table ---
    lines = [
        "| eta (nominal) | eta (effective) | Method | Admission decision | Pref. acc. | Margin | Unsafe refusal | Benign refusal | Retention | Train contam. | Admitted contam. | Harmful survival | Det. P | Det. R | Det. F1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['eta_nominal_pct']}% | {r['effective_eta']*100:.2f}% | {r['method']} | "
            f"{'yes' if r['emits_admission_decision'] else 'no'} | "
            f"{fmt(r['preference_acc'])} | {fmt(r['mean_margin'])} | "
            f"{fmt(r['unsafe_refusal'])} | {fmt(r['benign_refusal'])} | "
            f"{fmt(r['retention'], pct=True)} | {fmt(r['training_contamination'], pct=True)} | "
            f"{fmt(r['admitted_contamination'], pct=True)} | {fmt(r['harmful_survival'], pct=True)} | "
            f"{fmt(r['detector_precision'])} | {fmt(r['detector_recall'])} | {fmt(r['detector_f1'])} |"
        )
    (ROOT / "analysis/full_baseline_grid.md").write_text("\n".join(lines) + "\n")
    print(f"wrote analysis/full_baseline_grid.md")

    # --- compact table for direct posting ---
    c = [
        "| eta | Method | Pref. acc. | Unsafe ref. | Retention | Admitted contam. | Harmful surv. | Det. F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        c.append(
            f"| {r['eta_nominal_pct']}% | {r['method']} | {fmt(r['preference_acc'])} | "
            f"{fmt(r['unsafe_refusal'])} | {fmt(r['retention'], pct=True)} | "
            f"{fmt(r['admitted_contamination'], pct=True)} | "
            f"{fmt(r['harmful_survival'], pct=True)} | {fmt(r['detector_f1'])} |"
        )
    (ROOT / "analysis/full_baseline_grid_compact.md").write_text("\n".join(c) + "\n")
    print("wrote analysis/full_baseline_grid_compact.md")


if __name__ == "__main__":
    main()
