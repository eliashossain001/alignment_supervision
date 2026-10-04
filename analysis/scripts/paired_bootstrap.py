"""Paired example-level bootstrap for the Raw-DPO vs consensus-k3 contamination gap.

Regenerates, from first principles, the interval cited in the paper and recorded
in `analysis/EVIDENCE_MANIFEST.md` under
`paired-bootstrap raw-consensus harm gap eta={10,20}`. The code that originally
produced those numbers is not in the tree, so nothing here is inherited from it:
the statistic is rebuilt from the definitions in `discussion/metric_definitions.md`
and the admission rule in `src/consensus_filter_precomputed.py`.

Two differences are bootstrapped, both paired at the example level:

  1. admitted-data contamination (metric_definitions.md 1.2)
         alpha(method) = kept_corrupt / n_kept
     Raw DPO admits everything, so alpha(raw) is the effective eta of the
     analysed set. The reported gap is  alpha(raw) - alpha(consensus_k3),
     positive when consensus admits cleaner data than no filtering at all.

  2. true harmful survival (metric_definitions.md 1.3)
         s(method) = kept_corrupt / n_corrupt_total
     s(raw) = 1 by construction. The reported gap is s(raw) - s(consensus_k3),
     the fraction of injected corruption that consensus removes.

Pairing: one resample of example indices per replicate, with replacement, and BOTH
methods recomputed on that same index set. The two methods therefore see identical
example draws in every replicate, so the difference is not inflated by independent
sampling noise in the two arms.

Usage (full precompute):

    python analysis/scripts/paired_bootstrap.py \
        --precomputed results/verifier_outputs/hh_train.precomputed.jsonl

Usage (smoke test on the in-progress shards):

    cat results/verifier_outputs/hh_train.precomputed.shard*.jsonl > /tmp/pre.jsonl
    python analysis/scripts/paired_bootstrap.py --precomputed /tmp/pre.jsonl --etas 20

Rows of the corrupted file whose id is absent from the precomputed file are
dropped, and the number dropped is reported and stored. On a partial precompute
the regenerated numbers will NOT match the manifest; the comparison table printed
at the end makes any deviation explicit rather than hiding it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROLES, VSE  # noqa: E402

MANIFEST_ROW = re.compile(
    r"^\|\s*paired-bootstrap raw-consensus harm gap eta=(\d+)\s*\|\s*([0-9.]+)\s*\|"
)


# --- io ---------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list:
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{ln}: malformed JSON ({e}). "
                                 f"If the precompute is still running, the last "
                                 f"line may be partially written.")
    return rows


def load_precomputed(path: Path) -> dict:
    pre = {}
    for r in load_jsonl(path):
        pre[r["id"]] = r
    return pre


def read_manifest_published(path: Path) -> dict:
    """eta -> {'value': float, 'ci': [lo, hi]} parsed out of the evidence manifest."""
    published = {}
    if not path.exists():
        return published
    for line in path.read_text().splitlines():
        m = MANIFEST_ROW.match(line)
        if not m:
            continue
        eta, value = int(m.group(1)), float(m.group(2))
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        ci = None
        if cells and cells[-1].startswith("["):
            ci = [float(x) for x in cells[-1].strip("[]").split(",")]
        published[eta] = {"value": value, "ci": ci}
    return published


# --- statistic --------------------------------------------------------------

def chosen_scores(rows: list, pre: dict) -> np.ndarray:
    """(N, 4) verifier scores on the RECORDED-preference direction of each row.

    Direction selection follows src/consensus_filter_precomputed.py:52-55 and
    common.chosen_results: verifier_results_a when user_choice == "A", else _b.
    """
    out = np.empty((len(rows), len(ROLES)), dtype=float)
    for i, r in enumerate(rows):
        p = pre[r["id"]]
        vr = p["verifier_results_a"] if r["user_choice"] == "A" else p["verifier_results_b"]
        missing = [role for role in ROLES if role not in vr]
        if missing:
            raise SystemExit(f"id={r['id']}: precomputed row is missing roles {missing}")
        for j, role in enumerate(ROLES):
            out[i, j] = float(vr[role]["score"])
    return out


def contamination_and_survival(keep: np.ndarray, corrupt: np.ndarray):
    """(admitted contamination, true harmful survival) for one admission mask.

    admitted contamination = kept_corrupt / n_kept        (metric_definitions 1.2)
    true harmful survival  = kept_corrupt / n_corrupt     (metric_definitions 1.3)
    Both fall back to 0.0 on an empty denominator, matching
    src/consensus_filter_precomputed.py:107 and :98.
    """
    n_kept = int(keep.sum())
    n_corrupt = int(corrupt.sum())
    kept_corrupt = int(np.count_nonzero(keep & corrupt))
    alpha = kept_corrupt / n_kept if n_kept else 0.0
    surv = kept_corrupt / n_corrupt if n_corrupt else 0.0
    return alpha, surv


def paired_bootstrap(keep_cons: np.ndarray, corrupt: np.ndarray, n_resamples: int,
                     seed: int, rng_kind: str):
    """Paired example-level bootstrap of the two Raw-minus-consensus differences.

    Raw DPO admits every example, so within a replicate its admission mask is all
    ones over the SAME resampled index set used for consensus; both arms are
    recomputed per replicate rather than differenced across independent draws.
    """
    n = len(corrupt)
    keep_raw = np.ones(n, dtype=bool)
    if rng_kind == "legacy":
        rng = np.random.RandomState(seed)
        draw = lambda: rng.randint(0, n, size=n)  # noqa: E731
    else:
        rng = np.random.default_rng(seed)
        draw = lambda: rng.integers(0, n, size=n)  # noqa: E731

    d_alpha = np.empty(n_resamples, dtype=float)
    d_surv = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        idx = draw()
        c_b = corrupt[idx]
        a_raw, s_raw = contamination_and_survival(keep_raw[idx], c_b)
        a_con, s_con = contamination_and_survival(keep_cons[idx], c_b)
        d_alpha[b] = a_raw - a_con
        d_surv[b] = s_raw - s_con
    return d_alpha, d_surv


def summarize(point: float, reps: np.ndarray) -> dict:
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {
        "point": float(point),
        "ci95": [float(lo), float(hi)],
        "bootstrap_mean": float(reps.mean()),
        "bootstrap_se": float(reps.std(ddof=1)),
        "bias": float(reps.mean() - point),
    }


# --- driver -----------------------------------------------------------------

def run_eta(eta: int, regime: str, pre: dict, pre_path: Path, args) -> dict:
    corrupted_path = VSE / f"data/corrupted/hh_train_{regime}_eta{eta}.jsonl"
    all_rows = load_jsonl(corrupted_path)
    rows = [r for r in all_rows if r["id"] in pre]
    n_missing = len(all_rows) - len(rows)
    if not rows:
        raise SystemExit(f"eta={eta}: no corrupted row ids found in {pre_path}")

    scores = chosen_scores(rows, pre)
    corrupt = np.array([not bool(r.get("is_clean", True)) for r in rows])
    keep_cons = (scores >= args.threshold).sum(axis=1) >= args.k

    a_raw, s_raw = contamination_and_survival(np.ones(len(rows), dtype=bool), corrupt)
    a_con, s_con = contamination_and_survival(keep_cons, corrupt)
    d_alpha_reps, d_surv_reps = paired_bootstrap(
        keep_cons, corrupt, args.n_resamples, args.seed, args.rng)

    # effective eta as recorded for the whole file, for reference against the
    # analysed subset (they differ only when the precompute is incomplete)
    summary_path = corrupted_path.with_suffix("").with_suffix(".summary.json")
    file_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    n_corrupt = int(corrupt.sum())
    n_kept = int(keep_cons.sum())
    return {
        "eta_nominal": eta,
        "regime": regime,
        "admission_rule": {"method": "consensus_k3", "k": args.k,
                           "threshold": args.threshold, "roles": ROLES,
                           "direction": "recorded preference (user_choice)"},
        "counts": {
            "n_total": len(rows),
            "n_corrupt": n_corrupt,
            "n_kept": n_kept,
            "kept_corrupt": int(np.count_nonzero(keep_cons & corrupt)),
            "n_rows_in_corrupted_file": len(all_rows),
            "n_missing_from_precomputed": n_missing,
        },
        "effective_eta": a_raw,
        "effective_eta_full_file": file_summary.get("effective_eta"),
        "raw": {"admitted_contamination": a_raw, "true_harmful_survival": s_raw,
                "retention": 1.0},
        "consensus_k3": {"admitted_contamination": a_con, "true_harmful_survival": s_con,
                         "retention": n_kept / len(rows)},
        "gap_admitted_contamination": summarize(a_raw - a_con, d_alpha_reps),
        "gap_true_harmful_survival": summarize(s_raw - s_con, d_surv_reps),
        "n_resamples": args.n_resamples,
        "seed": args.seed,
        "rng": args.rng,
        "inputs": {
            "corrupted": {"path": str(corrupted_path.relative_to(VSE)),
                          "sha256": sha256(corrupted_path)},
            "precomputed": {"path": str(pre_path),
                            "sha256": sha256(pre_path)},
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precomputed",
                    default=str(VSE / "results/verifier_outputs/hh_train.precomputed.jsonl"),
                    help="precomputed both-direction verifier scores (jsonl)")
    ap.add_argument("--regime", default="structured_unsafe")
    ap.add_argument("--etas", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--k", type=int, default=3, help="consensus threshold, roles passing")
    ap.add_argument("--threshold", type=float, default=0.7, help="per-role score cutoff")
    ap.add_argument("--n-resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rng", choices=["pcg64", "legacy"], default="pcg64",
                    help="pcg64 = np.random.default_rng(seed); legacy = "
                         "np.random.RandomState(seed). The generator used for the "
                         "published interval is not recorded, so both are offered.")
    ap.add_argument("--manifest", default=str(VSE / "analysis/EVIDENCE_MANIFEST.md"))
    ap.add_argument("--out", default=str(OUT / "e1_paired_bootstrap.json"))
    args = ap.parse_args()

    pre_path = Path(args.precomputed).resolve()
    if not pre_path.exists():
        raise SystemExit(f"precomputed file not found: {pre_path}")
    pre = load_precomputed(pre_path)
    print(f"[in] precomputed  {pre_path}  ({len(pre)} ids)")

    results = {}
    for eta in args.etas:
        r = run_eta(eta, args.regime, pre, pre_path, args)
        results[str(eta)] = r
        c = r["counts"]
        print(f"\n== eta={eta}% ({args.regime}) ==")
        if c["n_missing_from_precomputed"]:
            print(f"  PARTIAL PRECOMPUTE: {c['n_missing_from_precomputed']} of "
                  f"{c['n_rows_in_corrupted_file']} rows have no verifier scores and "
                  f"were dropped. All numbers below are on the {c['n_total']} joined "
                  f"rows only and are NOT comparable to the published values.")
        print(f"  n_total={c['n_total']}  n_corrupt={c['n_corrupt']}  "
              f"n_kept={c['n_kept']}  kept_corrupt={c['kept_corrupt']}")
        print(f"  effective_eta (analysed set) = {r['effective_eta']:.4f}"
              + (f"   full-file = {r['effective_eta_full_file']}"
                 if r["effective_eta_full_file"] is not None else ""))
        print(f"  admitted contamination:  raw={r['raw']['admitted_contamination']:.4f}  "
              f"consensus_k3={r['consensus_k3']['admitted_contamination']:.4f}")
        print(f"  true harmful survival:   raw={r['raw']['true_harmful_survival']:.4f}  "
              f"consensus_k3={r['consensus_k3']['true_harmful_survival']:.4f}")
        for key, label in (("gap_admitted_contamination", "gap admitted contamination"),
                           ("gap_true_harmful_survival", "gap true harmful survival")):
            g = r[key]
            print(f"  {label:28s} = {g['point']:.4f}  "
                  f"95% CI [{g['ci95'][0]:.4f}, {g['ci95'][1]:.4f}]  "
                  f"(se {g['bootstrap_se']:.4f}, bias {g['bias']:+.4f})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Paired example-level bootstrap of Raw DPO minus consensus "
                       "(k=3 of 4) on admitted-data contamination and on true "
                       "harmful survival. Definitions: discussion/metric_definitions.md "
                       "sections 1.2 and 1.3.",
        "generated_by": "analysis/scripts/paired_bootstrap.py",
        "n_resamples": args.n_resamples,
        "seed": args.seed,
        "rng": args.rng,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n[saved] {out_path}")

    # --- regenerated vs published ------------------------------------------
    published = read_manifest_published(Path(args.manifest))
    print("\n== regenerated vs EVIDENCE_MANIFEST.md (admitted-contamination gap) ==")
    hdr = (f"{'eta':>5} {'published':>10} {'regen':>10} {'|dev|':>8}  "
           f"{'published CI':>20} {'regenerated CI':>20} {'|dev| lo/hi':>16}")
    print(hdr)
    print("-" * len(hdr))
    any_partial = False
    for eta in args.etas:
        r = results[str(eta)]
        any_partial |= bool(r["counts"]["n_missing_from_precomputed"])
        g = r["gap_admitted_contamination"]
        p = published.get(eta)
        if p is None:
            print(f"{eta:>5} {'n/a':>10} {g['point']:>10.4f} {'--':>8}  "
                  f"{'(not in manifest)':>20} "
                  f"[{g['ci95'][0]:.4f}, {g['ci95'][1]:.4f}]")
            continue
        dev = abs(g["point"] - p["value"])
        pci = p["ci"]
        if pci:
            dlo, dhi = abs(g["ci95"][0] - pci[0]), abs(g["ci95"][1] - pci[1])
            pci_s = f"[{pci[0]:.4f}, {pci[1]:.4f}]"
            dev_s = f"{dlo:.4f}/{dhi:.4f}"
        else:
            pci_s, dev_s = "n/a", "n/a"
        rci_s = f"[{g['ci95'][0]:.4f}, {g['ci95'][1]:.4f}]"
        print(f"{eta:>5} {p['value']:>10.4f} {g['point']:>10.4f} {dev:>8.4f}  "
              f"{pci_s:>20} {rci_s:>20} {dev_s:>16}")
    if any_partial:
        print("\nNOTE: at least one eta was computed on a PARTIAL precompute. The "
              "deviations above are expected to be nonzero and say nothing about "
              "whether the published interval reproduces. Rerun against the complete "
              "results/verifier_outputs/hh_train.precomputed.jsonl before citing.")
    else:
        print("\nBoth etas were computed on the full precompute. A nonzero deviation "
              "here is a real regeneration failure, not a data-coverage artifact.")
    print("\nThe true-harmful-survival gap has no published counterpart in the "
          "manifest; it is reported here for the first time.")


if __name__ == "__main__":
    main()
