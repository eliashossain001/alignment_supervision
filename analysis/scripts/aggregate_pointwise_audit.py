"""Aggregate and then unblind the independent pointwise audit.

Two stages, deliberately separated so that the claim "audit labels were fixed
before the corruption labels were consulted" is checkable from artifacts rather
than taken on trust.

  stage1  reads ONLY the two judges' derived preference files, forms the strict
          cross-family label, writes the blinded aggregate, and records a freeze
          file containing its sha256 and the sha256 of each input.
  stage2  verifies those hashes, then joins the hidden key and the corruption
          labels and reports per-stratum estimates.

Usage:
    python3 analysis/scripts/aggregate_pointwise_audit.py stage1
    python3 analysis/scripts/aggregate_pointwise_audit.py stage2
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
JUDGES = ("llama", "phi")
ETA = 20
N_BOOT = 2000
SEED = 42

DERIVED = {j: ROOT / f"analysis/model_audit_{j}_pointwise_derived.jsonl" for j in JUDGES}
BLINDED_OUT = ROOT / "analysis/model_audit_aggregated_blinded.csv"
FREEZE = ROOT / "analysis/model_audit_freeze.json"
KEY = ROOT / "analysis/rejection_audit_key.csv"
CORRUPTED = ROOT / f"data/corrupted/hh_train_structured_unsafe_eta{ETA}.jsonl"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_derived():
    out = {}
    for j, p in DERIVED.items():
        out[j] = {}
        for line in open(p):
            d = json.loads(line)
            out[j][d["audit_id"]] = d
    return out


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959963985
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - m) / d, (c + m) / d)


def stage1() -> int:
    der = load_derived()
    ids = sorted(set(der["llama"]) & set(der["phi"]))
    rows = []
    for aid in ids:
        l, p = der["llama"][aid], der["phi"][aid]
        lp = l["preferred_option"] if l.get("derivable") else None
        pp = p["preferred_option"] if p.get("derivable") else None
        if lp is None or pp is None:
            strict = "underivable"
        elif lp == pp:
            strict = lp
        else:
            strict = "unresolved"
        # Conservative confidence screen: minimum across both judges and both
        # scored responses, so a label is only "high confidence" when nothing
        # in the four underlying judgements was hedged.
        confs = []
        for side in (l, p):
            for k in ("option_1_score", "option_2_score"):
                s = side.get(k) or {}
                if isinstance(s.get("confidence"), int):
                    confs.append(s["confidence"])
        rows.append({
            "audit_id": aid,
            "llama_pref": lp, "phi_pref": pp,
            "strict_label": strict,
            "min_confidence": min(confs) if confs else "",
            "llama_margin": l.get("quality_margin", ""),
            "phi_margin": p.get("quality_margin", ""),
        })
    with open(BLINDED_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    freeze = {
        "stage": "1 (blinded)",
        "n_examples": len(rows),
        "inputs": {j: {"path": str(DERIVED[j].relative_to(ROOT)),
                       "sha256": sha256(DERIVED[j]),
                       "mtime": DERIVED[j].stat().st_mtime} for j in JUDGES},
        "aggregated_blinded": {"path": str(BLINDED_OUT.relative_to(ROOT)),
                               "sha256": sha256(BLINDED_OUT)},
        "note": "Written before any corruption label, filter decision or verifier "
                "score was joined. stage2 refuses to run if these hashes change.",
    }
    FREEZE.write_text(json.dumps(freeze, indent=2))

    from collections import Counter
    c = Counter(r["strict_label"] for r in rows)
    print(f"stage1: {len(rows)} examples")
    print("  strict label distribution:", dict(c))
    for j in JUDGES:
        jc = Counter(r[f"{j}_pref"] for r in rows)
        print(f"  {j:<6} preference distribution:", dict(jc))
    agree = sum(1 for r in rows if r["llama_pref"] == r["phi_pref"] and r["llama_pref"])
    both_dec = [r for r in rows if r["llama_pref"] in ("option_1", "option_2")
                and r["phi_pref"] in ("option_1", "option_2")]
    dec_agree = sum(1 for r in both_dec if r["llama_pref"] == r["phi_pref"])
    print(f"  raw cross-family agreement: {agree}/{len(rows)} = {agree/len(rows):.3f}")
    print(f"  both judges decisive: {len(both_dec)}; agreement among them: "
          f"{dec_agree}/{len(both_dec)}" + (f" = {dec_agree/len(both_dec):.3f}" if both_dec else ""))
    print(f"  freeze sha256: {freeze['aggregated_blinded']['sha256'][:16]}")
    print(f"[wrote] {BLINDED_OUT.relative_to(ROOT)}")
    print(f"[wrote] {FREEZE.relative_to(ROOT)}")
    return 0


def stage2() -> int:
    if not FREEZE.exists():
        print("ERROR: freeze file missing; run stage1 first.", file=sys.stderr)
        return 3
    fz = json.loads(FREEZE.read_text())
    if sha256(BLINDED_OUT) != fz["aggregated_blinded"]["sha256"]:
        print("ERROR: aggregated blinded file changed since freeze.", file=sys.stderr)
        return 4
    for j in JUDGES:
        if sha256(DERIVED[j]) != fz["inputs"][j]["sha256"]:
            print(f"ERROR: raw judge file {j} changed since freeze.", file=sys.stderr)
            return 4

    agg = {r["audit_id"]: r for r in csv.DictReader(open(BLINDED_OUT))}
    key = {r["audit_id"]: r for r in csv.DictReader(open(KEY))}
    corr = {}
    for line in open(CORRUPTED):
        d = json.loads(line)
        corr[d["id"]] = d

    recs = []
    for aid, a in agg.items():
        k = key[aid]
        row = corr[k["example_id"]]
        # user_choice "A" means option_1 (response_a) is the recorded preference.
        recorded = "option_1" if row["user_choice"] == "A" else "option_2"
        flipped = not row.get("is_clean", True)
        accepted = int(k["verifiers_passing"]) >= 3
        s = a["strict_label"]
        if s in ("unresolved", "underivable"):
            valid = s
        elif s == "tie":
            valid = "uncertain"
        elif s == recorded:
            valid = "yes"
        else:
            valid = "no"
        recs.append({**a, "example_id": k["example_id"], "stratum": k["stratum"],
                     "injected_flip": "yes" if flipped else "no",
                     "accepted": "yes" if accepted else "no",
                     "recorded_option": recorded,
                     "original_preference_valid": valid})

    out = ROOT / "analysis/model_audit_hidden_join.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)

    groups = {
        "rejected_not_flipped": [r for r in recs if r["accepted"] == "no" and r["injected_flip"] == "no"],
        "rejected_flipped":     [r for r in recs if r["accepted"] == "no" and r["injected_flip"] == "yes"],
        "accepted_flipped":     [r for r in recs if r["accepted"] == "yes" and r["injected_flip"] == "yes"],
        "accepted_not_flipped": [r for r in recs if r["accepted"] == "yes" and r["injected_flip"] == "no"],
    }
    lines = ["# Independent pointwise model audit: results", "",
             "This is an independent MODEL-BASED audit and explicitly not human ground truth.",
             "Judges are Llama-3.1-8B-Instruct and Phi-3.5-mini-instruct, neither of which",
             "appears in the shared-backbone pool, the cross-backbone pool, or as the trained",
             "policy. Each response was scored alone, with the other response never present in",
             "context, so no display order exists and position bias is removed by construction.",
             "Labels were frozen and hashed before the corruption labels were joined.", "",
             "| Group | n | judged valid | judged invalid | tie/uncertain | unresolved |",
             "|---|---:|---|---|---|---|"]
    print(f"\nstage2: {len(recs)} examples joined")
    for name, g in groups.items():
        n = len(g)
        if n == 0:
            lines.append(f"| {name} | 0 | | | | |"); continue
        def pct(pred):
            k_ = sum(1 for r in g if pred(r))
            lo, hi = wilson(k_, n)
            return f"{k_}/{n} = {k_/n:.3f} [{lo:.3f}, {hi:.3f}]"
        v = pct(lambda r: r["original_preference_valid"] == "yes")
        iv = pct(lambda r: r["original_preference_valid"] == "no")
        un = pct(lambda r: r["original_preference_valid"] == "uncertain")
        ur = pct(lambda r: r["original_preference_valid"] in ("unresolved", "underivable"))
        lines.append(f"| {name} | {n} | {v} | {iv} | {un} | {ur} |")
        print(f"  {name:<22} n={n:<4} valid={v}  invalid={iv}")
    (ROOT / "analysis/model_audit_results.md").write_text("\n".join(lines) + "\n")
    with open(ROOT / "analysis/model_audit_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n", "valid", "invalid", "uncertain", "unresolved"])
        for name, g in groups.items():
            n = len(g)
            counts = [sum(1 for r in g if r["original_preference_valid"] == v)
                      for v in ("yes", "no", "uncertain")]
            # "unresolved" must also absorb "underivable" (a judge produced no
            # derivable preference), otherwise the row does not sum to n and the
            # CSV silently disagrees with the markdown.
            unres = sum(1 for r in g
                        if r["original_preference_valid"] in ("unresolved", "underivable"))
            row = [name, n] + counts + [unres]
            assert sum(row[2:]) == n, f"{name}: counts {row[2:]} do not sum to n={n}"
            w.writerow(row)
    print(f"[wrote] analysis/model_audit_hidden_join.csv")
    print(f"[wrote] analysis/model_audit_results.md and .csv")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit(stage1() if cmd == "stage1" else stage2() if cmd == "stage2" else
             (print(__doc__) or 2))
