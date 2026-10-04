"""Construct a blinded, stratified sample for the independent rejection-set audit.

Replaces the previous rejection-set analysis, which was
circular: the pool that rejected the examples also characterised them. This
script only selects and blinds examples. The audit labels must come from
annotators or judge models that are not members of the filtering pool.

Two files are written:

  analysis/rejection_audit_sample.csv    given to annotators. Contains the
                                         prompt, the two responses in a
                                         randomised display order, and the
                                         recorded preference. It does NOT
                                         contain the injected-corruption status,
                                         the filter outcome, the verifier votes,
                                         or the method name.

  analysis/rejection_audit_key.csv       withheld from annotators. Maps the
                                         audit id back to stratum, injected
                                         status, vote count, and display order.

Usage (after verifier scores and filter outputs exist):
    python3 analysis/scripts/build_rejection_audit_sample.py --eta 20 --n 200
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THRESHOLD = 0.7
K = 3
ROLES = ["safety", "helpfulness", "factuality", "policy"]

# Stratum -> target share of the sample. Rejected non-flipped examples carry the
# most weight because they are the population the objection is about.
TARGETS = {
    "rejected_flipped": 0.15,
    "rejected_not_flipped": 0.40,
    "accepted_flipped": 0.15,
    "accepted_not_flipped": 0.10,
    "rejected_unanimous": 0.10,
    "rejected_split_vote": 0.05,
    "near_threshold": 0.05,
}


def n_pass(row: dict) -> int:
    """Passing verifiers on the recorded-preference direction."""
    key = "verifier_results_a" if row["user_choice"] == "A" else "verifier_results_b"
    return sum(1 for v in row[key].values() if v.get("score", 0.0) >= THRESHOLD)


def scores(row: dict) -> list[float]:
    key = "verifier_results_a" if row["user_choice"] == "A" else "verifier_results_b"
    return [row[key][r]["score"] for r in ROLES if r in row[key]]


def assign_strata(rows: list[dict]) -> dict[str, list[dict]]:
    strata: dict[str, list[dict]] = {k: [] for k in TARGETS}
    for r in rows:
        npass = n_pass(r)
        accepted = npass >= K
        flipped = not r.get("is_clean", True)
        if accepted:
            strata["accepted_flipped" if flipped else "accepted_not_flipped"].append(r)
        else:
            strata["rejected_flipped" if flipped else "rejected_not_flipped"].append(r)
            if npass == 0:
                strata["rejected_unanimous"].append(r)
            elif npass in (1, 2):
                strata["rejected_split_vote"].append(r)
        s = scores(r)
        if s and any(0.65 <= x < 0.75 for x in s):
            strata["near_threshold"].append(r)
    return strata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=int, default=20, choices=[0, 10, 20])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument(
        "--precomputed",
        default="results/verifier_outputs/hh_train.precomputed.jsonl",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)

    corrupted = ROOT / f"data/corrupted/hh_train_structured_unsafe_eta{args.eta}.jsonl"
    corrupt_rows = [json.loads(l) for l in open(corrupted)]
    pre = {}
    for line in open(ROOT / args.precomputed):
        d = json.loads(line)
        pre[d["id"]] = d

    missing = [r["id"] for r in corrupt_rows if r["id"] not in pre]
    if missing:
        raise SystemExit(
            f"{len(missing)} rows have no precomputed verifier scores "
            f"(first: {missing[:3]}). Complete the precompute before sampling."
        )

    # Merge the corrupted labels with the precomputed both-direction scores.
    rows = []
    for r in corrupt_rows:
        p = pre[r["id"]]
        rows.append({**r,
                     "verifier_results_a": p["verifier_results_a"],
                     "verifier_results_b": p["verifier_results_b"]})

    strata = assign_strata(rows)
    print(f"stratum sizes at eta={args.eta}%:")
    for k, v in strata.items():
        print(f"  {k:<24} {len(v)}")

    # Sample without replacement across strata, deduplicating by id.
    chosen: dict[str, dict] = {}
    stratum_of: dict[str, str] = {}
    for name, share in TARGETS.items():
        pool = [r for r in strata[name] if r["id"] not in chosen]
        want = min(round(args.n * share), len(pool))
        for r in rng.sample(pool, want):
            chosen[r["id"]] = r
            stratum_of[r["id"]] = name

    # Top up from rejected_not_flipped if rounding left the sample short.
    if len(chosen) < args.n:
        pool = [r for r in strata["rejected_not_flipped"] if r["id"] not in chosen]
        for r in rng.sample(pool, min(args.n - len(chosen), len(pool))):
            chosen[r["id"]] = r
            stratum_of[r["id"]] = "rejected_not_flipped"

    items = list(chosen.values())
    rng.shuffle(items)

    sample_path = ROOT / "analysis/rejection_audit_sample.csv"
    key_path = ROOT / "analysis/rejection_audit_key.csv"

    with open(sample_path, "w", newline="") as fs, open(key_path, "w", newline="") as fk:
        ws = csv.writer(fs)
        wk = csv.writer(fk)
        ws.writerow([
            "audit_id", "prompt", "response_1", "response_2",
            "recorded_preference",
            "label_direction",          # correct / wrong / ambiguous
            "label_both_acceptable", "label_both_poor", "label_insufficient_info",
            "label_rejection_is_safety_preserving", "label_harmful_false_rejection",
            "confidence", "notes",
        ])
        wk.writerow([
            "audit_id", "example_id", "stratum", "injected_flip",
            "verifiers_passing", "display_order", "recorded_preference_is_response",
        ])
        for i, r in enumerate(items):
            aid = f"A{i:04d}"
            # Randomise which response is shown first so display order carries
            # no information about the recorded preference.
            swap = rng.random() < 0.5
            r1, r2 = (r["response_b"], r["response_a"]) if swap else (r["response_a"], r["response_b"])
            pref_letter = r["user_choice"]                       # "A" or "B"
            pref_is_a = pref_letter == "A"
            shown = ("response_2" if pref_is_a else "response_1") if swap else \
                    ("response_1" if pref_is_a else "response_2")
            ws.writerow([aid, r["prompt"], r1, r2, shown,
                         "", "", "", "", "", "", "", ""])
            wk.writerow([aid, r["id"], stratum_of[r["id"]],
                         "yes" if not r.get("is_clean", True) else "no",
                         n_pass(r), "swapped" if swap else "original", shown])

    print(f"\nwrote {sample_path.relative_to(ROOT)} ({len(items)} examples, blinded)")
    print(f"wrote {key_path.relative_to(ROOT)} (withhold from annotators)")


if __name__ == "__main__":
    main()
