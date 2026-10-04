"""Build the blinded inputs for the independent model-based rejection audit.

Reuses the existing stratified 200-example sample (analysis/rejection_audit_key.csv)
so the stratification is unchanged. Produces:

  analysis/model_audit_sample_blinded.csv   what the judges see (no status fields)
  analysis/model_audit_randomization.json   per (judge, run) A/B order, with hashes

Design note. The judges are never shown the recorded preference. They state which
response they prefer, and `original_preference_valid` is derived afterwards by
comparing their preference against the recorded one. Showing the recorded
preference and asking whether it is valid would anchor the judge on that answer.
For the same reason the judges are not asked whether rejecting a pair would be
safety preserving, since that question reveals that a rejection decision exists.
Both fields are derived at aggregation time and are documented as derived.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ETA = 20
JUDGES = ["llama", "phi"]
RUNS = [1, 2]
# Independent seeds per (judge, run) so no two passes share an order.
SEEDS = {("llama", 1): 20260801, ("llama", 2): 20260802,
         ("phi", 1): 20260803, ("phi", 2): 20260804}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    key_path = ROOT / "analysis/rejection_audit_key.csv"
    with open(key_path) as f:
        key = list(csv.DictReader(f))
    audit_ids = [r["audit_id"] for r in key]
    id_of = {r["audit_id"]: r["example_id"] for r in key}

    corrupted = {}
    with open(ROOT / f"data/corrupted/hh_train_structured_unsafe_eta{ETA}.jsonl") as f:
        for line in f:
            d = json.loads(line)
            corrupted[d["id"]] = d

    # Blinded sample: canonical dataset order, no status fields of any kind.
    out_path = ROOT / "analysis/model_audit_sample_blinded.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "prompt", "option_1", "option_2"])
        for aid in audit_ids:
            row = corrupted[id_of[aid]]
            w.writerow([aid, row["prompt"], row["response_a"], row["response_b"]])

    # Per (judge, run) display order: does option_1 appear as displayed "A"?
    #
    # Run 1 is randomised per judge. Run 2 is the EXACT INVERSE of run 1, so
    # every example is seen by every judge in both display orders. The pilot
    # showed this is essential: with independently randomised runs only about
    # half of the examples happened to receive opposite orders, and on those the
    # Llama judge tracked the displayed letter on 6 of 6 while agreeing on
    # content 0 of 6. Guaranteeing both orders converts position dependence into
    # a per-example measurement instead of an unmeasured contaminant, since a
    # judge whose preference follows the letter now disagrees with itself and is
    # collapsed to "inconsistent" rather than contributing a spurious label.
    rand = {}
    for judge in JUDGES:
        rng = random.Random(SEEDS[(judge, 1)])
        first = {aid: ("A" if rng.random() < 0.5 else "B") for aid in audit_ids}
        rand[f"{judge}_run1"] = {
            "seed": SEEDS[(judge, 1)],
            "order_policy": "randomised",
            "option_1_shown_as": first,
        }
        rand[f"{judge}_run2"] = {
            "seed": SEEDS[(judge, 1)],
            "order_policy": "exact inverse of run 1",
            "option_1_shown_as": {aid: ("B" if v == "A" else "A")
                                  for aid, v in first.items()},
        }

    # Position-balance check: each pass should be near 50/50.
    for k, v in rand.items():
        n_a = sum(1 for x in v["option_1_shown_as"].values() if x == "A")
        v["option_1_shown_as_A_count"] = n_a
        v["n"] = len(audit_ids)

    payload = {
        "created": "2026-08-01",
        "eta_percent": ETA,
        "n_examples": len(audit_ids),
        "judges": JUDGES,
        "runs_per_judge": len(RUNS),
        "blinded_sample": {
            "path": "analysis/model_audit_sample_blinded.csv",
            "sha256": sha256_file(out_path),
        },
        "source_key": {
            "path": "analysis/rejection_audit_key.csv",
            "sha256": sha256_file(key_path),
            "note": "hidden key, never shown to judges, joined only after audit labels are frozen",
        },
        "randomization": rand,
    }
    (ROOT / "analysis/model_audit_randomization.json").write_text(
        json.dumps(payload, indent=2)
    )

    print(f"wrote {out_path.relative_to(ROOT)} ({len(audit_ids)} examples)")
    print("wrote analysis/model_audit_randomization.json")
    for k, v in rand.items():
        print(f"  {k:<12} option_1 shown as A in {v['option_1_shown_as_A_count']}/{v['n']}")
    print(f"blinded sample sha256: {payload['blinded_sample']['sha256'][:16]}")


if __name__ == "__main__":
    main()
